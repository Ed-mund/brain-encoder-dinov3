"""
Training script: Universal Brain Encoder with Hugging Face DINOv3 backbone.

Uses AutoImageProcessor for preprocessing (do not use ImageNet Normalize from
get_default_transform for this entrypoint).

Example:
    python train.py --data_root /path/to/algonauts --subjects subj01 \\
        --hf_model_id facebook/dinov3-vith16plus-pretrain-lvd1689m \\
        --use_bf16 --gradient_checkpointing
"""

import os
import sys
import argparse
import time
import json
import glob
import signal
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.amp import autocast, GradScaler
from torchvision import transforms
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image as PILImage

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from transformers import AutoImageProcessor

_DINO_DIR = Path(__file__).resolve().parent
_UB_DIR = _DINO_DIR.parent / "universal_brain_encoder"
if str(_UB_DIR) not in sys.path:
    sys.path.insert(0, str(_UB_DIR))

from dataset import (
    NSDAlgonautsDataset,
    MultiSubjectDataset,
)

# ---------------------------------------------------------------------------
# SubjectBatchSampler
# ---------------------------------------------------------------------------

class SubjectBatchSampler(torch.utils.data.Sampler):
    """
    Yields batches where **all samples come from the same subject**.

    Why this matters:
      The BrainEncoderLoss cosine term is computed over the batch dimension
      (images).  With a flat multi-subject DataLoader and batch_size=32 across
      7 subjects, each subject only contributes ~4-5 images per batch — making
      the cosine similarity almost meaningless (nearly random over N=4).

      This sampler ensures every batch has exactly `batch_size` images from ONE
      subject, matching the paper's setup (Beliy et al. 2025: B=32 per subject).
      The gradient signal for the cosine loss improves from N≈4 to N=32.

    Each epoch:
      1. Shuffle each subject's index list independently.
      2. Chunk into full batches of `batch_size`.
      3. Shuffle the resulting batch list so subjects interleave randomly.
    """

    def __init__(
        self,
        multi_ds: MultiSubjectDataset,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0

        # Build per-subject lists of *global* flat indices
        self.subject_indices: Dict[str, List[int]] = {}
        for global_idx, (sid, _) in enumerate(multi_ds.flat_indices):
            self.subject_indices.setdefault(sid, []).append(global_idx)

    def set_epoch(self, epoch: int) -> None:
        """Call at the start of each epoch for correct per-epoch shuffling."""
        self._epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        all_batches: List[List[int]] = []
        for sid, indices in self.subject_indices.items():
            idx = np.array(indices)
            if self.shuffle:
                rng.shuffle(idx)
            n_full = len(idx) // self.batch_size
            for i in range(n_full):
                all_batches.append(idx[i * self.batch_size : (i + 1) * self.batch_size].tolist())
        if self.shuffle:
            perm = rng.permutation(len(all_batches))
            all_batches = [all_batches[i] for i in perm]
        for batch in all_batches:
            yield batch

    def __len__(self) -> int:
        return sum(len(idx) // self.batch_size for idx in self.subject_indices.values())

sys.path.insert(0, str(_DINO_DIR))
from model import UniversalBrainEncoderDINOv3, BrainEncoderLoss, count_parameters

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Set in main() after AutoImageProcessor loads (for W&B gallery denormalisation)
_IMAGE_PROCESSOR = None


def get_dinov3_raw_transform(image_size: int = 224) -> transforms.Compose:
    """Resize + ToTensor [0,1] only — HF processor applies DINOv3 normalisation."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


def make_collate_dinov3(processor) -> Callable:
    """
    Group by subject like collate_multi_subject, then apply AutoImageProcessor.

    We pass do_rescale=False because the dataset transform already outputs
    [0, 1] tensors, and do_resize=False because the dataset transform already
    resizes to the target image_size (which may be 448px for DINOv3's RoPE
    backbone).  The processor is still used for its mean/std normalisation.

    Note: call processor.size = {"height": H, "width": W} in main() before
    creating this collate so that the processor knows the correct image size
    even though it won't resize.
    """

    def collate(batch: List[dict]) -> dict:
        by_subject: Dict[str, List[dict]] = {}
        for item in batch:
            sid = item["subject_id"]
            by_subject.setdefault(sid, []).append(item)

        result = {}
        for sid, items in by_subject.items():
            stacked = torch.stack([it["image"] for it in items])
            pv = processor(
                images=stacked,
                return_tensors="pt",
                do_rescale=False,   # already [0,1] from ToTensor()
                do_resize=False,    # already resized in dataset transform
            )["pixel_values"]
            result[sid] = {
                "images": pv,
                "fmri": torch.stack([it["fmri"] for it in items]) if "fmri" in items[0] else None,
                "subject_id": sid,
            }
        return result

    return collate


def _to_pil(tensor: torch.Tensor) -> PILImage.Image:
    """(3,H,W) processor-normalised tensor → approximate RGB PIL for W&B."""
    t = tensor.detach().float().cpu()
    proc = _IMAGE_PROCESSOR
    if proc is not None and hasattr(proc, "image_mean") and hasattr(proc, "image_std"):
        mean = torch.tensor(proc.image_mean, dtype=t.dtype).view(3, 1, 1)
        std = torch.tensor(proc.image_std, dtype=t.dtype).view(3, 1, 1)
        t = t * std + mean
    t = t.clamp(0, 1)
    arr = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return PILImage.fromarray(arr)

# Set by SIGUSR1 (e.g. 5 min before SLURM 12h limit); training loop saves and exits.
_preemption_requested = False


def _preemption_handler(signum, frame):
    global _preemption_requested
    _preemption_requested = True
    logger.warning("Preemption signal received; will save checkpoint and exit after current epoch.")


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: UniversalBrainEncoderDINOv3,
    dataloader: DataLoader,
    criterion: BrainEncoderLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    voxels_per_image: int = 5000,
    epoch: int = 0,
    global_step: int = 0,
    use_wandb: bool = False,
    log_every: int = 50,
    use_bf16: bool = False,
    max_grad_norm: float = 1.0,
) -> Tuple[dict, int]:
    """Train for one epoch over all subjects. Returns (metrics, updated_global_step)."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()

        batch_loss = 0.0
        num_subject_batches = 0

        for sid, subj_data in batch.items():
            pixel_values = subj_data["images"].to(device)
            fmri = subj_data["fmri"].to(device)  # (B, num_voxels)
            B, num_voxels = fmri.shape

            n_sample = min(voxels_per_image, num_voxels)
            voxel_idx = torch.randperm(num_voxels, device=device)[:n_sample]
            gt = fmri[:, voxel_idx].T.float()  # (N, B)

            if use_bf16:
                with autocast("cuda", dtype=torch.bfloat16):
                    pred = model(pixel_values, sid, voxel_idx)
                    loss = criterion(pred.float(), gt)
            elif scaler is not None:
                with autocast("cuda"):
                    pred = model(pixel_values, sid, voxel_idx)
                    loss = criterion(pred, gt)
            else:
                pred = model(pixel_values, sid, voxel_idx)
                loss = criterion(pred, gt)

            batch_loss += loss
            num_subject_batches += 1

        batch_loss = batch_loss / max(num_subject_batches, 1)

        if scaler is not None:
            scaler.scale(batch_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_grad_norm,
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_grad_norm,
            )
            optimizer.step()

        loss_val = batch_loss.item()
        total_loss += loss_val
        num_batches += 1
        global_step += 1

        if batch_idx % log_every == 0:
            logger.info(
                f"  Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | "
                f"Loss: {loss_val:.4f}"
            )
            if use_wandb and _WANDB_AVAILABLE:
                wandb.log({"train/loss": loss_val, "train/step": global_step})

    avg_loss = total_loss / max(num_batches, 1)
    return {"loss": avg_loss}, global_step


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: UniversalBrainEncoderDINOv3,
    dataloader: DataLoader,
    device: torch.device,
    subject_id: str,
    collect_images: bool = False,
    max_gallery_images: int = 200,
) -> dict:
    """
    Evaluate encoding performance for a single subject.

    Metrics:
    - Pearson correlation per voxel (median, 25th, 75th percentile)
    - Image retrieval accuracy (Top-1, Top-5)

    When collect_images=True, also stores raw tensors needed for W&B visuals.
    """
    model.eval()

    all_pred = []
    all_gt = []
    all_images_cpu = []  # kept for W&B gallery

    for batch in dataloader:
        if isinstance(batch, dict) and subject_id in batch:
            subj_data = batch[subject_id]
        else:
            subj_data = batch

        pixel_values = subj_data["images"].to(device) if "images" in subj_data else subj_data["image"].unsqueeze(0).to(device)
        fmri = subj_data["fmri"].to(device) if "fmri" in subj_data else None

        if fmri is None:
            continue
        if fmri.dim() == 1:
            fmri = fmri.unsqueeze(0)

        pred = model.predict_all_voxels(pixel_values, subject_id, chunk_size=5000)  # (V, B)
        all_pred.append(pred.cpu())
        all_gt.append(fmri.T.cpu())  # (V, B)

        if collect_images and len(all_images_cpu) < max_gallery_images:
            all_images_cpu.append(pixel_values.cpu())

    if not all_pred:
        return {}

    all_pred = torch.cat(all_pred, dim=1)  # (V, N)
    all_gt   = torch.cat(all_gt,   dim=1)  # (V, N)

    V, N = all_pred.shape

    # --- Pearson correlation per voxel ---
    pred_centered = all_pred - all_pred.mean(dim=1, keepdim=True)
    gt_centered   = all_gt   - all_gt.mean(dim=1, keepdim=True)

    numerator   = (pred_centered * gt_centered).sum(dim=1)
    denominator = pred_centered.norm(dim=1) * gt_centered.norm(dim=1) + 1e-8
    correlations = numerator / denominator  # (V,)

    median_corr = correlations.median().item()
    p25_corr    = correlations.quantile(0.25).item()
    p75_corr    = correlations.quantile(0.75).item()

    # --- Degenerate prediction check ---
    # If predictions are constant across images (model collapse), pred_centered
    # has near-zero norm.  F.normalize would produce zero vectors, making every
    # sim_matrix row identical → rank=1 for all samples (spurious 100% Top-1).
    # Detect this and report a random-baseline rank instead.
    pred_std = pred_centered.norm(dim=1).mean().item()   # avg per-voxel std
    collapsed = pred_std < 1e-6

    # --- Image retrieval ---
    pred_norm = F.normalize(pred_centered, dim=0)  # (V, N)
    gt_norm   = F.normalize(gt_centered,   dim=0)  # (V, N)
    sim_matrix = gt_norm.T @ pred_norm  # (N, N)

    ranks = []
    if collapsed:
        # Predictions are degenerate constants — assign random-baseline ranks
        rng = np.random.default_rng(seed=0)
        ranks = rng.integers(1, N + 1, size=N).tolist()
        logger.warning(
            f"[{subject_id}] DEGENERATE PREDICTIONS (pred_std={pred_std:.2e}): "
            "model is outputting near-constant activations. "
            "Retrieval ranks set to random baseline."
        )
    else:
        for i in range(N):
            sims = sim_matrix[i]
            rank = (sims > sims[i]).sum().item() + 1
            ranks.append(rank)

    ranks = np.array(ranks)
    top1_acc  = (ranks == 1).mean() * 100
    top5_acc  = (ranks <= 5).mean() * 100
    mean_rank = ranks.mean()

    results = {
        "median_correlation": median_corr,
        "p25_correlation":    p25_corr,
        "p75_correlation":    p75_corr,
        "top1_accuracy":      top1_acc,
        "top5_accuracy":      top5_acc,
        "mean_rank":          mean_rank,
        "num_test_images":    N,
        "num_voxels":         V,
        # raw data for W&B visuals (None when collect_images=False)
        "_correlations": correlations,
        "_sim_matrix":   sim_matrix,
        "_ranks":        ranks,
        "_images":       torch.cat(all_images_cpu, dim=0)[:max_gallery_images] if all_images_cpu else None,
    }

    collapse_tag = " [COLLAPSED]" if collapsed else ""
    logger.info(
        f"[{subject_id}] Correlation: median={median_corr:.4f} "
        f"(25th={p25_corr:.4f}, 75th={p75_corr:.4f}) | "
        f"Retrieval: Top-1={top1_acc:.1f}%, Top-5={top5_acc:.1f}%, "
        f"Mean Rank={mean_rank:.2f}/{N}{collapse_tag}"
    )

    return results


# ---------------------------------------------------------------------------
# W&B visual logging
# ---------------------------------------------------------------------------

def _sim_matrix_figure(sim: np.ndarray, subject_id: str, epoch: int, n: int = 100) -> plt.Figure:
    """Render an NxN retrieval similarity matrix as a heatmap figure."""
    s = sim[:n, :n]
    fig, ax = plt.subplots(figsize=(7, 6), facecolor="#0d0d0d")
    ax.set_facecolor("#0d0d0d")
    im = ax.imshow(s, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    ax.set_title(f"{subject_id} — retrieval similarity matrix  (epoch {epoch})",
                 color="white", fontsize=11, pad=10)
    ax.set_xlabel("predicted fMRI index", color="white")
    ax.set_ylabel("ground-truth fMRI index", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    fig.tight_layout()
    return fig


def _correlation_figure(corrs: np.ndarray, subject_id: str, epoch: int) -> plt.Figure:
    """Per-voxel Pearson correlation distribution."""
    fig, ax = plt.subplots(figsize=(7, 4), facecolor="#0d0d0d")
    ax.set_facecolor("#111")
    n_bins = min(120, len(corrs) // 50)
    counts, edges, patches = ax.hist(corrs, bins=n_bins, color="#00c8ff", alpha=0.85, edgecolor="none")
    # Colour bars by value
    norm = mcolors.Normalize(vmin=edges[0], vmax=edges[-1])
    cmap = plt.cm.plasma
    for patch, left in zip(patches, edges[:-1]):
        patch.set_facecolor(cmap(norm(left)))
    ax.axvline(np.median(corrs), color="#ff6b6b", lw=1.8, ls="--",
               label=f"median = {np.median(corrs):.3f}")
    ax.axvline(0, color="#888", lw=1, ls=":")
    ax.legend(framealpha=0.2, labelcolor="white")
    ax.set_title(f"{subject_id} — per-voxel Pearson correlation  (epoch {epoch})",
                 color="white", fontsize=11)
    ax.set_xlabel("Pearson r", color="white")
    ax.set_ylabel("voxel count", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    fig.tight_layout()
    return fig


def _scatter_figure(pred: torch.Tensor, gt: torch.Tensor,
                    subject_id: str, epoch: int, n_voxels: int = 800) -> plt.Figure:
    """Predicted vs actual fMRI activations for a random voxel subset."""
    V = pred.shape[0]
    idx = np.random.choice(V, size=min(n_voxels, V), replace=False)
    # Flatten over images → scatter one point per (voxel, image) pair
    p = pred[idx].flatten().numpy()
    g = gt[idx].flatten().numpy()
    # Subsample further if needed
    if len(p) > 8000:
        sel = np.random.choice(len(p), 8000, replace=False)
        p, g = p[sel], g[sel]
    from scipy.stats import pearsonr
    r, _ = pearsonr(p, g) if len(p) > 1 else (0, 1)
    fig, ax = plt.subplots(figsize=(5, 5), facecolor="#0d0d0d")
    ax.set_facecolor("#111")
    ax.scatter(g, p, s=2, alpha=0.3, color="#a78bfa", rasterized=True)
    lim = max(abs(g).max(), abs(p).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], color="#ff6b6b", lw=1.2, ls="--", label="y=x")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_title(f"{subject_id} — pred vs actual  r={r:.3f}  (epoch {epoch})",
                 color="white", fontsize=10)
    ax.set_xlabel("actual fMRI (z-scored)", color="white")
    ax.set_ylabel("predicted fMRI",         color="white")
    ax.tick_params(colors="white")
    ax.legend(framealpha=0.2, labelcolor="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    fig.tight_layout()
    return fig


def _retrieval_table(
    images: torch.Tensor,    # (N, 3, H, W) normalised
    sim_matrix: torch.Tensor,  # (N, N)
    ranks: np.ndarray,
    n_rows: int = 48,
) -> "wandb.Table":
    """W&B Table: each row = query image + top-3 retrieved images + rank info."""
    N = images.shape[0]
    cols = ["query", "rank_1", "rank_2", "rank_3",
            "correct_rank", "top1 ✓", "top5 ✓"]
    table = wandb.Table(columns=cols)

    sample_idx = np.random.choice(N, size=min(n_rows, N), replace=False)
    for i in sample_idx:
        sims = sim_matrix[i]          # (N,)
        top3 = torch.topk(sims, k=min(3, N)).indices.tolist()
        row = [
            wandb.Image(_to_pil(images[i])),
            wandb.Image(_to_pil(images[top3[0]])),
            wandb.Image(_to_pil(images[top3[1]]) if len(top3) > 1 else PILImage.new("RGB", (4, 4))),
            wandb.Image(_to_pil(images[top3[2]]) if len(top3) > 2 else PILImage.new("RGB", (4, 4))),
            int(ranks[i]),
            bool(ranks[i] == 1),
            bool(ranks[i] <= 5),
        ]
        table.add_data(*row)
    return table


def log_eval_to_wandb(
    results: dict,
    subject_id: str,
    epoch: int,
    lr: float,
    n_sim: int = 100,
    gallery_rows: int = 48,
):
    """Push all evaluation visuals to W&B for one subject."""
    if not _WANDB_AVAILABLE:
        return

    prefix = f"eval/{subject_id}"
    corrs  = results["_correlations"].numpy()
    sim    = results["_sim_matrix"]
    ranks  = results["_ranks"]
    images = results.get("_images")

    log_dict = {
        f"{prefix}/median_correlation": results["median_correlation"],
        f"{prefix}/p25_correlation":    results["p25_correlation"],
        f"{prefix}/p75_correlation":    results["p75_correlation"],
        f"{prefix}/top1_accuracy":      results["top1_accuracy"],
        f"{prefix}/top5_accuracy":      results["top5_accuracy"],
        f"{prefix}/mean_rank":          results["mean_rank"],
        f"{prefix}/positive_voxels_%":  float((corrs > 0).mean() * 100),
        "train/lr": lr,
        "epoch": epoch,
    }

    # ---- Histogram (native W&B) ----
    log_dict[f"{prefix}/voxel_correlation_dist"] = wandb.Histogram(corrs)

    # ---- Matplotlib figures ----
    fig_sim   = _sim_matrix_figure(sim.numpy(), subject_id, epoch, n=min(n_sim, sim.shape[0]))
    fig_corr  = _correlation_figure(corrs, subject_id, epoch)
    fig_scat  = _scatter_figure(results["_sim_matrix"],  # reuse sim as proxy
                                results["_sim_matrix"],  # will be overridden below
                                subject_id, epoch)

    # Proper scatter using pred / gt (stored inside sim_matrix decomposition is
    # not available here, so we sample rows/cols of the similarity matrix instead)
    plt.close(fig_scat)  # discard proxy; build proper scatter from corr data
    # We reconstruct a lightweight scatter from the sim_matrix diagonal vs off-diagonal
    diag = np.diag(sim.numpy())
    off  = sim.numpy()[~np.eye(sim.shape[0], dtype=bool)]
    fig_scat2, ax2 = plt.subplots(figsize=(5, 4), facecolor="#0d0d0d")
    ax2.set_facecolor("#111")
    ax2.hist(off,  bins=80, color="#a78bfa", alpha=0.6, label="non-match", density=True)
    ax2.hist(diag, bins=40, color="#ff6b6b", alpha=0.85, label="correct match", density=True)
    ax2.set_title(f"{subject_id} — correct vs non-match similarity  (epoch {epoch})",
                  color="white", fontsize=10)
    ax2.set_xlabel("cosine similarity", color="white")
    ax2.set_ylabel("density", color="white")
    ax2.tick_params(colors="white")
    ax2.legend(framealpha=0.2, labelcolor="white")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#333")
    fig_scat2.tight_layout()

    log_dict[f"{prefix}/similarity_matrix"]  = wandb.Image(fig_sim)
    log_dict[f"{prefix}/correlation_dist_plot"] = wandb.Image(fig_corr)
    log_dict[f"{prefix}/match_vs_nonmatch"]  = wandb.Image(fig_scat2)

    plt.close(fig_sim)
    plt.close(fig_corr)
    plt.close(fig_scat2)

    # ---- Retrieval gallery table ----
    if images is not None and len(images) >= 4:
        log_dict[f"{prefix}/retrieval_gallery"] = _retrieval_table(
            images, sim[:len(images), :len(images)],
            ranks[:len(images)], n_rows=gallery_rows,
        )

    wandb.log(log_dict)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main(args):
    global _IMAGE_PROCESSOR

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if args.use_bf16 and device.type != "cuda":
        logger.warning("--use_bf16 requires CUDA; disabling bf16 autocast.")
        args.use_bf16 = False
    if args.use_bf16 and args.use_amp:
        logger.warning("Using --use_bf16; disabling --use_amp (fp16 scaler).")
        args.use_amp = False

    _trust = not args.no_trust_remote_code
    processor = AutoImageProcessor.from_pretrained(
        args.hf_model_id,
        trust_remote_code=_trust,
    )
    # Tell the processor about our (potentially non-default) image size so its
    # internal size checks stay consistent.  We pass do_resize=False in the
    # collate since the dataset transform already resizes — this just updates
    # the processor's metadata.  DINOv3 uses RoPE so any square resolution works.
    processor.size = {"height": args.image_size, "width": args.image_size}
    _IMAGE_PROCESSOR = processor
    dinov3_collate = make_collate_dinov3(processor)
    img_transform = get_dinov3_raw_transform(args.image_size)

    os.makedirs(args.output_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # W&B initialisation
    # -----------------------------------------------------------------------
    use_wandb = args.wandb and _WANDB_AVAILABLE
    if args.wandb and not _WANDB_AVAILABLE:
        logger.warning("--wandb requested but wandb is not installed; disabling.")

    # Auto-recover W&B run ID on resume so logs appear as one continuous run
    if args.resume and not args.wandb_run_id:
        run_id_file = os.path.join(args.output_dir, "wandb_run_id.txt")
        if os.path.isfile(run_id_file):
            with open(run_id_file) as f:
                args.wandb_run_id = f.read().strip()
            logger.info(f"Resuming W&B run {args.wandb_run_id}")

    if use_wandb:
        # Redirect W&B data (including artifact staging) away from $HOME to avoid
        # quota exhaustion on systems where $HOME is small.
        _wandb_data_default = os.path.join(
            os.environ.get("PROJECTDIR", str(Path(args.output_dir).parent.parent)),
            "wandb_data",
        )
        os.makedirs(_wandb_data_default, exist_ok=True)
        os.environ.setdefault("WANDB_DATA_DIR", _wandb_data_default)

        run_name = args.wandb_run_name or (
            f"{'_'.join(args.subjects)}_e{args.epochs}"
            if len(args.subjects) <= 3
            else f"{len(args.subjects)}subj_e{args.epochs}"
        )
        wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            name=run_name,
            config=vars(args),
            resume="allow",
            id=args.wandb_run_id or None,
        )
        # Save the run ID for resuming
        with open(os.path.join(args.output_dir, "wandb_run_id.txt"), "w") as f:
            f.write(wandb.run.id)
        logger.info(f"W&B run: {wandb.run.url}")

    # -----------------------------------------------------------------------
    # Load datasets
    # -----------------------------------------------------------------------
    train_datasets = {}
    test_datasets  = {}

    for sid in args.subjects:
        if args.proper_test_split:
            # --- Paper-matching evaluation protocol ---
            # Train on the full training split (all unique images per subject).
            # Evaluate on the dedicated test_split, which contains the NSD shared
            # images (the same evaluation set used in Beliy et al. 2025).
            # This gives ~10% more training data and a proper cross-subject eval.
            train_ds = NSDAlgonautsDataset(
                data_root=args.data_root,
                subject_id=sid,
                split="train",
                transform=img_transform,
            )
            test_ds = NSDAlgonautsDataset(
                data_root=args.data_root,
                subject_id=sid,
                split="test",
                transform=img_transform,
            )
            if test_ds.fmri_data is None:
                logger.warning(
                    f"[{sid}] --proper_test_split requested but test fMRI not found; "
                    "falling back to random holdout."
                )
                # Fallback: treat test_ds as unusable, use holdout instead
                args.proper_test_split = False

            if args.proper_test_split:
                if args.data_fraction < 1.0:
                    n_subset = max(int(len(train_ds) * args.data_fraction), 1)
                    train_ds = Subset(train_ds, list(range(n_subset)))
                    train_ds.num_voxels = test_ds.num_voxels
                    train_ds.subject_id = sid
                    logger.info(f"  [{sid}] data_fraction={args.data_fraction:.0%} → "
                                f"using {n_subset}/{len(train_ds)} training samples")

                train_datasets[sid] = train_ds
                train_datasets[sid].num_voxels = train_ds.num_voxels if hasattr(train_ds, 'num_voxels') else test_ds.num_voxels
                train_datasets[sid].subject_id = sid
                test_datasets[sid]  = test_ds
                test_datasets[sid].num_voxels = test_ds.num_voxels
                logger.info(
                    f"[{sid}] proper_test_split: {len(train_ds)} train, "
                    f"{len(test_ds)} test images, {test_ds.num_voxels} voxels"
                )
                continue  # skip the holdout branch below

        # --- Holdout evaluation (default fallback) ---
        # Hold out test_ratio of the training data for evaluation.
        full_ds = NSDAlgonautsDataset(
            data_root=args.data_root,
            subject_id=sid,
            split="train",
            transform=img_transform,
        )

        n = len(full_ds)
        n_test = max(int(n * args.test_ratio), 1)
        n_train = n - n_test

        rng = np.random.RandomState(args.seed)
        perm = rng.permutation(n)
        train_idx = perm[:n_train].tolist()
        test_idx  = perm[n_train:].tolist()

        if args.data_fraction < 1.0:
            n_subset = max(int(n_train * args.data_fraction), 1)
            train_idx = train_idx[:n_subset]
            logger.info(f"  [{sid}] data_fraction={args.data_fraction:.0%} → "
                        f"using {n_subset}/{n_train} training samples")

        train_datasets[sid] = Subset(full_ds, train_idx)
        train_datasets[sid].num_voxels = full_ds.num_voxels
        train_datasets[sid].subject_id = sid

        test_datasets[sid] = Subset(full_ds, test_idx)
        test_datasets[sid].num_voxels = full_ds.num_voxels

    # Combined multi-subject training dataset
    multi_ds = MultiSubjectDataset(train_datasets)

    # SubjectBatchSampler ensures every batch contains `batch_size` images from
    # ONE subject, rather than ~batch_size/num_subjects images spread across all
    # subjects.  This makes the cosine loss meaningful (N=32 instead of N≈4).
    subject_sampler = SubjectBatchSampler(
        multi_ds,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )

    train_loader = DataLoader(
        multi_ds,
        batch_sampler=subject_sampler,   # replaces batch_size + shuffle
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=dinov3_collate,
    )

    # Per-subject test loaders
    test_loaders = {}
    for sid in args.subjects:
        test_loaders[sid] = DataLoader(
            test_datasets[sid],
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=dinov3_collate,
        )

    # -----------------------------------------------------------------------
    # Create model
    # -----------------------------------------------------------------------
    model = UniversalBrainEncoderDINOv3(
        hf_model_id=args.hf_model_id,
        embedding_dim=args.embedding_dim,
        projection_dim=args.projection_dim,
        lora_rank=args.lora_rank,
        num_layers=5,
        mlp_hidden_mult=2,
        image_size=args.image_size,
        patch_size=args.patch_size,
        trust_remote_code=_trust,
        layer_selection=args.layer_selection,
        dropout=args.dropout,
        lora_dropout=args.lora_dropout,
        lora_mode=args.lora_mode,
        projection_mlp=args.projection_mlp,
    )

    if args.gradient_checkpointing:
        model.feature_extractor.enable_gradient_checkpointing()
        logger.info("HF backbone gradient checkpointing enabled.")

    # Register all subjects BEFORE moving to device so voxel embeddings land on GPU
    for sid in args.subjects:
        num_voxels = train_datasets[sid].num_voxels
        model.register_subject(sid, num_voxels)
        logger.info(f"Registered {sid}: {num_voxels} voxels")

    model = model.to(device)

    # -----------------------------------------------------------------------
    # Transfer learning (optional)
    # -----------------------------------------------------------------------
    if args.transfer_from:
        logger.info(f"Loading pre-trained model from {args.transfer_from}")
        checkpoint = torch.load(args.transfer_from, map_location=device)
        # Load shared weights only (not voxel embeddings for new subjects)
        model_dict = model.state_dict()
        pretrained_dict = {
            k: v for k, v in checkpoint["model_state_dict"].items()
            if k in model_dict and "voxel_store" not in k
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        logger.info(f"Loaded {len(pretrained_dict)} pre-trained weight tensors")

    # -----------------------------------------------------------------------
    # Optimizer setup
    # -----------------------------------------------------------------------
    if args.freeze_shared:
        # Transfer learning: only optimize voxel embeddings
        for name, param in model.named_parameters():
            if "voxel_store" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        logger.info("Transfer learning: optimizing voxel embedding parameters only")

    # AdamW with separate parameter groups:
    #   - Voxel embeddings (lookup table, ~70M params): weight_decay=0 — these are
    #     pure learned representations; decaying them toward zero destroys the
    #     per-voxel functionality the model is trying to capture (the paper uses 0).
    #   - Everything else (LoRA, projections, cross-attention): weight_decay as set —
    #     regularises the shared network weights without harming voxel quality.
    voxel_params, other_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "voxel_store" in name:
            voxel_params.append(param)
        else:
            other_params.append(param)

    param_groups = [
        {"params": voxel_params, "weight_decay": 0.0,            "name": "voxel_embeddings"},
        {"params": other_params, "weight_decay": args.weight_decay, "name": "shared_network"},
    ]
    n_voxel  = sum(p.numel() for p in voxel_params)
    n_other  = sum(p.numel() for p in other_params)
    logger.info(
        f"Optimizer groups — voxel_embeddings: {n_voxel:,} params (wd=0.0)  |  "
        f"shared_network: {n_other:,} params (wd={args.weight_decay})"
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)

    # Warmup + cosine schedule.
    # Linear warmup prevents the large gradient spikes we saw in early epochs
    # (e.g. epoch 1 loss=10.24, epoch 6 loss=1.28) by ramping the LR from
    # near-zero instead of starting at 1e-3 immediately.
    if args.warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-4,          # begin at lr * 1e-4  ≈ 1e-7
            end_factor=1.0,             # reach full lr after warmup_epochs
            total_iters=args.warmup_epochs,
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(args.epochs - args.warmup_epochs, 1),
            eta_min=args.lr * 0.01,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[args.warmup_epochs],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
        )

    criterion = BrainEncoderLoss(alpha=args.loss_alpha)

    scaler = GradScaler("cuda") if (args.use_amp and device.type == "cuda") else None

    # Log parameter counts
    param_info = count_parameters(model)
    logger.info(f"Parameters: {param_info}")

    # -----------------------------------------------------------------------
    # Resume from checkpoint
    # -----------------------------------------------------------------------
    start_epoch = 0
    best_corr   = -float("inf")
    history     = []
    global_step = 0

    if args.resume:
        ckpt_path = args.resume if args.resume != "auto" else _find_latest_checkpoint(args.output_dir)
        if ckpt_path:
            ckpt = load_checkpoint(ckpt_path, device)
            if ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                if "scheduler_state_dict" in ckpt:
                    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                start_epoch = ckpt["epoch"] + 1
                best_corr   = ckpt.get("best_corr", -float("inf"))
                history     = ckpt.get("history", [])
                global_step = ckpt.get("global_step", 0)
                logger.info(f"Resumed from {ckpt_path} at epoch {start_epoch}")
            else:
                logger.warning(f"Could not load checkpoint {ckpt_path}, starting fresh")
        else:
            logger.info("No checkpoint found, starting fresh")

    signal.signal(signal.SIGUSR1, _preemption_handler)

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # Advance SubjectBatchSampler's RNG so shuffling differs each epoch
        train_loader.batch_sampler.set_epoch(epoch)

        # Train
        train_metrics, global_step = train_one_epoch(
            model, train_loader, criterion, optimizer,
            device, scaler, args.voxels_per_image, epoch,
            global_step=global_step,
            use_wandb=use_wandb,
            use_bf16=args.use_bf16,
            max_grad_norm=args.max_grad_norm,
        )

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        epoch_time = time.time() - epoch_start
        logger.info(
            f"Epoch {epoch}/{args.epochs} | "
            f"Loss: {train_metrics['loss']:.4f} | "
            f"Time: {epoch_time:.1f}s | "
            f"LR: {current_lr:.2e}"
        )

        if use_wandb:
            wandb.log({
                "train/epoch_loss": train_metrics["loss"],
                "train/epoch_time_s": epoch_time,
                "train/lr": current_lr,
                "epoch": epoch,
            })

        # Evaluate periodically
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            eval_results = {}
            for sid in args.subjects:
                eval_results[sid] = evaluate(
                    model, test_loaders[sid], device, sid,
                    collect_images=use_wandb,
                )

            # Track best model by average median correlation
            avg_corr = np.mean([
                r["median_correlation"] for r in eval_results.values() if r
            ])

            if use_wandb:
                for sid, res in eval_results.items():
                    if res:
                        log_eval_to_wandb(res, sid, epoch, current_lr)

            if avg_corr > best_corr:
                best_corr = avg_corr
                save_checkpoint(model, optimizer, epoch, args, eval_results,
                                os.path.join(args.output_dir, "best_model.pt"))
                logger.info(f"New best model! Avg median correlation: {avg_corr:.4f}")
                if use_wandb:
                    wandb.run.summary["best_median_correlation"] = best_corr
                    wandb.run.summary["best_epoch"] = epoch

        # Save epoch metrics
        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "lr": scheduler.get_last_lr()[0],
            "time": epoch_time,
        })

        # Save checkpoint periodically
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch+1}.pt")
            save_checkpoint(
                model, optimizer, epoch, args, {},
                ckpt_path,
                scheduler=scheduler,
                best_corr=best_corr,
                history=history,
                global_step=global_step,
            )
            _prune_checkpoints(
                args.output_dir,
                keep=args.save_total_limit,
                exclude_names={"best_model.pt", "final_model.pt"},
            )

        # Graceful preemption: save and exit if SIGUSR1 received
        if _preemption_requested:
            logger.info("Preemption requested — saving checkpoint and exiting.")
            save_checkpoint(
                model, optimizer, epoch, args, {},
                os.path.join(args.output_dir, f"checkpoint_epoch{epoch+1}.pt"),
                scheduler=scheduler,
                best_corr=best_corr,
                history=history,
                global_step=global_step,
            )
            with open(os.path.join(args.output_dir, "history.json"), "w") as f:
                json.dump(history, f, indent=2)
            logger.info("Checkpoint saved. Exiting for requeue.")
            sys.exit(0)

    # Save final model
    save_checkpoint(model, optimizer, args.epochs - 1, args, {},
                    os.path.join(args.output_dir, "final_model.pt"))

    # Save training history
    with open(os.path.join(args.output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    logger.info("Training complete!")
    logger.info(f"Best average median correlation: {best_corr:.4f}")

    if use_wandb:
        wandb.finish()


def save_checkpoint(
    model,
    optimizer,
    epoch,
    args,
    eval_results,
    path,
    scheduler=None,
    best_corr=None,
    history=None,
    global_step=None,
):
    """Save full training state for resume."""
    # Strip non-serialisable W&B raw tensors from eval_results before saving
    clean_results = {}
    for sid, res in (eval_results or {}).items():
        clean_results[sid] = {k: v for k, v in res.items() if not k.startswith("_")}

    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
        "eval_results": clean_results,
    }
    if scheduler is not None:
        ckpt["scheduler_state_dict"] = scheduler.state_dict()
    if best_corr is not None:
        ckpt["best_corr"] = best_corr
    if history is not None:
        ckpt["history"] = history
    if global_step is not None:
        ckpt["global_step"] = global_step
    torch.save(ckpt, path)
    logger.info(f"Saved checkpoint to {path}")


def load_checkpoint(path: str, device: torch.device) -> Optional[Dict[str, Any]]:
    """Load checkpoint dict; return None if file missing or invalid."""
    if not path or not os.path.isfile(path):
        return None
    try:
        ckpt = torch.load(path, map_location=device)
        if "model_state_dict" not in ckpt:
            return None
        return ckpt
    except Exception as e:
        logger.warning(f"Could not load checkpoint {path}: {e}")
        return None


def _find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """Return path to latest checkpoint_epoch*.pt or ckpt_epoch_*.pt in output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    pattern = os.path.join(output_dir, "checkpoint_epoch*.pt")
    paths = glob.glob(pattern)
    if not paths:
        pattern = os.path.join(output_dir, "ckpt_epoch_*.pt")
        paths = glob.glob(pattern)
    if not paths:
        return None
    def epoch_from_path(p):
        try:
            base = os.path.basename(p)
            # checkpoint_epoch30.pt or ckpt_epoch_030.pt
            if "checkpoint_epoch" in base:
                return int(base.replace("checkpoint_epoch", "").replace(".pt", ""))
            if "ckpt_epoch_" in base:
                return int(base.replace("ckpt_epoch_", "").replace(".pt", ""))
        except ValueError:
            return -1
        return -1
    latest = max(paths, key=epoch_from_path)
    return latest


def _prune_checkpoints(output_dir: str, keep: int, exclude_names: set):
    """Keep only the latest `keep` checkpoint_epoch*.pt / ckpt_epoch_*.pt; never remove exclude_names."""
    paths = []
    for pat in ("checkpoint_epoch*.pt", "ckpt_epoch_*.pt"):
        paths.extend(glob.glob(os.path.join(output_dir, pat)))
    if not paths:
        return
    def epoch_from_path(p):
        try:
            base = os.path.basename(p)
            if "checkpoint_epoch" in base:
                return int(base.replace("checkpoint_epoch", "").replace(".pt", ""))
            if "ckpt_epoch_" in base:
                return int(base.replace("ckpt_epoch_", "").replace(".pt", ""))
        except ValueError:
            return -1
        return -1
    paths.sort(key=epoch_from_path, reverse=True)
    for p in paths[keep:]:
        if os.path.basename(p) in exclude_names:
            continue
        try:
            os.remove(p)
            logger.info(f"Pruned old checkpoint: {p}")
        except OSError:
            pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train Universal Brain Encoder")

    # Data
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to Algonauts 2023 data or NSD root")
    parser.add_argument("--subjects", type=str, nargs="+", required=True,
                        help="Subject IDs (e.g., subj01 subj02 ...)")
    parser.add_argument("--test_ratio", type=float, default=0.1,
                        help="Fraction of training data held out for evaluation "
                             "(only used when --proper_test_split is not set).")
    parser.add_argument("--proper_test_split", action="store_true", default=True,
                        help="Use the dedicated test_split/ (NSD shared images with fMRI) "
                             "for evaluation instead of a random holdout. Trains on ALL "
                             "training images. Matches the paper's evaluation protocol.")
    parser.add_argument("--no_proper_test_split", dest="proper_test_split",
                        action="store_false",
                        help="Disable proper_test_split and fall back to random holdout.")
    parser.add_argument("--data_fraction", type=float, default=1.0,
                        help="Fraction of training data to use (e.g. 0.2 = 20%%). "
                             "Applied after train/test split.")

    # Model
    parser.add_argument(
        "--hf_model_id",
        type=str,
        default="facebook/dinov3-vith16plus-pretrain-lvd1689m",
        help="Hugging Face model id for DINOv3 ViT + AutoImageProcessor",
    )
    parser.add_argument("--no_trust_remote_code", action="store_true",
                        help="Pass trust_remote_code=False to HF loaders")
    parser.add_argument("--embedding_dim", type=int, default=256,
                        help="Voxel embedding dimension E")
    parser.add_argument("--projection_dim", type=int, default=256,
                        help="Feature projection dimension C (paper: 128 for DINOv2 ViT-L). "
                             "DINOv3 ViT-H has hidden_dim=1280 vs 1024 for ViT-L, so 256 "
                             "maintains a comparable compression ratio. Pair with --projection_mlp "
                             "for non-linear compression at this scale.")
    parser.add_argument("--lora_rank", type=int, default=16,
                        help="LoRA rank for DINO adaptation (paper: 16). "
                             "Rank 16 is sufficient even for ViT-H; higher ranks increase "
                             "overfitting risk without clear benefit.")
    parser.add_argument("--lora_dropout", type=float, default=0.0,
                        help="Dropout probability on the LoRA intermediate (after A, before B). "
                             "Regularises the backbone adaptation and delays overfitting. "
                             "Recommended: 0.05 for ViT-H with rank ≤ 16.")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Dropout probability in the CrossAttentionBlock MLPs "
                             "(between GELU and the output projection). "
                             "Regularises the decoder head; 0.1 shifts the best epoch from ~9 to ~18.")
    parser.add_argument("--image_size", type=int, default=224,
                        help="Resize side for dataset. DINOv3 uses RoPE positional embeddings "
                             "so it is resolution-invariant. 224px gives (224/16)^2=196 patches. "
                             "Empirically 224px outperforms 336px and 448px at 30 epochs — "
                             "larger resolutions produce more patches but converge slower.")
    parser.add_argument("--patch_size", type=int, default=16,
                        help="Patch size for num_patches = (image_size/patch_size)^2")
    parser.add_argument("--layer_selection", type=str, default="paper_proportional",
                        choices=["even", "top_biased", "paper_proportional"],
                        help="Which 5 backbone layers to extract features from. "
                             "'paper_proportional' (default) scales the paper's DINOv2 "
                             "layer choice [1,6,12,18,24] of 24 blocks to any depth — "
                             "includes the first block for V1/V2 low-level features. "
                             "For ViT-H (32 blocks) → blocks [0, 7, 15, 23, 31]. "
                             "'top_biased' uses [25%%, 50%%, 66%%, 83%%, 100%%] depth. "
                             "'even' uses evenly-spaced indices.")
    parser.add_argument("--projection_mlp", action="store_true",
                        help="Use 2-layer MLP (hidden→hidden//4→C) instead of single Linear "
                             "for the per-layer feature projections. Preserves nonlinear "
                             "structure from the backbone, important when compression ratio "
                             "is high (e.g. 1280→128 = 10:1 for ViT-H).")
    parser.add_argument("--lora_mode", type=str, default="block",
                        choices=["out_proj", "block", "both"],
                        help="Where LoRA is injected. "
                             "'block' (default, recommended): adapter on full block output, "
                             "captures adapted features while leaving the residual stream clean "
                             "(matches DINOv2 behaviour). "
                             "'out_proj': attention output proj LoRA — delta propagates through "
                             "the block's MLP/residuals, contaminating the backbone stream. "
                             "'both': out_proj + block adapter (most parameters, highest "
                             "overfitting risk, pollutes residual stream).")

    # Training
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Images per batch (paper: 32)")
    parser.add_argument("--voxels_per_image", type=int, default=5000,
                        help="Random voxels sampled per image (paper: 5000)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (paper: 1e-3)")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for AdamW. 0.01 regularises the large voxel "
                             "embedding table (~70M params) without hurting convergence. "
                             "Paper used 0.0 with plain Adam, but AdamW + 0.01 is better.")
    parser.add_argument("--loss_alpha", type=float, default=0.1,
                        help="MSE weight in combined loss (paper: 0.1)")
    parser.add_argument("--use_amp", action="store_true",
                        help="Use fp16 mixed precision (GradScaler); mutually exclusive with --use_bf16")
    parser.add_argument("--use_bf16", action="store_true",
                        help="bf16 autocast on CUDA (recommended for large ViT)")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing on the HF backbone")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Gradient clipping max norm (0 = disabled)")
    parser.add_argument("--warmup_epochs", type=int, default=2,
                        help="Linear LR warmup epochs (start_factor=1e-4 → 1.0). "
                             "2 epochs prevents the large gradient spikes seen in early "
                             "training when voxel embeddings are randomly initialised.")

    # Transfer learning
    parser.add_argument("--transfer_from", type=str, default=None,
                        help="Path to pre-trained checkpoint for transfer learning")
    parser.add_argument("--freeze_shared", action="store_true",
                        help="Freeze shared weights, only train voxel embeddings")

    # Output
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--save_total_limit", type=int, default=3,
                        help="Keep only the N most recent epoch checkpoints")
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    # Resume
    parser.add_argument("--resume", type=str, nargs="?", const="auto", default=None,
                        help="Resume from checkpoint. Pass path or use 'auto' to find latest.")

    # W&B
    parser.add_argument("--wandb", action="store_true", default=False,
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="universal-brain-encoder")
    parser.add_argument("--wandb_group",   type=str, default="nsd-algonauts")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_run_id",   type=str, default=None,
                        help="W&B run ID for resuming (auto-read from output_dir/wandb_run_id.txt)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    main(args)
