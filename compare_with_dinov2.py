#!/usr/bin/env python3
"""
Evaluate a DINOv2 (universal_brain_encoder) checkpoint and a DINOv3 checkpoint
on the same held-out batches. Reports Pearson r and MSE vs ground-truth fMRI.

Caveats:
  - Patch counts differ (DINOv2 ViT-L/14: 256 vs DINOv3 ViT-H+ @ 224/16: 196).
  - Input preprocessing differs (ImageNet norm vs HF AutoImageProcessor).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

_DINO = Path(__file__).resolve().parent
_UB = _DINO.parent / "universal_brain_encoder"
sys.path.insert(0, str(_UB))

from dataset import NSDAlgonautsDataset, MultiSubjectDataset  # noqa: E402
from model import UniversalBrainEncoder as UBE_v2  # noqa: E402

sys.path.insert(0, str(_DINO))
from model import UniversalBrainEncoderDINOv3  # noqa: E402
from train import get_dinov3_raw_transform  # noqa: E402
from transformers import AutoImageProcessor  # noqa: E402


def _imagenet_norm(x: torch.Tensor) -> torch.Tensor:
    mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (x - mean) / std


def make_collate_dual(processor):
    """Produce DINOv3 pixel_values and DINOv2-style ImageNet-normalised tensors."""

    def collate(batch: List[dict]) -> dict:
        by_subject: Dict[str, List[dict]] = {}
        for item in batch:
            sid = item["subject_id"]
            by_subject.setdefault(sid, []).append(item)
        out = {}
        for sid, items in by_subject.items():
            stacked = torch.stack([it["image"] for it in items])
            pv = processor(
                images=stacked,
                return_tensors="pt",
                do_rescale=False,
            )["pixel_values"]
            out[sid] = {
                "pixel_values_d3": pv,
                "images_d2": _imagenet_norm(stacked),
                "fmri": torch.stack([it["fmri"] for it in items]),
                "subject_id": sid,
            }
        return out

    return collate


@torch.no_grad()
def _batch_metrics(
    pred: torch.Tensor,
    gt: torch.Tensor,
) -> Dict[str, float]:
    """pred, gt: (N_vox, B)"""
    mse = F.mse_loss(pred, gt).item()
    pred_c = pred - pred.mean(dim=1, keepdim=True)
    gt_c = gt - gt.mean(dim=1, keepdim=True)
    num = (pred_c * gt_c).sum(dim=1)
    den = pred_c.norm(dim=1) * gt_c.norm(dim=1) + 1e-8
    r = num / den
    return {
        "mse": mse,
        "median_r": r.median().item(),
        "mean_r": r.mean().item(),
    }


@torch.no_grad()
def run_comparison(
    loader: DataLoader,
    model_v2: UBE_v2,
    model_v3: UniversalBrainEncoderDINOv3,
    device: torch.device,
    voxels_sample: int,
    max_batches: int,
) -> None:
    model_v2.eval()
    model_v3.eval()

    acc_v2 = {"mse": [], "median_r": [], "mean_r": []}
    acc_v3 = {"mse": [], "median_r": [], "mean_r": []}

    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        for sid, subj in batch.items():
            pv = subj["pixel_values_d3"].to(device)
            im2 = subj["images_d2"].to(device)
            fmri = subj["fmri"].to(device)
            B, nv = fmri.shape
            n_s = min(voxels_sample, nv)
            idx = torch.randperm(nv, device=device)[:n_s]
            gt = fmri[:, idx].T

            p2 = model_v2(im2, sid, idx)
            p3 = model_v3(pv, sid, idx)

            m2 = _batch_metrics(p2, gt)
            m3 = _batch_metrics(p3, gt)
            for k in acc_v2:
                acc_v2[k].append(m2[k])
                acc_v3[k].append(m3[k])

    def _avg(d):
        return {k: float(np.mean(v)) for k, v in d.items()}

    print("DINOv2 (same batches):", _avg(acc_v2))
    print("DINOv3 (same batches):", _avg(acc_v3))


def parse_args():
    p = argparse.ArgumentParser(description="Compare DINOv2 vs DINOv3 brain encoders")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--subjects", type=str, nargs="+", default=["subj01"])
    p.add_argument("--dinov2_ckpt", type=str, required=True)
    p.add_argument("--dinov3_ckpt", type=str, required=True)
    p.add_argument("--hf_model_id", type=str, default="facebook/dinov3-vith16plus-pretrain-lvd1689m")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--test_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--voxels_sample", type=int, default=2048)
    p.add_argument("--max_batches", type=int, default=50)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--patch_size_d3", type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoImageProcessor.from_pretrained(args.hf_model_id, trust_remote_code=True)
    collate_fn = make_collate_dual(processor)
    img_tf = get_dinov3_raw_transform(args.image_size)

    test_datasets = {}
    for sid in args.subjects:
        full_ds = NSDAlgonautsDataset(
            args.data_root, sid, split="train", transform=img_tf,
        )
        n = len(full_ds)
        n_test = max(int(n * args.test_ratio), 1)
        rng = np.random.RandomState(args.seed)
        perm = rng.permutation(n)
        test_idx = perm[-n_test:].tolist()
        test_datasets[sid] = Subset(full_ds, test_idx)
        test_datasets[sid].num_voxels = full_ds.num_voxels

    multi = MultiSubjectDataset(test_datasets)
    loader = DataLoader(
        multi,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )

    model_v2 = UBE_v2(
        embedding_dim=256,
        projection_dim=128,
        lora_rank=16,
        num_layers=5,
        mlp_hidden_mult=2,
        image_size=224,
        patch_size=14,
    )
    for sid in args.subjects:
        model_v2.register_subject(sid, test_datasets[sid].num_voxels)

    model_v3 = UniversalBrainEncoderDINOv3(
        hf_model_id=args.hf_model_id,
        embedding_dim=256,
        projection_dim=128,
        lora_rank=16,
        num_layers=5,
        mlp_hidden_mult=2,
        image_size=args.image_size,
        patch_size=args.patch_size_d3,
        trust_remote_code=True,
    )
    for sid in args.subjects:
        model_v3.register_subject(sid, test_datasets[sid].num_voxels)

    ck2 = torch.load(args.dinov2_ckpt, map_location=device)
    ck3 = torch.load(args.dinov3_ckpt, map_location=device)
    model_v2.load_state_dict(ck2["model_state_dict"], strict=True)
    model_v3.load_state_dict(ck3["model_state_dict"], strict=True)

    model_v2.to(device)
    model_v3.to(device)

    run_comparison(loader, model_v2, model_v3, device, args.voxels_sample, args.max_batches)


if __name__ == "__main__":
    main()
