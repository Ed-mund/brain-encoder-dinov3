"""
GPU tests for brain_encoder_dinov3. Intended to run on a Slurm GPU allocation.

Requires:
  - CUDA
  - Hugging Face access to gated DINOv3 weights: set HF_TOKEN or huggingface-cli login
  - conda env brain_encoder (transformers, torch CUDA)

Optional real-data smoke (BRAIN_ENCODER_TEST_DATA_ROOT or default path).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

_DINO = Path(__file__).resolve().parent
_UB = _DINO.parent / "universal_brain_encoder"
sys.path.insert(0, str(_UB))
sys.path.insert(0, str(_DINO))

MID = os.environ.get(
    "TEST_DINOV3_MODEL_ID",
    "facebook/dinov3-vith16plus-pretrain-lvd1689m",
)
DATA_ROOT = os.environ.get(
    "BRAIN_ENCODER_TEST_DATA_ROOT",
    "/projects/b6ac/brain/algonauts_prepared_data",
)


def _hf_login_from_env() -> None:
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        from huggingface_hub import login

        login(token=tok, add_to_git_credential=False)


@pytest.fixture(scope="module", autouse=True)
def hf_login():
    _hf_login_from_env()


def test_cuda_available():
    assert torch.cuda.is_available(), "This suite must run on a GPU node (sbatch --gpus=1)"
    props = torch.cuda.get_device_properties(0)
    assert props.total_memory > 0


def test_lorad_dinov3_forward_backward_bf16():
    from transformers import AutoImageProcessor
    from dinov3_backbone import LoRADINOv3

    device = torch.device("cuda")
    proc = AutoImageProcessor.from_pretrained(MID, trust_remote_code=True)
    m = LoRADINOv3(
        model_id=MID,
        projection_dim=128,
        lora_rank=8,
        freeze_backbone=True,
        trust_remote_code=True,
    ).to(device)
    m.train()
    m.enable_gradient_checkpointing()

    x01 = torch.rand(1, 3, 224, 224, device=device)
    pv = proc(images=x01, return_tensors="pt", do_rescale=False)["pixel_values"].to(device)

    trainable = [p for p in m.parameters() if p.requires_grad]
    assert trainable, "LoRA/projections should be trainable"
    opt = torch.optim.Adam(trainable, lr=1e-4)

    opt.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        feat = m(pv)
    assert feat.dim() == 4
    assert feat.shape[0] == 1
    assert feat.shape[1] == 5
    loss = feat.float().pow(2).mean()
    loss.backward()
    opt.step()
    assert loss.item() == loss.item()  # finite


def test_universal_encoder_one_step_synthetic():
    from transformers import AutoImageProcessor
    from model import BrainEncoderLoss, UniversalBrainEncoderDINOv3

    device = torch.device("cuda")
    proc = AutoImageProcessor.from_pretrained(MID, trust_remote_code=True)
    ub = UniversalBrainEncoderDINOv3(
        hf_model_id=MID,
        projection_dim=128,
        lora_rank=8,
        image_size=224,
        patch_size=16,
        trust_remote_code=True,
    ).to(device)
    ub.feature_extractor.enable_gradient_checkpointing()
    ub.train()
    ub.register_subject("subj01", 512)

    x01 = torch.rand(2, 3, 224, 224, device=device)
    pv = proc(images=x01, return_tensors="pt", do_rescale=False)["pixel_values"].to(device)
    nv = 512
    idx = torch.randperm(nv, device=device)[:256]
    gt = torch.randn(256, 2, device=device)

    params = [p for p in ub.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=1e-4)
    crit = BrainEncoderLoss(alpha=0.1)

    opt.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        pred = ub(pv, "subj01", idx)
        loss = crit(pred.float(), gt)
    loss.backward()
    opt.step()
    assert torch.isfinite(loss).item()


@pytest.mark.integration
def test_train_dataloader_one_batch_real_data():
    root = Path(DATA_ROOT)
    if not root.is_dir():
        pytest.skip(f"No dataset at {DATA_ROOT}; set BRAIN_ENCODER_TEST_DATA_ROOT")

    from torch.utils.data import DataLoader, Subset
    from transformers import AutoImageProcessor

    from dataset import NSDAlgonautsDataset, MultiSubjectDataset
    from train import get_dinov3_raw_transform, make_collate_dinov3

    sid = "subj01"
    sub_root = root / sid
    if not sub_root.is_dir():
        pytest.skip(f"Missing {sub_root}")

    proc = AutoImageProcessor.from_pretrained(MID, trust_remote_code=True)
    tfm = get_dinov3_raw_transform(224)
    ds = NSDAlgonautsDataset(str(root), sid, split="train", transform=tfm)
    n = min(16, len(ds))
    sub = Subset(ds, list(range(n)))
    sub.num_voxels = ds.num_voxels
    multi = MultiSubjectDataset({sid: sub})
    loader = DataLoader(
        multi,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=make_collate_dinov3(proc),
    )
    batch = next(iter(loader))
    assert sid in batch
    imgs = batch[sid]["images"]
    fmri = batch[sid]["fmri"]
    assert imgs.ndim == 4 and fmri.ndim == 2
    device = torch.device("cuda")
    from model import UniversalBrainEncoderDINOv3

    ub = UniversalBrainEncoderDINOv3(
        hf_model_id=MID,
        projection_dim=128,
        lora_rank=8,
        image_size=224,
        patch_size=16,
        trust_remote_code=True,
    ).to(device)
    ub.register_subject(sid, ds.num_voxels)
    ub.eval()
    pv = imgs.to(device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        pred = ub.predict_all_voxels(pv, sid, chunk_size=1000)
    assert pred.shape[0] == ds.num_voxels
    assert pred.shape[1] == pv.shape[0]
