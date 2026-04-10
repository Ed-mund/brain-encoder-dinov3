# Universal Brain Encoder — DINOv3 (Hugging Face)

Vision trunk: **`facebook/dinov3-vith16plus-pretrain-lvd1689m`** (~0.8B parameters) via `transformers`, with LoRA on five depth-sliced patch-token features and the same cross-attention head as [`universal_brain_encoder`](../universal_brain_encoder/).

## Requirements

- **`transformers>=4.56`** (DINOv3 support and `AutoModel` / `AutoImageProcessor`).
- PyTorch with CUDA (bf16 recommended for this backbone).
- Same NSD / Algonauts data layout as the DINOv2 trainer.

## Token / shape notes

- Default **224×224**, **patch 16** → **P = 14×14 = 196** patch tokens (plus CLS and register tokens inside the backbone; those are stripped before the head).
- The cross-attention block uses `num_patches=(image_size // patch_size) ** 2`; keep **`--image_size` / `--patch_size` aligned with the processor and backbone**.

## Preprocessing

Training uses **`AutoImageProcessor.from_pretrained(hf_model_id)`** with dataset tensors in **[0, 1]** (`Resize` + `ToTensor` only). **Do not** use ImageNet `Normalize` from `get_default_transform()` for this pipeline.

## Train

From the repo root (with `PYTHONPATH` including this directory and `universal_brain_encoder`):

```bash
python brain/brain_encoder_dinov3/train.py \
  --data_root /path/to/algonauts \
  --subjects subj01 \
  --use_bf16 \
  --gradient_checkpointing \
  --output_dir ./checkpoints/dinov3_run
```

Slurm: see [`run_train_dinov3.slurm.sh`](../../run_train_dinov3.slurm.sh).

## GPU tests on Slurm

DINOv3 checkpoints are **gated** on Hugging Face: set `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`) before submit, or run `huggingface-cli login`.

```bash
export PROJECTDIR=/projects/b6ac   # if needed
export HF_TOKEN=hf_...
sbatch run_test_dinov3_gpu.slurm.sh
```

Optional: only the real-data dataloader + `predict_all_voxels` check:

```bash
sbatch run_test_dinov3_gpu.slurm.sh --integration-only
```

Override model id (e.g. smaller ViT) with `TEST_DINOV3_MODEL_ID`.

## Compare DINOv2 vs DINOv3 checkpoints

On the **same random mini-batches** (different `num_patches` and preprocessing — interpret as a rough sanity check, not apples-to-apples architecture parity):

```bash
python brain/brain_encoder_dinov3/compare_with_dinov2.py \
  --data_root /path/to/algonauts \
  --subjects subj01 \
  --dinov2_ckpt /path/to/best_model.pt \
  --dinov3_ckpt /path/to/dinov3_best.pt
```

## References

- [DINOv3 in Transformers](https://huggingface.co/docs/transformers/model_doc/dinov3)
- [DINOv3 model collection](https://huggingface.co/collections/facebook/dinov3)
