#!/usr/bin/env bash
# ============================================================================
# Universal Brain Encoder — DINOv3 backbone  (Isambard-AI / Cray Shasta)
# ============================================================================
#
# Usage:
#   sbatch run_train.slurm.sh                          # fresh 30-epoch run
#   sbatch run_train.slurm.sh --resume                 # resume latest checkpoint
#   sbatch run_train.slurm.sh --run_name my_exp --epochs 50
#
# All unknown flags are forwarded verbatim to train.py.
#
# ============================================================================
#SBATCH --job-name=brain-enc-d3
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=/projects/b6ac/.logs/%x-%j.out
#SBATCH --error=/projects/b6ac/.logs/%x-%j.err
#SBATCH --signal=USR1@300

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CONDA_ENV="brain_encoder"
DINO3_DIR="${PROJECTDIR}/brain/brain_encoder_dinov3"
DATA_ROOT="${PROJECTDIR}/brain/algonauts_prepared_data"
DATE_TAG=$(date +%Y%m%d)
OUTPUT_DIR="${PROJECTDIR}/brain/checkpoints/brain_encoder_dinov3_${DATE_TAG}"

echo "=== $(date) === Job ${SLURM_JOB_ID} on $(hostname) ==="

# ---------------------------------------------------------------------------
# CUDA
# ---------------------------------------------------------------------------
module load cuda/12.6 2>/dev/null \
  || module load cudatoolkit/24.11_12.6 2>/dev/null \
  || true
echo "CUDA_HOME: ${CUDA_HOME:-not set}"
nvidia-smi --list-gpus

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

# ---------------------------------------------------------------------------
# Cache / data dirs — all on project storage, never $HOME
# ---------------------------------------------------------------------------
export HF_HOME="${PROJECTDIR}/.cache/hf"
export HF_DATASETS_CACHE="${PROJECTDIR}/.cache/hf/datasets"
export TORCH_HOME="${PROJECTDIR}/.cache/torch"
export TRITON_CACHE_DIR="${SCRATCHDIR:-${PROJECTDIR}/.cache}/triton_cache"
export WANDB_DATA_DIR="${PROJECTDIR}/wandb_data"

mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TORCH_HOME}" \
         "${TRITON_CACHE_DIR}" "${WANDB_DATA_DIR}" \
         "${OUTPUT_DIR}" /projects/b6ac/.logs

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------
GPU_IDS=$(nvidia-smi --list-gpus | awk '{print NR-1}' | tr '\n' ',' | sed 's/,$//')
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available'
d = torch.cuda.get_device_properties(0)
print(f'CUDA OK: {d.name}  {d.total_memory/1024**3:.0f} GB  bf16={torch.cuda.is_bf16_supported()}')
"

echo "DINOv3 train dir: ${DINO3_DIR}"
echo "Data root:        ${DATA_ROOT}"
echo "Output dir:       ${OUTPUT_DIR}"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
RESUME_FLAG=""
EXTRA_ARGS=()
RUN_NAME=""
EPOCHS=30

while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume)    RESUME_FLAG="--resume"; shift ;;
        --run_name)  RUN_NAME="$2"; shift 2 ;;
        --epochs)    EPOCHS="$2"; shift 2 ;;
        # If resuming, allow pointing at an existing output dir
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        *)           EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# When resuming, default to the most recent checkpoint dir
if [[ -n "${RESUME_FLAG}" && -z "${RUN_NAME}" ]]; then
    LATEST=$(ls -dt "${PROJECTDIR}/brain/checkpoints/brain_encoder_dinov3_"* 2>/dev/null | head -1 || echo "")
    if [[ -n "${LATEST}" ]]; then
        OUTPUT_DIR="${LATEST}"
        echo "Resuming from: ${OUTPUT_DIR}"
    fi
fi

RUN_NAME_ARG=""
[[ -n "${RUN_NAME}" ]] && RUN_NAME_ARG="--wandb_run_name ${RUN_NAME}"

echo "Extra args:       ${EXTRA_ARGS[*]+"${EXTRA_ARGS[*]}"}"
echo ""

# ---------------------------------------------------------------------------
# Signal handler: forward SIGUSR1 to Python subprocess
# ---------------------------------------------------------------------------
PY_PID=""
forward_signal() {
    echo "[slurm] Forwarding SIGUSR1 to Python (pid ${PY_PID})..."
    kill -USR1 "${PY_PID}" 2>/dev/null || true
}
trap 'forward_signal' USR1
trap 'forward_signal' TERM

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
cd "${DINO3_DIR}"

python train.py \
    --data_root "${DATA_ROOT}" \
    --output_dir "${OUTPUT_DIR}" \
    --subjects subj01 subj02 subj03 subj04 subj05 subj06 subj07 \
    --epochs "${EPOCHS}" \
    --batch_size 32 \
    --lr 1e-3 \
    --weight_decay 0.01 \
    --warmup_epochs 2 \
    --image_size 224 \
    --patch_size 16 \
    --layer_selection paper_proportional \
    --lora_rank 16 \
    --lora_dropout 0.05 \
    --lora_mode block \
    --projection_dim 256 \
    --projection_mlp \
    --use_bf16 \
    --gradient_checkpointing \
    --max_grad_norm 1.0 \
    --eval_every 2 \
    --save_every 5 \
    --num_workers 8 \
    --wandb \
    --wandb_project universal-brain-encoder-dinov3 \
    ${RESUME_FLAG} \
    ${RUN_NAME_ARG} \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" &

PY_PID=$!
wait "${PY_PID}"
EXIT_CODE=$?

echo ""
echo "=== $(date) === Done ==="
exit "${EXIT_CODE}"
