#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# ===== Conda 虚拟环境 + PyTorch (CUDA 11.8) =====
CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-funad}"
# shellcheck source=/dev/null
source "$CONDA_SH"
conda activate "$CONDA_ENV"

# ===== User-configurable =====
DATA_PATH="/media/honeywell/D/bhy/dataset/MVTec"
# Under plot/ so behavior matches older "cd plot" runs
OUTPUT_DIR="plot/tsne_outputs_no_cls_token"
# CLASSES="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"
CLASSES="screw"
# Set your checkpoint here. If empty, script runs without checkpoint.
CKPT="/media/honeywell/E/bhy/FUNAD/save_results/muti_class_residual_correst_dinov3_kl/mvtec/10%/gaussian_True_noise_10%_balancing_True_oto_True_weight_2.5_multiclass_residual_localnet.pt"
REPORT_STAGE="after"  # before | after

# Common options
COMMON_ARGS=(
  --data_path "$DATA_PATH"
  --output_dir "$OUTPUT_DIR"
  --classes $CLASSES
  --backbone dinov3
  --image_size 512
  --crop_size 448
  --batch_size 16
  --num_workers 4
  --max_per_group 1000
  --perplexity 30
  --num_reference_images 4
  --output tsne_residual.png
  --distance_output tsne_distance_distribution.png
)

if [[ -n "${CKPT}" ]]; then
  python plot/tsne_residual_features.py \
    "${COMMON_ARGS[@]}" \
    --checkpoint "$CKPT" \
    --checkpoint_key net \
    --report_stage "$REPORT_STAGE" \
    --difference_output tsne_center_shift.png
else
  python plot/tsne_residual_features.py "${COMMON_ARGS[@]}"
fi
