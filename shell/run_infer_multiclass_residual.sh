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

# ===== User-configurable（必填：CHECKPOINT_PATH、OUTPUT_DIR）=====
DATA_PATH="/media/honeywell/D/bhy/dataset/MVTec_overlap/MVTec_noisy10"
DATASET="mvtec" # mvtec | visa
CHECKPOINT_PATH="/media/honeywell/E/bhy/FUNAD/save_results/muti_class_residual_correst_dinov3_kl/mvtec/20%/gaussian_True_noise_20%_balancing_True_oto_True_weight_2.5_multiclass_residual_localnet.pt"
OUTPUT_DIR="./output_20"       # 例如: /path/to/infer_outputs/run1

BATCH_SIZE=16
NUM_WORKERS=4
FEATURE_MODEL="dino" # dino | clip
SEED=0
IMAGE_SIZE=512
CROP_SIZE=448

# 可选：限制每类图像数；-1 表示不限制
MAX_IMAGES_PER_CLASS=-1

# 可选：只推理部分类别；留空数组则全部类别
CLASS_NAMES=()
# 例如: CLASS_NAMES=(bottle cable)

# 额外参数示例（取消注释即可）：
# EXTRA_ARGS+=(--save_overlay --overlay_alpha 0.45)
# EXTRA_ARGS+=(--faiss_cpu_index)
# EXTRA_ARGS+=(--use_cls_token True)
EXTRA_ARGS=()

if [[ -z "$CHECKPOINT_PATH" || -z "$OUTPUT_DIR" ]]; then
  echo "请在本脚本中设置 CHECKPOINT_PATH 与 OUTPUT_DIR。" >&2
  exit 1
fi

CLASS_ARG=()
if ((${#CLASS_NAMES[@]} > 0)); then
  CLASS_ARG=(--class_names "${CLASS_NAMES[@]}")
fi

python src/inference/infer_multiclass_residual.py \
  --data_path "$DATA_PATH" \
  --dataset "$DATASET" \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --feature_model "$FEATURE_MODEL" \
  --seed "$SEED" \
  --image_size "$IMAGE_SIZE" \
  --crop_size "$CROP_SIZE" \
  --max_images_per_class "$MAX_IMAGES_PER_CLASS" \
  "${CLASS_ARG[@]}" \
  "${EXTRA_ARGS[@]}"
