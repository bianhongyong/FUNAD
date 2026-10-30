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

# ===== User-configurable（可用环境变量覆盖）=====
# 训练输出目录（best checkpoint 与其 raw-reference-bank sidecar 必须同目录）
TRAIN_DIR="${TRAIN_DIR:-/media/honeywell/E/bhy/FUNAD/save_results/class_patch/mvtec/10%}"

DATA_PATH="${DATA_PATH:-/media/honeywell/D/bhy/dataset/MVTec_overlap/MVTec_noisy10}"
DATASET="${DATASET:-mvtec}" # mvtec | visa
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${TRAIN_DIR}/gaussian_True_noise_10%_balancing_True_oto_True_weight_2.5_multiclass_residual_fusion_class_patch_layers_3-6-9-12-15-18-21-24_adapter_legacy_match_prior_top1_localnet.pt}"
# 指标输出目录（class_metrics.csv 与可视化面板）
OUTPUT_DIR="${OUTPUT_DIR:-/media/honeywell/E/bhy/FUNAD/save_results/class_patch_infer/mvtec/10%/prior_top1_screw_lastlayer}"

BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
FEATURE_MODEL="${FEATURE_MODEL:-dinov3_vitl16}"
SEED="${SEED:-0}"
IMAGE_SIZE="${IMAGE_SIZE:-512}"
CROP_SIZE="${CROP_SIZE:-448}"

# 必须与训练一致：本 checkpoint 训练时使用 --moe_use_cls_token，
# --use_cls_token 保持默认 False。
USE_CLS_TOKEN="${USE_CLS_TOKEN:-False}"
MOE_USE_CLS_TOKEN="${MOE_USE_CLS_TOKEN:-True}"

# 只影响保存的可视化面板数量，不影响指标（指标始终用全部测试图）。
# 设为 -1 可保存每类全部图像。
MAX_IMAGES_PER_CLASS="${MAX_IMAGES_PER_CLASS:-20}"

# 可选：只推理部分类别，逗号分隔；留空则全部类别。
# 例如: CLASSES=screw  或  CLASSES=screw,bottle
CLASSES="${CLASSES:-}"

# 其他可选参数
IMG_SCORE_TOPK_RATIO="${IMG_SCORE_TOPK_RATIO:-0.01}"
GAUSSIAN_SIGMA="${GAUSSIAN_SIGMA:-4.0}"
SAVE_OVERLAY="${SAVE_OVERLAY:-False}"

if [[ ! -f "$CHECKPOINT_PATH" ]]; then
  echo "checkpoint 不存在: $CHECKPOINT_PATH" >&2
  exit 1
fi
if [[ -z "$OUTPUT_DIR" ]]; then
  echo "请设置 OUTPUT_DIR。" >&2
  exit 1
fi

CLASS_ARG=()
if [[ -n "$CLASSES" ]]; then
  IFS=',' read -r -a CLASS_VALUES <<< "$CLASSES"
  CLASS_ARG=(--class_names "${CLASS_VALUES[@]}")
fi

EXTRA_ARGS=(--moe_use_cls_token "$MOE_USE_CLS_TOKEN")
if [[ "$SAVE_OVERLAY" == "True" ]]; then
  EXTRA_ARGS+=(--save_overlay --overlay_alpha 0.45)
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
  --use_cls_token "$USE_CLS_TOKEN" \
  --img_score_topk_ratio "$IMG_SCORE_TOPK_RATIO" \
  --gaussian_sigma "$GAUSSIAN_SIGMA" \
  --max_images_per_class "$MAX_IMAGES_PER_CLASS" \
  "${CLASS_ARG[@]}" \
  "${EXTRA_ARGS[@]}"
