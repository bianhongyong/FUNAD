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
export MPLBACKEND=Agg

# ===== User-configurable =====
NOISE="${NOISE:-10%}"         # 0% | 1% | 2% | 3% | 5% | 10% | 20% | 25%
DATA_PATH="${DATA_PATH:-/media/honeywell/D/bhy/dataset/MVTec_overlap/MVTec_noisy10}"
SAVE_PATH="${SAVE_PATH:-/media/honeywell/E/bhy/FUNAD/save_results/codex_fusion_screw_mad_4}"
DATASET="${DATASET:-mvtec}"   # mvtec | visa
EPOCH="${EPOCH:-20}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-2e-5}"
SEED="${SEED:-0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
FEATURE_MODEL="${FEATURE_MODEL:-dinov3_vitl16}"

# learned_class: one softmax weight vector per class (main setting)
# learned_shared: one shared softmax weight vector (E1 ablation)
FUSION_MODE="${FUSION_MODE:-learned_class}"

# Leave empty to use DINO_FUSION_CANDIDATE_LAYERS[FEATURE_MODEL].
# Example for vitl16: FUSION_LAYERS="3 6 9 12 15 18 21 24"
FUSION_LAYERS="${FUSION_LAYERS:-}"

# Fused reference cache refresh schedule.
#   epoch / 1: default (refresh once per epoch)
#   step  / 1: exact-NN reference mode (slow; for controlled comparison)
REFRESH_UNIT="${REFRESH_UNIT:-epoch}"
REFRESH_INTERVAL="${REFRESH_INTERVAL:-1}"
REFERENCE_STORAGE_DTYPE="${REFERENCE_STORAGE_DTYPE:-float16}"

FUSION_ARGS=(
  --fusion_mode "$FUSION_MODE"
  --fusion_temperature "${FUSION_TEMPERATURE:-1.0}"
  --fusion_norm "${FUSION_NORM:-none}"
  --reference_refresh_unit "$REFRESH_UNIT"
  --reference_refresh_interval "$REFRESH_INTERVAL"
  --reference_storage_dtype "$REFERENCE_STORAGE_DTYPE"
  --residual True
)
if [[ -n "$FUSION_LAYERS" ]]; then
  # shellcheck disable=SC2206
  FUSION_ARGS+=(--fusion_layers $FUSION_LAYERS)
fi

# Optional switches:
# - Probe cached-vs-current NN mismatch every 100 optimizer steps:
#   FUSION_ARGS+=(--fusion_stale_probe_interval 100)
# - Resume a learned-fusion run (keep --fusion_mode/--fusion_layers consistent):
#   FUSION_ARGS+=(--resume /path/to/..._train_checkpoint.pt)
EXTRA_ARGS=(--save_log --kl --weight 2.5 --use_mad_threshold --noise_threshold 0.8
      --eval_interval 1 --greedy_keep_images 10 --use_moe_discriminator --moe_hard_class_gate --moe_use_cls_token
      --moe_expert_vis_enable
      --normal_sample_selection quantile --normal_sample_quantile 0.2
      --memory_bank_freeze_start_epoch -1
      --pseudo_label_scoring pca
)

python self_train_ad_multiclass_dinov3.py \
  --data_path "$DATA_PATH" \
  --save_path "$SAVE_PATH" \
  --dataset "$DATASET" \
  --noise "$NOISE" \
  --epoch "$EPOCH" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --seed "$SEED" \
  --num_workers "$NUM_WORKERS" \
  --feature_model "$FEATURE_MODEL" \
  "${FUSION_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
