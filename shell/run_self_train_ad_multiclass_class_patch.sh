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
SAVE_PATH="${SAVE_PATH:-/media/honeywell/E/bhy/FUNAD/save_results/class_patch}"
DATASET="${DATASET:-mvtec}"   # mvtec | visa
EPOCH="${EPOCH:-20}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-2e-5}"
FUSION_LR="${FUSION_LR:-}"    # empty: same as LR
SEED="${SEED:-0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
FEATURE_MODEL="${FEATURE_MODEL:-dinov3_vitl16}"

# V2-A: legacy adapter (isolates the dynamic fusion gain)
# V2-C: class_residual (beta/FiLM class-strength adapter)
FEATURE_ADAPTER_MODE="${FEATURE_ADAPTER_MODE:-legacy}"

# Leave empty to use DINO_FUSION_CANDIDATE_LAYERS[FEATURE_MODEL].
FUSION_LAYERS="${FUSION_LAYERS:-}"

# Query-conditioned coarse Top-K rerank (K persisted in the checkpoint).
FUSION_TOPK="${FUSION_TOPK:-32}"
FUSION_QUERY_CHUNK="${FUSION_QUERY_CHUNK:-256}"
FUSION_CANDIDATE_CHUNK="${FUSION_CANDIDATE_CHUNK:-8}"

# Patch gate.
GATE_EMBEDDING_DIM="${GATE_EMBEDDING_DIM:-32}"
GATE_HIDDEN_DIM="${GATE_HIDDEN_DIM:-64}"
GATE_NORM="${GATE_NORM:-layer_norm}"
GATE_LAMBDA_INIT="${GATE_LAMBDA_INIT:-0.1}"
GATE_USE_CONTEXT="${GATE_USE_CONTEXT:-False}"

# V2-C adapter.
ADAPTER_HIDDEN_DIM="${ADAPTER_HIDDEN_DIM:-128}"
ADAPTER_BETA_INIT="${ADAPTER_BETA_INIT:-0.15}"

# Fused reference cache refresh schedule.
#   epoch / 1: default (refresh once per epoch)
#   step  / 1: exact-NN reference mode (slow; for controlled comparison)
REFRESH_UNIT="${REFRESH_UNIT:-epoch}"
REFRESH_INTERVAL="${REFRESH_INTERVAL:-1}"
REFERENCE_STORAGE_DTYPE="${REFERENCE_STORAGE_DTYPE:-float16}"

FUSION_ARGS=(
  --fusion_mode class_patch
  --feature_adapter_mode "$FEATURE_ADAPTER_MODE"
  --fusion_temperature "${FUSION_TEMPERATURE:-1.0}"
  --fusion_norm "${FUSION_NORM:-none}"
  --fusion_topk "$FUSION_TOPK"
  --fusion_query_chunk "$FUSION_QUERY_CHUNK"
  --fusion_candidate_chunk "$FUSION_CANDIDATE_CHUNK"
  --gate_embedding_dim "$GATE_EMBEDDING_DIM"
  --gate_hidden_dim "$GATE_HIDDEN_DIM"
  --gate_norm "$GATE_NORM"
  --gate_lambda_init "$GATE_LAMBDA_INIT"
  --gate_use_context "$GATE_USE_CONTEXT"
  --adapter_hidden_dim "$ADAPTER_HIDDEN_DIM"
  --adapter_beta_init "$ADAPTER_BETA_INIT"
  --reference_refresh_unit "$REFRESH_UNIT"
  --reference_refresh_interval "$REFRESH_INTERVAL"
  --reference_storage_dtype "$REFERENCE_STORAGE_DTYPE"
  --residual True
)
if [[ -n "$FUSION_LAYERS" ]]; then
  # shellcheck disable=SC2206
  FUSION_ARGS+=(--fusion_layers $FUSION_LAYERS)
fi
if [[ -n "$FUSION_LR" ]]; then
  FUSION_ARGS+=(--fusion_lr "$FUSION_LR")
fi

# Optional diagnostics:
# - Cached-vs-current coarse NN drift every 100 steps:
#   FUSION_ARGS+=(--fusion_stale_probe_interval 100)
# - Top-K recall vs exact dynamic NN (very expensive):
#   FUSION_ARGS+=(--fusion_topk_recall_probe_interval 500)
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
