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

# Reference matching mode:
#   prior_top1   : class prior selects a single j*; patch alpha only shapes
#                  the residual (default, schema v3).
#   dynamic_topk : historical coarse Top-K + query-conditioned rerank.
MATCH_MODE="${MATCH_MODE:-prior_top1}"
FUSION_QUERY_CHUNK="${FUSION_QUERY_CHUNK:-256}"
FUSION_REF_CHUNK="${FUSION_REF_CHUNK:-1024}"
# dynamic_topk only:
FUSION_TOPK="${FUSION_TOPK:-32}"
FUSION_CANDIDATE_CHUNK="${FUSION_CANDIDATE_CHUNK:-8}"

# Patch gate.
GATE_EMBEDDING_DIM="${GATE_EMBEDDING_DIM:-32}"
GATE_HIDDEN_DIM="${GATE_HIDDEN_DIM:-64}"
GATE_NORM="${GATE_NORM:-layer_norm}"
GATE_LAMBDA_INIT="${GATE_LAMBDA_INIT:-0.1}"
GATE_USE_CONTEXT="${GATE_USE_CONTEXT:-False}"

# Classes whose class prior is pinned to the last candidate layer (weight=1,
# no learned inter-layer fusion). The patch gate may still deviate per patch.
# Empty string disables the pin (e.g. for datasets without screw).
# Use ${VAR-default} (not :-) so an explicit empty value disables the pin.
NO_LAYER_FUSION_CLASSES="${NO_LAYER_FUSION_CLASSES-screw}"
LAST_LAYER_PRIOR_LOGIT="${LAST_LAYER_PRIOR_LOGIT:-8.0}"

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
  --fusion_match_mode "$MATCH_MODE"
  --feature_adapter_mode "$FEATURE_ADAPTER_MODE"
  --fusion_temperature "${FUSION_TEMPERATURE:-1.0}"
  --fusion_norm "${FUSION_NORM:-none}"
  --fusion_query_chunk "$FUSION_QUERY_CHUNK"
  --fusion_ref_chunk "$FUSION_REF_CHUNK"
  --gate_embedding_dim "$GATE_EMBEDDING_DIM"
  --gate_hidden_dim "$GATE_HIDDEN_DIM"
  --gate_norm "$GATE_NORM"
  --gate_lambda_init "$GATE_LAMBDA_INIT"
  --gate_use_context "$GATE_USE_CONTEXT"
  --fusion_no_layer_fusion_classes "$NO_LAYER_FUSION_CLASSES"
  --fusion_last_layer_prior_logit "$LAST_LAYER_PRIOR_LOGIT"
  --adapter_hidden_dim "$ADAPTER_HIDDEN_DIM"
  --adapter_beta_init "$ADAPTER_BETA_INIT"
  --reference_refresh_unit "$REFRESH_UNIT"
  --reference_refresh_interval "$REFRESH_INTERVAL"
  --reference_storage_dtype "$REFERENCE_STORAGE_DTYPE"
  --residual True
)
if [[ "$MATCH_MODE" == "dynamic_topk" ]]; then
  FUSION_ARGS+=(
    --fusion_topk "$FUSION_TOPK"
    --fusion_candidate_chunk "$FUSION_CANDIDATE_CHUNK"
  )
fi
if [[ -n "$FUSION_LAYERS" ]]; then
  # shellcheck disable=SC2206
  FUSION_ARGS+=(--fusion_layers $FUSION_LAYERS)
fi
if [[ -n "$FUSION_LR" ]]; then
  FUSION_ARGS+=(--fusion_lr "$FUSION_LR")
fi
if [[ "${FUSION_PROFILE:-False}" == "True" ]]; then
  FUSION_ARGS+=(--fusion_profile True)
fi

# Optional diagnostics:
# - Cached-vs-current coarse NN drift every 100 steps:
#   FUSION_ARGS+=(--fusion_stale_probe_interval 100)
# - Top-K recall vs exact dynamic NN (dynamic_topk only, very expensive):
#   FUSION_ARGS+=(--fusion_topk_recall_probe_interval 500)
# - prior_top1 stage timings (CUDA-synchronized; disable for max throughput):
#   FUSION_PROFILE=True bash shell/run_self_train_ad_multiclass_class_patch.sh
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
