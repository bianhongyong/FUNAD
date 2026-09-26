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
SAVE_PATH="/media/honeywell/E/bhy/FUNAD/save_results/codex"
DATASET="mvtec"         # mvtec | visa
EPOCH=20
BATCH_SIZE=16
LR=2e-5
SEED=0
NUM_WORKERS=4
FEATURE_MODEL="dinov3_vitb16"    # dinov3_vits16plus | dinov3_vitb16 | dinov3_vitl16 | dinov3_vitl16plus | dinov3_vith16plus | dinov3_vit7b16

# Optional switches:
# - Uncomment to force CPU FAISS index:
# EXTRA_ARGS+=(--faiss_cpu_index)
# - Uncomment to enable class adaptive threshold:
# EXTRA_ARGS+=(--use_class_adaptive_threshold --adaptive_threshold_quantile 0.7)
# - Uncomment to enable per-class PCA pseudo-label scorer:
# EXTRA_ARGS+=(--pseudo_label_scoring pca --pseudo_label_pca_ev 0.99)
# - Or set fixed PCA dimension instead of explained variance:
# EXTRA_ARGS+=(--pseudo_label_scoring pca --pseudo_label_pca_dim 128)


# EXTRA_ARGS=(--save_log --kl --weight 2.5 --threshold 0.4 --noise_threshold 0.8
#       --use_moe_discriminator --moe_num_expert 16 --gate_aux_weight 0.5 --moe_top_k 1 --moe_use_cls_token
#       --eval_interval 10 --greedy_keep_images 2                 
#       --moe_expert_vis_enable
#       --pseudo_label_distance_norm percentile --pseudo_label_distance_norm_percentile 99 --pseudo_label_scoring nn
# )

# Normal sample selection mode in Phase 2:
#   --normal_sample_selection threshold  (default, uses norm_score < 0.5)
#   --normal_sample_selection quantile   (uses top N% lowest scores, set via --normal_sample_quantile)
# Example: EXTRA_ARGS+=(--normal_sample_selection quantile --normal_sample_quantile 0.3)

EXTRA_ARGS=(--save_log --kl --weight 2.5 --use_mad_threshold --noise_threshold 0.8
      --eval_interval 1 --greedy_keep_images 10 --use_moe_discriminator --moe_hard_class_gate --moe_use_cls_token
      --moe_expert_vis_enable
      --normal_sample_selection quantile --normal_sample_quantile 0.2
      --memory_bank_freeze_start_epoch -1
      --pseudo_label_scoring pca
      #--resume /media/honeywell/E/bhy/FUNAD/save_results/muti_class_residual_correst_dinov3vitl16_moe_discriminator_hard_gate/0617/visa/10%/gaussian_True_noise_10%_balancing_True_oto_True_weight_2.5_multiclass_residual_train_checkpoint.pt
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
  "${EXTRA_ARGS[@]}"
