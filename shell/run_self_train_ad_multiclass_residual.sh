#!/usr/bin/env bash
set -euo pipefail

cd /media/honeywell/D/bhy/my_research/FUNAD

# ===== Conda 虚拟环境 + PyTorch (CUDA 11.8) =====
CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-funad}"
# shellcheck source=/dev/null
source "$CONDA_SH"
conda activate "$CONDA_ENV"

# ===== User-configurable =====
DATA_PATH="/media/honeywell/D/bhy/dataset/MVTec_overlap/MVTec_noisy10"
SAVE_PATH="/media/honeywell/E/bhy/FUNAD/save_results/muti_class_residual_correst_moe_no_cls_token"
DATASET="mvtec"         # mvtec | visa
NOISE="10%"         # 0% | 1% | 2% | 3% | 5% | 10% | 20%
EPOCH=50
BATCH_SIZE=16
LR=2e-5
SEED=0
NUM_WORKERS=4
FEATURE_MODEL="dino"    # dino | clip

# Optional switches:
# - Uncomment to force CPU FAISS index:
# EXTRA_ARGS+=(--faiss_cpu_index)
# - Uncomment to enable class adaptive threshold:
# EXTRA_ARGS+=(--use_class_adaptive_threshold --adaptive_threshold_quantile 0.7)
EXTRA_ARGS=(--save_log --kl --weight 2.5 --threshold 0.15 --noise_threshold 0.85 --use_moe_discriminator --moe_num_expert 16 --gate_aux_weight 0.1 --eval_interval 10) 

python self_train_ad_multiclass_residual_dinov3.py \
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
