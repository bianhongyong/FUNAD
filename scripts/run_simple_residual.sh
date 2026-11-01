#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${DATA_PATH:?Set DATA_PATH to the clean MVTec-layout dataset root}"
python self_train_ad_multiclass_residual_simple.py \
  --data_path "$DATA_PATH" \
  --save_path "${SAVE_PATH:-./save_results/simple_residual}" \
  --dataset mvtec \
  --feature_model dinov3_vitl16 \
  --feature_layer -1 \
  --noise_std 0.015 \
  --synthesis_mode cras --cras_beta 0.3 \
  --epoch 200 --batch_size 16 --eval_interval 10 \
  "$@"
