#!/usr/bin/env bash
set -euo pipefail

cd /media/honeywell/D/bhy/my_research/FUNAD

# ===== User-configurable =====
DATA_PATH="/media/honeywell/D/bhy/dataset/MVTec"
OUTPUT_DIR="tsne_outputs_no_cls_token"
CLASSES="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"

# Set your checkpoint here. If empty, script runs without checkpoint.
CKPT="/media/honeywell/E/bhy/FUNAD/save_results/multi_redusial_correst/mvtec/10%/gaussian_True_noise_10%_balancing_True_oto_False_weight_0_multiclass_residual_localnet.pt"
REPORT_STAGE="after"  # before | after

# Common options
COMMON_ARGS=(
  --data_path "$DATA_PATH"
  --output_dir "$OUTPUT_DIR"
  --classes $CLASSES
  --batch_size 16
  --num_workers 4
  --max_per_group 50
  --perplexity 30
  --num_reference_images 4
  --output tsne_residual.png
  --distance_output tsne_distance_distribution.png
)

if [[ -n "${CKPT}" ]]; then
  python tsne_residual_features.py \
    "${COMMON_ARGS[@]}" \
    --checkpoint "$CKPT" \
    --checkpoint_key net \
    --report_stage "$REPORT_STAGE" \
    --difference_output tsne_center_shift.png
else
  python tsne_residual_features.py "${COMMON_ARGS[@]}"
fi
