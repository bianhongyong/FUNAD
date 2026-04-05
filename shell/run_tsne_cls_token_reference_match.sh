#!/usr/bin/env bash

# CLS-token 参考库匹配 + t-SNE 可视化 一键脚本

set -e

PROJECT_ROOT="/media/honeywell/D/bhy/my_research/FUNAD"
DATA_ROOT="/media/honeywell/D/bhy/dataset/MVTec"

cd "$PROJECT_ROOT" || exit 1

# ===== Conda 虚拟环境 + PyTorch (CUDA 11.8) =====
CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-funad}"
# shellcheck source=/dev/null
source "$CONDA_SH"
conda activate "$CONDA_ENV"

PYTHON_BIN="python"

# ===== 实验参数 =====
DATASET="mvtec"
# 想跑哪些类别就写哪些；留空表示所有类别
CLASSES=""

NUM_REF_PER_CLASS=4          # 每类参考图像数量
NUM_TRIALS=5                 # 重复随机抽参考库次数
BATCH_SIZE=16
NUM_WORKERS=4
PERPLEXITY=30
MAX_TSNE_POINTS_PER_GROUP=800
SEED=42

OUTPUT_DIR="tsne_cls_token_outputs"
TSNE_FIGURE="tsne_cls_token_reference_vs_test.png"

# ===== 组装 --classes 参数 =====
CLASS_ARGS=()
for c in $CLASSES; do
  CLASS_ARGS+=(--classes "$c")
done

# ===== 运行 Python 脚本 =====
"$PYTHON_BIN" -m plot.tsne_cls_token_reference_match \
  --data_path "$DATA_ROOT" \
  --dataset "$DATASET" \
  "${CLASS_ARGS[@]}" \
  --num_reference_per_class "$NUM_REF_PER_CLASS" \
  --num_trials "$NUM_TRIALS" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --perplexity "$PERPLEXITY" \
  --max_tsne_points_per_group "$MAX_TSNE_POINTS_PER_GROUP" \
  --seed "$SEED" \
  --output_dir "$OUTPUT_DIR" \
  --tsne_figure "$TSNE_FIGURE" \
  --save_npz true

