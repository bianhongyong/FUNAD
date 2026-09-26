# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

FUNAD (WACV 2025) 的全无监督异常检测复现 + 扩展。核心思路：在无标签且含噪的训练数据中，通过**迭代重构记忆库 (IRMB)** 生成伪标签，配合**互平滑损失 (mutual smoothness loss)** 进行自训练。主要扩展了多类别残差特征 + 混合专家 (MoE) 判别器分支。

## 分支与 PR 工作流

- `codex/main` 是本项目后续开发的主线，远端对应 `origin/codex/main`。
- 开始新开发前，先获取远端更新，以最新的 `origin/codex/main` 为基线新建开发分支（默认命名为 `codex/<主题>`）。
- 在开发分支上提交和推送改动，创建以 `codex/main` 为目标分支的 PR；通过 PR 合并回主线。
- 不直接在 `codex/main` 上提交功能改动；同步主线时保留现有工作区改动。

## 训练入口

| 脚本 | 说明 |
|------|------|
| `self_train_ad.py` | 原始 FUNAD，单类别含噪自训练 |
| `self_train_ad_multiclass_dinov3.py` | 残差特征 + 多类自训练，主干使用 DINOv3 (torch.hub) |

Shell 脚本启动示例（`shell/` 目录）：
```bash
bash shell/run_self_train_ad_multiclass_residual.sh
```

单测：
```bash
python tests/test_fmoe_transformer_mlp_forward.py --num-expert 4 --top-k 2
```

## 项目结构

```
FUNAD/
├── self_train_ad*.py          # 训练主脚本（入口）
├── src/
│   ├── model/
│   │   ├── model.py           # localnet / globalnet / MoEStack / MoEDiscriminator / UNetModel
│   │   ├── model_utils.py     # AttentionBlock, ResBlock, Downsample, normalization
│   │   └── moeblock/          # FMoE 实现（来自 FastMoE 的适配）
│   │       ├── transformer.py # FMoETransformerMLP — 主 MoE 层
│   │       ├── layers.py      # FMoE 核心（scatter/gather, all-to-all dispatch）
│   │       ├── gates/         # 多种门控：Naive, GShard, Noisy, Switch, DC, Zero, Faster, Swipe
│   │       ├── linear.py      # 专家内部的 FMoELinear
│   │       └── fastermoe/     # 高性能调度策略
│   ├── train/
│   │   ├── epoch_precompute.py      # 伪标签预计算 (precompute_pseudo_labels_*)
│   │   └── pseudo_label_scorers.py  # 打分子：NN (FAISS), Mahalanobis, PCA
│   └── inference/
│       ├── inference.py             # 原始推理
│       └── infer_multiclass_residual.py  # 残差多类推理
├── dataset/
│   ├── dataset.py              # 原始单类数据集（含 mixup/cutpaste 增广）
│   ├── dataload.py             # ImageDataset
│   ├── dataset_extract.py      # MyDataset（用于推理 / 特征提取）
│   ├── feature_extract.py      # 特征提取器（DINO / CLIP）
│   └── multiclass_feature_dataset.py  # MultiClassFeatureDataset + 类别名
├── utils/
│   ├── loss.py                 # 损失函数：balanced BCE, one-to-one loss, log-barrier OCC
│   ├── train_utils.py          # FAISS 索引构建, 距离计算, topk 特征更新, 种子设定
│   ├── evaluate.py             # 评估：ROC-AUC, PRO, F1-max, per-region overlap
│   ├── sampler.py              # GreedyCoreset / ApproximateGreedyCoreset 抽样
│   ├── print.py                # 训练打印辅助
│   └── memory_bank_stats.py    # 记忆库异常 patch 诊断统计
├── plot/                       # 可视化脚本（t-SNE, MNN pair ratio）
└── docs/                       # 相关论文 + 迁移文档
```

## 核心架构（推理流程）

```
图像 → Feature Extractor (DINOv3/CLIP) → patch features [B, 784, C]
  → (可选) 残差计算: feat - nearest_class_reference → residual features
  → localnet.adaptor (Linear+LeakyReLU) → adapted features
  → localnet.discriminator (MLP or MoEDiscriminator) → patch anomaly scores
```

### 伪标签流水线（每个 epoch 开始前执行）

1. **Phase 1** — 用当前 localnet 对所有训练样本打分，按类 min-max 归一化得到 image_norm_scores
2. **Phase 2** — 按类筛选 norm_score < 0.5 的样本，构造 memory bank（每类通过 GreedyCoreset 压缩到约 2 张图的 patch 量）
3. **Phase 3** — 对全量 patch，计算其与对应类 memory bank 的距离（k-NN / Mahalanobis / PCA / blend），按类归一化得到 distance_map

### 损失函数（每个 batch）

- **BCE loss**: 以 distance_map 阈值为软标签，监督 localnet 输出的 anomaly score
- **One-to-one loss (互平滑)**: 对同类内互相为最近邻的 patch 对，拉近它们的 score
- **(可选) MoE gate aux loss**: 专家负载均衡损失

### MoE 判别器

`FMoETransformerMLP` 替代原始的 3 层 MLP。每个 expert 是 3 层 Linear+LeakyReLU。支持 cls_token 作为门控路由输入。

## 常用命令行选项

- `--use_moe_discriminator` - 启用 MoE 判别器
- `--moe_num_expert 16 --moe_top_k 1` - 16 个 expert，top-1 路由
- `--gate_aux_weight 0.5` - 门控辅助损失权重
- `--pseudo_label_scoring [nn|mahalanobis|blend]` - 伪标签打分方式
- `--use_class_adaptive_threshold` - 启用每类自适应伪标签阈值
- `--feature_model` - `dino` / `clip` / `dinov3_vitl16` 等
- `--eval_interval 10` - 每 N 个 epoch 评估一次
- `--threshold 0.15 --noise_threshold 0.85` - 伪标签二值化阈值与不确定区间上界

## 关键环境

- Python 3.9 + PyTorch 2.0.1 + CUDA 11.8
- FAISS (GPU) 用于 k-NN 搜索
- torch.hub 加载 DINOv3 (facebookresearch/dinov3)
- AnomalyCLIP_lib 作为 CLIP 特征提取器
- 依赖见 `requirements.txt`
