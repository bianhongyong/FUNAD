# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

FUNAD (WACV 2025) 的全无监督异常检测复现 + 扩展。核心思路：在无标签且含噪的训练数据中，通过**迭代重构记忆库 (IRMB)** 生成伪标签，配合**互平滑损失 (mutual smoothness loss)** 进行自训练。主要扩展了多类别残差特征 + 混合专家 (MoE) 判别器分支。

## 训练 / 推理入口

| 脚本 | 说明 |
|------|------|
| `self_train_ad.py` | 原始 FUNAD，单类别含噪自训练（README 示例对应此脚本） |
| `self_train_ad_multiclass_dinov3.py` | **主力脚本**：残差特征 + 多类自训练，主干 DINOv3 (torch.hub)，可选 MoE 判别器 |
| `self_train_ad_multiclass_dinov3_clean.py` | 上者的精简版（去除实验性分支） |
| `self_train_ad_singleclass_dinov3.py` | DINOv3 单类自训练 |
| `memory_score_generation_multiclass.py` | **蒸馏 Step 1**：用 memory bank 的 cosine distance 生成 per-patch anomaly score 图（teacher 信号），按类 ensemble 平均后存盘 |
| `self_train_ad_distillation.py` | **蒸馏 Step 2**：用 Step 1 的 score 图作软监督蒸馏 student localnet（MSE/L1，identity 残差） |
| `src/inference/infer_multiclass_residual.py` | 多类残差推理（加载 `*_localnet.pt` checkpoint，输出 anomaly map / overlay） |
| `count_dino_localnet_params.py` | 统计 localnet / MoE 参数量（用 NaiveGate 绕过 SwitchGate 的 top_k==1 限制） |

Shell 脚本启动示例（`shell/` 目录，注意 `shell/` 在 `.gitignore` 中）：
```bash
bash shell/run_self_train_ad_multiclass.sh   # 多类残差自训练
bash shell/run_infer_multiclass_residual.sh  # 多类残差推理
```
> 这些 shell 脚本内置 `conda activate funad` 与硬编码的 `/media/honeywell/...` 数据/checkpoint 路径，换机器需先改路径。

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
│   │   ├── epoch_precompute.py        # 伪标签预计算 (precompute_pseudo_labels_*)
│   │   ├── pseudo_label_scorers.py    # 打分子：NN (FAISS), Mahalanobis, PCA ensemble
│   │   └── pseudo_label_normalizers.py# 距离→[0,1] 归一化：MinMax / Percentile / ...
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
│   ├── loss.py                 # 损失函数：balanced BCE, one-to-one loss, origin regularizer, log-barrier OCC
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
3. **Phase 3** — 对全量 patch，计算其与对应类 memory bank 的距离（k-NN / Mahalanobis / PCA ensemble，见 `--pseudo_label_scoring`），按类归一化得到 distance_map

### 损失函数（每个 batch）

- **BCE loss**: 以 distance_map 阈值为软标签，监督 localnet 输出的 anomaly score
- **One-to-one loss (互平滑)**: 对同类内互相为最近邻的 patch 对，拉近它们的 score
- **(可选) Origin regularizer**: 正常 patch 推向零，异常 patch 保留原始特征
- **(可选) MoE gate aux loss**: 专家负载均衡损失

### MoE 判别器

`FMoETransformerMLP` 替代原始的 3 层 MLP。每个 expert 是 3 层 Linear+LeakyReLU。支持 cls_token 作为门控路由输入。

## 常用命令行选项（`self_train_ad_multiclass_dinov3.py`）

- `--feature_model dinov3_vitl16` - 主干，默认 `dinov3_vitb16`；可选 vits16plus/vitb16/vitl16/vitl16plus/vith16plus/vit7b16
- `--use_moe_discriminator` - 启用 MoE 判别器
- `--moe_num_expert 16 --moe_top_k 1` - expert 数与路由 top-k
- `--moe_hard_class_gate` - 每类硬路由到专属 expert（替代软门控）
- `--moe_use_cls_token` - 用 cls_token 作门控路由输入
- `--gate_aux_weight 0.5` - 门控辅助（负载均衡）损失权重
- `--pseudo_label_scoring [nn|mahalanobis|pca]` - 伪标签打分方式（`pca` = PCA ensemble）
- `--ensemble_size 100 --memory_sampling_ratio 0.1` - PCA ensemble 模型数与每次采样比例
- `--normal_sample_selection [threshold|quantile]` - Phase 2 正常样本筛选；`quantile` 配 `--normal_sample_quantile 0.1`
- `--use_mad_threshold --mad_k 3` - 用 MAD（中位数绝对偏差）自适应伪标签阈值；`--mad_k_per_class screw:10.0` 按类覆盖
- `--threshold_stage1 0.15 --threshold_stage2 0.1 --threshold_epoch_split 1` - 分阶段伪标签二值化阈值
- `--noise_threshold 0.8` - 不确定（忽略）区间上界
- `--use_class_adaptive_threshold --adaptive_threshold_quantile 0.7` - 每类自适应阈值
- `--memory_bank_freeze_start_epoch -1` - 从该 epoch 起冻结 memory bank（-1 = 不冻结）
- `--eval_interval 3` - 每 N 个 epoch 评估；`--resume <ckpt>` 断点续训

> **已弃用 / 向后兼容**（在部分脚本中仍接受但被忽略）：`--greedy_keep_images`、`--k_number`、旧版 `--pseudo_label_scoring blend`。以脚本内 argparse 与 `shell/` 注释为准。

## 蒸馏两阶段流程（distillation）

`self_train_ad.py` 系列是端到端自训练；另有一条独立的 teacher-student 蒸馏路线：

1. `memory_score_generation_multiclass.py` — 逐类用随机 memory bank 的 cosine distance ensemble 生成稳定 anomaly score 图，存为 `memory_scores.pth`
2. `self_train_ad_distillation.py` — 以该 score 图（按类 min-max 归一化）为 teacher 软监督，MSE/L1 蒸馏 student localnet，直接预测 per-patch [0,1] 分数（无残差，identity 输入）

## 关键环境

- Python 3.9 + PyTorch 2.0.1 + CUDA 11.8
- FAISS (GPU) 用于 k-NN 搜索
- torch.hub 加载 DINOv3 (facebookresearch/dinov3)
- AnomalyCLIP_lib 作为 CLIP 特征提取器
- 依赖见 `requirements.txt`（注意 `requirements.txt` 不含 torch/faiss，需按 README 用 conda 单独装）

## 设计文档（根目录 `*.md`，含实现细节，改相关模块前先读）

- `docs_iterative_memory_bank_adapter_ref.md` — IRMB / adapter 参考实现
- `memory_bank_self_evolution_methods.md` — memory bank 自演化方法
- `evaluation_batch_metrics_devdoc.md` — 批量评估指标实现
- `self_train_ad_multiclass_residual_params.md` — 多类残差参数说明
> `/docs/` 目录（论文 PDF + 迁移说明）在 `.gitignore` 中，不随仓库提交。
