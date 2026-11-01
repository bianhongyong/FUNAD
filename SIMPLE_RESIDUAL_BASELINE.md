# 最简多类别残差基线（借鉴 CRAS 合成与损失）

独立入口 `self_train_ad_multiclass_residual_simple.py`，默认沿用 ViT-L/16 和 448 裁剪，方便与现有实验比较。

```text
冻结 DINOv3 最后一层 Patch 特征 q
  → 同类别正常参考库中最近邻 m（L2，所有位置可匹配）
  → r = q - m
  → 共享 MLP：C → 1024 → 128 → 1
```

训练使用 CRAS 的 Distance-guided Anomaly Feature Synthesis（DAFS）：

```text
g ~ N(0, noise_std² I)
t_p = ||g_p||₂ / max(||r_p||₂, eps)
alpha_p = 1 + beta * (t_p / max(mean(t), eps) - 1)
r_syn = r + alpha * g
```

`mean(t)` 按官方代码对当前 batch 的全部 Patch（包括不同类别）计算，不是每张图或每类单独归一化。
`noise_std=0.015` 是标准差，`beta=0.3` 是 CRAS 官方默认值；不直接使用原文中不严谨的“variance=0.015”表述。
新增 `eps=1e-8` 防止零残差除零，并把 beta 限制在 [0,1]，避免负缩放；没有额外的尺度裁剪、梯度上升或 LOIO。
噪声和尺度不参与反向传播。所有 Patch 都合成一个异常副本；不生成空间 Mask。

损失遵循官方实现：原始副本标签为 0，合成副本标签为 1。

```text
L_normal = mean(BCEWithLogits(normal_logits, 0))
L_synthetic = mean(BCEWithLogits(synthetic_logits, 1))
L = L_normal + L_synthetic
```

两组权重 1:1，与官方 `Sigmoid + BCELoss` 数学等价。两组数量相等，因此该和是拼接样本全局平均 BCE 的两倍；不再乘类别权重或正样本数量补偿。
推理时只输入真实残差，输出 Sigmoid 分数。

没有多层融合、Patch Gate、类别 Adapter、MoE、LOIO、空间 Mask、重建、OTO 或伪标签。
类别 ID 仅用于选择参考库；所有类别共享同一个 MLP。正常参考图从训练查询中排除，每个类别至少保留一张训练查询。
每类默认选择最多 8 张参考图。仅使用干净 `train/good`；发现 `noisy*` 文件会报错。

这是“残差基线 + CRAS DAFS/BCE 借鉴”，不是完整 CRAS 复现。对于固定参考 m，`(q + alpha*g) - m = r + alpha*g`，因此直接在残差上实现该扰动。
完整 CRAS 的判别输入是 `[原特征, 残差]` 拼接；本实验按“只保留残差”的要求仅输入 r。
Backbone、正常参考库、L2 检索、优化器和学习率也保持简化基线配置，不宣称复现 CRAS 论文精度。

参考：[CRAS 原文 Eq.4–8](https://arxiv.org/html/2505.17551v1#S3.SS4)、[官方训练实现](https://github.com/cqylunlun/CRAS/blob/main/cras.py)。

## 启动

Linux / 服务器：

```bash
DATA_PATH=/path/to/MVTec bash scripts/run_simple_residual.sh
```

或者直接运行（Windows 和 Linux 均可）：

```bash
python self_train_ad_multiclass_residual_simple.py --data_path /path/to/MVTec --feature_model dinov3_vitl16 --feature_layer -1 --synthesis_mode cras --noise_std 0.015 --cras_beta 0.3
```

`noise_std=0.015, beta=0.3` 来自 CRAS，其适合当前 DINOv3 残差空间的程度仍需验证，不是性能保证。
日志记录正常残差 RMS、正常/合成 BCE、缩放均值/范围、实际扰动 RMS。
用 `--synthesis_mode gaussian` 可回到最初固定高斯噪声对照；`--cras_beta 0` 也退化为固定高斯。
VisA 需要与仓库原有训练相同的 MVTec 风格目录，并加 `--dataset visa`。
依赖沿用项目训练环境及 DINOv3 权重；CPU 核心测试不需要这些权重。

## 结果与独立评估

默认输出 `save_results/simple_residual/mvtec/seed_0/`：

- `config.json`：解析后的实验参数，记录实际单层编号。
- `references.pt`：固定正常参考特征及来源图像路径。
- `last.pt`：最后一轮 MLP 和合成/损失配置，不根据测试集指标选模型；仍兼容旧版仅高斯基线的独立评估。
- `metrics_epoch_*.csv`：每类及宏平均 I/P-AUROC、I/P-AP、I/P-F1；`--eval_compute_pro` 可增加 AUPRO。
- `run_stdout.log`：训练、评估日志。

独立评估使用同一个入口，自动恢复 checkpoint 的 Backbone、层号、预处理和评分配置：

```bash
python self_train_ad_multiclass_residual_simple.py --data_path /path/to/MVTec --checkpoint ./save_results/simple_residual/mvtec/seed_0/last.pt
```

移动 checkpoint 时必须同时携带 `references.pt`。该格式独立于旧融合模型，不交给原来的融合推理脚本加载。
每次实验应使用独立 `--save_path`，防止覆盖同一种子已有结果。独立评估输出 `eval_metrics_epoch_*.csv`。

CPU 验证：

```bash
python -m pytest -p no:cacheprovider tests/test_simple_residual.py -q
```
