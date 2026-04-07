# DAFS 模块迁移说明（CRAS）

本文档整理 `Center-aware Residual Anomaly Synthesis (CRAS)` 中 `DAFS`（Distance-guided Anomaly Feature Synthesis）模块的可迁移实现，便于在其他项目中独立复用。

## 1. 原仓库中 DAFS 对应代码

在 `cras.py` 的 `CRAS._train_discriminator()` 中，`Anomaly Synthesis` 代码段就是 DAFS 核心：

```python
# 3.Anomaly Synthesis
dist_norm = torch.norm(true_feats - center_feats, dim=1)
noise = torch.normal(0, self.noise, true_feats.shape).to(self.device)
noise_norm = torch.norm(noise, dim=1)
scale = noise_norm / dist_norm
scale = scale / scale.mean()
scale = (scale - 1) * self.k + 1
noise = scale.unsqueeze(1) * noise
fake_feats = true_feats + noise.detach()
```

## 2. 方法含义（简述）

- 输入：
  - `residual_feats`：残差特征（你当前项目已提前构造好）
- 思想：
  - 直接计算残差特征的 L2 范数 `res_norm`
  - 采样高斯噪声并计算 `noise_norm`
  - 用 `noise_norm / res_norm` 得到每个样本的噪声缩放系数，再归一化并通过 `k` 控制强度
  - 把缩放后的噪声直接加到 `residual_feats`，得到 `fake_residual_feats`（伪异常残差特征）

## 3. 可直接迁移的最小实现

> 该实现与原逻辑一致，并补充了数值稳定性（`eps`）处理。

```python
import torch


class DAFS(torch.nn.Module):
    """
    Distance-guided Anomaly Feature Synthesis (DAFS)

    Args:
        noise_std: 高斯噪声标准差（对应原 CRAS 的 self.noise）
        k: 缩放强度（对应原 CRAS 的 self.k）
        eps: 数值稳定项，防止除 0
        detach_noise: 是否对噪声分支 detach（原实现是 True）
    """

    def __init__(self, noise_std=0.015, k=0.3, eps=1e-8, detach_noise=True):
        super().__init__()
        self.noise_std = noise_std
        self.k = k
        self.eps = eps
        self.detach_noise = detach_noise

    def forward(self, residual_feats: torch.Tensor):
        """
        Args:
            residual_feats: [N, C]
        Returns:
            fake_residual_feats: [N, C]
            noise: [N, C]
            scale: [N]
        """
        res_norm = torch.norm(residual_feats, dim=1).clamp_min(self.eps)

        noise = torch.normal(
            mean=0.0,
            std=self.noise_std,
            size=residual_feats.shape,
            device=residual_feats.device,
            dtype=residual_feats.dtype,
        )
        noise_norm = torch.norm(noise, dim=1).clamp_min(self.eps)

        scale = noise_norm / res_norm
        scale = scale / scale.mean().clamp_min(self.eps)
        scale = (scale - 1.0) * self.k + 1.0

        noise = scale.unsqueeze(1) * noise
        if self.detach_noise:
            noise = noise.detach()

        fake_residual_feats = residual_feats + noise
        return fake_residual_feats, noise, scale
```

## 4. 训练中接入方式

```python
dafs = DAFS(noise_std=0.015, k=0.3).to(device)
fake_residual_feats, noise, scale = dafs(residual_feats)
```

其中：
- `residual_feats`：你已有的残差特征（例如 `true_feats - center_feats`）

## 5. 迁移时注意事项

- 输入应为残差特征 `residual_feats`（通常 `[N, C]`）。
- 若特征是 `[B, P, C]`，可先 reshape 为 `[B*P, C]` 后再送入 DAFS。
- 建议保留 `detach_noise=True`，以复现原始训练行为。
- `noise_std` 与 `k` 是核心超参，建议先沿用论文默认设置再微调。

---

如需，我可以继续补一份 `dafs.py` 独立源码文件 + `pytest` 单测模板（shape、梯度、数值稳定性检查）。
