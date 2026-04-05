# 多元高斯（马氏距离）实现说明 — 分块迁移用

下文按 **逻辑块** 列出与 SoftPatch 一致的实现，便于复制到其他项目；并说明张量形状与 **单一代表集**（例如 \(N\) 个 \(D\) 维正常特征、不区分空间位置）的用法。

**依赖**：`torch`、`torch.nn`，以及 `typing` 中的 `Any, List, Optional`。协方差逆使用 `torch.linalg.inv`。

---

## 背景

### 原版在做什么

原版把特征组织成 **`[patch, batch, channel]`**：在每个 **patch 索引** 上，沿 **batch**（多张图）把向量当样本，拟合一个多元高斯，再用马氏距离给该位置上的样本打分（用于训练期 patch 权重等）。**检测阶段的 memory bank 近邻与此无关。**

### 迁移到新项目时的需求（与原版假设的差异）

在新项目中，**不要求「同一空间位置」对齐**：特征来自各类 **内存库（memory bank）** 中的代表向量（例如 coreset、多图多 patch 混池后的 \(N\) 条 \(D\) 维向量），**整体上视为同一批正常样本**，只拟合 **一个** 多元高斯 \(\mathcal{N}(\mu, \Sigma)\)，用马氏距离衡量待检特征相对 **整库分布** 的偏离。实现上对应令 **`patch = 1`**，把内存库堆成 **`[1, N, D]`** 再调用下文 **块一** 的 `fit`（见 **块四**）。这与原版「每个空间索引单独一个高斯」不同，但复用同一套 `_cov`、正则与求逆代码。

---

## 块一：`MultiVariateGaussian` 完整类

内含 **`_cov`**（样本协方差）与 **`forward` / `fit`**（按 patch 循环估计均值、协方差、`0.01I` 正则、求逆）。

**输入 `embedding` 形状：`[patch, batch, channel]`**  
**输出**：`mean` 形状 `[channel, patch]`，`inv_covariance` 形状 `[patch, channel, channel]`。

```python
"""Multi Variate Gaussian Distribution."""

from typing import Any, List, Optional

import torch
from torch import Tensor, nn


class MultiVariateGaussian(nn.Module):
    """Multi Variate Gaussian Distribution."""

    def __init__(self, n_features, n_patches):
        super().__init__()

        self.register_buffer("mean", torch.zeros(n_features, n_patches))
        self.register_buffer("inv_covariance", torch.eye(n_features).unsqueeze(0).repeat(n_patches, 1, 1))

        self.mean: Tensor
        self.inv_covariance: Tensor

    @staticmethod
    def _cov(
        observations: Tensor,
        rowvar: bool = False,
        bias: bool = False,
        ddof: Optional[int] = None,
        aweights: Tensor = None,
    ) -> Tensor:

        if observations.dim() == 1:
            observations = observations.view(-1, 1)

        if rowvar and observations.shape[0] != 1:
            observations = observations.t()

        if ddof is None:
            if bias == 0:
                ddof = 1
            else:
                ddof = 0

        weights = aweights
        weights_sum: Any

        if weights is not None:
            if not torch.is_tensor(weights):
                weights = torch.tensor(weights, dtype=torch.float)
            weights_sum = torch.sum(weights)
            avg = torch.sum(observations * (weights / weights_sum)[:, None], 0)
        else:
            avg = torch.mean(observations, 0)

        if weights is None:
            fact = observations.shape[0] - ddof
        elif ddof == 0:
            fact = weights_sum
        elif aweights is None:
            fact = weights_sum - ddof
        else:
            fact = weights_sum - ddof * torch.sum(weights * weights) / weights_sum

        observations_m = observations.sub(avg.expand_as(observations))

        if weights is None:
            x_transposed = observations_m.t()
        else:
            x_transposed = torch.mm(torch.diag(weights), observations_m).t()

        covariance = torch.mm(x_transposed, observations_m)
        covariance = covariance / fact

        return covariance.squeeze()

    def forward(self, embedding: Tensor) -> List[Tensor]:
        device = embedding.device
        patch, _, channel = embedding.shape
        embedding_vectors = embedding.permute(1, 2, 0)

        self.mean = torch.mean(embedding_vectors, dim=0)
        covariance = torch.zeros(size=(channel, channel, patch), device=device)
        identity = torch.eye(channel).to(device)
        for i in range(patch):
            covariance[:, :, i] = self._cov(embedding_vectors[:, :, i], rowvar=False) + 0.01 * identity

        self.inv_covariance = torch.linalg.inv(covariance.permute(2, 0, 1))

        return [self.mean, self.inv_covariance]

    def fit(self, embedding: Tensor) -> List[Tensor]:
        return self.forward(embedding)
```

**说明摘要**

- `_cov`：`observations` 为 `[num_samples, num_features]`，无权重时 `fact = n - ddof`，默认无偏协方差（\(n-1\) 分母）。  
- `forward`：先 `permute` 成 `[batch, channel, patch]`，对 **每个 patch 列** 用 `_cov` 得到 `[channel, channel]`，加 `0.01 * I` 后按 patch 维堆叠并逐块 `inv`。  
- `n_features` / `n_patches` 仅初始化 buffer；**真实维度由 `embedding` 决定**。

---

## 块二：马氏距离（与原版 `_compute_distance_with_gaussian` 等价）

下面写成 **独立函数**，便于在非类里调用。逻辑与原版一致。

**输入**

- `embedding`：`[patch, batch, channel]`  
- `mean`：来自 `fit`，`[channel, patch]`  
- `inv_covariance`：来自 `fit`，`[patch, channel, channel]`

**返回**：`[patch, batch]`，每个位置为 \(\sqrt{(x-\mu)^\top \Sigma^{-1}(x-\mu)}\)。

```python
def mahalanobis_distance(
    embedding: torch.Tensor,
    mean: torch.Tensor,
    inv_covariance: torch.Tensor,
) -> torch.Tensor:
    embedding = embedding.permute(1, 2, 0)

    delta = (embedding - mean).permute(2, 0, 1)

    distances_sq = (torch.matmul(delta, inv_covariance) * delta).sum(2)
    return torch.sqrt(distances_sq)
```

**说明**

- `embedding.permute(1, 2, 0)` → `[batch, channel, patch]`，与 `mean` 的 `[channel, patch]` 广播对齐。  
- `delta` 回到 `[patch, batch, channel]`，与 `inv_covariance` 的 batch 矩阵乘一致。  
- `(delta @ inv) * delta` 再对最后一维求和即马氏距离平方。

原版在 **高斯权重分支**里还会对结果做 `transpose`、与常数相加等，那是 **采样权重** 业务逻辑；**仅算距离时不需要**。

---

## 块三：原版「高斯权重」管线中的调用顺序（逻辑）

仅说明数据流，不涉及工程目录：

1. 得到 `patch_features`，形状 `[patch, batch, channel]`。  
2. `gaussian = MultiVariateGaussian(channel, patch)`，`stats = gaussian.fit(patch_features)`。  
3. `dist = mahalanobis_distance(patch_features, stats[0], stats[1])`，得到 `[patch, batch]`。  
4. 若要做权重，再按需 `transpose`、`+ 1` 等。

---

## 块四：单一代表集 — \(N\) 个点、\(D\) 维、**一个**高斯

没有「同空间位置」时，令 **`patch = 1`**：把代表矩阵 **`[N, D]`** 变成 **`[1, N, D]`**，则 `forward` 内部只对 **一个** patch 循环，得到 **全局** \(\mu\) 与 \(\Sigma^{-1}\)。

```python
# representative: (N, D)，例如 N=1568, D=768
representative: torch.Tensor  # [N, D]
embedding = representative.unsqueeze(0)  # [1, N, D]

gaussian = MultiVariateGaussian(D, 1)
mean, inv_covariance = gaussian.fit(embedding)
# mean: (D, 1), inv_covariance: (1, D, D)

queries: torch.Tensor  # [M, D]
scores = mahalanobis_distance(queries.unsqueeze(0), mean, inv_covariance).squeeze(0)  # [M]
```

**说明**：`mean` 为 `(D, 1)`、`inv_covariance` 为 `(1, D, D)`，与块二广播一致。

---

## 块五（可选）：不用 `nn.Module` 的极简等价

与 `_cov` + `0.01I` + `inv` + 马氏距离数学一致时可手写：

```python
def fit_global_gaussian(representative: torch.Tensor, ridge: float = 0.01):
    """representative: [N, D]"""
    n, d = representative.shape
    mu = representative.mean(dim=0)
    centered = representative - mu
    cov = centered.t().mm(centered) / (n - 1)
    cov = cov + ridge * torch.eye(d, device=cov.device, dtype=cov.dtype)
    inv_cov = torch.linalg.inv(cov)
    return mu, inv_cov


def mahalanobis_global(queries: torch.Tensor, mu: torch.Tensor, inv_cov: torch.Tensor) -> torch.Tensor:
    """queries: [M, D], mu: [D]"""
    delta = queries - mu
    m = (delta @ inv_cov) * delta
    return torch.sqrt(m.sum(dim=-1))
```

若要与块二 **`mean (D,1)` / `inv (1,D,D)`** 混用，注意把 `mu` reshape 或与 `mahalanobis_distance` 的广播规则对齐。

---

## 迁移时注意

- **样本数**：满协方差需要 \(N > D\) 更稳妥；保留 **`0.01 * I`**（或更大 ridge）减轻病态。  
- **内存**：`inv_covariance` 为 **`O(patch · D²)`**；单高斯时为 **`O(D²)`**。  
- **设备**：`fit` 与 `mahalanobis_distance` 的张量需同一 `device`、`dtype`。

---

## 小结

| 场景 | `embedding` 形状 | 结果含义 |
|------|------------------|----------|
| 原版按位置 | `[patch, batch, channel]` | 每个 patch 一个高斯，输出 `[patch, batch]` 距离 |
| 单一代表集 | `[1, N, D]` | 一个高斯，对查询 `[M,D]` 得 `[M]` 距离 |

**最小迁移**：**块一**（整类）+ **块二**（距离函数）；**块四**为你的代表集用法。
