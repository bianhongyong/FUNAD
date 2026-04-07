# PCA 拟合与推理迁移拆解（SubspaceAD）

本文把当前项目中与 **PCA 拟合**、**PCA 推理打分** 直接相关的源码链路拆解出来，便于迁移到其他项目。

---

## 1. 你真正需要迁移的最小模块

按依赖关系，最小可运行链路如下：

1. `FeatureExtractor.extract_tokens(...)`
2. `PCAModel.fit(...)` 或 `KernelPCAModel.fit(...)`
3. `calculate_anomaly_scores(...)`
4. （可选）`post_process_map(...)`
5. （可选）patch 推理：`process_image_patched(...)`

对应源码文件：

- `src/subspacead/core/extractor.py`
- `src/subspacead/core/pca.py`
- `src/subspacead/post_process/scoring.py`
- `src/subspacead/core/patching.py`（仅 patch 模式需要）

---

## 2. 数据与张量形状约定（迁移时最重要）

- `extract_tokens` 输出：
  - `tokens`: `[B, H_p, W_p, C]`（numpy）
  - `grid_size`: `(H_p, W_p)`
  - `saliency_mask`: `[B, H_p, W_p]`（numpy）
- PCA 拟合输入（标准 PCA）：
  - 一批批的 `X_batch`: `[N_batch, C]`
- 推理打分输入：
  - `X`: `[N, C]`
- 推理打分输出：
  - `scores`: `[N]`，再 reshape 回 `[B, H_p, W_p]`

---

## 3. 训练阶段：PCA 如何拟合出子空间

### 3.1 入口（main 流程）

主流程会先构造一个 `feature_generator`，每次 `yield` 一批特征（`[N_batch, C]`），然后调用：

```python
pca_model = PCAModel(k=args.pca_dim, ev=args.pca_ev, whiten=args.whiten)
pca_params = pca_model.fit(
    feature_generator,
    feature_dim,
    total_tokens,
    num_batches,
)
```

如果启用核 PCA：

```python
pca_model = KernelPCAModel(
    k=args.pca_dim,
    kernel=args.kernel_pca_kernel,
    gamma=args.kernel_pca_gamma,
)
pca_params = pca_model.fit(all_train_tokens)  # all_train_tokens: [N, C]
```

---

### 3.2 标准 PCA（`src/subspacead/core/pca.py`）

`PCAModel.fit(...)` 采用两遍流式算法：

1. Pass1 求均值 `mu`
2. Pass2 求协方差 `cov`
3. `torch.linalg.eigh(cov)` 特征分解
4. 按 `k` 或 `ev_ratio` 选主成分
5. 构建 `pca_params`

关键产物（迁移必须保持字段兼容）：

```python
pca_params = {
    "mu": ...,           # [C]
    "components": ...,   # [C, k]
    "eigvals": ...,      # [k]
    "sqrt_eig": ...,
    "k": k,
    "whiten": whiten,
    "eps": eps,
    "cov_Z_inv": diag(1/(eigvals+eps)),
}
```

> 说明：当前打分主路径用到的是 `mu/components/eigvals/k/eps`，其余字段主要是兼容或扩展用途。

---

### 3.3 训练特征构造逻辑（可按需保留）

当前项目在训练 PCA 前做了以下增强/筛选（迁移时可简化）：

- 多图批量抽特征
- k-shot 与数据增强（可选）
- 背景掩码筛选训练 token（`dino_saliency` 可选）
- patch 模式下按 patch 产出 token

如果你的新项目只想快速跑通，可先保留最简版：

- 不做 mask
- 不做 augmentation
- 不做 patch
- 直接整图 token 拟合 PCA

---

## 4. 推理阶段：如何用 PCA 计算异常图

### 4.1 非 patch 模式最小主线

```python
# 1) 提特征
tokens, (h_p, w_p), saliency = extractor.extract_tokens(...)
# tokens: [B, H_p, W_p, C]

# 2) 展平并打分
X = tokens.reshape(B * h_p * w_p, C)
scores = calculate_anomaly_scores(X, pca_params, method, drop_k)
# scores: [B*H_p*W_p]

# 3) 还原异常图
anomaly_maps = scores.reshape(B, h_p, w_p)

# 4) (可选) 背景置零 + 后处理
anomaly_map_final = post_process_map(anomaly_maps[i], image_res)
```

---

### 4.2 `calculate_anomaly_scores` 支持的打分方式

定义在 `src/subspacead/post_process/scoring.py`：

- `reconstruction`
  - 重建 `X_recon`，分数为 `||X - X_recon||^2`
- `mahalanobis`
  - PCA 子空间系数 `Z` 经特征值缩放后的二次型
- `euclidean`
  - PCA 子空间系数 `Z` 的平方和
- `cosine`
  - `X` 与 `X_recon` 的余弦距离

`drop_k` 逻辑：

- 会丢掉前 `k` 个主成分后再打分，常用于减弱最共性方向的影响。

---

### 4.3 patch 模式推理（可选）

`src/subspacead/core/patching.py` 中 `process_image_patched(...)` 的流程：

1. 计算 patch 坐标（支持 overlap）
2. 每批 patch 抽特征并打分
3. 每个 patch 异常图 resize 回 patch 尺寸
4. 按坐标拼接到整图画布
5. 重叠区域做平均

这是迁移时最容易漏掉的点：**必须用计数图做重叠平均**，否则边缘区域分数会偏高。

---

## 5. 参数对迁移行为的直接影响

与 PCA 拟合和推理直接相关的参数：

- PCA：
  - `pca_dim`（优先于 `pca_ev`）
  - `pca_ev`
  - `use_kernel_pca`
  - `kernel_pca_kernel`
  - `kernel_pca_gamma`
- 打分：
  - `score_method`
  - `drop_k`
- 输入特征：
  - `layers`
  - `agg_method`
  - `grouped_layers`
  - `image_res`
  - `docrop`
  - `use_clahe`
- 可选 mask：
  - `bg_mask_method`
  - `mask_threshold_method`
  - `percentile_threshold`
  - `dino_saliency_layer`
- patch：
  - `patch_size`
  - `patch_overlap`

---

## 6. 迁移到新项目的建议目录结构

建议至少拆成以下 4 个文件：

1. `feature_extractor.py`
   - 提供 `extract_tokens(...) -> tokens, grid, saliency`
2. `pca_model.py`
   - 提供 `fit_pca(...) -> pca_params`
3. `scoring.py`
   - 提供 `calculate_anomaly_scores(...)`
4. `inference.py`
   - 串起提特征、打分、reshape、后处理

这样可先跑通非 patch，再按需加 `patching.py`。

---

## 7. 最小迁移伪代码（可直接当模板）

```python
# train.py
extractor = FeatureExtractor(model_ckpt)

def feature_generator(train_images, batch_size):
    for imgs in batched(train_images, batch_size):
        tokens, (h_p, w_p), _ = extractor.extract_tokens(
            imgs, res=image_res, layers=layers, agg_method=agg_method
        )
        B, Hp, Wp, C = tokens.shape
        yield tokens.reshape(B * Hp * Wp, C)

pca_model = PCAModel(k=pca_dim, ev=pca_ev, whiten=False)
pca_params = pca_model.fit(feature_generator, feature_dim=C, total_tokens=N, num_batches=M)
save_pickle(pca_params, "pca_params.pkl")
```

```python
# infer.py
extractor = FeatureExtractor(model_ckpt)
pca_params = load_pickle("pca_params.pkl")

tokens, (h_p, w_p), saliency = extractor.extract_tokens(
    [pil_img], res=image_res, layers=layers, agg_method=agg_method
)
B, Hp, Wp, C = tokens.shape
scores = calculate_anomaly_scores(tokens.reshape(B * Hp * Wp, C), pca_params, "reconstruction", drop_k=0)
anomaly_map = scores.reshape(B, Hp, Wp)[0]
anomaly_map = post_process_map(anomaly_map, image_res)
```

---

## 8. 迁移时常见坑（结合当前源码）

1. `extract_tokens` 依赖模型输出 `attentions`
   - 某些 attention 实现下会拿不到 attention，`dino_saliency` 就不可用。
2. `pca_normality` 与 patch 模式不兼容
   - 当前主流程显式禁止该组合。
3. `drop_k >= k` 会退化
   - 打分可能变全 0（代码有告警）。
4. Kernel PCA 内存开销大
   - 需要先把所有训练特征拼成大矩阵。
5. 包路径命名不一致
   - 仓库里有 `src.subspacead...` 与 `subspacead...` 两种导入写法，迁移时统一包名。

---

## 9. 建议的迁移顺序

1. 先迁移非 patch、无 mask 的标准 PCA `reconstruction` 路径（最稳）。
2. 确认单图异常图结果正确后，再加入 `drop_k` 和其他 score method。
3. 再加入 `dino_saliency` 背景掩码。
4. 最后加 patch 拼接和 specular 过滤（如果你确实需要）。

---

## 10. 本文档覆盖范围

已覆盖：

- PCA 拟合链（标准/核）
- 推理打分链（整图/patch）
- 参数与行为映射

未展开：

- 指标计算（AUROC/AUPR/F1/AUPRO）
- 可视化存图逻辑
- 数据集读取与类别循环

这些部分不是迁移 PCA 子空间与推理核心所必需。

