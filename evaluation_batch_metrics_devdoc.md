# `evaluation_batch` 指标设计开发文档（可移植版）

本文档面向“将 Dinomaly 的评估逻辑迁移到其他项目”的开发者/AI 代理，聚焦以下 5 个指标在 `evaluation_batch` 中的**设计方法与实现细节**：

- Image-level: `AP`、`F1-max`
- Pixel-level: `AP`、`F1-max`、`AUPRO`

同时补充这些指标依赖的前置流程（异常图生成、平滑、样本级打分聚合），确保可完整复现。

---

## 1. 目标函数与总体设计

`evaluation_batch` 的目标是：  
在一次 dataloader 全量推理后，统一输出 image-level 与 pixel-level 的多种排名/阈值无关指标，尽量避免单阈值偶然性。

返回顺序如下（7 个值）：

1. `auroc_sp`（image-level AUROC）
2. `ap_sp`（image-level AP）
3. `f1_sp`（image-level F1-max）
4. `auroc_px`（pixel-level AUROC）
5. `ap_px`（pixel-level AP）
6. `f1_px`（pixel-level F1-max）
7. `aupro_px`（pixel-level AUPRO）

> 虽然你本次重点关注 AP/F1-max/AUPRO，但迁移时通常会整段复制，建议保持返回结构一致。

---


## 5. Image-level 指标设计细节

以下基于：

- `y_true_img = gt_list_sp`（0/1）
- `y_score_img = pr_list_sp`（连续分数）

## 5.1 AP（Average Precision）

实现：

```python
ap_sp = average_precision_score(y_true_img, y_score_img)
```

设计要点：

- AP 是 PR 曲线下的面积型指标，聚焦正类（异常类）检索质量
- 对类别不平衡更敏感也更合理（AD 场景常见负样本更多）
- 阈值无关，适合模型横向比较

## 5.2 F1-max（最大 F1）

实现函数：`f1_score_max(y_true, y_score)`

算法步骤：

1. `precs, recs, thrs = precision_recall_curve(y_true, y_score)`
2. `f1s = 2 * precs * recs / (precs + recs + 1e-7)`
3. `f1s = f1s[:-1]`（与阈值数量对齐）
4. 返回 `f1s.max()`

关键细节：

- `precision_recall_curve` 返回的 `precs/recs` 比 `thrs` 多 1 个点，所以必须去掉最后一项
- 加 `1e-7` 防除零
- F1-max 代表“在所有可行阈值中的最优 F1”，不是固定阈值 F1

---

## 6. Pixel-level 指标设计细节

先将批次累计后的像素 GT/预测组织为：

- `gt_list_px`: 先是 `[N,H,W]`，后续 `ravel()` 成一维
- `pr_list_px`: 同上

## 6.1 Pixel-level AP

实现：

```python
ap_px = average_precision_score(y_true_px_flat, y_score_px_flat)
```

解释：

- 把所有测试图像像素视为同一检索集合
- 衡量像素异常排序质量
- 对小缺陷/大背景不平衡仍有较好辨识能力

## 6.2 Pixel-level F1-max

实现与 image-level 相同，只是输入换成 flatten 后的像素序列：

```python
f1_px = f1_score_max(y_true_px_flat, y_score_px_flat)
```

解释：

- 给出“最佳阈值下”的像素分割 F1 上限
- 常用于报告模型可达到的分割上界表现

---

## 7. Pixel-level AUPRO 设计细节（核心）

函数：`compute_pro(masks, amaps, num_th=200)`  
输入必须是：

- `masks`: `ndarray[N,H,W]`，二值 `{0,1}`
- `amaps`: `ndarray[N,H,W]`，连续异常分数

## 7.1 设计目标

PRO（Per-Region Overlap）强调**按缺陷连通域**统计召回，防止大缺陷主导指标。  
AUPRO 则是在不同阈值下，`PRO-FPR` 曲线的面积。

## 7.2 具体算法（逐阈值扫描）

1. 取阈值范围：`min_th = amaps.min()` 到 `max_th = amaps.max()`
2. 线性步进：`delta = (max_th - min_th) / num_th`
3. 对每个阈值 `th`：
   - 二值化预测：`binary_amaps = (amaps > th)`
   - 对每张图的 GT mask 做连通域标记：`measure.label(mask)`
   - 对每个 region：
     - 取该 region 内被预测为 1 的像素数 `tp_pixels`
     - 计算 region 召回 `tp_pixels / region.area`
   - 全 region 均值作为该阈值下 `pro`
   - 背景误检率：
     - `inverse_masks = 1 - masks`
     - `fpr = logical_and(inverse_masks, binary_amaps).sum() / inverse_masks.sum()`

4. 收集所有 `(pro, fpr, threshold)` 点

## 7.3 0~0.3 FPR 截断与归一化

实现保留 `fpr < 0.3` 的点，并将该段 `fpr` 线性归一到 `[0,1]`：

```python
df = df[df["fpr"] < 0.3]
df["fpr"] = df["fpr"] / df["fpr"].max()
pro_auc = auc(df["fpr"], df["pro"])
```

解释：

- 只关注低误报区（工业检测更关心低 FPR）
- 归一化后面积可横向比较

---

## 8. 迁移到其他项目的最小实现骨架

```python
def eval_metrics(all_gt_masks, all_pred_maps, all_img_labels, all_img_scores):
    # image-level
    ap_sp = average_precision_score(all_img_labels, all_img_scores)
    f1_sp = f1_score_max(all_img_labels, all_img_scores)

    # pixel-level
    gt_px = all_gt_masks.ravel().astype(np.uint8)
    pr_px = all_pred_maps.ravel().astype(np.float32)
    ap_px = average_precision_score(gt_px, pr_px)
    f1_px = f1_score_max(gt_px, pr_px)
    aupro_px = compute_pro(all_gt_masks.astype(np.uint8), all_pred_maps.astype(np.float32))

    return ap_sp, f1_sp, ap_px, f1_px, aupro_px
```

---

## 9. 迁移检查清单（给 AI/开发者）

- 数据契约  
  - `masks/amaps` 必须同 shape 的 `N,H,W`
  - `masks` 必须是严格二值 `{0,1}`

- 分数方向  
  - 分数越大越异常（`amaps > th` 判为异常），不要反向

- flatten 时机  
  - AP/F1-max 在 pixel-level 计算前需 `ravel()`
  - AUPRO 不要 flatten，保留 `N,H,W`

- PR 曲线对齐  
  - `f1s = f1s[:-1]` 防止 `threshold` 维度错位

- resize 策略  
  - 预测图用 `bilinear`
  - GT mask 用 `nearest`

- 平滑一致性  
  - 如果想复现原始数值，保持同样高斯核参数（`5, sigma=4`）

- top-k image score  
  - `max_ratio > 0` 时注意 `k=int(H*W*max_ratio)` 至少为 1（可自行加保护）

---

## 10. 已知实现风险与建议修正（迁移时可顺手优化）

1. `np.bool` 已废弃  
   - 原实现 `dtype=np.bool`，建议改为 `dtype=bool` 或 `np.bool_`

2. `DataFrame.append` 已废弃且慢  
   - 建议先收集 list，再 `pd.DataFrame(records)`

3. 阈值步进可能除零  
   - 若 `max_th == min_th`，`delta=0` 会异常；需提前返回或兜底

4. 低 FPR 过滤为空  
   - 若 `df[df["fpr"]<0.3]` 为空，`auc` 会失败；需兜底返回 0 或 NaN

5. `top-k` 可能为 0  
   - 小分辨率 + 很小 `max_ratio` 时 `int(H*W*max_ratio)==0`，要 `max(1,k)`

---

## 11. 与原始代码保持一致的关键结论

- Image-level `AP/F1-max` 是在样本级分数（max/top-k 聚合）上计算  
- Pixel-level `AP/F1-max` 是在全测试集像素 flatten 后计算  
- Pixel-level `AUPRO` 保留三维 mask/map，按连通域计算 PRO，再在 `FPR<0.3` 上做 AUC  
- 预测图默认经过固定高斯平滑，且可能在评估前 resize

按以上约束迁移，可最大程度复现 `evaluation_batch` 的行为与指标数值分布。

