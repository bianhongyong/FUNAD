# 内存库自我进化（Memory Bank Self-Evolution）实现方案说明

本文面向当前代码基线：
- `self_train_ad_multiclass.py`
- `self_train_ad_multiclass_residual.py`

目标是将“每轮重建 memory bank”进一步升级为“保留历史高置信 normal 原型”的稳定进化机制，使伪标签在迭代过程中更稳、更抗噪。

---

## 1. 背景与问题定义

当前流程本质上是：
1) 用当前模型筛选更像 normal 的样本；
2) 用筛选结果构建每类 memory bank；
3) 用 patch 到 bank 的距离生成伪标签；
4) 用伪标签训练下一轮模型。

这已经是“联合进化（model <-> bank <-> pseudo labels）”。

但当前常见痛点是：
- 每轮 bank 完全替换，跨轮不连续，伪标签会抖动；
- 若某轮 normal 筛选被污染，会引入确认偏差；
- 后期损失继续下降，但 pixel 指标进入平台或轻微回落。

因此核心改进方向是：**在保持适应性的同时，给 bank 加“历史稳定记忆”**。

---

## 2. 总体设计原则

不论采用哪种具体方法，都建议遵循以下原则：

- **分离快慢记忆**：快速跟随当前模型（fresh）与慢速积累稳定原型（stable）同时存在。
- **置信门控写入**：只有高置信 normal 才进入长期记忆，减少污染。
- **跨轮平滑更新**：用 EMA / 动量 / 投票机制降低单轮误差影响。
- **类别条件化**：每类独立维护状态，防止类间混淆。
- **可回退性**：提供一组开关参数，可快速退回原始“每轮重建”基线。

---

## 3. 方案 A：类级 EMA 原型（最低工程成本）

### 3.1 思想
每类维护一个（或少量）原型向量，不直接保存大量历史 patch。
每轮根据高置信 normal 的均值更新原型：

`mu_c <- m * mu_c + (1 - m) * mean(f_c_t)`

其中：
- `mu_c`：类 `c` 的稳定原型
- `f_c_t`：第 `t` 轮筛出的类 `c` 高置信 normal 特征
- `m`：动量（建议 0.95~0.995）

### 3.2 接入点
放在 `precompute_pseudo_labels_multiclass(...)` 的 Phase 2 后：
- 当前轮拿到 class-wise normal features
- 先更新 `mu_c`
- Phase 3 计算距离时，可混合：
  - 仅对 `mu_c` 距离
  - 或 `d = lambda * d(mu_c) + (1-lambda) * d(fresh_bank)`

### 3.3 优缺点
- 优点：实现快、显存开销极低、便于讲述“历史记忆”。
- 缺点：原型表达能力弱，复杂多模态 normal 类会欠拟合。

### 3.4 推荐参数
- `proto_momentum`: 0.99
- `proto_mix_lambda`: 0.3~0.7
- `min_samples_per_class_for_update`: 64（过少则跳过更新）

---

## 4. 方案 B：双库融合（Stable + Fresh，推荐首选）

### 4.1 思想
每类维护两套 bank：
- **Fresh bank**：每轮重建，快速响应当前模型；
- **Stable bank**：跨轮保留，仅吸收高置信 normal，慢速更新。

伪标签距离融合：
`d = lambda * d_stable + (1 - lambda) * d_fresh`

### 4.2 Stable bank 更新策略
- 输入：本轮高置信 normal 特征（类内）
- 写入：动量并库 or 容量受限追加
- 淘汰：FIFO 或按低置信/离群优先淘汰

### 4.3 接入点
在当前 Phase 2 产出 `normal_features_by_class` 后：
1) 直接建立 Fresh bank（原流程）；
2) 更新 Stable 状态（新增状态容器）；
3) Phase 3 搜索时对两库都 search 并融合距离。

### 4.4 优缺点
- 优点：稳定性和适应性兼顾，叙事最完整，通常效果最稳。
- 缺点：计算量约增加 1.5x~2x（多一次 search/索引维护）。

### 4.5 推荐参数
- `stable_momentum`: 0.99
- `stable_mix_lambda`: 0.4~0.7
- `stable_capacity_per_class`: 20k~100k patch 特征（按显存调）
- `stable_write_quantile`: 图像级最低 20%~40% 才允许写入

---

## 5. 方案 C：跨轮一致性投票（Consensus Gate）

### 5.1 思想
不要求某轮“看起来 normal”就立刻进入 stable。
给每个样本（或 patch）维护计数器，只有在最近 `T` 轮中至少 `K` 次被判高置信 normal，才允许写入 stable。

### 5.2 两种粒度
- **图像粒度投票**：简单，维护成本低，适合先做。
- **patch 粒度投票**：更细，但状态规模大很多。

### 5.3 接入点
在 Phase 1 或 Phase 2 的筛选处新增：
- `consistency_state[sample_id]` 更新
- `eligible_for_stable = votes >= K`

### 5.4 优缺点
- 优点：对偶发误判非常鲁棒，抗噪明显。
- 缺点：冷启动慢，前期可用 stable 样本少。

### 5.5 推荐参数
- `consistency_window_T`: 5
- `consistency_min_K`: 3
- `warmup_epochs_without_gate`: 3~5（热启动后再启用）

---

## 6. 方案 D：容量受限持久池（Reservoir / FIFO Memory）

### 6.1 思想
每类维护一个固定大小的长期特征池，不每轮清空：
- 新特征进入时，如果超容量则淘汰旧或低置信特征。

### 6.2 淘汰策略
- FIFO（最简单）
- Least-confidence first（需要缓存写入置信度）
- Outlier first（距离类中心最远先淘汰）

### 6.3 接入点
Phase 2 结束后更新持久池；
Phase 3 直接使用持久池建索引（可按 N 轮重建一次 FAISS）。

### 6.4 优缺点
- 优点：保留多样性比单原型强，历史信息完整。
- 缺点：工程复杂度中等，索引维护和内存管理需仔细。

### 6.5 推荐参数
- `pool_capacity_per_class`: 30k~150k
- `rebuild_index_every_n_epochs`: 1~5
- `eviction_policy`: `fifo`（先起步）

---

## 7. 方案 E：EMA Teacher 建库（隐式历史平滑）

### 7.1 思想
引入 teacher 网络（localnet 参数 EMA）：
- Student 用伪标签训练；
- Teacher 不反传，仅用于生成筛选分数和建 bank 特征；
- Teacher 参数更新：`theta_t <- tau * theta_t + (1 - tau) * theta_s`

### 7.2 接入点
- 复制一份 `localnet_teacher`
- `precompute_pseudo_labels_multiclass` 改用 teacher 前向
- 每次 student optimizer step 后更新 teacher

### 7.3 优缺点
- 优点：大幅减少伪标签抖动，方法叙事标准、容易被审稿理解。
- 缺点：实现量高于 A/B，算力基本不降。

### 7.4 推荐参数
- `teacher_ema_tau`: 0.999
- `teacher_warmup_epochs`: 1~3

---

## 8. 方案 F：Core + Cloud 分层原型

### 8.1 思想
每类两层记忆：
- **Core prototypes**：少量稳定簇中心（慢更新）
- **Cloud bank**：大规模近期样本（快更新）

伪标签距离融合：
- `d = min(d_core, d_cloud)` 或
- `d = alpha * d_core + (1-alpha) * d_cloud`

### 8.2 适用场景
normal 分布显著多模态（角度、材质、纹理变化大），单库难兼顾。

### 8.3 优缺点
- 优点：表达力最强，兼顾稳定与细节。
- 缺点：实现与调参负担最大。

---

## 9. 与当前代码结构的对齐建议

### 9.1 建议改造最小闭环（优先）
优先实现 **方案 B（Stable + Fresh）**，再叠加 **方案 C（一致性投票）**。

理由：
- 不改变主训练逻辑和损失定义；
- 只改 `precompute_pseudo_labels_multiclass(...)` 和少量状态管理；
- 最容易做出“伪标签逐轮变稳”的可视化证据。

### 9.2 代码级新增状态（建议）
- `stable_memory_state`（按类维护长期特征）
- `consistency_state`（按样本 id 维护跨轮投票）
- `epoch_context`（传入当前 epoch，支持 warmup 和调度）

### 9.3 与 `residual` 版本协同
在 `self_train_ad_multiclass_residual.py` 中，残差特征已经降低类内漂移；
再加 stable 机制通常更稳，推荐优先落地 residual 版本。

---

## 10. 训练与评估中的关键监控项

为验证“自我进化是否真的提升伪标签精度”，建议新增以下日志：

- 每类每轮：
  - 进入 Fresh / Stable 的样本数与 patch 数
  - Stable 写入拒绝率（被门控拒绝比例）
  - 距离分布统计（mean/std/quantile）
- 伪标签质量代理指标：
  - `positive patch ratio`（全局与按类）
  - 连续两轮伪标签 IoU（稳定性）
- 结果指标：
  - per-class image AUROC
  - per-class pixel AUROC
  - 均值和标准差（多 seed）

---

## 11. 推荐消融实验矩阵（可直接执行）

### 11.1 第一阶段：确认稳定性收益
- Baseline：原始每轮重建
- A：EMA 原型
- B：Stable+Fresh（lambda=0.5）
- C：Stable+Fresh + consistency vote

固定其余超参，先跑 3 seeds，对比：
- pixel mean / img mean
- 指标方差
- 最差 3 类提升

### 11.2 第二阶段：参数敏感性
对最佳方案扫：
- `stable_mix_lambda`: [0.3, 0.5, 0.7]
- `stable_momentum`: [0.95, 0.99, 0.995]
- `consistency (T, K)`: [(3,2), (5,3), (7,4)]

---

## 12. 风险与防护

- **风险 1：Stable 被早期错误污染**
  - 防护：warmup 期间不写 stable；仅最低分位写入；加入一致性门控。

- **风险 2：过稳导致适应慢**
  - 防护：减小 `stable_mix_lambda`，提高 fresh 权重；降低 momentum。

- **风险 3：类不平衡导致某些类 stable 很稀疏**
  - 防护：按类最小写入量保障；类自适应写入阈值。

---

## 13. 实施优先级建议

若目标是“尽快验证思路 + 工程可控”，推荐顺序：
1) 先做方案 B（双库融合）
2) 再加方案 C（一致性投票）
3) 若仍有抖动，再考虑方案 E（EMA teacher）

这样能在最小改动下验证“memory 自我进化”是否真正带来伪标签净化。

---

## 14. 一段可用于论文/报告的摘要表述

我们将迭代重建 memory bank 扩展为一种带历史约束的自我进化机制：通过并行维护快速适应的 fresh memory 与跨轮稳定的 stable memory，并结合一致性门控的高置信写入策略，显式抑制单轮筛选误差的传播。该机制使伪标签从“瞬时估计”转为“时序平滑估计”，从而在噪声训练数据下获得更稳定的异常分数学习信号。

