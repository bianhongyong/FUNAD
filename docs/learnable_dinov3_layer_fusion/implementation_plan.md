# FUNAD 可学习 DINOv3 层融合：实施与测试路线图

> 文档状态：**计划，不包含代码实现**。核心公式和范围以 [requirements_and_design.md](./requirements_and_design.md) 为准；引用见 [references.md](./references.md)。

## 核心实施 Pipeline（已确认）

**原始特征库固定 + 融合特征库/NN 索引周期性动态刷新 + 融合参数持续训练**，不是每个 Batch 都重新运行 DINOv3。

```text
初始化：
  frozen DINOv3(reference images)
    → Raw Reference Bank[c]: [N_c,L,C]（逐层 Patch 原始特征，一次提取，训练期间固定）
  Fusion(theta_0)(Raw Reference Bank)
    → Fused Reference Cache[c]: [N_c,C] + NN Index（初次构建）

每次训练迭代：
  frozen DINOv3(query) → raw_query[B,P,L,C]
  Fusion(theta_current)(raw_query) → fused_query[B,P,C]
  NN(fused_query, Fused Reference Cache[category]) → selected raw_reference_id
  Raw Bank[selected raw_reference_id] → Fusion(theta_current)，保留梯度
  residual = fused_query(theta_current) - selected_reference_fused(theta_current)
  residual → original Class Adapter → MoE → FUNAD loss
  backward → optimizer.step()：更新 theta、Adapter、MoE

刷新触发器（默认 Epoch 开始，可改为每 N 个 optimizer Step）：
  current theta + fixed Raw Bank
    → 重新融合全部/受影响类别 reference Patch
    → 构建新 NN Index
    → 原子切换 Fused Reference Cache / NN Index
    → 保存 cache_fusion_step / version 等元数据

最终评估 / 独立推理：
  载入 Fusion(theta_checkpoint) → 强制重新融合 Raw Bank 并重建 NN Index
  冻结参数 → NN → fused residual → Adapter / MoE
```

**明确区分**：Raw Bank 是持久化的数据源；Fused Cache 是训练期间周期刷新、可丢弃并重建的派生索引；伪标签记忆库是 FUNAD 现有的另一套记忆库，按自训练 Phase1/2/3 单独更新。

- 默认正式模式：`--reference_refresh_unit epoch --reference_refresh_interval 1`，每 Epoch 开始刷新；可配为 step/N（如每 100 steps）。
- 严格正确性基准：`--reference_refresh_unit step --reference_refresh_interval 1`，每次 optimizer 更新后下一次 NN 搜索前更新缓存。**此为实验对照，不是唯一生产路径**。
- 周期缓存中查询以 `theta_current` 融合，而 NN 缓存来自 `theta_refresh`；**这是受控的近似检索**。取得最近邻索引后，始终从 Raw Bank 取回对应原始参考 Patch，再以 `theta_current` 融合并计算残差，以便回传梯度。这样无法修复陈旧索引误选近邻，需记录 NN mismatch 和性能差异。
- 最终报告指标时必须针对当前 checkpoint 强制刷新参考缓存；不要混用过期训练索引。
- 缓存刷新和 `memory_bank_freeze_start_epoch`（伪标签库冻结）分属不同机制。

## 第一阶段：接口和一致性（P0）

1. **新增逐层特征提取**

   - 文件建议：dataset/feature_extract.py。
   - 新接口 extract_dinov3_multilayer_feature_batch() 接收统一候选层（非类别人工层表），输出 [B,P,L,C]、可选 CLS [B,C]；冻结 DINOv3，仅 backbone 前向用 no_grad。
   - 实现层索引的 deterministic 排序及 1-based 到 0-based 规范转换；旧 extract_dinov3_feature_batch() 不能改变返回形状/默认行为。
   - Test：选 L=1 等价于单层直接输出；同一图像与参考提取的逐层 tensor 对齐；多层平均复原旧模型输出。
2. **实现 ClassConditionedLayerFusion**

   - 新文件建议：src/model/layer_fusion.py。
   - 接口 forward(x: [B,P,L,C], class_idx: [B]) -> [B,P,C]，亦支持参考库 [N,L,C] 与单一类别 c。
   - 参数 theta[num_classes,L]，softmax(theta/tau)，可选无仿射 LayerNorm；确保优化器更新。
   - Test：theta 全 0 且关闭新归一化时，输出与 candidate layers mean 相同；不同类别权重可不同；梯度有限非零。
3. **构建原始多层参考库**

   - 文件建议：utils/train_utils.py。
   - 保留原每类参考图选择规则；新的 raw_reference_memory_by_class[c] = [N_c,L,C]，不 Mean；保存层列表、backbone、图像预处理、类别顺序、缓存精度、版本号。
   - Test：每个 memory row 的 L 份向量必须来自同一个 reference Patch；存取无损/容差内；原参考样本仍从训练数据中移除。
4. **动态融合参考缓存 + 融合后最近邻与残差**

   - Raw Bank 常驻持久存储，只需 DINOv3 提取一次；Fused Cache 在刷新触发时用同一 fusion 模块获得 [N_c,C] 并重建 NN Index；查询每步用当前 theta 获得 [M,C]。
   - 在最近一次刷新生成的 fused feature 上 NN：j*=argmin ||Z_q(theta_current)−Z_m(theta_refresh)||；再取选中 raw_m[j*]，以当前 theta 重新有梯度地 fused 出参考 z_sel，返回 residual=Z_q(theta_current)−z_sel(theta_current)。周期性缓存可能导致 j* 不是当前 theta 的精确 NN。
   - 模块要支持按类别批量、查询分块、参考分块，避免 torch.cdist 的 [query_patches, reference_patches] 全距离矩阵爆显存。
   - Refresh trigger：支持 epoch/step 两种刷新单位、interval 可配置；缓存记录版本、刷新步数，刷新后索引与缓存同时切换；恢复训练允许从 raw 库重新生成融合缓存。
   - Test：人工构造已知最近邻；每步严格刷新 vs 周期刷新；统计近邻匹配一致率、残差偏差、计时和显存；融合梯度不断更新；只更新 theta 而不刷新时显式识别 staleness。
5. **将 P0 接入训练、预计算、评估**

   - self_train_ad_multiclass_dinov3.py：参数/开关/optimizer/保存；
   - src/train/epoch_precompute.py：Phase1/2/3 全部基于新 residual 的输出，原有 PCA/coreset/伪标签后端不改；
   - 现有评估：与训练一致的 fusion 状态，且 eval 阶段禁用 fusion 参数更新；
   - Test：至少完成一轮 smoke test，梯度正常，评估结果可重复。
6. **独立推理与 checkpoint**

   - src/inference/infer_multiclass_residual.py：根据 checkpoint schema 识别 raw memory 与 fusion state；eval() 后融合 raw reference 一次构建 FAISS；强校验 backbone、candidate layers、类别顺序、preprocess；
   - 必须支持旧 localnet.pt（仅 fused reference + net）或清楚提供显式兼容模式；
   - raw reference 一般可存文件/外部 sidecar，checkpoint 需保存可移植引用/哈希；不能简单依赖训练机器绝对路径。
   - Test：相同 checkpoint、同一张图的训练内评估与独立推理残差/异常图一致；加载错误元数据应立即报错。

## 必须优先修复：Fusion 梯度接入（P0 阻断项）

源码审核发现 [self_train_ad_multiclass_dinov3.py](https://github.com/bianhongyong/FUNAD/blob/codex/main/self_train_ad_multiclass_dinov3.py#L563-L698) 的 Gaussian 分支使用 `_copy = x.detach().clone()`，同时 BCE 使用 `gaussian_pred = localnet(_copy)`，所以 BCE 不能经 `x` 反传到拟新增的融合模块。命令行 `--gaussian` 使用 `store_false`，不传时反而是 True；当前 shell 启用 `--kl`，OTO 可能经另一路提供部分梯度，但**不应以 OTO 代替 BCE 的可微链路检查**。详细审计见 [requirements_and_design.md §5.1.1](./requirements_and_design.md#511-反向传播审计已有-funad-主训练存在阻断-bce--fusion-梯度的路径2026-10-22)。

实施顺序：

1. 保留噪声生成的 mask/std/采样逻辑，但使 Gaussian 输入为 `x_perturbed = x + noise.detach()`（其中 noise 根据 x.detach() 等统计生成），**不得 detach 整个 x**；修正实现时需验证维度、mask 和 Hard Class Gate 行为。
2. 在 Fusion 独立于 localnet 定义时，将其参数显式加入所有负责训练上游特征的 optimizer，并更新 checkpoint 保存/恢复；若嵌入 localnet，检查 parameters() 实际包含 theta。
3. 伪标签预计算的 no_grad 保持不变；主训练的 DINOv3 可冻结但融合器不能放入 no_grad。
4. NN 搜索用 detached 缓存选 j*，从 Raw Bank 索引 j* 后**重新以当前 theta 有梯度地融合选中参考**；不能直接用 cached_fused[j*] 算最终 residual（会失去参考端梯度）。
5. 增加严格门禁：关闭 KL 单测仅 BCE 的 theta.grad（Gaussian 开/关）；启用 KL 后分别记录 BCE/OTO 对 theta 的贡献；对比有梯度的选中参考与 detached cache；验证 optimizer.step 确实更新 theta、部分类别缺席批次不会虚构梯度。
6. 小型 autograd sanity test 已验证：完全重融合参考 theta 梯度与直接减缓存 theta 梯度不同；若 residual 整体 detach，theta.grad 为 None。因此“loss.backward() 调用成功”不是足够的验收条件。

## 第二阶段：训练策略和算力优化（P1）

- 初次训练：推荐保留现有 train loss、类别 MoE、PCA pseudo-label 配置，alpha 从近似均匀初始化；可先固定伪标签让融合层学到非零梯度，再打开全量自训练。
- 原始参考库 CPU float16 + 分块融合，匹配计算 FP32；单纯改成 raw 多层 reference 可能极大增加内存，不应无估算直接 all-layer 常驻 GPU。
- **P0 即实现动态刷新架构**：主模式每 Epoch 开始重建派生 Fused Cache/NN Index，支持每 N Step；每步重建仅作为严格 NN 结果基准。记录索引版本滞后与刷新耗时，并作刷新间隔消融。
- 训练日志输出每类别 alpha[l]、熵、selected references、伪标签分布、I-AUROC、P-AUROC、吞吐量、峰值内存。
- checkpoint 训练恢复必须同步 fusion、optimizer、随机状态与上一次伪标签库 snapshot，并校验 snapshot 对应的特征版本。

## 第三阶段：实验与论文材料（P1/P2）

建议必须完成：

- B0 原人工层+Mean 现状；
- B1 新 API 复刻 B0（相同层集合、相同归一化）；
- B2 统一候选层+固定 Mean；
- E1 统一候选层+共享可学习权重；
- E2 统一候选层+类别条件可学习权重；
- E3 归一化开关与不同候选层集合；
- E4（可选）类别+Patch 动态融合；
- C1（仅对照）逐层独立 NN 后融合残差。

报告至少三个种子、每类别 I/P-AUROC、宏平均和方差、正常候选污染率、层权重热力图、运行资源；统计仅能从已跑出的结果归纳。

## 关键设计决策清单

| 决策               | 当前建议                                             | 原因                                  |
| ------------------ | ---------------------------------------------------- | ------------------------------------- |
| 主融合顺序         | **learnable fusion → NN → fused residual**   | 用户明确指定                          |
| raw reference 存储 | 固定 [N_c,L,C]，DINOv3 只提取一次                    | 训练 theta 时无需重提 DINOv3          |
| fused cache        | 周期性按 theta_refresh 重新融合原始库并更新 NN Index | 加速近邻搜索，避免每 batch 全库重融合 |
| theta 更新         | 每个 optimizer.step() 持续更新                       | 学到类别自适应层权重                  |
| 默认刷新           | 每 Epoch 一次，可选每 N Step                         | 与训练开销平衡，允许受控滞后          |
| 精确对照           | 每 Step 一次                                         | 量化近似缓存导致的最近邻误差          |
| 候选层集           | 统一可配置，覆盖多个深度                             | 避免手动类别层组合的研究偏差          |
| 检索空间           | query/ref 使用同一 theta                             | 避免表征不一致                        |
| 最近邻             | 每个 fused Patch 只选一个 reference Patch            | 保持旧 NN residual 语义               |
| 梯度               | 选中参考后重融合，不能透过 argmin                    | 正确优化融合器                        |
| P0 门控            | 类别特定 softmax 层权重                              | 参数少、可解释、易消融                |
| 硬类 MoE           | 保持不变                                             | 控制研究变量                          |
| 伪标签库           | 随 fusion / adapter 更新                             | 保证迭代训练不使用过期表示            |
| 旧模型             | fixed_mean 默认兼容                                  | 保证现有 FUNAD 可复现                 |

## DoD：交付验收

- [ ] fixed_mean 回归与 codex/main 误差在允许容差内；
- [ ] learned_class 融合层确有梯度、实际更新；
- [ ] 原始参考库保持逐层结构，同 Patch 不错位；
- [ ] NN 在融合特征空间搜索，动态刷新所用 theta_refresh 的版本可追踪；
- [ ] 原始库固定、融合缓存周期刷新、融合 theta 持续学习；
- [ ] 每 Epoch / N Step 刷新可配，支持每 Step 精确模式，记录 stale-NN 率、刷新耗时和资源；
- [ ] 训练中从 raw 库重新融合选中参考保持梯度；评估/推理强制精确刷新；
- [ ] 训练/伪标签/评估/独立推理路径全部打通；
- [ ] checkpoint/metadata 无静默错配；
- [ ] 记录至少一轮可复现的对照结果和完整消融方案；
- [ ] 合理引用并区分 R1–R7 的直接相关性和局限。
