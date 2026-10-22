# FUNAD：DINOv3 可学习的类别条件多层特征融合——需求与设计方案

> 状态：Design Proposal / 尚未修改训练代码
>
> 创建日期：2026-10-22
>
> 基线分支：codex/main；设计分支：feature/learnable-dinov3-layer-fusion-design
>
> 相关文献见 [references.md](./references.md)，工程拆解见 [implementation_plan.md](./implementation_plan.md)。

## 0. 结论 / 不可偏离的核心需求

目标是替换 FUNAD 当前的「按类别手工选 DINOv3 层 + 简单 Mean Fusion」为**由异常检测任务学习出来的多层融合模块**。先对查询图和参考记忆库的**原始逐层特征分别使用同一个可学习融合器**，然后在**融合后的特征空间寻找最近邻 Patch**，最后执行 **融合特征减去融合参考特征**，将结果交给现有 Class Adapter → MoE → 伪标签/检测链路。

**指定的操作顺序：Raw Multi-layer Features → Shared Learnable Fusion → Class-wise NN Search → Fused Residual → Adapter → MoE。**

**不采用**「逐层各找最近邻 → 逐层计算残差 → 最后才融合残差」作为主方案；那是可选的对比实验，不是本需求。

参考记忆库**持久化原始、同位置对齐的逐层 Patch 特征**，不提前 Mean，不保存会随融合器训练而过期的唯一固定融合表示。检索时仍需按**当前**融合器参数生成融合参考表示，否则查询和参考不在相同空间。

## 1. 研究动机、假设与边界

### 1.1 来自实验的观察

用户对 FUNAD 的 DINOv3 特征层进行了类别相关实验，发现不同产品类别的最优层组合影响最终异常检测结果明显。当前人工指定层范围既需要重复调参，也限制模型发现潜在有效的层。

**待验证假设 H1**：类别条件、任务驱动学习的层选择/权重优于固定的手选层 + Mean 基线。
**待验证假设 H2**：类别特定权重优于所有类别共享的层权重。
**待验证假设 H3（后续）**：在类别权重之上增加 Patch 内容调节，能够进一步提高某些纹理/结构异常的检测精度。

这些是待验证假设，不能提前写成“已证明有效”或“创新性已经成立”。

### 1.2 第一版明确保留的设计

- 冻结 DINOv3 backbone；不修改现有训练任务、异常评分定义、Class Adapter、Hard Class Gate MoE 和自训练主损失，除非为了打通新特征接口而必须更改。
- 已知产品类别 c 作为融合条件；首版仅**类别条件静态权重**，暂不引入自由学习型类别 Gate、Patch Gate 或昂贵 Cross-Attention。
- 参考 Patch 的跨层对应关系必须保留：同一 memory row 的各层特征来自同一参考图像、同一 Patch 空间位置。
- 最近邻搜索按类别进行；与当前流程一致，不在多类别参考库中混搜。
- 原版可通过开关完整复现，不允许为新方案破坏旧 checkpoint 的加载和评估。
- 统一训练、测试、独立推理中的 backbone/层集合/归一化/类别索引/融合参数。

## 2. 代码基线（codex/main，2026-10-22 核对）

| 当前行为 | 源码位置 | 对本需求的影响 |
| --- | --- | --- |
| 按类别选 DINOv3 层并进行无权重 Mean | [dataset/feature_extract.py, 318–379](https://github.com/bianhongyong/FUNAD/blob/codex/main/dataset/feature_extract.py#L318-L379) | 需要新增返回逐层特征的接口，旧接口保留 |
| 手动类别层索引表；ViT-L/16 常规类 [11..16]，screw/VisA 为 [11,12,13,22,23,24] | [dataset/multiclass_feature_dataset.py, 78–125](https://github.com/bianhongyong/FUNAD/blob/codex/main/dataset/multiclass_feature_dataset.py#L78-L125) | 学习模式绕过按类别硬编码选层，使用全类别共享、覆盖浅中深的候选层集合 |
| 每类默认选 8 张参考图片，排除文件名前缀 noisy；将所有参考 Patch 展平进按类别的参考库 | [utils/train_utils.py, 268–318](https://github.com/bianhongyong/FUNAD/blob/codex/main/utils/train_utils.py#L268-L318)、[375–407](https://github.com/bianhongyong/FUNAD/blob/codex/main/utils/train_utils.py#L375-L407) | 修改储存格式为 [N_ref_patches, L, C]，不做 Mean |
| 训练残差 = 查询融合特征 − 类别最近参考；主训练路径使用 torch.cdist/argmin，而不是其构建的 FAISS 索引 | [utils/train_utils.py, 332–363](https://github.com/bianhongyong/FUNAD/blob/codex/main/utils/train_utils.py#L332-L363) | 同一个融合器作用于查询/参考，保留检索定义 |
| 融合函数的当前 DINOv3 特征提取置于 torch.no_grad | [dataset/feature_extract.py, 326–379](https://github.com/bianhongyong/FUNAD/blob/codex/main/dataset/feature_extract.py#L326-L379) | 冻结 backbone 没问题，**但新可学习融合层必须位于 no_grad 外** |
| 主训练在建立参考库后移除参考图像，并建立 Class Adapter + MoE | [self_train_ad_multiclass_dinov3.py, 868–932](https://github.com/bianhongyong/FUNAD/blob/codex/main/self_train_ad_multiclass_dinov3.py#L868-L932) | 保留样本分离策略；将融合器加入 optimizer、checkpoint 和评估调用 |
| 伪标签 Phase 1 打分、Phase 2 按类别筛选正常样本并缓存 **Adapter 输出**、GreedyCoreset，Phase 3 评分 | [src/train/epoch_precompute.py, 868–895](https://github.com/bianhongyong/FUNAD/blob/codex/main/src/train/epoch_precompute.py#L868-L895)、[1031–1145](https://github.com/bianhongyong/FUNAD/blob/codex/main/src/train/epoch_precompute.py#L1031-L1145)、[1193–1255](https://github.com/bianhongyong/FUNAD/blob/codex/main/src/train/epoch_precompute.py#L1193-L1255) | 与原始参考库是**两套库**；前者可重用，伪标签库必须随特征空间变更刷新 |
| 当前 Shell 启用 Hard Class Gate、PCA 伪标签、按类最低 20% 评分图像选择正常候选 | [shell/run_self_train_ad_multiclass.sh, 52–57](https://github.com/bianhongyong/FUNAD/blob/codex/main/shell/run_self_train_ad_multiclass.sh#L52-L57) | 首轮尽量保持相同，避免多变量改动 |
| 独立推理从 checkpoint 加载融合后的参考库并用 FAISS 搜索 | [src/inference/infer_multiclass_residual.py, 606–668](https://github.com/bianhongyong/FUNAD/blob/codex/main/src/inference/infer_multiclass_residual.py#L606-L668)、[705–745](https://github.com/bianhongyong/FUNAD/blob/codex/main/src/inference/infer_multiclass_residual.py#L705-L745) | 新格式必须从保存的原始逐层特征重建当前融合空间的 FAISS 索引 |

补充：当前 [DINO_LAYER_FUSION_INDICES](https://github.com/bianhongyong/FUNAD/blob/codex/main/dataset/multiclass_feature_dataset.py#L161-L183) 是另一组 mid/tail 配置，不能误认为上述主训练与推理路径已经启用了可学习融合。

## 3. 数据定义与形状协议

设：
- B：查询 batch size；P：每张图的 Patch 数（448 / 16 = 28，P=784）；
- L：候选 DINOv3 Transformer 层数；C：各层 patch embedding 维度；
- c：类别整数 ID，采用 get_all_class_names 确定的固定顺序；
- N_c：类别 c 的参考 Patch 总数（通常为参考图数 × P）。

### 3.1 查询张量

~~~text
raw_query:    float[B, P, L, C]   # 原始逐层 Patch Token（不提前 mean）
class_idx:    int64[B]
fused_query:  float[B, P, C]      # learnable fusion 输出
~~~

### 3.2 持久化参考库（不可丢失 L 轴）

~~~text
raw_reference_memory_by_class = {
  c: Tensor[N_c, L, C]
}
~~~

若要保留图像/空间索引和对齐审计信息，可另存：
- reference_indices_by_class；
- reference_image_id / patch_id；
- feature_model、candidate_layers（1-based，解析时转换到 0-based）；
- 预处理（Resize/CenterCrop/Normalize）；
- storage_dtype 与 memory_format_version。

查询与参考必须完全采用**相同的候选层顺序**、相同 Token 后处理、相同融合器参数。多层特征数据不可按类别从不同长度列表直接 stack；首版统一候选层集合，各类别由权重决定是否使用某一层。

CLS 路由分支与 Patch 融合分开管理，保持当前默认 use_cls_token=False 的数据维度；hard class gate 的路由仍由 class_idx 决定。

## 4. 第一版融合模块（P0 / MVP）

名称暂定：**Class-Conditioned Learnable DINOv3 Layer Fusion (CC-LF)**。

每类别 c 持有可学习 logits theta[c, l]，经过温度 softmax 得到层权重：

~~~text
alpha[c, l] = softmax(theta[c, :] / temperature)[l]
Z[b, p, :] = sum_l alpha[c_b, l] * Norm_l(F[b, p, l, :])
~~~

推荐：
1. 每层先执行维度不变、相同方法的 LayerNorm（先测试 affine=False，避免额外的层特定可训练变换掩盖权重作用；也测 no_norm）。
2. Logits 初始为 0；初始化是“全部候选层均匀权重”，不是人工预设目标类别权重。另设置基线兼容模式复现手动选层的均匀融合。
3. 温度 tau 作为 CLI 超参，可调度；从 tau=1 开始，避免起步硬 Top-1。
4. 不强制每层均匀使用；不在 P0 使用 MoE 式负载均衡损失。
5. 类别 logits 是 learnable parameters，必须在优化器中；权重经 state_dict 保存/恢复。
6. 初始 output dimension C 保持不变，避免额外改变 Class Adapter 和 MoE 输入维度。
7. 训练时记录类别 × 层的 alpha 矩阵，绘制可解释性热力图。

**候选层集合**：独立于现有 DINO_CLASS_LAYER_INDICES。P0 提供明确 --fusion_layers，例如 vitl16 的 [3,6,9,12,15,18,21,24] 或可配置所有 24 层；这只是覆盖深度的实验起点，**不是学习出的最终选择**，也不应宣称未经实验验证的候选层“最优”。

### 4.1 可选后续（P1 / P2）

- P1 共享 vs 类别特异：shared alpha[l]、class alpha[c,l]、class embedding + MLP gate（参数共享）。
- P2：类别先验与 Patch 内容联合门控。动态 Patch 门控要防范异常 Patch 根据自身内容“掩盖异常”及参考/query 融合空间不稳定。
- P2：Sparsemax、Entmax、可微 top-k 或温度退火选择稀疏层集；这是稀疏选择的增强实验，不作为 P0 前置条件。
- P2：可学习对齐投影、低秩层变换；要额外消融，以区分“选层”与“投影网络容量”。
- P2：参考分布引导的权重，但须明确训练目标及与现有记忆库方法的实质差异，不能预设新颖性。

## 5. 正确的残差定义与参考库生命周期

**目标不是逐层检索。** 每个 Patch 首先统一融合，然后选中**一个**参考 Patch：

~~~text
raw_q = DINOv3_multilayer(image_q)           # [B, P, L, C]
raw_m = raw_reference_memory_by_class[c]     # [N_c, L, C]，离线存一次

z_q   = fusion_theta(raw_q, class=c)         # [B, P, C]
z_m   = fusion_theta(raw_m, class=c)         # [N_c, C]
j*    = argmin_j || z_q[p] - z_m[j] ||_2
res_p = z_q[p] - z_m[j*]
~~~

这保留了“**先融合 → 最近邻 → 融合残差**”的严格顺序，且参考库是原始层特征而非上一个 epoch 的平均值。

### 5.1 梯度与最近邻

- DINOv3 仍处于 torch.no_grad；融合器前向**不得**包在 torch.no_grad 里。
- argmin / FAISS 选出的**索引**本身不具备可微梯度；索引切换是离散的。
- 推荐在当前 theta 下检索索引，然后从 raw_m 以 j* 取回原始多层参考特征，用同一 fusion_theta **在有梯度上下文中**重新融合选中参考，计算 res = z_q - z_m[j*]。这样损失可经 residual 的两端回传至 theta，但**不能声称梯度穿过了 argmin 的选择决策**。
- 如果为省显存，在索引构建/候选搜索阶段用 no_grad 缓存融合向量，仍须保证**当前** theta 的版本，并在最终残差计算处有上面描述的梯度路径。
- 注意：若损失仅鼓励最小残差，可能导致融合权重偏向“天然容易匹配、但不擅长检测缺陷”的层；应观察层权重塌缩并通过最终 anomaly loss 与严格消融检验。

### 5.2 缓存一致性

**硬约束**：原始参考库与冻结 DINOv3 相关，可以跨 epoch 固定；**融合参考特征及其 NN 索引与 theta 相关**，不能固定不变。

推荐实现分档：
- P0 正确性模式：每次 optimizer 更新后的新 batch、仅对所需类别，用当前 theta 对原始参考库重新融合并检索（必要时分块算距离）；最容易验证一致性，但开销高。
- P1 加速模式：精确记录 fusion 参数版本 / refresh_step，重建对应版本的 FAISS 索引；明确 staleness/更新间隔和对结果的影响，不可无声使用旧索引。
- 推理模式：fusion.eval() 加载 checkpoint 后只需融合 raw_m 一次并构建索引；保持测试全程参数冻结。
- 对照模式：固定权重、相同层集时，融合结果与原 Mean 数值近似一致，才能认为回归测试通过。

### 5.3 存储与精度成本

参考库的核心数据量约为 N_c × L × C × bytes_per_element。举例：每类 8 张、每张 784 Patch、ViT-L/16 C=1024、候选 L=12，FP16 约 147 MiB/类（仅特征存储）；若 L=24，则约 294 MiB/类。跨 15 类会达到 GB 级，因此不能无评估地默认全部层同时常驻 GPU。

建议原始库 CPU float16（写入 metadata）、查询/检索阶段根据资源分块迁移，计算距离采用 float32；记录显存、CPU RAM、磁盘和耗时。量化/压缩属于后续实验，须先验证精度。

## 6. 与 FUNAD 自训练的集成

区分两个记忆库：

1. **原始参考库**（Reference Memory）：从按类别选定的 clean 参考图提取、冻结 DINOv3 的逐层特征，供在线融合后的 NN Residual 使用。它不经 Adapter、不会在训练中用新的 theta 覆盖原始数据。
2. **伪标签记忆库**（Pseudo-label Memory）：现有 Phase 1 评分 → Phase 2 根据模型估计选取正常样本并通过 Adapter 提取特征 → GreedyCoreset → PCA/NN/Mahalanobis 拟合 → Phase 3 生成每 Patch 分数。由于 Adapter 的输入残差随 theta 更新，伪标签库随 theta/Adapter 状态变化，需要显式的重新预计算策略。

第一版尽量沿用当前 --pseudo_label_scoring pca、--normal_sample_selection quantile 0.2、Hard Class Gate 等条件，只替换特征融合部分。

### 6.1 防止自训练反馈不稳定

- 建议先做**冻结伪标签阶段**或先在稳定 baseline 上生成伪标签，验证 layer logits 确实得到梯度，再开启联合自训练。
- 保留现有完整 Phase 1/2/3 每 epoch 重建的选项，初期禁止冻结记忆库（--memory_bank_freeze_start_epoch -1）。
- 测试每 epoch 的 alpha 分布、残差尺度、PCA 异常分数分布、伪标签正负比以及 clean 候选纯度，观察闭环是否崩溃。
- 类别排序、参考样本选择、数据预处理、LayerNorm 及温度必须由 checkpoint 元数据固定，避免单独推理时依据不同 Shell 默认值重建错误空间。

## 7. 需求优先级与验收标准

### P0 必须实现
- [ ] 独立开关：fixed_mean（旧行为）/ learned_class（新行为），默认行为可复现；
- [ ] 完整多层 raw feature extraction API，明确 [B,P,L,C] 层序；
- [ ] raw reference memory 格式、构建、序列化、metadata 校验；
- [ ] shared fusion：查询/参考严格同一参数与候选层；
- [ ] 融合后 NN，再融合残差；不能误做逐层 NN；
- [ ] Theta 有非零、有限梯度且 optimizer 实际更新；
- [ ] 当前 theta 的参考索引一致性可自动检验；
- [ ] 训练、epoch_precompute 的三阶段、评估、独立推理全部可调用新路径；
- [ ] checkpoint 同时保存 localnet、fusion 参数、候选层、参考 raw 库、版本、preprocess/backbone；
- [ ] 旧 checkpoint 仍能以原 fixed_mean 路径读取或明确报出可操作的兼容诊断；
- [ ] 在选定对照数据集、随机种子、污染条件下记录 I-AUROC、P-AUROC、性能均值/方差、资源使用。

### P1 可选
- [ ] shared vs class-specific gate 消融；
- [ ] 不同候选层集合与逐层归一化消融；
- [ ] 每步更新索引 vs 每 epoch 更新索引精度/性能对比；
- [ ] 跨类权重可视化（有无层偏好），统计权重熵及 top-k 质量；
- [ ] 逐层 NN → 残差 → 融合，仅作为独立对照而非主路线。

### P2 研究探索
- [ ] Patch 动态 Gate、稀疏门控；
- [ ] 参考分布条件化 fusion；
- [ ] 防止梯度/检测敏感性塌缩的正则与训练课程方案。

## 8. 消融实验矩阵（严格单变量）

| 组别 | DINOv3 候选层 | 融合策略 | 检索位置 | 目的 |
| --- | --- | --- | --- | --- |
| B0 | 原手选类别层 | 现有 Mean | Mean 后 NN | 原始 FUNAD 基线 |
| B1 | 与 B0 相同的每类层集合 | 固定均匀 Mean（新接口） | 融合后 NN | 校验重构等价 |
| B2 | 统一候选层 | 固定均匀 Mean | 融合后 NN | 排除候选层增多造成的效果 |
| E1 | 同 B2 | 全类别共享可学习权重 | 融合后 NN | 学习权重本身是否有效 |
| E2（主模型） | 同 B2 | 类别条件可学习权重 | 融合后 NN | 类别差异是否带来增益 |
| E3 | 同 B2 | 类别条件可学习权重 + 归一化变化 | 融合后 NN | 特征尺度影响 |
| E4（可选） | 同 B2 | 类别 + Patch 动态权重 | 融合后 NN | Patch 自适应是否有价值 |
| C1（对照） | 同 B2 | 按层 NN → 残差 → 最后融合 | 层内 NN | 验证共享 NN 与独立 NN 假设 |

比较须固定数据划分、参考图 IDs、种子、后端参数、污染/噪声比、训练轮次与打分协议，至少 3 个独立种子报告均值±标准差。特别注意 B0 与 E2 在候选层集合上不同，**不可直接把任何提升全归因于“可学习类别融合”**；判断类别条件的关键比较是 E1 vs E2，判断学习效果的关键比较是 B2 vs E1/E2。

## 9. 风险与可预期失败模式

| 风险 | 原因 | 缓解 |
| --- | --- | --- |
| NN 特征空间过期 | theta 每步更新，索引仍是旧融合表示 | P0 重算，后续带版本号刷新 |
| 伪标签不稳定 | theta 改变 Adapter 特征分布，PCA/coreset 基础变化 | 分阶段训练，定期全量 Phase 1/2/3 |
| 注意力退化 / 忽略敏感层 | 训练信号倾向容易重建的层 | 记录权重熵和逐类指标，考虑约束但先不预设正则 |
| 检索不是端到端可微 | argmin/FAISS 离散 | 选中参考重新融合以回传梯度；可微近邻仅做 P2 |
| 显存/时间爆炸 | raw 逐层 reference size 扩大 L 倍 | CPU FP16 + chunk + 按类别加载，实测峰值 |
| 训练推理不对齐 | Backbone/预处理/类别顺序/候选层不一致 | checkpoint metadata 校验、拒绝隐式错配 |
| 名称夸大研究结论 | 单纯 Softmax 融合已有丰富先例 | 以实证类别依赖与自训练 NN Residual 的联合机制为目标，文献对比见 references.md |

## 10. 暂定决策与后续评审问题

本设计文件是待实现的**需求确认版**，不代表已经在新分支完成代码变更或实验证明。

暂定默认：首版类别特定 softmax 层加权，先融合、再类别 NN、再残差；缓存原始逐层参考特征；用旧后端与旧损失；先验收 old-mode 和 gradient，再运行正式消融。

实施前需最终确认：候选层数 L 与模型型号；每类 reference 原始缓存预算；是否将 train optimizer 每步准确重算作为验收基准；PCA 伪标签是否先冻结几轮；是否同时支持训练恢复和独立推理。

## 11. 文献与论证边界

- DINOv3 是基础的自监督 dense feature backbone [R1]。
- PatchCore 支持正常 Patch memory / 近邻异常检测及效率讨论，但**不是**对本方案“可学习类别融合”效果的直接证明 [R2]。
- Dinomaly 是多类别无监督工业异常检测的重要对照背景，但研究框架不同 [R3]。
- SKNet / ASFF 提供“数据驱动加权选择多分支/多尺度特征”的先例，因此不能把 Softmax 层权重本身作为新颖性主张 [R4,R5]。
- ALF (ICIP 2026) 已专门讨论工业异常检测里的自适应层融合；它采用**无学习参数的图像级可靠性权重**，和本方案任务驱动、类别条件、融合后 NN Residual 的路径不同；必须做差异对比，而非宣称“首个自适应层融合” [R6]。
- FPN 为多层视觉信息的融合方法提供一般背景，非 DINOv3 异常检测直接前作 [R7]。

所有条目的会议、日期、出处、可用链接见 [references.md](./references.md)。