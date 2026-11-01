# FUNAD V2：类别—Patch 层次化动态融合 + 类别强度 Adapter（HCPAF）代码修改方案

> **当前推荐路径（2026-10-29 更新）：** class_patch 新训练默认使用 `--fusion_match_mode prior_top1`：
> 类别先验只负责选择一个参考 Patch `j*`，Patch Gate 只参与最终残差的双侧融合。
> 见 [prior_top1_patch_residual.md](prior_top1_patch_residual.md)。
> 本文描述的“类别先验粗检索 Top-K + Query 条件精排”保留为
> `--fusion_match_mode dynamic_topk` 的历史/研究消融路径，不应无记录地删除。

> **状态：Implementation Proposal / 仅设计，尚未实现、训练或验证。**
>
> **目标分支：** feature/learnable-dinov3-layer-fusion-design
>
> **核对基线：** 8b6766bf0383cf39acabaac1f66e849b85a5a77d（2026-10-29）
>
> **日期：** 2026-10-29
>
> **本提交范围：** 只新增可执行修改方案，不改现有训练、推理、模型和参数默认行为。
>
> 关联原设计：[learnable_dinov3_layer_fusion/requirements_and_design.md](../learnable_dinov3_layer_fusion/requirements_and_design.md)。
>
> **核心研究假设（待实验验证）：** 同一产品类别内的不同 Patch 依赖的 DINOv3 特征层可能不同；在类别层融合先验之上，引入 Patch 动态偏移和可解释的类别 Adapter 调整强度，有望改善工业异常检测的判别性，同时保留原有参考残差匹配与自训练框架。没有预先假设性能必然改善。

## 0. 目标、优先级与不可违反的约束

### 0.1 目标

1. 将静态的类别融合权重 alpha[c,l] 升级为类别 + Patch 双层权重 alpha[c,p,l]。
2. **不同查询 Patch 可以采用不同融合权重**，但其与候选参考 Patch 计算残差时，必须**使用同一个查询权重 alpha[c,p,l]**。
3. 为解决每个 Query 使用不同度量导致无法直接复用唯一的预融合参考索引，设计“类别先验粗检索 Top-K + Query 条件精排”。
4. 统一 Fusion 与 Adapter 的类别条件：类别嵌入同时控制类别先验、局部偏离幅度、适配强度，适配网络保留 identity residual shortcut。
5. 首先推出 **V2-A：Patch Fusion + 原有 Adapter/MoE**；验证后才推出 **V2-C：Patch Fusion + 新的类别强度 Adapter**，不要一次修改所有模块。
6. 保留固定 DINOv3、逐层原始参考库、类别内检索、BCE+OTO、自训练 PCA/MAD 与旧分支可复现性。
7. 给出代码接口、部署资源边界、兼容迁移、测试、可解释性日志与实验设计，供后续 Codex 分步实现。

### 0.2 不变量（不得违反）

- 保留 Raw Reference Bank 的对齐层维度：[N_c, L, C]；第 j 行的 L 个特征均来自同一参考图像、同一 Patch。
- 不使用“8 层分别查最近邻再融合层残差”作为主方案；每个 Query 只选一个最终参考 j*。
- **精排时 Query 与参考候选统一使用当前 Query 的 alpha[c,p,:]**，不能拿 Query 动态 alpha 和 Reference 自身另一个动态 alpha 直接相减。
- 最近邻离散索引不做伪造的可微化宣称；在固定 j* 下保留 Query 和重融合 Reference 两条梯度路径；未来 soft-NN 属于可选消融。
- Backbone 冻结；全流程训练、Epoch 伪标签计算、测试、独立推理保持同一个模型/融合协议。
- 不允许用 MVTec AD 测试 GT 调节 alpha、beta、K、正则系数或选最佳权重；测试集只评估。
- **保留原模式：** fixed_mean / learned_class / learned_shared；V2 由新 flag 显式开启。不能悄悄覆盖已有 checkpoint。
- 本分支最新默认融合特征归一化是 none（可选 layer_norm），不是旧讨论里的默认 LayerNorm。

## 1. 当前代码与需求缺口（基线 SHA 对齐）

| 文件与当前实现 | 现状 | V2 应改 |
|---|---|---|
| src/model/layer_fusion.py：ClassConditionedLayerFusion | theta[num_classes,L]，每类别每 Patch 同一 Softmax；查询与参考共用 theta | **保留 V1**，新增 V2 类别+Patch Gate 与适配模块，不覆盖老类 |
| dataset/multiclass_feature_dataset.py | ViT-L/16 默认层 [3,6,9,12,15,18,21,24] | 保持默认和候选层校验；可扩展实验选层，避免对比混杂 |
| dataset/feature_extract.py | 输出 raw [B,P,L,C] | 继续复用，Gate 接入原始层特征；不要额外重复运行 Backbone |
| utils/train_utils.py：FusedReferenceCache / compute_fused_residual_batch | 按融合缓存 Top-1 查 NN；当前 theta 重融合选中原始参考；参考缓存默认按 Epoch 刷新 | 扩展带候选 id 的 Top-K 检索与 Query 动态精排；实现分块算法 |
| src/model/model.py：ClassCrossAttentionAdapter / localnet | 单类别 token 的 MHA，后接 GELU；下游 Hard Class Gate MoE | V2-A 不改原 Adapter；V2-C 替换为强度门控残差 Adapter，避免双重适配 |
| self_train_ad_multiclass_dinov3.py | 现有 FusionRuntime、CLI、RMSprop 2e-5、Fusion 参数和 localnet 参数并列优化 | 新增 V2 模式、初始化/参数组/日志/冷启动与 checkpoint 版本校验 |
| src/train/epoch_precompute.py | Phase 1 图像得分；Phase 2 正常样本适配特征库；Phase 3 PCA 等伪标签 + MAD | 保证 3 阶段全走同一 V2 residual/adapt 回调，控制伪标签更新频率 |
| utils/evaluate.py | 走训练时提供的特征提取和 residual 回调，返回 I-/P-AUROC | 统一 V2 前向与缓存刷新，不改变 score 聚合以避免混杂 |
| src/inference/infer_multiclass_residual.py | 加载 V1 参数/参考库与计算残差 | 增加 V2 checkpoint 分派，保证独立推理与验证数值一致 |
| tests/test_layer_fusion.py | V1 融合、梯度、参考缓存测试 | 保留全部测试，另建 V2 Gate/Top-K/Adapter/Checkpoint 测试 |

**已观察到的两个风险：**
1. 当前类别 Softmax 权重对初始化敏感，训练时权重几乎不动；引入 Patch Gate 不会自动解决其梯度可辨识性。
2. 现有 ClassCrossAttentionAdapter 每类别仅一个 Key/Value Token，Cross-Attention Softmax 恒为 1，其输出主要相当于类别相关偏置；V2-C 不需要保留这个伪内容注意力结构。

## 2. 总体结构与张量协议

使用当前 ViT-L/16、448×448 输入为例：P=28×28=784，L=8，C=1024。具体数值由模型和预处理动态推断，不可在新模块内部写死。

~~~mermaid
flowchart TD
    I["Query image [B,3,H,W]"] --> F["Frozen DINOv3<br/>Raw Query Q [B,P,L,C]"]
    REF["Raw Reference Bank<br/>M_c [N_c,L,C]"] --> CC["Class-prior fused cache + per-class NN index"]
    CID["Known class id c"] --> E["Class embedding e_c"]
    E --> PR["Global prior logits theta_c<br/>alpha^G_c"]
    E --> GATE["Patch gate amplitude lambda_c"]
    F --> PG["Patch content/context encoder<br/>delta_c,p [B,P,L]"]
    PR --> ALPHA["alpha_c,p = softmax((theta_c + lambda_c delta_c,p)/tau)"]
    GATE --> ALPHA
    PG --> ALPHA
    F --> COARSE["Coarse fused Query with alpha^G_c"]
    PR --> COARSE
    PR --> CC
    CC --> K["Coarse Top-K candidate ids"]
    COARSE --> K
    ALPHA --> FINE["Query-conditioned fine rerank<br/>same alpha_c,p for Q and each candidate M"]
    K --> FINE
    REF --> FINE
    F --> FINE
    FINE --> R["Single matched ref j*(c,p)<br/>residual R = sum_l alpha_c,p,l (Q_l - M_j*,l)"]
    R --> AD["V2-A: legacy Adapter<br/>V2-C: r + beta_c H(r,e_c)"]
    E --> AD
    AD --> MOE["Hard Class Gate -> class expert -> Sigmoid"]
    MOE --> SCORE["Patch anomaly score [B,P]"]
    AD --> PSEUDO["FUNAD Phase 1/2/3 adapted-memory / PCA / MAD"]
    SCORE --> PSEUDO
    PSEUDO --> LOSS["BCE + 2.5 * OTO (+ optional gated regularizers)"]
    LOSS --> ALPHA
    LOSS --> AD
    LOSS --> MOE
~~~

该图中 **Top-K 粗检索采用全局类别权重；最终精排和残差采用每个 Query 的动态权重**。类别 Adapter 位于残差计算之后，不改变第一版用于查 NN 的空间；这是有意设计的隔离边界。

### 2.1 推荐 API / 输入输出

- raw_query: [B,P,L,C]；class_idx: [B]；raw_ref_bank_by_class[c]: [N_c,L,C]。
- ClassPatchFusionAdapter.global_weights(class_idx) -> [B,L]。
- ClassPatchFusionAdapter.patch_weights(raw_query,class_idx) -> [B,P,L]。
- ClassPatchFusionAdapter.fuse_query(raw_query,alpha) -> [B,P,C]。
- ClassPatchFusionAdapter.fuse_refs_for_query(raw_refs,alpha) -> [M,K,C]，M 是已展开的 Query 数量，raw_refs [M,K,L,C]。
- DynamicReferenceMatcher.topk_by_global_prior(global_query,class_idx,K) -> [B,P,K]。
- DynamicReferenceMatcher.rerank_and_residual(raw_query,alpha,topk_ids,class_idx) -> residual [B,P,C]、best_ids [B,P]、可选 distance/margin。
- ClassPatchFusionAdapter.adapt_residual(residual,class_idx) -> adapted [B,P,C]。
- V2-A 仅调用新 Fusion 部分，继续使用现有 localnet.adaptor；V2-C 只调用新 adapt_residual，旧 adaptor 不得重复执行。
- class_idx 可以混类别 batch，必须独立分组，在各自类别的参考库中取候选。

## 3. 类别先验与 Patch Gate

### 3.1 全局类别先验

直接保留 V1 的显式可学习 theta_c [L]，便于从 V1 初始化并观测 alpha：

alpha^G_c = softmax(theta_c / tau)。

统一类条件向量 e_c = Embedding(c), 建议 embedding_dim=32（可配置，非定值）。

**为什么先保留 theta 而不是立即完全替换成 W_alpha e_c？** 可以从 V1 权重精确初始化，减少变量，更容易判断 Patch 动态化是否真的改善性能。共享 embedding 仍用来输出动态门控尺度和 Adapter 参数；后续可加全生成式先验作为实验。

### 3.2 Patch 上下文与动态偏移

对 Query 的多层特征 Q_{b,p,l,:}，构造轻量局部编码器。推荐首版：

- 对每层特征使用独立于主融合的 norm（仅 Gate 内部，可选 LayerNorm），避免 raw feature 尺度主导 Gate；不改变 V1 默认的特征融合 norm=none。
- 使用共享线性投影 W_s: C -> d（例如 d=64），形成 [B,P,L,d]，可加 learned layer-index embedding。
- 在层维度做 attention/MLP 聚合成 patch token h_{b,p} [B,P,d]；可以加廉价的 3×3 depthwise conv 空间上下文（先关，作为单独消融）。
- 拼接/调制类别 embedding e_c，输出 delta_{c,p}=G_psi(Q_{p,1:L},context,e_c) [L]。
- Gate 最后一层的 weight 和 bias **零初始化**，保证新模型初始 alpha 与 V1 完全相同。
- 新增可学习幅度 lambda_c，建议 lambda_c=softplus(a_c) 或 sigmoid(a_c)；首版从小正值（例如 0.1）启动，**不要零值导致门控梯度被长期阻断**。
- 可选对 delta 的层维度去均值（Softmax 对公共平移不敏感），提升日志可解释性。

核心：

alpha_{c,p,l} = softmax_l((theta_{c,l} + lambda_c delta_{c,p,l}) / tau)

初始 delta=0 时：alpha_{c,p,:}=alpha^G_c，严格退化到 V1（仅对相同 layers/norm/cache 的融合子模块而言）。

**不要在第一版给 Reference 特征另行预测 alpha_ref[j,:]**。匹配单个 Query 时，其候选 Reference 均须按 alpha_query[p,:] 投影到相同空间。

### 3.3 避免动态 Gate 的退化解

- 记录 E_p||alpha_{c,p} - alpha^G_c||_1，以及各类别 Patch 权重方差/熵。
- 分别统计高置信正常/伪异常 Patch 的 alpha；在测试集只能事后分析，不使用 GT 调参。
- Gate 可能通过选择“最好匹配正常参考”的层来隐藏真实异常；因此不能仅以更低训练 BCE、更多层权重分化判断成功。

## 4. Query-consistent Top-K 检索与动态重融合

### 4.1 第一阶段：全局先验粗检索

采用类别先验 alpha^G_c 构造可缓存融合参考：

m^G_{c,j} = sum_l alpha^G_{c,l} M_{c,j,l}。

查询粗特征：
q^G_{c,p} = sum_l alpha^G_{c,l} Q_{p,l}。

通过类别内 Top-K 近邻得到 N_K(c,p)。
这一路仍复用现有 Raw Bank + 派生 Fused Cache 的设计；cache 只依赖 theta_c，不依赖每个 Query 的 delta。

**接口改动：** 当前 FusedReferenceCache.nearest_ids 只输出 1 个 id。新增 nearest_topk_ids(...,k) [B,P,K]，支持 per-class batch + chunking；K 不超过该类别库大小；对空库报清楚的错误。若后端使用 FAISS，Top-K 应走其 search(k)；当前 torch chunked 距离实现应正确维护跨 chunk 的全局最小 K，不能简单取单一 chunk 的 Top-K。

### 4.2 第二阶段：每个 Query 使用自己的权重精排

对每个 Query p 以及粗检索候选 j∈N_K：

q_{c,p} = sum_l alpha_{c,p,l} Q_{p,l}

m_{c,j|p} = sum_l alpha_{c,p,l} M_{c,j,l}

d_{c,p,j} = ||q_{c,p} - m_{c,j|p}||_2^2

j*(c,p) = argmin_{j ∈ N_K} d_{c,p,j}

r_{c,p} = sum_l alpha_{c,p,l}(Q_{p,l} - M_{c,j*,l})

**必要实现细节：**

1. Top-K 粗检索 id 作为离散数据，detach/no_grad。
2. 精排候选距离允许按当前 alpha 计算，但 argmin id 不可微；根据 id 选中原始多层参考后，再**以当前 alpha 重算所选参考**，确保 Query 和 Reference 两端对 theta、Gate 有梯度。
3. 参考特征固定，无需对 DINOv3/Raw Bank 求导；不得对 patch_weights() 的输出做 detach。
4. 如粗缓存 theta 是旧版本，记下 cache_version/cache_step；验证/推理必须先刷新。
5. 优先以 flattened Query 分块执行，避免建立整个 [B,P,K,L,C] 全量 GPU 张量。建议 Query-chunk × K-chunk：计算投影/距离并立即维护最优值和 id。
6. 输入原始参考库可使用 CPU float16 存储；参与差值与平方距离时明确转换为与查询一致的计算 dtype（必要时 float32），避免负距离或精度扰动。
7. 保持一 Query 只选一个参考 Patch；禁止逐层独立 NN。

### 4.3 近似检索误差必须实测

粗检索 Top-K 可能漏掉 query-dynamic metric 下的真实最近邻。设全库精确 Query 特定最近邻为 j_exact，定义：

recall@K = mean[ j_exact ∈ N_K(c,p) ]。

在固定抽样 Query 上测试 K∈{1,8,16,32,64}、精确动态匹配结果与时间/显存，先选折中值（建议初始 K=32，仅实验默认，不能宣称最优）。运行时 K 必须持久化到 config/checkpoint。

**两个有意义的基准：**
- K=1 + delta=0 应尽量数值复现 V1 NN/残差（注意 tie 和浮点实现差别）。
- K=全类别参考库 + 当前 alpha，必须等价于 brute-force 全库动态 NN（容忍浮点误差/距离 tie）。

## 5. 类别强度 Adapter：V2-C 再引入

### 5.1 为什么替换原 Adapter

现有 ClassCrossAttentionAdapter 的 K=V=单类别 token，softmax over one key 恒为 1，因此主要相当于类别偏移 + GELU；它不是内容相关的多 Token 注意力。

V2-C 建议统一类别 embedding e_c，并引入显式 beta_c∈(0,1)：

beta_c = sigmoid(w_beta^T e_c + b_beta)。

z_{c,p} = r_{c,p} + beta_c * H_phi(r_{c,p}, e_c)。

H_phi 采用轻量残差 MLP + FiLM：

u = LN(r)

(gamma_c, shift_c) = FiLMHead(e_c)

u_c = (1+gamma_c) ⊙ u + shift_c

H_phi = W_up GELU(W_down u_c), 例如 C -> 128 -> C（可配置）。

**beta 是门控系数，不等同于实际调整幅度**；另外记录：

rho_c = E_p[ ||z_{c,p}-r_{c,p}||_2 / (||r_{c,p}||_2 + eps) ]。

初始化建议 beta≈0.1–0.2，FiLM gamma/shift 从 0 开始，H 的输出投影**小幅非零初始化**以便初期有 beta 梯度；仍需通过消融检查稳定性。

### 5.2 V2-A 与 V2-C 的明确边界

- **V2-A**：新类别+Patch Fusion + Top-K 精排 + **原 ClassCrossAttentionAdapter + 原 Hard Class Gate MoE**。只测试动态融合本身。
- **V2-B（可选）**：类别融合 V1 + 新类别强度 Adapter。只测试强度适配本身。
- **V2-C**：新动态 Fusion + 新类别强度 Adapter + 原 Hard Class Gate MoE。新模块共享 e_c。
- 未来研究是否必要让 beta_{c,p} 也动态预测，**不属于本次主线**。先避免 alpha、beta 双重动态化导致无法归因。

### 5.3 对模型注册的工程约束

推荐 ClassPatchFusionAdapter **只注册到一个 nn.Module owner（例如 localnet.feature_transform）**，动态 reference runtime 保留指向该同一实例的非所有权引用。训练参数只计入 optimizer **一次**，避免同一实例同时通过 localnet 和独立 fusion 参数列表重复注册、重复优化或 checkpoint 含糊。

V2-A 可以把该模块作为 fusion-only owner，localnet.forward 仍应用 legacy adaptor；V2-C 让 localnet.forward 改为 feature_transform.adapt_residual()，并显式跳过旧 adaptor。两版均保证 residual -> adapted_features -> discriminator 是**一次且只有一次**适配。

## 6. 分阶段代码实施清单（一次只更改一类变量）

### PR / Commit 1：模块 + Gate，保持 V1 退化等价

新建：
- src/model/class_patch_fusion_adapter.py：ClassPatchFusionAdapter，包含 theta、class_embedding、patch encoder、delta head、lambda gate、可选 beta/FiLM head。
- tests/test_class_patch_fusion_adapter.py。

修改：
- self_train_ad_multiclass_dinov3.py：新增 --fusion_mode class_patch；解析 Gate embedding dim、hidden dim、lambda init、norm、context 等 flag。

**验收：** delta=0 的 patch_weights 与 learned_class 相等，Query/Ref 融合输出逐元素对齐；梯度能到 theta / patch Gate；混类别 batch 有效；fixed_mean/V1 完全不变。

### PR / Commit 2：Top-K + Query-conditioned fine reranking

修改：
- utils/train_utils.py：FusedReferenceCache.topk 扩展；新增 compute_class_patch_fused_residual_batch 与 query chunk helper。
- self_train_ad_multiclass_dinov3.py：新增 V2 runtime（粗缓存 refresh、alpha 日志、查询级 NN drift/Top-K recall probes）。

新建：
- tests/test_class_patch_reference_match.py。

**验收：** 候选 ref 与 Query 使用同一 alpha；全库 K 与 brute force 匹配；k=1/delta=0 可复现旧路径；ref 与 query 对 Gate 保留梯度；跨类别不混库；chunking 与非 chunking 结果一致。

### PR / Commit 3：主流程、自训练、独立推理打通

修改：
- self_train_ad_multiclass_dinov3.py：完整训练/评估 wiring、参数分组、阶段配置、checkpoint schema version。
- src/train/epoch_precompute.py：Phase 1/2/3 均复用同一 V2 residual callback；保持评分协议不变。
- src/inference/infer_multiclass_residual.py、shell/run_infer_multiclass_residual.sh：恢复 V2 配置、Gate/Adapter 权重及重建缓存。
- utils/evaluate.py：检查评估回调一致性，无必要不改评分定义。
- 新增 shell/run_self_train_ad_multiclass_class_patch.sh；旧脚本默认行为不变。

**验收：** Train/Eval/Standalone Infer 对同一图像 patch score 数值一致；存储并恢复 raw bank、层列表顺序、类名索引、norm、lambda、beta、K、阶段/伪标签设置；旧 checkpoint 仍可加载到旧模式。

### PR / Commit 4：强度 Adapter 合并

修改：
- src/model/model.py：支持 feature_adapter_mode=legacy|class_residual（不能双重适配）。
- 新模块补全 beta + FiLM residual MLP，和 Gate 共用类别 embedding。
- 添加训练阶段/冻结开关、实际 rho、beta 记录。

**验收：** beta≈0 时近似恒等、beta=1 时按定义使用新 MLP 修正；梯度可达 beta/FiLM/MLP；Class Adapter 只执行一次；MoE 输入仍 [B,P,C]。

### PR / Commit 5：收敛性分析、效率、消融

- 冻结伪标签与固定 NN id 两类诊断开关（严禁把“冻结 Phase1/2 库”误记成“完整冻结伪标签”）。
- 记录 theta 有效增量、alpha Patch 方差、delta/lambda、beta/rho、Gate gradient consistency、NN switch、Top-K recall、伪标签翻转率、每类 I/P-AUROC。
- K/显存/时间 profiling、seed 重复、多层候选公平对照。
- 完善文档、复现实验命令与图表脚本。

## 7. 优化器、伪标签和渐进训练方案

当前 Shell 用 RMSprop，LocalNet/Fusion 共用 lr=2e-5；BCE+OTO 可能在下游拟合伪标签后给层选择很弱的区分信号。V2 应增加参数分组，但不要假定增大学习率必然改善。

推荐可选阶段配置：

| 阶段 | alpha^G/theta | patch Gate | beta/Adapter | MoE | 伪标签 |
|---|---|---|---|---|---|
| A 类别先验/基线 | 训练或加载 V1 | 冻结 delta=0 | V2-A 原 Adapter | 按基线 | 固定诊断 + 正常训练对照 |
| B Patch Gate | 小 LR 或冻结 | 训练 | V2-A 原 Adapter | 可固定作诊断 | 短阶段固定伪标签 |
| C 强度 Adapter | 先冻结 | 先冻结 | 训练 beta+H | 固定或低 LR | 仍优先稳定伪标签 |
| D 联合微调 | 低 LR | 低 LR | 低 LR | 低 LR | 逐渐恢复自训练刷新 |

实现可选 CLI（准确命名在编码阶段统一）：fusion_lr、patch_gate_lr、adapter_lr、fusion_topk、fusion_query_chunk、gate_lambda_init、gate_prior_reg、gate_consistency_reg、fusion_stage、freeze_pseudo_labels、diagnostic_fixed_nn_ids。

**必须显式冻结“每个样本的伪标签、阈值及其来源”**才能称为固定伪标签；仅设置 memory_bank_freeze_start_epoch 不够，因为 Phase 3 仍会产生新的得分/距离图。

选用正则时仅在 V2 诊断/消融开启：

L_total = L_BCE + 2.5 L_OTO + mu * E_{c,p} KL(alpha_{c,p} || alpha^G_c) + nu * L_consistency。

- KL prior 防止早期 Patch Gate 远离类别先验，但 mu 过大将使所有 Patch 相同；建议 warm-up / anneal，可选仅对高置信正常 Patch 施加。
- consistency 只在已正确对齐的图像几何增强下比较同一物理位置 Patch 的 alpha，避免错误对齐。
- 单纯压低 alpha 熵并不代表学到了更有利于异常检测的层；没有验证前不加入强制稀疏正则。
- 训练过程中**不要**根据测试 I-AUROC 反向选权重；如现有脚本使用测试集分数选 best checkpoint，应把此行为视为历史可复现设置，V2 主实验另设不依赖测试 GT 的模型选择协议。

## 8. 具体诊断指标与公式

- **类别先验：** alpha^G[c,:]、entropy(alpha^G)、centered theta（减去类别内均值后再比较）。
- **Patch 差异：** mean_p alpha[c,p,:]、std_p alpha[c,p,:]、mean_p ||alpha[c,p,:]-alpha^G[c,:]||_1。
- **实际调整强度：** beta[c] 和 rho[c]；并报告正常与伪异常的 rho。
- **梯度稳定性：** consistency = ||sum_t g_t|| / (sum_t ||g_t|| + eps)，分别记录 theta、Patch Gate head、beta。
- **近邻质量：** coarse cache version、Top-K recall vs brute force、query NN switch rate、top1/top2 distance margin；真实 Query 集合不能拿库内点自匹配代替。
- **伪标签非平稳性：** 对固定训练样本集合统计 pseudo_label_flip_rate，记录 MAD 阈值/normal selection 变化。
- **分类型异常检测：** MVTec AD 逐类 I-AUROC/P-AUROC，重点 screw；图像分数仍采用相同 top-k 聚合以避免比较混淆。
- **训练资源：** 平均/峰值显存、单 Batch 用时、Epoch 用时、粗检索与精排分别用时、K 与 recall 的折衷曲线。
- **Feature 调整退化：** ||z-r||/||r||、Gate 输出方差；确认新模型没有仅靠 MoE 补偿而把 Gate 学成无效分支。

## 9. 测试矩阵与完成标准

### 9.1 单元测试（至少）

- B=1 与 B>1、同类别与混类别、P=1 与 P=784、L=1 与 L=8、不同 C。
- delta 全零：alpha_patch == alpha_global；alpha.sum(-1)==1，非负且无 NaN。
- Patch 不同：给不同输入产生不同权重；相同类 ID 不强制各 Patch 权重相同。
- 同 Query 对所有候选参考使用相同 alpha；改变 ref 自身内容不会偷偷改变 ref 的权重计算规则。
- K=全库时，与精确逐 Query 加权距离的 argmin 相同；K=1 与旧 V1 退化对齐。
- 显式通过 torch.autograd 验证 theta、Patch Gate 参数和选中 Reference 路径梯度；离散 argmin 没有梯度是预期。
- Raw Bank/Cache 与全部模式重载；norm=none 与 norm=layer_norm；不同候选层顺序的 checkpoint 必须拒绝不兼容加载。
- 各阶段 train/eval/infer 的相同输入预测一致；固定随机性下数值差受控。
- chunked 与 full 的精排索引/残差一致；Top-K tie 与空库异常明确。
- 原 tests/test_layer_fusion.py 全部仍通过；旧训练/推理脚本无需修改即可运行 V1。

### 9.2 研究消融（同层集、同预处理、同 seed、同训练预算）

| 编号 | Fusion | Adapter | 主要验证 |
|---|---|---|---|
| E0 | V1 固定 Mean | 原 Adapter | 历史基线 |
| E1 | V1 learned_class | 原 Adapter | 当前可学习融合 |
| E2 | V2 alpha^G + patch delta，K 动态精排 | 原 Adapter | **Patch 动态融合的独立增益** |
| E3 | V1 learned_class | 类别 beta + FiLM residual | 类别调整强度的独立增益 |
| E4 | V2 动态融合 | 类别 beta + FiLM residual | 联合模型 |
| E5 | E2，无 Patch 上下文 | 原 Adapter | 空间上下文是否必要 |
| E6 | E2，不同 K / 参考库 FP32 | 原 Adapter | 动态粗检索误差与精度 |
| E7 | E2，固定伪标签 / 固定 NN | 原 Adapter | 优化不稳定原因 |

**公平性特别注意：** 旧手选层 Mean 可能对 screw 使用 [11,12,13,22,23,24]，而 V1 ViT-L 候选为 [3,6,9,12,15,18,21,24]；对比必须把“候选层改变”单独列为因素，否则无法将指标差异归因于动态 Fusion。

每个主要实验使用多个固定 seed，报告 mean ± std、类别分指标以及训练权重曲线；在验证集上选择配置，测试集只报告最终结果。

### 9.3 交付验收门槛

- V2-A 真正实现每 Patch 不同 alpha；Query 与被选 Reference 一致投影。
- 退化 V1 通过数值回归；绝不破坏原功能分支既有行为。
- 全库 Query 动态最近邻的正确性用 brute force test 确认；Top-K 近似误差可量化。
- 动态 Gate 和 Adapter 梯度存在，训练与独立推理预测可复现，checkpoint 自描述。
- 能输出 alpha^G、alpha_patch 均值/方差、lambda、beta、rho、NN recall、伪标签翻转率与资源成本。
- 是否提升 I/P-AUROC 是**实验结果**而非工程完成的预设承诺。

## 10. 暂不做的事项

- 不将参考库每个 patch 独立 Gate 的 alpha_ref 当作唯一固定库权重与 Query alpha_query 混用。
- 不将 Patch-specific NN 误写为“一次预融合全库即可精确查到全部动态 Query NN”。
- 不在第一版增加 Patch 级 beta、可学习 MoE 路由、End-to-End Backbone 训练或可微 ANN。
- 不声称 KL 正则、稀疏注意力或降低 Softmax 温度一定能解决初始化敏感。
- 不依靠测试标签挑选权重或提前宣布创新性与性能结论。

## 11. 建议编码顺序（一句话）

**先实现 V2-A（Patch Gate + 同权重 Query/Ref Top-K 精排 + 梯度/精度测试），验证动态层选择真实有效且训练稳定；再实现 V2-C（共享类别 embedding 的 beta/FiLM Adapter），最后处理分阶段伪标签与优化问题。**

参考代码入口：
- [layer_fusion.py](../../src/model/layer_fusion.py)
- [model.py](../../src/model/model.py)
- [train_utils.py](../../utils/train_utils.py)
- [self_train_ad_multiclass_dinov3.py](../../self_train_ad_multiclass_dinov3.py)
- [epoch_precompute.py](../../src/train/epoch_precompute.py)
- [infer_multiclass_residual.py](../../src/inference/infer_multiclass_residual.py)
- [test_layer_fusion.py](../../tests/test_layer_fusion.py)
