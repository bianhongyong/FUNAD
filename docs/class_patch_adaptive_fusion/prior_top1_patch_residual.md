# FUNAD 代码修改方案：类别先验 Top-1 检索 + Query/Patch 动态融合残差

> **性质**：基于当前 GitHub 源码的实施设计；尚未修改远程仓库，也未运行训练/单元测试。  
> **仓库**：`bianhongyong/FUNAD`  
> **基线分支**：`feature/learnable-dinov3-layer-fusion-design`  
> **核验提交**：`800354e892af940bd08d09bdd3999f91830fa3f0`（以 SHA 固定避免后续分支漂移）  
> **目标**：Patch 动态权重只影响最终 DINOv3 融合残差，不影响最近邻参考 Patch 的选择；取消 Top-K Query 条件精排，保留训练梯度及旧 checkpoint 兼容。

## 1. 核心决策

**不再使用**“类别先验粗检索 Top-K → Query 权重对候选精排 → 选择最近邻”的 V2 默认路径。

改为：

1. 固定 DINOv3 提取原始多层特征：`Q: [B,P,L,C]`；原始参考库 `M_c: [N_c,L,C]`。
2. 仅以类别先验 `alpha_global[c,:]` 融合 Query 与参考，寻找**唯一**最近邻 `j*(c,p)`。参考的类别先验融合库可缓存并建立 Top-1 索引。
3. 由 Patch Gate 对每个 Query Patch 生成 `alpha_patch[c,p,:]`。
4. 用**相同的 `alpha_patch`**分别融合当前 Query 和它的唯一参考 Patch；两者相减得到 `residual: [B,P,C]`。
5. 后续 Adapter（legacy/class_residual）、Hard Class Gate MoE、BCE+OTO、伪标签与评估逻辑保持原状。

### 1.1 数学定义

类别先验：
\[
\alpha^G_{c,l}=\operatorname{softmax}_l(\theta_{c,l}/\tau)
\]

匹配阶段（与 Patch Gate 无关）：
\[
z^G_{q,p}=\sum_l\alpha^G_{c,l}T(Q_{p,l}),\qquad
z^G_{m,j}=\sum_l\alpha^G_{c,l}T(M_{c,j,l})
\]
\[
j^*(c,p)=\arg\min_{j\in\{1,\ldots,N_c\}}\|z^G_{q,p}-z^G_{m,j}\|_2^2.
\]

这里 `T` 是 Fusion 使用的逐层归一化（`none` 或 `layer_norm`）。

Patch 权重：
\[
\alpha_{c,p,l}=\operatorname{softmax}_l
\left((\theta_{c,l}+\lambda_c\delta_{c,p,l})/\tau\right).
\]

最终融合残差：
\[
r_{c,p}=\sum_l\alpha_{c,p,l}
\left[T(Q_{p,l})-T(M_{c,j^*(c,p),l})\right].
\]

**特别注意**：不要写成逐层单独 NN；所有层对应同一个参考行 `j*`。当 `T=none` 时，先融合再相减与先对**同一个**参考 Patch 逐层相减再线性加权在代数上等价；`layer_norm` 时必须分别对 Query 与 Reference 作 `T`，不能先做差后 LayerNorm。

### 1.2 与旧算法的研究关系

- 旧 `dynamic_topk K=32`：Patch 权重同时影响选择参考 Patch 和最终残差。
- 新 `prior_top1`：Patch 权重**只**影响最终残差；参考由类别先验决定。
- `K=1` 的旧动态 Top-K 路径与新路径在同一缓存/归一化/权重条件下应数值等价；但旧实现仍执行冗余候选精排循环。
- 新算法不是旧 K=32 的无损加速；两者可能选中不同参考，因此异常检测指标可能变化，需要公平实验。

## 2. 仓库现状（已实际读取 800354e 源码）

| 文件 | 现状 | 改动 |
| --- | --- | --- |
| `src/model/class_patch_fusion_adapter.py` | `ClassPatchFusionAdapter` 有 `global_weights()`、`patch_weights()`、`fuse_query()`、可选 `adapt_residual()`；`forward()` 是类别先验融合 | 主体保留；规范 `global` 只用于参考匹配、`patch` 只用于残差；无需重写 Gate |
| `utils/train_utils.py` | `FusedReferenceCache.nearest_topk_ids()`、`compute_class_patch_fused_residual_batch()` 包含 Top-K 粗排 + 分块 rerank + 二次加载 | 新增/替代为轻量 `compute_class_patch_prior_top1_residual_batch()`；调用 `nearest_ids()`；删除新模式 rerank 循环 |
| `self_train_ad_multiclass_dinov3.py` | `ClassPatchFusionRuntime.residual()` 调用 `compute_class_patch_fused_residual_batch(topk=...)`；CLI、日志、schema 绑定 K | 新增/切换匹配模式 `prior_top1`，默认使用；维护老 `dynamic_topk` 作为对照 |
| `src/train/epoch_precompute.py` | Phase 1、2、3 都通过相同 `compute_residual_feature_batch_fn` 回调处理图像 | 接口不变；实际改为统一新 callback；添加可选阶段计时 |
| `utils/evaluate.py` | 评估经 residual callback | 不更改评分定义，只验证新路径确实生效 |
| `src/inference/infer_multiclass_residual.py` | `class_patch` checkpoint 根据 `fusion_topk` 调用 Top-K 动态精排，重建 FAISS 缓存 | 根据 checkpoint `match_mode` 分派；prior_top1 调用新函数，FAISS `.search(k=1)` |
| `shell/run_self_train_ad_multiclass_class_patch.sh` | 默认 `FUSION_TOPK=32`、`FUSION_CANDIDATE_CHUNK=8` | 新默认 `MATCH_MODE=prior_top1`，只设置单参考 `REF_CHUNK`；把 K 配置放在 legacy dynamic_topk 分支 |
| `tests/test_class_patch_reference_match.py` | 测试 Top-K、全库精排、旧 K=1 与 V1 等价、两端梯度等 | 增加 prior_top1 专属测试；保留旧 Top-K 消融测试 |

精排热点：当前 `compute_class_patch_fused_residual_batch` 在 `utils/train_utils.py` 中以 `q_chunk × k_chunk` 循环，对 K 个候选不断调用 `fusion_cache.raw_rows()` 并重融合；选中最终参考后，又执行一轮 `raw_rows()` 与有梯度融合。新方案删除前一轮所有 K 候选查询与精排。

## 3. 推荐新 API 和关键伪代码

建议保留旧函数用于旧实验，**新增**清晰的无精排 API：

```python
def compute_class_patch_prior_top1_residual_batch(
    raw_query, class_idx_batch, fusion_cache, fusion,
    query_chunk_size=512,
    return_neighbor_ids=False,
    return_diagnostics=False,
):
    # shape: raw_query [B,P,L,C], class_idx_batch [B]
    validate_inputs(raw_query, class_idx_batch, fusion_cache, fusion)

    # A. 仅类别先验进行匹配：不使用 Patch Gate
    with torch.no_grad():
        # 优先：使用与 fused cache 同一刷新版本的类别先验权重/归一化快照
        coarse_query = fusion_cache.fuse_query_with_cached_prior(raw_query)
        nn_ids = fusion_cache.nearest_ids(coarse_query, class_idx_batch)  # [B,P]

    # B. Query Patch 动态权重只参与最终残差
    alpha = fusion.patch_weights(raw_query, class_idx_batch)  # [B,P,L], 保留梯度

    # C. 按类别、Query chunk 处理每个查询的唯一参考行
    residual = empty([B,P,C], dtype=alpha.dtype, device=raw_query.device)
    for cls in unique(class_idx_batch):
        for q_chunk in class_query_chunks(cls, query_chunk_size):
            ids = nn_ids_for_chunk(nn_ids, cls, q_chunk)
            # 可先 unique(ids) 以降低 CPU→GPU 重复传输，
            # 但每个 Query 的动态权重仍不同，不能复用同一预融合向量。
            unique_ids, inverse = torch.unique(ids, return_inverse=True)
            raw_unique = fusion_cache.raw_rows(cls, unique_ids)
            raw_ref = raw_unique.index_select(0, inverse)  # [M,L,C]

            q = query_rows_for_chunk(raw_query, cls, q_chunk)
            a = patch_alpha_for_chunk(alpha, cls, q_chunk)
            z_q = fusion.fuse_query(q, a)
            z_m = fusion.fuse_query(raw_ref, a)
            write_chunk(residual, z_q - z_m, cls, q_chunk)

    # D. 返回的单一索引 j* 完全由类别先验决定
    return residual, nn_ids, optional_alpha_diagnostics
```

上述是算法级伪代码，`validate_inputs`、`class_query_chunks`、`write_chunk` 需按代码库真实接口实现，**不能直接作为可运行补丁粘贴**。

### 3.1 重要的梯度约束

- DINOv3/Raw Bank 不需要梯度；检索及 `argmin`/FAISS 返回的 `nn_ids` 无梯度。
- `patch_weights()` 和最终 `fusion.fuse_query(query, alpha)`、`fusion.fuse_query(selected_ref, alpha)` **必须**处于 autograd 上下文内（主训练阶段）。
- 不得将最终 `z_m` 换成类别先验 `fused_cache[j*]`，否则违反“每个 Query 用同一动态权重融合 Q/M”的语义，且破坏参考侧融合权重的梯度。
- Phase 1/2/3、评估和推理在 `no_grad()` 下调用相同公式；仅跳过梯度图，不跳过动态参考融合。
- 保留 `fusion.norm_type="none"` 的现有默认值；支持 `layer_norm` 的完整数值回归。

### 3.2 NN 缓存版本的一致性（必要）

仓库已有 `refresh(epoch/step)` 和 stale-NN 诊断。当前 Top-K 代码的粗排 Query 使用当前 `fusion.global_weights()`，参考缓存可能使用上次刷新时的 `theta`。

**建议修正**：`FusedReferenceCache.refresh()` 记录查找索引所用的类别先验快照 `prior_alpha_at_refresh`（如果 `fusion_norm_affine=True`，归一化仿射参数也需同版本快照，或要求精确刷新/冻结参数）。粗排 Query 使用相同快照，避免两端投影不一致。

- 当前默认 `fusion_norm=none`，只保存类别先验 alpha 即可。
- 若选择“始终用最新类别先验检索”，则每次 theta 更新后需相应刷新参考索引；代价更高。
- 训练默认 epoch/1 是近似的**滞后先验**，不能称为每步精确的当前类别先验检索；评估/推理前强制刷新。
- `nn_ids` 的选取始终不由 `delta` 或 `lambda` 控制；对 `delta/lambda` 做干预后 `nn_ids` 应保持不变。

## 4. 分文件修改任务（Codex 可依序提交）

### PR1 / Commit 1：新增 Prior-Top1 Residual 函数

修改 `utils/train_utils.py`：
1. 增加 `compute_class_patch_prior_top1_residual_batch(...)`，复用 `_validate_class_patch_inputs`。
2. 查询用类别先验查 `nearest_ids`，不调用 `nearest_topk_ids(k>1)`、`exact_dynamic_nearest_ids()` 或精排。
3. 参考只读取 `nn_ids` 指向的原始多层特征；`unique(ids)` 减少 CPU→GPU 传输，重展开后逐 Query 动态融合。
4. `FusedReferenceCache.nearest_ids()` 优化 Top-1 fast path：对 Torch 分块使用 `_chunked_nearest_ids`；对 FAISS 使用 `search(...,k=1)`；保留已有 Top-K 方法供旧模式使用。
5. 封装 `fuse_query_with_cached_prior` 和在 `refresh` 保存匹配先验快照；只允许 NN 检索调用其 no-grad 路径。
6. 统计 `nn_search_ms/ref_gather_ms/patch_gate_ms/residual_fusion_ms`（计时时同步 CUDA，基准关闭过度频繁的逐 Batch 日志）。

验收：`nn_ids` 与类别先验全库 brute-force Top1 相同；返回 `[B,P,C]`；混合类别 Batch 不跨库。

### PR2 / Commit 2：训练运行时与参数协议

修改 `self_train_ad_multiclass_dinov3.py`：
1. 新增 `--fusion_match_mode {prior_top1,dynamic_topk}`。新训练的 class_patch **默认** `prior_top1`；旧 `dynamic_topk` 可显式选用。
2. `ClassPatchFusionRuntime.residual()`：
   - prior_top1 → 新函数；
   - dynamic_topk → 保留旧函数，作为历史比较。
3. 新模式删除 `_accumulate_patch_stats` 对 `best_ids != coarse_ids[...,0]` 的 switch-rate 统计（理论上恒为 0、失去意义）；仍记录 Patch alpha 均值/标准差/与类别先验 L1/熵、lambda、beta/rho。
4. `maybe_probe_topk_recall` 只在 dynamic_topk 运行；prior_top1 可记录检索缓存 drift、Phase 1/2/3/训练前向耗时。
5. 不修改 BCE+OTO、PCA/MAD、Hard Class Gate MoE、Adapter 设置以及 DINOv3 的冻结策略。
6. Phase 1/2/3 走同一个 residual callback；不用分别重写三套 NN。

### PR3 / Commit 3：checkpoint 兼容与独立推理

修改 `self_train_ad_multiclass_dinov3.py` 和 `src/inference/infer_multiclass_residual.py`：
1. 新 class_patch checkpoint 设置 `schema_version=3`、`match_mode="prior_top1"`；不再要求 `fusion_topk`。
2. 历史 `schema_version=2` 的 `class_patch` checkpoint **缺省解释为 `dynamic_topk`**；不可默默以新语义加载。
3. 新模式训练恢复时，核对 `match_mode` 与 checkpoint；不一致直接报错，避免继续训练后可复现性混乱。
4. 如希望复用历史模型权重：另提供显式的 *weights-only initialization* 路径，加载网络参数但不沿用旧优化器、自训练伪标签状态及历史指标；这是新实验，不是原训练续跑。
5. 独立推理按 checkpoint 自动分派；载入原始参考库和 Fusion state，刷新类别先验索引；prior_top1 调新函数。
6. 训练 eval/独立 infer 的相同图像 residual/score 数值一致（固定参数与预处理），旧 V1/旧 V2 checkpoint 回归不退化。

### PR4 / Commit 4：脚本、文档与可解释性

- `shell/run_self_train_ad_multiclass_class_patch.sh`：默认 `MATCH_MODE=prior_top1`；新模式不传 `FUSION_TOPK` 和 `FUSION_CANDIDATE_CHUNK`；需要对照时设置 `MATCH_MODE=dynamic_topk` 再传 K。
- `docs/class_patch_adaptive_fusion/implementation_plan.md`：新增本方案为当前推荐路径，旧 Top-K 作为历史/研究消融，不要不留痕迹地删除。
- 新建 `docs/class_patch_adaptive_fusion/prior_top1_patch_residual.md`，记录本决策、公式、API、风险与实验。
- 日志区分 `match_mode`、先验检索漂移、Gate 变化和 Adapter 变化；不要把旧 `nn_switch_rate` 误当成新架构的指标。

## 5. 测试矩阵

新增 `tests/test_class_patch_prior_top1.py`（也可扩展旧 test 文件）：

1. **固定类别 NN**：构造少量 Raw Bank，独立 brute-force 按 `alpha_global` 计算期望 ids，要求新函数输出一致。
2. **Gate/先验解耦**：保持 `theta` 和缓存不变，随机化/修改 `delta_head`、`lambda_raw`，`nn_ids` 必须逐元素不变；残差可以变化。
3. **动态残差一致**：逐 Patch 断言 `r == fusion.fuse_query(Q,alpha) - fusion.fuse_query(M[j*],alpha)`。
4. **退化一致**：`delta=0` 时，新模式与 V1 类别融合在相同原始库、层顺序、归一化、索引版本下对齐。
5. **旧 K=1 数值回归**：在对齐缓存条件下，新 prior_top1 与旧 dynamic_topk `K=1` 的 ids/residual 近似一致。
6. **梯度测试**：BCE 可微路径检查 `theta`, `delta_head`, `lambda_raw`, `class_embedding` 梯度；参考端不 detach；NN 索引不求导。
7. **特殊情况**：混类别、小参考库、空库错误、重复 ids、`L=1`、`norm=none/layer_norm`、FP16 Raw 与 FP32 距离、CPU/CUDA。
8. **索引快照**：更新 theta 但不刷新缓存时 Query/Reference 均使用同一快照；刷新后对新先验一致。
9. **checkpoint**：新模式恢复/推理、旧 schema 2 仍走 dynamic_topk、不同 match_mode 不允许无提示 resume。
10. **全链路**：Phase 1/2/3、主训练、eval、independent infer 使用同一 residual 定义，输出形状、数值和异常分数无意外变化。

建议执行（提交后）：

```bash
python -m unittest tests.test_class_patch_fusion_adapter -v
python -m unittest tests.test_class_patch_reference_match -v
python -m unittest tests.test_class_patch_prior_top1 -v
python -m unittest tests.test_layer_fusion -v
```

## 6. 性能定位与优化顺序

**预计**：从 K=32 候选精排变成单参考动态融合，可减少候选参考读取和逐候选投影成本；并不意味着端到端速度提高 32 倍，因为 DINOv3、全库类别先验 NN、Adapter/MoE 和数据加载仍存在。

第一版先做正确性与计时，随后按数据决策：

1. **删除精排**：不再遍历 K 候选；不再执行二次候选读取。
2. **Top-1 fast path**：原 `nearest_ids()` 目前委托给 `nearest_topk_ids(k=1)`，可改为 `min` 专用搜索，减少不必要的 `topk + cat + topk`。
3. **参考行读取**：CPU FP16 Raw Bank 保留；先对匹配 id `unique` 后读取；多个 Patch 可用不同动态 alpha，不能复用已融合向量。
4. **按类别的 GPU Raw Bank 热缓存（可选）**：仅在显存足够时启用，减少 CPU→GPU。每个类别存储约 `N_c*L*C*2 bytes`（FP16），计算预算后配置 LRU/分块，不要盲目全类别上 GPU。
5. **GPU 计时**：对 Phase 1/2/3、主训练、eval 分别记录 `DINOv3 / coarse NN / gate / ref IO / residual fusion / localnet / total`；用 CUDA Event 或同步边界计时。
6. **避免频繁 GPU→CPU 日志同步**：`_accumulate_patch_stats` 当前每 Batch `alpha.detach().cpu().numpy()`；可选先在 GPU 累加，按 Epoch 输出。

### 当前三个最显著的代码级性能风险

- V2 动态精排每个 Query 的 K 个候选都可能调用 `raw_rows()` 从 CPU 取原始多层参考。
- 当前 `nearest_ids()` 通过 Top-K(k=1) 路径，Torch chunk search 需要运行两轮 `torch.topk` 维护候选集合；改成专用 Top-1 `min` 更合适。
- V2 Patch 权重统计每 Batch 将 `alpha`、indices 转 CPU 并触发同步，可能额外影响阶段吞吐。

## 7. 实验矩阵及判定标准

实验需统一候选 DINOv3 层、原始参考库/类别索引、网络、训练预算、随机种子、loss、伪标签、数据预处理。

| 实验 | NN 选择 | 动态层权重用于残差 | 作用 |
|---|---|---|---|
| B0 | 类别先验 Top-1 | 否（类权重） | V1 learned_class baseline |
| B1 | 类别先验 Top-1 | 是 | **新方案 prior_top1** |
| B2 | 类别先验 Top-K + 动态精排，K=1 | 是 | 新方案数值回归/实现开销对照 |
| B3 | 类别先验 Top-K + 动态精排，K=16/32 | 是 | 旧 V2 的效果/速度上限对照 |
| B4 | B1 + GPU 参考行缓存 | 是 | 纯工程提速对照 |
| B5 | B1 + 类别强度 Adapter | 是 | 功能模块独立消融（可选） |

至少记录：I-AUROC / P-AUROC（逐类与均值）、Phase 1/2/3 耗时、训练 Epoch 耗时、评估推理耗时、峰值显存、Patch alpha 的均值/方差/偏离先验、Gate 梯度、Reference NN 漂移。

**验收门槛**：
- 固定类别先验和参考缓存后，任意修改 Patch Gate（不修改 theta）都不能改变 NN id。
- 选中唯一参考 `j*` 后，Query 和 Reference 使用同一 alpha；所有自训练/推理阶段一致。
- `delta=0` 与类别融合基线数值对齐；训练有正确 Fusion 梯度；旧 checkpoint 继续可复现。
- 新路径不再执行 `K>1` 候选精排。
- 性能指标与加速幅度应由真实 benchmark 给出，不提前宣称。

## 8. 实施优先顺序（给 Codex）

**先 PR1 核心 Top-1 路径 + 单元测试 → PR2 训练 Runtime/日志 → PR3 推理/Checkpoint/兼容 → PR4 性能优化与文档。**

在所有测试通过之前不要删掉旧的 `compute_class_patch_fused_residual_batch()`；保留 `dynamic_topk` 明确开关，既可复现实验，也方便隔离新目标改变造成的效果差异。

### 用于 Codex 的一句任务描述

> 以 `800354e892af940bd08d09bdd3999f91830fa3f0` 为基线，新增 `class_patch` 的 `prior_top1` 匹配模式（新训练默认），类别先验在可刷新预融合索引上只选一个参考 Patch `j*`；Patch Gate 输出 `alpha_{c,p}` 不参与最近邻选择，只用于分别融合当前 Query 和 `M[j*]` 并计算 DINOv3 残差。移除该模式 Top-K rerank 的循环/候选读取，保留双侧 Fusion 梯度、Phase1/2/3/训练/评估/独立推理一致性；新增 schema v3 + match_mode，旧 schema v2 缺省为 dynamic_topk；添加单元测试、独立 benchmark、性能日志，不更改现有 MoE/Adapter/伪标签算法。