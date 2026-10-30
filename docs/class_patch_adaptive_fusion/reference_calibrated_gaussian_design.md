# 参考库标定的类别固定 Gaussian 扰动设计（方案文档）

> **状态**：设计方案 / 待编码；本提交**只增加文档**，不修改训练代码、不宣称已经测试或提升性能。  
> **目标分支**：feature/learnable-dinov3-layer-fusion-design  
> **范围**：仅改进 FUNAD 不确定 Patch 的 Gaussian 特征合成。**不增加真实异常挖掘、不修改伪标签定义、不替换 MAD/noise_threshold、不改变 Fusion/MoE/OTO 和推理路径。**  
> **核心决策**：由每个类别的干净参考图，在训练开始前计算一个固定的残差噪声基准；训练时只按 Patch 所处不确定区间的位置进行轻量调节。

## 1. 问题与设计原则

现有 utils/train_utils.py 的 apply_gaussian_feature_noise 在未传入显式 std 时使用：

    x_std = x.detach().std(dim=0).max(dim=0)[0]

这里 x 是当前 Batch 的 **Fusion/最近邻匹配后的 residual，进入 LocalNet adaptor 之前**，形状为 [B,P,D]。该标准差沿 Batch 维统计，再取 Patch 位置上的最大值，因此：

- 不同 Batch 的噪声尺度会变化；
- 多类别 Batch 可能混合类别，使噪声尺度受到其他类别影响；
- 该统计并不是每个类别独有、可复现的正常先验；
- 关闭 Gaussian 已经观察到明显性能恶化，所以应保留扰动机制，避免大规模改动训练信号。

目标：为每个类别 c 预计算一个**固定标量** sigma_ref[c]；它只由该类正常参考库、冻结 DINOv3 特征与**初始** Fusion 匹配规则决定，不在 Epoch/Batch 中更新。对位于不确定区间的 Patch 使用 gamma[c,p] 轻量缩放；保留原 BCE 标签、训练梯度和 Gaussian 分支。

**注意**：sigma_ref 是正常特征变化的**统计尺度先验**，不是可证明最优的合成异常强度。其有效性仍以自然异常测试性能为准。

## 2. 严格保持现有三段式规则

定义 d[c,p] 为 Phase 3 已计算的**按类别归一化**的动态最近邻距离；T_mad[c] 为对应类别归一化 MAD 阈值；T_noise = args.noise_threshold（当前默认 0.8）。

| Patch 区间 | 现有训练处理 | 本方案 |
| --- | --- | --- |
| d < T_mad | 原始特征，伪标签 0 | 完全不变 |
| T_mad < d < T_noise | Gaussian 扰动，伪标签 1 | 仅将 Gaussian 尺度改为固定类别基准 × Patch 系数 |
| d > T_noise | 原始特征，伪标签 1 | 完全不变 |

边界等号严格沿用代码原有实现（当前扰动 mask 为 d > T_mad 且 d < T_noise；伪标签仍按 d > T_mad 生成），不得在改动中意外重定义阈值。

**不确定区间在训练中仍是合成异常监督来源**；不能把该区间原始 GT 正常比例当作扰动后标签的错误率。该方案不涉及原始真实异常伪标签提升或额外去噪。

当前分支已经存在 build_uncertain_patch_mask / Gaussian 关闭时的 uncertain 样本排除逻辑；必须沿用，不恢复旧版本的“关闭 Gaussian 仍硬标异常”行为。

## 3. 参考库统计：Leave-One-Image-Out 正常残差

### 3.1 统计空间必须与噪声注入空间相同

学习模式下已有 raw_reference_memory_by_class，其每类数组约为 [N_c,L,C]，保留每个 Patch 的全部 DINOv3 层特征。**不能直接取 raw DINOv3 特征的 std**：Gaussian 实际施加在 Fusion + NN residual 的 [B,P,D] 上，两者量纲与分布不同。

必须在模型初始状态构造“干净参考图之间的 residual”，形状 [N_c,D]，并基于该 residual 估计 sigma_ref。此步骤在任何训练 optimizer.step() 之前、完成 reference bank 初始化与 Fusion 初始化之后执行；使用 torch.no_grad()，计算完成立即冻结。

### 3.2 与当前 class_patch / prior_top1 计算规则对齐

对类别 c 的每一张参考图 I_i 的 Patch q：

1. 候选参考集合只能包含该类别**其他参考图的全部 Patch**，必须排除 I_i 的全部 Patch，不仅排除同一 Patch。
2. 使用初始化的类别先验 alpha_global[c,:] 融合 Query 与候选参考，选取**唯一 Top-1** 最近邻 j*，与现有 prior_top1 模式保持一致。screw 等“最后层固定先验”按当前 Fusion 配置处理。
3. 对 Query Patch 计算其初始化的 alpha_patch[c,p,:]；用**相同的 alpha_patch**分别融合 Query 和所选参考行，再作差：

   r_ref[c,p] = sum_l alpha_patch[c,p,l] * ( T(F_query[p,l]) - T(F_ref[j*,l]) )

   这里 T 指 Fusion 内的逐层归一化；若启用 LayerNorm，必须分别对 Query / Reference 做 T，不允许先减后归一化。alpha_patch 只影响残差，不改变步骤 2 的近邻选择。
4. 保留 [D] 正常残差向量，仅用于标定；不改变原始参考库、不污染动态记忆库。

**实现注意**：

- 当前 raw reference 存储按参考图的 Patch 顺序展平，标定代码必须显式记录并核验 image_id / image_offsets / patches_per_image；不要仅靠匿名行号猜测图片边界。
- 至少需要每类别 2 张干净参考图，正常设置为约 8 张；少于 2 张时报明确错误，禁止自匹配回退。
- 应分块计算 Top-1 距离，避免一次构造整个 N_c × N_c 距离矩阵；该代价只在训练开始前支付一次。
- 若使用非 class_patch 的 learned_class / fixed_mean 模式，保持其当前初始融合规则，仍要计算同一 residual 输入空间，不强制调用 class_patch 专属接口。
- 参考图的预处理必须与训练特征提取一致，避免额外随机增强改变先验。

### 3.3 每类别固定标量的推荐公式

设 D 为 Gaussian 注入点的 residual 通道数，S_c 为所有参考图留一匹配所得的正常残差范数集合：

   S_c = { ||r_ref[c,p]||_2 : p 属于类别 c 的干净参考 Patch }

推荐：

   **sigma_ref[c] = Q_0.75(S_c) / sqrt(D)**

选择 Q75 而非最大值，是为了降低少量极端正常 Patch 对尺度的影响；除以 sqrt(D) 是因为 D 维各向同性 Gaussian 的典型 L2 噪声尺度约为 sigma*sqrt(D)。

这是从残差**范数**反推**每通道各向同性标准差**的尺度标定，不代表原始残差各通道严格独立同分布。

- sigma_ref[c] 采用 float32 保存，必须有限且 >0；如果分布退化为零或 NaN，发出明确错误，不默默使用当前 Batch 标准差，以避免“固定先验”名不副实。
- 可保留一个默认固定全局倍数 gaussian_ref_scale=1.0，使最终基准 sigma_c=gaussian_ref_scale*sigma_ref[c]；该倍数属于实验配置，**不**由 Batch/GT 动态确定。
- 本版本只采用一个标量/类别，不估计全协方差、也不训练 sigma 网络；保持设计简单。

## 4. Patch 级轻量扰动系数

对已满足 uncertain_mask 的 Patch 计算：

   u[c,p] = clip((d[c,p]-T_mad[c])/(T_noise-T_mad[c]), 0, 1)

   gamma[c,p] = 1 - 0.25*u[c,p]    # 范围 [0.75, 1.0]

最终扰动：

   epsilon[c,p,k] ~ Normal(0, sigma_c^2)

   z_perturbed[c,p,k] = z[c,p,k] + gamma[c,p]*epsilon[c,p,k]

其中 k 为特征通道。仅在 uncertain_mask 上加噪，其余位置保持原值。

**为何这样定**：保留当前已验证有效的 Gaussian 路径，近 MAD 边界使用原始基准强度，接近 T_noise 时适当减弱扰动；不再使用固定上限以上的激进增幅。系数 0.25 是保守设计初值，不是实证最优值。

实现约束：

- 当 T_noise <= T_mad 时对应类别没有有效不确定区间；该类别不施加此类噪声，避免除零。
- 计算 gamma 时 detach 距离/阈值；噪声采样置于 no_grad，最后必须采用 z + detached_noise 的形式，**保留 BCE 到 z / Fusion 的梯度**。
- class_idx: [B] 映射 sigma_ref[class_idx]: [B,1,1]；gamma/mask: [B,P,1]；与 x: [B,P,D] 显式广播。避免跨类别 Batch 统计和形状错配。
- 如果 args.beta 额外拼接 synthetic row，原有单独生成的 synthetic row 与 label 维度必须维持一致；**只对真正的原始训练图 [B,P,D] 注入此 Gaussian**，之后按原代码拼接 synthetic row。
- 不更改原始 loss 权重、OTO、MoE Gate、动态记忆库、reference matching、P-AUROC/I-AUROC 计算、推理分支。

## 5. 接口、缓存与兼容性

建议新增 CLI：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| --gaussian_prior_mode {batch,reference} | batch | 默认完全复现旧版；reference 才启用类别固定参考先验 |
| --gaussian_ref_quantile | 0.75 | 正常参考残差范数分位数 |
| --gaussian_ref_scale | 1.0 | 对所有类别先验统一的固定缩放 |
| --gaussian_gamma_min | 0.75 | 令 gamma=1-(1-gamma_min)*u |

- 原参数 --gaussian 是 action=store_false：**不带此参数才开启 Gaussian；带上才关闭**。不得擅自修改兼容语义。
- 在 reference 模式中，如用户同时显式设置 --std，建议直接报参数冲突，并说明 reference 模式已自行定义固定 sigma；不要静默覆盖其任一个值。
- 默认 batch 模式下不改变当前 Gaussian 的随机采样、阈值逻辑和任何训练行为。
- 保存一次性标定结果：class_name/class_idx -> sigma_ref、quantile、scale、D、ref image IDs/offsets、raw reference fingerprint、backbone/layers、预处理、fusion 初始配置或其 fingerprint、match_mode、fusion_norm、代码版本标识。
- sigma_ref 与 checkpoint 一同持久化，并允许独立 JSON sidecar 供查看。resume 在 reference 模式下必须恢复**原来固定**的 sigma_ref，不能用恢复训练时已经更新的 Fusion 参数重新标定。
- 对 sidecar/checkpoint 的类别顺序、reference hash、feature_dim、fusion config 做一致性校验；不兼容时明确失败或要求重新开始，不能静默混用。
- 旧 checkpoint / 未启用该模式的训练保持原行为。

## 6. 建议代码修改点（仅设计，不实施）

1. **utils/train_utils.py**：
   - 新增 estimate_class_reference_gaussian_prior(...)：从原始多层参考行和初始化 Fusion 做 Leave-One-Image-Out Top-1 匹配，输出每类别 sigma_ref 与统计摘要；分块处理、明确排除整张参考图。
   - 扩展 apply_gaussian_feature_noise(...)：增加可选 sigma_by_class、class_idx、gamma_min；只有 reference 模式走新实现，旧参数路径保持原样。
2. **self_train_ad_multiclass_dinov3.py**：
   - 增加 CLI 参数与冲突检查；
   - 在创建并刷新初始 Fusion reference cache 后、首个训练 step 前计算/恢复 sigma_ref；
   - 在 train loop 的 Gaussian 调用中传入 sigma_ref 和 class_idx；不改变 local_label、pred_for_loss、BCE、OTO；
   - 统一保管/保存/恢复参考先验元数据。
3. **单元测试**（建议新建 tests/test_reference_calibrated_gaussian.py）：
   - toy references：排除同图全部 Patch，Top-1 参考选择及残差公式正确；
   - Q75/sqrt(D) 数值可重复，类别 A/B 统计独立；
   - 参考模式不同 Batch 排列不改变 sigma_ref；
   - gamma 在 MAD 处为 1.0、接近 noise_threshold 为 0.75，mask 外噪声严格为 0；
   - backward：输入 residual 获得非零有效梯度；
   - beta 拼接行对齐，CPU/GPU、空 uncertain 区域、MAD>=noise、resume 一致性；
   - 默认 batch 模式行为与老版一致。

## 7. 必须区分的日志与评价指标

新增一次性日志：

    [RefGaussian] class=bottle n_ref_images=8 n_ref_patches=6272 q75_residual_norm=... D=... sigma_ref=... scale=...

每 Epoch 可记录 uncertain Patch 数量、按类 gamma 的均值/范围、实际注入噪声的 L2 范数，不引入 GT 参与训练计算。

**评价边界**：
- 原始 Patch 的 MAD / noise_threshold 高置信判定质量不因本方案直接改变；
- 合成异常标签是对**扰动后特征**的训练目标，不能直接用原始 GT mask 宣称其正确率；
- 可能影响最终 I-AUROC/P-AUROC，但目前没有效果保证；若性能下滑，优先检查固定 sigma 与原 Batch 标准差的数量级差异及 Fusion 训练后 residual 尺度漂移。

## 8. 任务完成判据

- 设计按本文实施后，默认配置与原模型行为一致；显式 --gaussian_prior_mode reference 才启用。
- 一次标定得到每个产品类别唯一固定 sigma；整个训练过程不调用 Batch std 计算参考模式噪声尺度。
- 不改变三段式伪标签、Gaussian 加噪范围、BCE/OTO 目标、MoE 和推理。
- 参考图自匹配和跨类别统计得到排除，参考残差与注入空间严格对齐。
- 同一 checkpoint resume 后 sigma_ref 不变且校验元数据。
- 完成上面最小单元测试和运行日志检查，才可称为“已实施”；本文件提交**不代表代码已实现**。

## 9. 设计来源与相关代码

- 当前工程：utils/train_utils.py（Gaussian、reference bank、class_patch prior_top1 residual）、self_train_ad_multiclass_dinov3.py（Fusion 初始化、训练调用、Checkpoint）、src/model/class_patch_fusion_adapter.py（类别与 Patch 层权重）。
- [既有 prior_top1 设计](./prior_top1_patch_residual.md)；[已有多层 Fusion 参考文献](../learnable_dinov3_layer_fusion/references.md)。
- 相关工作：SimpleNet（CVPR 2023，Gaussian 特征合成）；GLASS（ECCV 2024，合成异常距离/难度控制）；PatchCore（CVPR 2022，正常 Patch 最近邻记忆库）。本文不是上述论文的复现，Q75/sqrt(D) 与 gamma 为本项目拟定的工程假设，尚需验证。
