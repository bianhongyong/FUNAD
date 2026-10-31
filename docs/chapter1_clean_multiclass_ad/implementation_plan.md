# 第一章：Clean 多类别异常检测代码修改方案

> **状态：Implementation Plan / 仅方案，尚未实现、训练或验证**
>
> **用途：** 供论文第一章方案审阅；本文件不代表实验结果或性能承诺。
>
> **基线仓库：** \`bianhongyong/FUNAD\`
>
> **基线分支：** \`feature/learnable-dinov3-layer-fusion-design\`
>
> **基线 Commit：** \`2051e69b7935f4bd2c3e2e0e9188df6fe8d40f24\`
>
> **方案分支：** \`feature/chapter1-clean-multiclass-ad-plan\`
>
> **方案日期：** 2026-10-31
>
> **本次范围：** 只新增本方案文档；不修改 Python、Shell、模型、训练、推理代码，不运行实验，不创建 PR。

---

## 0. 已确认的设计决策

第一章的训练数据全部为干净正常样本（clean normal），不包含真实异常污染。通过特征空间异常合成构造训练时的异常监督。

| 模块 | 已选方案 | 本方案中的约束 |
|---|---|---|
| 特征融合 | \`class_patch\` | 类别先验选择唯一最近邻；Patch 动态权重只参与最终残差计算 |
| 最近邻模式 | \`prior_top1\` | 不使用 \`dynamic_topk\` 作为主路径 |
| 异常合成 | \`feature\` | 在 DINOv3 多层特征融合后的残差空间合成，不在 RGB 图像空间做 CutPaste |
| 判别器 | \`legacy\` | 使用现有 MLP 判别器；不启用 MoE、不增加专家和 Gate 辅助损失 |
| 特征适配器 | 现有 legacy 路径 | 保留当前 \`localnet\` 的类别条件 Adapter；不启用 V2-C 的额外 Beta/FiLM Adapter |
| 训练损失 | Balanced BCE | 仅正常 Patch 与合成异常 Patch 的二分类监督 |
| OTO | 不使用 | 第一章不计算 one-to-one / mutual smoothness loss |
| IRMB / 伪标签 | 不使用 | 第一章不运行 noisy-AD 的伪标签重构和自训练流程 |
| Backbone | 冻结 DINOv3 | 不对 DINOv3 反向传播；只训练融合模块、现有 Adapter 和 MLP |

**术语区分：** 本文中的 \`legacy discriminator\` 指 \`localnet\` 中原有的三层 MLP 异常判别头；\`feature_adapter_mode=legacy\` 指继续使用 \`localnet\` 自己的现有 Adapter。两者不是同一个开关。

## 1. 第一章的研究目标与定位

### 1.1 研究目标

第一章解决无污染条件下的多类别工业异常检测：

1. 只使用每个类别的 clean normal 训练图像，不利用真实异常训练样本。
2. 重点改善像素级 Average Precision（P-AP），同时保持较高的 I-AUROC、P-AUROC。
3. 相比 Dinomaly 等强基线，尽量减少额外可训练参数、计算量和推理开销。
4. 形成可以被第二章 noisy-AD 直接承接的特征表示、类别融合、参考库和残差计算接口。

第一章的核心不是重新设计一个大型 Backbone，而是研究：**如何利用 DINOv3 的多层特征和类别/局部 Patch 信息，构造更适合像素级异常定位的残差表示，并通过低成本的特征级异常合成训练判别器。**

### 1.2 与第二章的关系

- 第一章：clean normal data + feature-level synthetic anomalies + Balanced BCE。
- 第二章：在第一章模型结构基础上，面对含噪训练数据，使用 FUNAD 的伪标签、记忆库更新和自训练策略。
- 两章共享：DINOv3 特征抽取协议、类别层次的参考库、\`class_patch\` 融合模块、\`prior_top1\` 最近邻选择、残差定义和模型推理接口。
- 两章不同：训练数据假设、异常监督来源、损失函数和训练阶段。

第一章不应该提前引入第二章的 OTO、MAD、IRMB 或 noisy pseudo-label 机制。这样可以让两章的贡献边界清晰。

## 2. 当前代码审查与需要补齐的部分

### 2.1 当前 clean 训练入口

重点文件：\`self_train_ad_multiclass_dinov3_clean.py\`

当前代码已经具备：

- 使用 \`MultiClassFeatureDataset\`，从每个类别的 \`train/good\` 读取训练图像。
- 按类别选择 clean reference images，默认每类最多 8 张，并从训练 DataLoader 中移除这些参考图，避免训练查询与自身参考图直接匹配。
- 冻结 DINOv3，提取 Patch 特征。
- 构建类别参考特征库，计算 Query 与最近邻参考 Patch 的残差。
- 使用现有 \`localnet\` 完成特征适配和 Patch 异常评分。
- 使用 BCE 训练，且没有 OTO。

当前最关键的问题：

**训练循环将所有训练 Patch 的标签全部设为 0。** 这意味着判别器只收到正常监督，没有明确的异常正样本；即使代码可以正常运行，也不能形成有效的正常/异常二分类训练信号。

此外，当前 clean 入口尚未接入 learned fusion，也没有 feature-level anomaly synthesis。它使用的是按类别指定的 DINO 层特征和旧的单层残差计算。

### 2.2 当前分支已经有、应该复用的能力

本方案基于的分支已经实现了以下模块，不应另起一套不兼容的实现：

| 文件 | 已有能力 | 第一章如何使用 |
|---|---|---|
| \`src/model/class_patch_fusion_adapter.py\` | 类别先验、Patch Gate、动态融合权重 | 直接复用 \`ClassPatchFusionAdapter\`，关闭额外 class-residual Adapter |
| \`src/model/layer_fusion.py\` | 类别级静态层融合 | 作为消融基线，不作为第一章主模型 |
| \`utils/train_utils.py\` | Raw Reference Bank、\`FusedReferenceCache\`、残差检索 | 复用 raw bank 和缓存刷新机制 |
| \`utils/train_utils.py\` | \`compute_class_patch_prior_top1_residual_batch\` | 作为第一章主残差计算路径 |
| \`self_train_ad_multiclass_dinov3.py\` | \`ClassPatchFusionRuntime\`、融合配置、缓存生命周期 | 把可复用的初始化和 runtime 接入 clean 训练入口 |
| \`src/inference/infer_multiclass_residual.py\` | class_patch checkpoint 识别、配置校验和推理 | 尽量复用，保证训练和推理使用同一残差定义 |
| \`utils/loss.py\` | \`compute_balanced_bce_loss\` | 复用已有正负 Patch 平衡 BCE |

第一章的主要新增算法应集中在**特征级异常合成**，而不是重复实现 class_patch 的检索与融合。

## 3. 第一章最终模型架构

### 3.1 总体流程

~~~mermaid
flowchart TD
    A["Clean normal image x, class c"] --> B["Frozen DINOv3"]
    B --> C["Raw multi-layer Query Q: [B,P,L,C]"]
    R["Clean Raw Reference Bank M_c: [N_c,L,C]"] --> D["Class-prior reference matching"]
    C --> D
    D --> E["One shared reference index j* per Query Patch"]
    C --> F["Patch Gate: delta(c,p)"]
    F --> G["Patch weights alpha(c,p,l)"]
    E --> H["Multi-layer residual r_p"]
    G --> H
    C --> H
    H --> I["Feature-level anomaly synthesis"]
    I --> J["Normal residuals + synthetic residuals"]
    J --> K["Legacy feature Adapter"]
    K --> L["Legacy MLP discriminator"]
    L --> M["Patch anomaly scores"]
    M --> N["Balanced BCE"]
    N --> F
    N --> K
    N --> L
~~~

### 3.2 class_patch 的匹配与残差定义

对类别 \(c\)，类别级先验权重为：

\[
\alpha^G_c=\operatorname{Softmax}(\theta_c/\tau)
\]

对于 Query Patch \(p\)，先使用类别先验权重选择唯一参考 Patch：

\[
j^*_{c,p}=
\arg\min_j
\left\|
\sum_{l=1}^{L}
\alpha^G_{c,l}(Q_{p,l}-M_{c,j,l})
\right\|_2^2
\]

Patch Gate 根据当前 Query Patch 的多层特征和类别嵌入生成偏移：

\[
\alpha_{c,p}=
\operatorname{Softmax}
\left(
\frac{\theta_c+\lambda_c\delta_{c,p}}{\tau}
\right)
\]

最终残差：

\[
r_{c,p}=
\sum_{l=1}^{L}
\alpha_{c,p,l}
\left(Q_{p,l}-M_{c,j^*_{c,p},l}\right)
\]

必须遵守以下约束：

1. 一个 Query Patch 只选一个参考 Patch \(j^*\)，所有层共享该索引。
2. \(\alpha^G_c\) 负责最近邻选择；Patch Gate 不参与最近邻搜索。
3. Patch 动态权重只影响最终残差，不改变已经选定的 \(j^*\)。
4. Query 和选中的 Reference 必须使用同一个 \(\alpha_{c,p}\) 做融合后再相减。
5. Raw Reference Bank 保留原始多层特征，不把参考库永久压缩成一个固定融合结果。
6. 训练、验证、测试和独立推理必须采用完全相同的残差定义。

这与现有分支的 \`prior_top1\` 路径一致。第一章不启用历史 \`dynamic_topk\` 精排主路径。

### 3.3 模块职责

| 模块 | 输入 | 输出 | 是否训练 |
|---|---|---|---|
| DINOv3 | 图像 | 多层 Patch 特征 | 否，冻结 |
| 类别先验 \(\theta_c\) | 类别索引 | 全局层权重 | 是 |
| Patch Gate | Query 多层特征、类别嵌入 | \(\delta_{c,p}\) | 是 |
| Raw Reference Bank | clean normal 特征 | 每类原始参考 Patch | 否，固定 |
| Fused Reference Cache | Raw Bank、类别先验 | NN 检索索引 | 否，按 epoch 刷新 |
| Feature Synthesis | 正常残差、类别尺度、空间 mask | 合成异常残差 | 否，无可学习参数 |
| Legacy Adapter | 残差特征、类别索引 | 适配后特征 | 是 |
| Legacy MLP | 适配后 Patch 特征 | Patch anomaly score | 是 |

## 4. Feature-level Anomaly Synthesis：第一章重点改进

### 4.1 为什么在残差空间合成

不建议直接在 RGB 图像上做 CutPaste，也不建议在原始 DINOv3 特征上直接加入噪声。

第一章的异常判别器实际接收的是参考残差。因此，合成异常应该作用于**已经完成类别参考匹配和 class_patch 融合的残差特征**：

\[
r\in\mathbb{R}^{B\times P\times C}
\]

这样合成扰动与模型真正使用的表示空间一致，同时避免额外运行 Backbone 或增加图像级合成管线。

合成操作放在残差计算之后、现有 localnet Adapter 之前：

\[
\text{DINOv3}
\rightarrow \text{class\_patch residual}
\rightarrow \text{feature synthesis}
\rightarrow \text{legacy Adapter}
\rightarrow \text{MLP}
\]

### 4.2 类别相关的扰动尺度

不同工业类别的正常残差分布并不相同。若所有类别使用同一个固定 Gaussian 标准差，可能导致某些类别合成异常过于容易，另一些类别的扰动又不足。

因此使用每个类别独立的固定尺度 \(\sigma_c\)。

**校准方法：基于 clean reference images 的 Leave-One-Image-Out（LOIO）残差分布。**

1. 为每个类别选取 clean reference images。
2. 对每张参考图像，将该图像的全部 Patch 从候选参考库中排除。
3. 使用其他参考图像为它计算 \`prior_top1\` 残差。
4. 汇总该类别的参考残差向量范数。
5. 使用第 75 百分位数进行鲁棒标定：

\[
\sigma_c=
\frac{
Q_{0.75}\left(\left\{\|r^{ref}_{c,p}\|_2\right\}_p\right)
}{\sqrt{C}}
\]

数值实现时增加极小正数下限，避免 \(\sigma_c=0\)。

**关键约束：**

- 必须排除同一参考图像的全部 Patch，而不仅仅排除同一个 Patch，防止近似自匹配使尺度严重偏小。
- Raw Reference Bank 需要保留 Patch 到原始图像的 ID 映射。
- LOIO 计算使用分块检索，避免一次性构造完整的 \(N\times N\) 距离矩阵。
- \(\sigma_c\) 在训练开始前校准一次，训练期间固定，不跟随每个 batch 或 epoch 实时变化。
- 校准数据只来自 clean train/reference，不读取测试集异常标签。
- 将每类 \(\sigma_c\) 写入 checkpoint，确保实验可复现。

### 4.3 合成公式

对正常残差 \(r_{c,p}\)，采样标准高斯噪声：

\[
\epsilon_{c,p}\sim\mathcal{N}(0,I_C)
\]

采样扰动系数：

\[
\gamma\sim U(0.5,1.5)
\]

构造 Patch mask \(M_{c,p}\in\{0,1\}\)，得到：

\[
\boxed{
r^{syn}_{c,p}
=
r_{c,p}
+
M_{c,p}\cdot\gamma\sigma_c\epsilon_{c,p}
}
\]

其中：

- \(M_{c,p}=1\)：该 Patch 被合成为异常。
- \(M_{c,p}=0\)：该 Patch 保持原始正常残差。
- \(\sigma_c\)：类别特定的固定扰动尺度。
- \(\gamma\)：每个合成样本随机采样的强度系数。

噪声采样、mask 和 \(\sigma_c\) 不参与梯度更新；但合成表达式中的原始残差 \(r\) 必须保留计算图，使 BCE 梯度仍然能够传回 class_patch 融合模块。

### 4.4 空间 mask 设计

只使用独立随机 Patch 噪声容易产生不符合工业缺陷空间结构的训练信号。第一版使用低成本的连通区域 mask：

| Mask 类型 | 目标形态 | 作用 |
|---|---|---|
| Blob | 不规则块状区域 | 模拟局部材料/表面异常 |
| Line | 细长条状区域 | 模拟划痕、裂纹等细长缺陷 |
| Spot | 小面积局部区域 | 模拟点状缺陷 |
| Irregular | 不规则连通区域 | 增加形状变化，避免只学习固定几何模式 |

建议初始异常区域面积比例：

\[
\rho=\frac{\sum_p M_p}{P}\in[0.005,0.10]
\]

对 \(28\times28\) 的 Patch 网格，即大约 4–78 个 Patch。具体 Patch 数根据实际 \(P\) 动态计算，不在代码中硬编码 784。

Irregular mask 第一版可以使用低分辨率随机场上采样后阈值化，并做简单连通区域筛选，不需要引入复杂图像生成网络或额外 Backbone。

**第一版不做：**

- RGB CutPaste、Perlin 图像合成。
- 额外的 GAN、扩散模型或生成式 Decoder。
- 多个复杂特征生成器叠加。
- 真实异常图像参与训练。

先验证“类别尺度 + 空间结构 + 残差扰动”是否足够有效，再考虑更复杂的异常合成算法。

### 4.5 一个训练样本如何形成监督

对一个 clean normal 输入，先得到正常残差 \(r\)，再生成一个合成残差 \(r^{syn}\)。

训练输入可以拼接为：

\[
X=[r;\ r^{syn}]
\]

标签为：

\[
Y=
\left[
\underbrace{\mathbf{0}}_{\text{原始正常 Patch}};
\underbrace{M}_{\text{合成样本 Patch mask}}
\right]
\]

因此：

- 原始 clean residual 的全部 Patch 标签为 0。
- 合成 residual 中 mask 内 Patch 标签为 1。
- 合成 residual 中 mask 外 Patch 标签仍为 0。

这样模型同时学习正常残差分布、局部异常响应和异常区域边界。

## 5. 判别器与损失函数

### 5.1 Legacy discriminator

使用当前 \`src/model/model.py\` 中 \`localnet\` 的原有 MLP 判别头：

\[
C\rightarrow1024\rightarrow128\rightarrow1
\]

中间使用 GELU，输出使用 Sigmoid。

明确关闭：

- \`use_moe_discriminator\`
- MoE Gate 和专家路由
- Gate auxiliary loss
- Hard Class Gate
- 新增的专家网络

现有类别条件 Adapter 继续走 legacy 路径；不额外启用 \`ClassPatchFusionAdapter.enable_adapter=True\` 的 V2-C Beta/FiLM 分支，避免同时叠加两套特征适配机制。

### 5.2 Balanced BCE

第一章仅需要正常/合成异常两组监督：

\[
\mathcal{L}_{neg}
=
\operatorname{BCE}(s_{neg},0)
\]

\[
\mathcal{L}_{pos}
=
\operatorname{BCE}(s_{pos},1)
\]

\[
\boxed{
\mathcal{L}_{clean}
=
\mathcal{L}_{neg}+\mathcal{L}_{pos}
}
\]

每组分别取均值，避免正常 Patch 数量远大于异常 Patch 数量时，正样本梯度被淹没。

优先复用 \`utils/loss.py\` 的 \`compute_balanced_bce_loss\`。若某个 batch 意外缺少正样本或负样本，对空组安全跳过；正常情况下，feature synthesis 应保证两组都存在。

第一章不计算：

\[
\mathcal{L}_{OTO}
\]

也不引入：

- IRMB 伪标签损失。
- KL/互平滑损失。
- noisy label BCE。
- MoE Gate auxiliary loss。
- 第二章专用的 MAD / 不确定区间处理。

## 6. 训练流程与参数更新

### 6.1 初始化阶段

1. 加载 clean-only 的多类别训练集。
2. 冻结 DINOv3。
3. 按类别选取 clean reference images。
4. 提取参考图像的原始多层 DINOv3 特征。
5. 建立 Raw Reference Bank，保留图像 ID、Patch ID 和层对齐关系。
6. 初始化 \`ClassPatchFusionAdapter\`，Patch Gate 的零初始化保证初始时不会突然破坏类别先验。
7. 使用 LOIO 残差分布计算每类 \(\sigma_c\)。
8. 初始化 \`FusedReferenceCache\` 和 \`ClassPatchFusionRuntime\`，主匹配模式设为 \`prior_top1\`。
9. 初始化 legacy Adapter 和 MLP discriminator。

### 6.2 每个 Epoch

1. 按当前类别先验刷新 Fused Reference Cache。
2. 对 clean normal batch 提取 DINOv3 多层特征。
3. 使用 \`prior_top1\` 为每个 Query Patch 选择唯一参考 Patch。
4. 通过 Patch Gate 计算 \(\alpha_{c,p}\)，生成最终残差 \(r\)。
5. 使用固定的 \(\sigma_c\)、随机 \(\gamma\) 和空间 mask 生成 \(r^{syn}\)。
6. 拼接正常/合成异常特征和标签。
7. 通过 legacy Adapter 与 MLP 得到 Patch anomaly scores。
8. 计算 Balanced BCE。
9. 反向传播并更新 Fusion、Adapter、MLP 参数；DINOv3 和 Raw Reference Bank 不更新。
10. 记录每类融合权重、Patch 权重统计、扰动尺度和损失。
11. 验证前将 Fusion 切换到 eval，并按当前参数刷新缓存，再进行评估。

### 6.3 训练/验证/推理一致性

必须统一以下计算路径：

\[
\text{Raw DINO features}
\rightarrow
\text{class\_patch prior\_top1 residual}
\rightarrow
\text{legacy Adapter}
\rightarrow
\text{legacy MLP}
\]

Feature synthesis **只在训练阶段开启**。验证、测试和独立推理都不合成异常，直接对真实输入计算残差和异常分数。

## 7. Backbone 与第二章权重承接：必须明确的兼容性问题

当前仓库存在一个不能忽略的配置差异：

| 入口 | 当前默认 Backbone | 默认候选层 |
|---|---|---|
| \`self_train_ad_multiclass_dinov3_clean.py\` | \`dinov3_vitb16\` | 旧的 per-class layer table |
| \`shell/run_self_train_ad_multiclass_class_patch.sh\` | \`dinov3_vitl16\` | \`[3,6,9,12,15,18,21,24]\` |
| \`get_fusion_candidate_layers("dinov3_vitb16")\` | \`dinov3_vitb16\` | \`[2,4,6,8,10,12]\` |

这意味着当前 clean 入口与 noisy 入口默认并不能直接进行完整 checkpoint 权重迁移。

**必须统一 Backbone、候选层顺序、特征维度、输入预处理和类别顺序。** 否则 Fusion 的 \(\theta\)、Patch Gate、Adapter 和参考库都可能发生维度或语义不匹配。

本方案建议：

- 若将轻量化和降低特征提取开销放在第一优先级，则以 DINOv3-B/16、候选层 \([2,4,6,8,10,12]\) 为统一目标，并同步规划第二章迁移到相同 Backbone。
- 若第二章必须保留现有 ViT-L/16 实验配置，则第一章也必须使用 ViT-L/16 和相同的 8 个候选层；不能一章 B/16、一章 L/16 后仍宣称可以直接加载完整权重。

**建议将 Backbone 统一作为代码实现前的硬性配置，不要通过隐式维度适配来掩盖差异。**

### 7.1 第一章到第二章的权重迁移

当前 \`utils/train_utils.py\` 的 \`load_weights_only_checkpoint\` 对 \`localnet\` 和 Fusion 使用 strict load。第一章使用 MLP 判别器，而第二章当前 noisy 配置可能使用 MoE，因此不能直接把第一章整个 \`net\` strict-load 到第二章 MoE 模型。

建议新增显式的迁移策略：

- \`strict\`：架构完全相同时，严格加载所有模型权重。
- \`shared_modules\`：只加载经过校验的共享模块，包括 Fusion state 和兼容的 legacy Adapter；当目标判别器是 MoE 时，跳过第一章的 MLP discriminator。
- 迁移前校验 Backbone、候选层、特征维度、类别顺序、预处理、Fusion schema 和 Adapter 配置。
- 不匹配时明确报错，不能静默忽略关键参数。
- 第二章的 noisy reference bank 默认重新从其训练数据构建，不盲目复用第一章 checkpoint 中的参考库。
- 第二章的 OTO、伪标签和 Memory Bank 状态从新实验开始，不继承第一章的优化器或训练状态。

这样第一章可以为第二章提供可复用的特征融合和适配初始化，同时不强行迁移不兼容的 MLP 判别器。

## 8. 需要修改或新增的文件清单

本节只描述未来的代码修改范围，不在本次提交中实施。

| 文件 | 计划修改 |
|---|---|
| \`self_train_ad_multiclass_dinov3_clean.py\` | 接入多层特征提取、class_patch runtime、raw reference bank、feature synthesis、Balanced BCE 和 Fusion checkpoint；删除“全 Patch 标签恒为 0”的唯一训练路径 |
| \`src/train/feature_anomaly_synthesis.py\` | **新增**：LOIO 类别尺度校准、Blob/Line/Spot/Irregular mask、残差特征扰动和标签生成 |
| \`utils/train_utils.py\` | 复用现有 raw bank/cache/prior_top1；必要时扩展参考 Patch 的 image ID 元数据和 shared-module 权重迁移 |
| \`utils/evaluate.py\` | 保留现有 AUROC 计算，补充 I-AP、P-AP、F1-max、AUPRO 的统一评估和结果返回 |
| \`src/inference/infer_multiclass_residual.py\` | 复用 class_patch checkpoint；校验 clean checkpoint 的 Fusion schema；推理阶段关闭 feature synthesis |
| \`shell/run_self_train_ad_multiclass_clean_class_patch.sh\` | **新增**：第一章 clean 训练启动脚本，显式配置 Backbone、候选层、Fusion、合成参数和 legacy MLP |
| \`tests/test_feature_anomaly_synthesis.py\` | **新增**：尺度标定、mask、标签、梯度和边界测试 |
| \`tests/test_clean_class_patch_training.py\` | **新增**：clean-only 数据流、prior_top1 一致性、Balanced BCE 和 checkpoint 测试 |

不建议直接修改第二章 noisy 训练主流程来适配第一章。第二章只需要在确认第一章架构和 Backbone 后，增加受控的 shared-module 初始化入口。

## 9. 实验设计：按最小增量验证，不一次堆叠所有模块

### 9.1 主实验顺序

| 实验 | Fusion | Feature synthesis | 目的 |
|---|---|---|---|
| E0 | 固定均值/现有残差 | 全局尺度 Gaussian | 建立有效的 clean 合成异常训练基线 |
| E1 | learned_class | 同 E0 | 验证类别级层融合是否有效 |
| E2 | class_patch | 同 E0 | 验证 Patch 动态融合是否带来收益 |
| E3 | class_patch | 类别专属 \(\sigma_c\) | 验证类别残差统计标定的作用 |
| E4 | class_patch | 类别专属 \(\sigma_c\) + 结构化 mask | 完整模型，验证空间结构合成的贡献 |

说明：

- E0 不能简单等同于当前“所有 Patch 标签都是 0”的旧脚本；那只是代码基线，不是有效的正负监督训练基线。
- E0–E4 尽可能固定 Backbone、参考图像、参考库规模、预处理、训练轮数、优化器和评估后处理。
- 先完成 E0，再逐步增加 Fusion 和合成策略，避免无法归因性能变化。

### 9.2 必做消融

1. 全局统一 \(\sigma\) vs 类别独立 \(\sigma_c\)。
2. 无空间 mask 的独立 Patch 扰动 vs 结构化 mask。
3. 固定 \(\gamma\) vs \(\gamma\sim U(0.5,1.5)\)。
4. 不同异常区域面积区间。
5. learned_class vs class_patch。
6. Patch Gate 关闭 vs 开启。
7. reference residual vs 不使用 residual（用于确认残差路径的贡献）。

第一轮不建议同时搜索大量超参数。先固定 \(\gamma\in[0.5,1.5]\)、面积比例 \([0.005,0.10]\) 和 \(Q_{0.75}\) 标定，再根据验证结果决定是否扩大搜索范围。

### 9.3 评估指标

P-AP 是第一章重点关注指标，但不能只报告它。

| 指标 | 作用 | 优先级 |
|---|---|---|
| P-AP | 像素级 Precision-Recall 表现，重点指标 | 最高 |
| P-AUROC | 像素级排序能力 | 最高 |
| I-AUROC | 图像级异常检测能力 | 最高 |
| I-AP | 图像级 PR 表现 | 高 |
| F1-max | 最佳阈值下的综合表现 | 高 |
| AUPRO | 区域级定位能力 | 高 |
| Params / FLOPs / Latency | 模型复杂度与部署代价 | 最高 |

P-AP 必须在相同的 resize、Gaussian smoothing 和 score map 后处理条件下比较。图像级分数的 top-k ratio 也要固定，不能因为某个模型单独调整后处理来获得更高指标。

### 9.4 复杂度报告

分别报告：

1. 冻结 Backbone 参数量和特征提取耗时。
2. 新增可训练 Fusion 参数量。
3. Adapter 参数量。
4. Legacy MLP 参数量。
5. 总可训练参数量。
6. 单张图像推理延迟、显存占用和参考库大小。

不要只报告不包含 Backbone 的参数量而称模型整体轻量。可以同时给出“冻结 Backbone + 可训练检测头”的分项统计，让复杂度比较更透明。

### 9.5 数据与防泄漏约束

- 训练阶段只读取 clean train/good。
- reference bank 和 \(\sigma_c\) 校准只使用 clean training images。
- 测试集异常图像及像素 GT 仅用于最终评估。
- 不允许根据测试集 P-AP 反向选择 \(\gamma\)、mask 面积、\(\sigma_c\) 分位数或融合权重。
- 建议固定至少 3 个随机种子，报告各类结果和宏平均值。

## 10. 日志与可解释性

每个 Epoch 至少记录：

- 每个类别的 \(\alpha^G_c\)。
- 每个类别 Patch 权重均值、标准差和熵。
- Patch 权重相对类别先验的平均 L1 偏移。
- 每个类别的 \(\lambda_c\)。
- 每个类别固定的 \(\sigma_c\)。
- 各 mask 类型采样比例、实际 mask 面积比例。
- \(\gamma\) 的均值、标准差和范围。
- Normal BCE、Synthetic BCE、总 BCE。
- P-AP、P-AUROC、I-AUROC、I-AP、AUPRO。
- Cache 刷新耗时、训练耗时、推理耗时。

重点关注两个现象：

1. 如果 Patch 权重熵长期接近 \(\ln L\)，且 Patch 间权重标准差很小，说明 Patch Gate 可能没有真正学到局部差异。
2. 如果合成异常训练指标持续变好，但真实异常 P-AP 不提升，说明模型可能只学会识别人工 Gaussian 扰动，需要调整异常合成分布，而不是继续扩大判别器。

## 11. 测试与验收标准

### 11.1 功能测试

- [ ] clean Dataset 只读取 train/good，不读取真实异常训练样本。
- [ ] Raw Reference Bank 的所有候选层来自同一图像、同一 Patch，层顺序一致。
- [ ] LOIO 校准时排除同一参考图像的全部 Patch。
- [ ] 每个类别只产生一个固定 \(\sigma_c\)，且写入 checkpoint。
- [ ] mask 形状为 \([B,P]\)，面积符合配置区间。
- [ ] mask 外的合成残差保持不变。
- [ ] mask 内的合成残差按类别尺度和 \(\gamma\) 扰动。
- [ ] Feature synthesis 不切断 residual 到 Fusion 的梯度。
- [ ] \`prior_top1\` 的最近邻索引不依赖 Patch Gate 输出。
- [ ] Query 和 Reference 使用相同 Patch 权重计算最终残差。
- [ ] 训练损失只有 Balanced BCE，不调用 OTO。
- [ ] 不启用 MoE、Gate auxiliary loss 或 IRMB。
- [ ] 验证/测试/推理不执行异常合成。

### 11.2 兼容性测试

- [ ] checkpoint 保存 Fusion state、fusion_config、Raw Reference Bank 元数据和类别尺度。
- [ ] 独立推理能够恢复正确的 Backbone、候选层、类别顺序和预处理。
- [ ] 训练与独立推理对同一输入计算的残差一致。
- [ ] 第二章 shared-module 初始化能识别 MLP/MoE 判别器不匹配，并只迁移共享模块。
- [ ] Backbone 或候选层不匹配时明确报错，不允许静默加载。

### 11.3 研究验收

- [ ] E0–E4 能够独立复现。
- [ ] P-AP、P-AUROC、I-AUROC 等指标统一计算。
- [ ] 复杂度统计包含 Backbone 与检测头的分项数据。
- [ ] 至少完成 3 个随机种子的核心对比。
- [ ] 不以尚未完成的实验结果提前宣称超过 SOTA。

## 12. 风险与边界

| 风险 | 可能表现 | 处理方式 |
|---|---|---|
| Gaussian 特征扰动过于简单 | 合成异常容易识别，但真实异常 P-AP 不提升 | 比较结构化 mask、类别尺度和不同区域面积 |
| 类别尺度估计偏差 | 某些类别异常过强/过弱 | LOIO + 75% 分位数 + 每类统计 |
| Patch Gate 退化为类别先验 | Patch 权重变化很小 | 监控权重方差、熵、L1 偏移和 \(\lambda_c\) |
| Cache 与当前参数不一致 | NN 索引陈旧，训练和评估残差不一致 | Epoch 刷新；评估前强制刷新 |
| Backbone 不一致 | 第一章 checkpoint 无法初始化第二章 | 统一 Backbone、候选层和预处理 |
| MLP 与 MoE 结构不一致 | strict load 失败 | 使用显式 shared_modules 迁移策略 |

## 13. 明确不做的事情

第一章不做以下改动：

- 不修改 DINOv3 Backbone，不做端到端微调。
- 不引入新的 MoE 专家结构。
- 不引入复杂 Decoder 或大型生成网络。
- 不做 RGB 图像级异常合成。
- 不使用真实异常训练图像。
- 不引入 OTO、IRMB、MAD、不确定区间伪标签或 noisy self-training。
- 不改变第二章已有 noisy-AD 主训练逻辑。
- 不把 Patch 动态权重用于最近邻索引选择。
- 不在本次方案分支中实现代码。

---

## 14. 最终建议

第一章应围绕两个核心贡献组织：

**贡献一：类别—Patch 层次化的多层残差融合。**  
类别先验负责稳定地选择唯一参考 Patch，Patch Gate 只负责调整最终残差的层融合权重，在不改变参考匹配定义的前提下提高局部表征的适应能力。

**贡献二：基于类别残差统计的轻量级特征异常合成。**  
利用 clean reference 的 LOIO 残差估计每类扰动尺度，在残差空间内通过 Gaussian 扰动与结构化 Patch mask 生成异常监督，避免引入额外生成网络。

两者共同服务于 P-AP 和像素级定位能力。Legacy MLP 保持判别器简单，第一章不引入 OTO 或 MoE。第二章在相同的 Backbone、Fusion、参考残差和 Adapter 协议上，进一步引入 noisy-AD 自训练机制。

**建议实现顺序：先打通 E0 的特征合成 + Balanced BCE，再接入 class_patch，最后做 checkpoint 到第二章的共享模块迁移。** 不建议一次性同时重构训练入口、Fusion、判别器和第二章自训练逻辑。
