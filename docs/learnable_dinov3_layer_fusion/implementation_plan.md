# FUNAD 可学习 DINOv3 层融合：实施与测试路线图

> 文档状态：**计划，不包含代码实现**。核心公式和范围以 [requirements_and_design.md](./requirements_and_design.md) 为准；引用见 [references.md](./references.md)。

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

4. **融合后最近邻与残差**
   - 查询和参考都用同一个 fusion 模块得到 [M,C] 与 [N_c,C]。
   - 在 fused feature 上 NN：j*=argmin ||Z_q−Z_m||，再取选中 raw_m[j*]，重新有梯度地 fused 出参考 z_sel，返回 residual=z_q−z_sel。
   - 模块要支持按类别批量、查询分块、参考分块，避免 torch.cdist 的 [query_patches, reference_patches] 全距离矩阵爆显存。
   - Test：人工构造已知最近邻的例子；对比 brute-force NN；统计融合模块学习后的梯度；改变 theta 后参考融合向量必须改变。

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

## 第二阶段：训练策略和算力优化（P1）

- 初次训练：推荐保留现有 train loss、类别 MoE、PCA pseudo-label 配置，alpha 从近似均匀初始化；可先固定伪标签让融合层学到非零梯度，再打开全量自训练。
- 原始参考库 CPU float16 + 分块融合，匹配计算 FP32；单纯改成 raw 多层 reference 可能极大增加内存，不应无估算直接 all-layer 常驻 GPU。
- **数据一致性优先于速度**：第一版训练每个迭代使用当前 theta 的融合参考；后续做索引 refresh interval，明确失效窗口。
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

| 决策 | 当前建议 | 原因 |
| --- | --- | --- |
| 主融合顺序 | **learnable fusion → NN → fused residual** | 用户明确指定 |
| raw reference 存储 | [N_c,L,C] | 训练 theta 时无需重提 DINOv3 |
| 候选层集 | 统一可配置，覆盖多个深度 | 避免手动类别层组合的研究偏差 |
| 检索空间 | query/ref 使用同一 theta | 避免表征不一致 |
| 最近邻 | 每个 fused Patch 只选一个 reference Patch | 保持旧 NN residual 语义 |
| 梯度 | 选中参考后重融合，不能透过 argmin | 正确优化融合器 |
| P0 门控 | 类别特定 softmax 层权重 | 参数少、可解释、易消融 |
| 硬类 MoE | 保持不变 | 控制研究变量 |
| 伪标签库 | 随 fusion / adapter 更新 | 保证迭代训练不使用过期表示 |
| 旧模型 | fixed_mean 默认兼容 | 保证现有 FUNAD 可复现 |

## DoD：交付验收

- [ ] fixed_mean 回归与 codex/main 误差在允许容差内；
- [ ] learned_class 融合层确有梯度、实际更新；
- [ ] 原始参考库保持逐层结构，同 Patch 不错位；
- [ ] 最近邻严格在融合特征空间计算；
- [ ] 训练/伪标签/评估/独立推理路径全部打通；
- [ ] checkpoint/metadata 无静默错配；
- [ ] 记录至少一轮可复现的对照结果和完整消融方案；
- [ ] 合理引用并区分 R1–R7 的直接相关性和局限。
