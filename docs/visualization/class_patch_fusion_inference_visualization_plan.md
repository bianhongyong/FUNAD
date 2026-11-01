# FUNAD 训练权重推理可视化——六张核心图脚本编写方案

> 目标仓库：[bianhongyong/FUNAD](https://github.com/bianhongyong/FUNAD/tree/feature/learnable-dinov3-layer-fusion-design)  
> 基础分支：`feature/learnable-dinov3-layer-fusion-design`  
> 范围：**只读取训练完成的 checkpoint 进行推理、统计与绘图；不训练、不更新参数、不重建训练参考库。**  
> 推荐提交路径：`docs/visualization/class_patch_fusion_inference_visualization_plan.md`（已写入远程 GitHub）。

## 1. 现有代码核对与技术约束

| 代码位置 | 已有能力 | 新脚本应如何复用 |
|---|---|---|
| `src/inference/infer_multiclass_residual.py` | 读取 `checkpoint['net']`、`checkpoint['fusion_state']`、`checkpoint['fusion_config']`、`checkpoint['raw_reference_manifest']`，重建 `localnet` / `ClassPatchFusionAdapter` / `FusedReferenceCache`；从测试图像推理异常得分 | **以该脚本为推理基准**；抽取/复用其加载、严格配置检查、残差匹配与 score map 后处理，不另写独立算法 |
| `src/model/class_patch_fusion_adapter.py` | `get_layer_weights()`、`global_weights()`、`patch_weights()`、`get_patch_lambda()`；类别固定最后一层先验 `no_layer_fusion_classes` | 导出全局先验、逐 Patch 权重，严禁重新初始化后拿随机权重绘图 |
| `dataset/feature_extract.py` | `extract_dinov3_multilayer_feature_batch()` 输出 `[B,P,L,C]`；层顺序为 checkpoint 所记载的候选层 | 提取原始多层 Query 特征，不做额外平均融合 |
| `dataset/dataset_extract.py` | `MyDataset(..., is_train=False)` 返回 `(image, label, mask)`；`test_set.x` 保存测试图像路径 | 用于正常/异常案例与 GT mask；保留图像路径以保证可追溯 |
| `utils/train_utils.py` | `compute_class_patch_prior_top1_residual_batch()`、`compute_class_patch_fused_residual_batch()` | 根据 checkpoint 中 `match_mode` 选择对应函数，确保检测结果与独立推理相符 |
| `shell/run_infer_multiclass_residual.sh` | 已有推理超参数示例，包括 `--moe_use_cls_token`、图像尺寸等 | 新 shell 脚本参数与现有推理对齐，显式提供 checkpoint / 数据集 / 输出目录 |

**关键差异**：schema v3 `class_patch` 的 `match_mode=prior_top1` 代表全局先验决定参考 Patch，动态 alpha **只影响残差**；旧 schema v2 的缺省模式则应按现有 `resolve_fusion_match_mode()` 解释为 `dynamic_topk`。不得把当前模型描述成始终采用“动态权重精排 Top-K”。

**类别固定先验**：在 `no_layer_fusion_classes` 中的类别（如某些配置下的 `screw`），`get_layer_weights()` 为最后候选层 one-hot 权重，但 `patch_weights()` 依旧允许动态偏离。图中用 `pinned prior` 标注，不应认为它是自动学习到的全局先验。

## 2. 文件结构建议

```text
src/inference/
  visualize_class_patch_fusion.py          # 独立推理入口：参数/加载/前向/数据导出
utils/
  fusion_viz_stats.py                      # 统计累计器与验证（NumPy/Torch）
  fusion_viz_plots.py                      # 六张图的 Matplotlib 绘图函数
shell/
  run_visualize_class_patch_fusion.sh     # 一键推理 + 导出
tests/
  test_fusion_viz_stats.py                 # 数值/shape/边界测试
  test_fusion_viz_inference.py             # 小规模集成一致性测试
```

注：建议优先**复用**现有推理文件内可直接导入的辅助函数，例如 `detect_learned_fusion`、`resolve_fusion_match_mode`、`validate_fusion_inference_config`、`validate_raw_bank_metadata`、`infer_model_params_from_checkpoint` 等。加载逻辑当前主要放在其 `main()` 内；如需重构，应仅抽出公共 helper 并用回归测试保证原推理输出不变。也可以先在新脚本中封装一个薄加载层，避免大范围改动原脚本。

## 3. 权重推理主流程（冻结模式）

```text
输入：可信任的训练 checkpoint + 同目录 raw-reference-bank sidecar + 测试集
  |
  +-- 读取 checkpoint / fusion_config；验证 schema、类别顺序、层顺序、预处理、backbone
  +-- 构建 frozen DINOv3 + ClassPatchFusionAdapter + localnet
  +-- strict=True 加载 fusion_state / net；eval()；torch.inference_mode()
  +-- 从 raw_reference_manifest 加载参考库，验证并 rebuild fused reference cache（不训练）
  |
  +-- 不依赖测试图像：导出所有类别 alpha_global、lambda、候选层序号、pinned 标记
  |
  +-- 按测试集类别、图像批次迭代
       +-- 图像 -> DINOv3 多层原始特征 [B,P,L,C]；保留相同预处理的 RGB/GT
       +-- fusion.patch_weights(raw, class_idx) -> alpha_patch [B,P,L]
       +-- fusion.global_weights(class_idx) -> prior [B,L]
       +-- 统计每个 Patch 的熵、L1、主导层，按类别流式汇总
       +-- checkpoint match_mode 对应的匹配函数 -> residual [B,P,C]
       +-- localnet(residual, cls_token, patch_class_idx, class_idx) -> score [B,P]
       +-- 按现有 infer_multiclass_residual.py 的 resize + Gaussian 后处理 -> score map
       +-- 选定案例保存 image/GT/alpha/score 等最小必要数组
  |
  +-- 导出 CSV / NPZ / JSON + 6 种图（PNG 和 PDF 或 SVG）
```

必须始终满足：

1. 不创建 optimizer / scheduler，不调用 `train()`、`backward()`、`optimizer.step()`；不写回 checkpoint。
2. `torch.inference_mode()` 包裹前向，`feature_extractor.eval()`、`fusion.eval()`、`localnet.eval()`；若使用 V2-C `class_residual`，按原推理入口将 fusion 挂到 `localnet`，且只应用一次 adapter。
3. `torch.load(..., weights_only=False)` 仅允许用户信任的本地 checkpoint；不加载不可信来源的 pickle。
4. `candidate_layers`、类别索引、`use_cls_token`、`feature_model`、`image_size`、`crop_size`、`moe_use_cls_token`、`gate_use_context` 等必须与 checkpoint / 原推理一致；不可写死层数或 Patch 边长。
5. `P = H_p * W_p`；若当前实现只支持平方网格，检查 `sqrt(P)` 是否为整数，并明确报错；不得无提示 reshape。
6. 对分数图仍采用现有推理中的 `cv2.resize(..., INTER_LINEAR)` 及 `gaussian_filter(..., sigma=args.gaussian_sigma)`；禁止每张图单独 min-max 归一化后再用于跨图比较，色条范围应可配置且记录。

## 4. 六张核心图：定义、数据、绘图规则

### Fig.1 类别 × DINOv3 候选层全局先验热力图

- **数据**：`A_global = fusion.get_layer_weights().cpu().numpy()`，形状 `[C_class,L]`。
- **横轴**：`fusion_config['candidate_layers']` 所记录的真实 1-based DINOv3 Block 编号（如 3、6、9、12…），不要笼统写“第 1～8 层”。
- **纵轴**：checkpoint `class_names` 顺序；颜色固定 `[0,1]`；可在格子上标 2 位小数。
- **特殊标注**：在被 pinned 的类别名后标 `*`，图注说明 `*` 为固定最后层先验，不是学到的结果。
- **文件**：`fig1_global_prior_heatmap.png/pdf`，`tables/global_prior.csv`。
- **验收**：每个类别的权重之和误差 ≤ 1e-5，所有权重有限且位于 `[0,1]`。

### Fig.2 各类别 Patch 权重均值 ± 标准差

- **数据**：测试集推理得到的 `alpha_patch [B,P,L]`；按类别累计 `sum(alpha)`、`sum(alpha²)` 和 Patch 个数，不必缓存全部图像。
- **计算**：`mean_l = Σ_p alpha[p,l] / N`；`std_l = sqrt(max(Σ_p alpha[p,l]^2 / N - mean_l^2, 0))`。
- **绘图**：类别小 multiples（默认选 `leather,capsule,transistor`；可切换所有类别），横轴真实 Block 编号，纵轴 0～1，误差棒表示 Patch 间标准差，不是置信区间。
- **辅助**：可叠加相同类别的全局先验虚线以示区别。
- **文件**：`fig2_patch_mean_std.png/pdf`，`tables/patch_statistics.csv`。
- **验收**：不同 batch_size 的统计结果允许微小数值误差，但不应随 batch_size 系统性变化。

### Fig.3 单张测试图像 Patch 主导层空间图

- **输入**：指定案例的一张图及 `[P,L]` 权重。
- **计算**：`dominant[p] = argmax(alpha_patch[p,:])`；从候选层索引映射为真实 DINOv3 Block 编号。
- **绘图**：原 RGB（CenterCrop 对齐版本） + 主导层离散色彩图；使用**离散**色条且所有案例保持相同层到颜色的映射；可选网格版 + nearest-neighbor 放大版；不要用双线性插值离散层标签。
- **诊断**：标题打印该图最高占比层及其 Patch 占比；若 100% 主导层相同，正常展示为单色，不要人为调色制造“动态性”。
- **文件**：`fig3_dominant_layer_<class>_<sample_id>.png/pdf`；保存对应 `alpha_patch.npy` 或 `npz`。

### Fig.4 单张测试图像 Patch 熵热图

- **计算**：`H[p] = -Σ_l alpha[p,l] * log(max(alpha[p,l], 1e-12))`。
- **绘图**：原图 + 熵热图；跨类别、跨样本色条固定 `[0, ln(L)]`，而非每张图独立归一化；可同时输出 `H/ln(L)` 作为可跨不同 L 对比的归一化熵。
- **解释**：低熵表示融合权重集中；高熵表示分布分散，**不能直接等同模型置信度/异常置信度**。
- **文件**：`fig4_patch_entropy_<class>_<sample_id>.png/pdf`。

### Fig.5 类别级 Patch-to-Prior L1 距离统计

- **计算**：`D[p] = Σ_l |alpha_patch[p,l] - alpha_global[class,l]|`，再对每类所有 Patch 求平均。
- **绘图**：15 类柱状图，纵轴固定 `[0,2]`；可配置排序或原类别顺序。数值输出到 CSV；可加类别的 Patch weight Std 作为辅助信息，但不和 L1 混成同一含义。
- **重要解释限定**：**L1 高不等于“不同 Patch 之间差异大”**——如果所有 Patch 都偏向相同最后一层，同时类别先验均匀，L1 也可以很高。必须与 Fig.2 的 std、Fig.3 空间图联合解释。
- **文件**：`fig5_l1_to_prior_by_class.png/pdf`，`tables/patch_statistics.csv`。

### Fig.6 异常检测与 Patch Fusion 联动案例图

- **布局**：同一测试样本横向排列 `RGB | GT mask | anomaly score map | 某个候选层 alpha map | patch-to-prior L1 map`（建议附一个 alpha-叠加原图作为第六面板，仍属于 Fig.6 一张组合图）。
- **alpha 层选择**：参数 `--focus_layer` 使用真实 Block 编号；默认自动选择**预先指定的规则**（如该类别 Patch 平均 alpha 最大的层），并在图注写明。不要根据 GT 区域挑最漂亮的层而不披露。
- **对齐**：所有空间图由相同的 Patch grid 与相同 CenterCrop 得到；连续权重图允许 nearest/linear 显示，但记录插值方式；GT 使用 nearest；最终 anomaly score 必须保持原推理定义和后处理。
- **比较**：至少展示同类 1 正常 + 1 异常；异常样本注明 GT 与 score，不把 alpha 的高低解释成异常因果证据。
- **文件**：`fig6_fusion_detection_<class>_<sample_id>.png/pdf`，`cases/<class>/<id>.npz`。

## 5. 建议数据接口

全局统计保存为：

```text
outputs/
  metadata.json                          # checkpoint、schema、match_mode、candidate_layers、类别顺序、预处理、采样策略
  tables/
    global_prior.csv                     # class,block3,...,block24,pinned_prior,lambda,beta
    patch_statistics.csv                 # class,n_images,n_patches,mean_layer_*,std_layer_*,entropy_mean,l1_mean,lambda
  cases/
    leather/good_<id>.npz                # image_rgb, gt_mask, alpha_patch, alpha_global, score_patch, score_map, src_path
    leather/anomaly_<id>.npz
    ...
  figures/
    fig1_global_prior_heatmap.png
    fig2_patch_mean_std.png
    fig3_dominant_layer_*.png
    fig4_patch_entropy_*.png
    fig5_l1_to_prior_by_class.png
    fig6_fusion_detection_*.png
```

可选导出 `figures/*.pdf`（矢量图适合论文；带 RGB 的 panel 会包含栅格图片）。按类别只累计轻量统计；只为选中的 case 保存逐 Patch `alpha_patch`、原图及分数，避免对全量测试集保存 `[N,P,L,C]` 原始特征。

推荐统计器：

```python
# 伪代码：不应替代仓库原推理中的严格加载和匹配
@torch.inference_mode()
def evaluate_batch(images, class_idx_batch, context):
    raw, cls = extract_dinov3_multilayer_feature_batch(
        images, context.feature_extractor, context.candidate_layers,
        use_cls_token=context.use_cls_token, return_cls_token=True,
    )
    alpha = context.fusion.patch_weights(raw, class_idx_batch)  # [B,P,L]
    prior = context.fusion.global_weights(class_idx_batch)      # [B,L]
    entropy = -(alpha * alpha.clamp_min(1e-12).log()).sum(-1)    # [B,P]
    l1 = (alpha - prior[:, None, :]).abs().sum(-1)                # [B,P]
    dominant = alpha.argmax(-1)                                  # [B,P]

    # 必须与已存在的 infer_multiclass_residual.py 分支一致
    if context.match_mode == 'prior_top1':
        residual = common_utils.compute_class_patch_prior_top1_residual_batch(
            raw, class_idx_batch, context.cache, context.fusion,
            query_chunk_size=context.query_chunk,
        )
    elif context.match_mode == 'dynamic_topk':
        residual = common_utils.compute_class_patch_fused_residual_batch(
            raw, class_idx_batch, context.cache, context.fusion,
            topk=context.topk, query_chunk_size=context.query_chunk,
            candidate_chunk_size=context.candidate_chunk,
        )
    else:
        raise ValueError(f'Unsupported match_mode: {context.match_mode}')

    # 以下继续严格复用原推理的 localnet forward、MoE 路由、
    # score reshape、resize、Gaussian smoothing；不要省略 cls_token。
    return alpha, prior, entropy, l1, dominant, cls, residual
```

**注意**：`lambda` 是类别级可学习幅度，不是每个 Patch 的独立参数。`entropy` 是“逐 Patch 求熵后平均”，不是“平均 alpha 求熵”。`l1_mean` 是“逐 Patch 到同一类别先验的 L1 后平均”，不是仅 `||mean(alpha)-prior||1`。

## 6. CLI 与一键运行建议

新脚本建议支持：

```bash
python src/inference/visualize_class_patch_fusion.py \
  --checkpoint_path /path/to/trained_localnet.pt \
  --data_path /path/to/MVTec \
  --dataset mvtec \
  --output_dir ./outputs/fusion_viz \
  --batch_size 8 \
  --feature_model dinov3_vitl16 \
  --image_size 512 \
  --crop_size 448 \
  --use_cls_token False \
  --moe_use_cls_token True \
  --case_class_names leather capsule transistor \
  --case_types good anomaly \
  --cases_per_class 2 \
  --focus_layer 24 \
  --gaussian_sigma 4.0 \
  --export_formats png pdf \
  --seed 0
```

此命令仅作接口设计示例：**具体参数应以实际 checkpoint 为准**。推荐将统计范围与案例范围分开：`--stat_class_names` 默认全部类别，`--case_class_names` 只选择案例类别（上例为 leather/capsule/transistor）；因此 Fig.1、Fig.2、Fig.5 仍覆盖全部测试类别，而 Fig.3、Fig.4、Fig.6 仅为指定类别生成案例。若显式指定 `--stat_class_names` 为子集，应在图中标注统计覆盖范围。

另外提供 `--case_selection first|seeded_random|explicit`、`--case_paths ...`；首选固定随机种子或显式样本路径，避免挑图的主观性。不存在目标标签时记录 warning，并跳过该 case，不得把正常图当异常图。

## 7. 实现优先级

- **P0：数据和复现**：实现 checkpoint / sidecar / 模型严格加载，类别顺序和预处理检查；一次完整 test forward；保存全局先验 CSV；保存一个 case NPZ。
- **P1：六张图**：封装流式统计器；先绘 Fig.1、Fig.2、Fig.5，再绘 Fig.3、Fig.4、Fig.6；保证全部图色条、分辨率、字体和命名一致。
- **P2：论文质量**：同一类别 good vs anomaly 成对排版、论文图注、导出 PDF/SVG、可复现 manifest、报告统计样本数。
- **可选后续**：另做 `delta` 的 class mean / patch std、Gate 输出固定偏置检验、正常区域与异常区域的 alpha 差异，以及 `dynamic_topk` 专属 NN switch 分析；这不是核心六图的必需功能。

## 8. 测试与验收标准

1. **形状与数值**：`alpha [B,P,L]`、`prior [B,L]`；权重在 `[0,1]`，各 Patch 权重和为 1；`entropy∈[0,ln(L)]`；`l1∈[0,2]`（允许 1e-5 浮点容差）。
2. **权重一致**：加载 checkpoint 后直接读取的 `fusion.get_layer_weights()` 应与同一状态的直接模块调用一致；pinned 类别在 Fig.1 显示 one-hot，Fig.3 仍按实际 patch alpha 绘图。
3. **推理分数一致**：同一 checkpoint、同一图像和配置下，新脚本的 `score_patch` 与原 `infer_multiclass_residual.py` 数值 `allclose`；后处理 score map 同样一致，不改变 MoE / CLS / Adapter 路径。
4. **统计准确**：用手工构造的两个类别、若干 Patch 做标准答案；对比流式统计与直接 `numpy.mean/std`；熵和 L1 逐 Patch 统计一致。
5. **模式兼容**：分别覆盖 schema v2 缺省 `dynamic_topk`、schema v3 `prior_top1` 和 schema v3 `dynamic_topk`；对旧 checkpoint 不擅自套用新模式。
6. **不训练**：执行前后对 `net`、`fusion` 全量 `state_dict` 的张量值比较不变；无梯度、无 optimizer、无 checkpoint 改写。
7. **结果可复现**：同 seed / 相同权重 / 相同输入样本，CSV 数值和样本选择稳定；metadata 记录 Git commit、checkpoint 路径与哈希（如成本允许）、数据路径、推理配置与时间。
8. **可视化合规**：离散主导层图不用双线性插值；GT 与 score 对齐；热图不对单样本做隐式独立归一化；缺少案例时如实报错/跳过。

## 9. 最终交付定义

完成后执行一次命令即可从**已训练 checkpoint**生成 Fig.1～Fig.6、统计 CSV、少量案例 NPZ 与复现信息。训练代码及模型语义保持不变；任何性能指标只作为已有推理结果的对照，不利用 GT 对输入特征或融合权重做调参。

论文核心论证链条应当是：**类别先验差异（Fig.1）→ Patch 权重统计与空间差异（Fig.2～4）→ 偏离先验的幅度（Fig.5）→ 与异常检测输出的关系（Fig.6）**。如发现某类所有 Patch 主导层几乎相同，应如实报告“类别偏置主导，Patch 自适应性有限”，而不将 L1 较大单独当作动态性证据。