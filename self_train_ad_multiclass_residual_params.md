# `self_train_ad_multiclass_residual.py` 参数说明

本文档对应脚本：`self_train_ad_multiclass_residual.py`  
用途：多类别异常检测的残差特征自训练。

## 1) 路径与数据

- `--data_path` (str, 默认: `/media/honeywell/D/bhy/dataset/MVTec_overlap/MVTec_noisy10`)  
  数据集根目录。
- `--save_path` (str, 默认: `/media/honeywell/E/bhy/FUNAD/save_results/muti_class_residual`)  
  结果与模型保存目录。
- `--dataset` (str, 默认: `mvtec`, 可选: `mvtec`/`visa`)  
  数据集类型。
- `--noise` (str, 默认: `10%`)  
  仅用于实验标识/保存路径命名（不是训练中动态加噪开关）。

## 2) 训练基础超参

- `--seed` (int, 默认: `0`)  
  随机种子。
- `--lr` / `-l` (float, 默认: `2e-5`)  
  学习率。
- `--epoch` (int, 默认: `200`)  
  训练轮数。
- `--batch_size` / `-b` (int, 默认: `64`)  
  batch 大小。
- `--num_workers` (int, 默认: `4`)  
  DataLoader 的 worker 数。
- `--eval_interval` (int, 默认: `1`)  
  每多少个 epoch 评估一次。
- `--save_log` (flag, 默认: 关闭)  
  是否写入日志文件。

## 3) 伪标签与阈值相关

- `--threshold` / `-t` (float, 默认: `0.5`)  
  伪标签二值化阈值（distance 大于该值视作异常 patch）。
- `--noise_threshold` / `-n` (float, 默认: `0.9`)  
  高斯扰动使用的不确定区间上界。
- `--memory_bank_score_quantile` (float, 默认: `0.1`)  
  memory bank 构建时，每类（或 legacy 特征路径下全局）按归一化图像分数取最低的该比例图像作为候选（`ceil(n * q)` 张，至少 1 张）。
- `--bank_sample_ratio` (float, 默认: `0.1`, 可选: `0.05`/`0.1`)  
  precompute 前的全局抽样比例。
- `--max_bank_images` (int, 默认: `128`)  
  进入 memory bank 的图片上限（控制算力与内存）。
- `--k_number` (int, 默认: `2`)  
  最近邻搜索时的 k（用于 patch distance）。

### 残差脚本新增（推荐重点关注）

- `--num_reference_images_per_class` (int, 默认: `2`)  
  每类用于构建参考记忆的 clean 图数量。
- `--strict_clean_reference` (flag, 默认: 关闭)  
  若某类找不到 clean 参考图则直接报错；不加该参数时仅 warning 并继续。
- `--use_class_adaptive_threshold` (flag, 默认: 关闭)  
  开启后使用“每类自适应分位数阈值”替代固定 `threshold` 做 local_label。
- `--adaptive_threshold_quantile` (float, 默认: `0.7`)  
  每类阈值分位数（例如 0.7 表示该类当前 batch 的 70% 分位）。

## 4) 损失与训练策略

- `--kl` (flag, `store_false`)  
  注意：默认是 `True`，传入该参数后变为 `False`（关闭 one-to-one 约束项）。
- `--oto_loss` (str, 默认: `mae`, 可选: `kl`/`mae`/`mse`)  
  one-to-one 对齐损失类型。
- `--weight` (float, 默认: `0`)  
  one-to-one 损失权重。
- `--iter` (int, 默认: `0`)  
  从第几个 iteration 开始启用 one-to-one 损失。
- `--alternative` (flag, 默认: 关闭)  
  使用替代优化流程（与默认联合优化不同）。
- `--balancing` (flag, `store_false`)  
  注意：默认是 `True`，传入该参数后变为 `False`（不做正负样本平衡 BCE）。
- `--llambda` (float, 默认: `1`)  
  兼容保留参数（当前主流程中几乎不直接影响关键逻辑）。

## 5) 高斯扰动

- `--gaussian` (flag, `store_false`)  
  注意：默认是 `True`，传入该参数后变为 `False`（关闭高斯扰动分支）。
- `--std` (float, 默认: `None`)  
  高斯噪声标准差；`None` 时按 batch 统计量自适应。

## 6) 特征提取与主干（DINOv3）

- `--feature_model` (str, 默认: `dinov3_vitb16`)  
  DINOv3 变体，对应 `torch.hub` 入口名；可选值见脚本内 `DINOV3_FEATURE_MODEL_REGISTRY`（如 `dinov3_vits16`、`dinov3_vitl16` 等）。  
  表中每项含 `hub_entry`（传给 `torch.hub.load` 的模型名）与可选的 `hub_repo_dir`（本地 hub 目录；为 `None` 时不写死路径）。
- 本地 hub 目录解析顺序：`hub_repo_dir`（若该项在注册表中配置了有效路径）→ 环境变量 `DINOV3_HUB_DIR` → `torch.hub.get_dir()/facebookresearch_dinov3_main` → `~/.cache/torch/hub/facebookresearch_dinov3_main`；均无效时从 GitHub `facebookresearch/dinov3` 加载。
- `--use_cls_token` (bool, 默认: `True`)  
  是否拼接 CLS token 到 patch token。
- `--dino_layer_indices` (int list, 默认: `[22,23,24,25,26,27,28]`)  
  按论文层编号（1-based）选择多层 ViT block 输出 token 并做均值聚合（Middle-7 Mean Pool）；层数不足时会自动回退到可用的中间层。

## 7) FAISS 与性能/显存

- `--faiss_cpu_index` (flag, 默认: 关闭)  
  开启后强制使用 CPU FAISS（可降显存峰值，速度可能变慢）。
- `--faiss_gpu_temp_mem_mb` (int, 默认: `256`)  
  FAISS GPU 临时显存池大小（MB）。

## 8) 兼容参数（当前多类残差脚本中未启用核心逻辑）

- `--synthetic` (flag)  
  当前脚本会直接报错：暂不支持。
- `--hist` (flag)  
  保留参数，当前主流程中未见关键分支使用。

---

## 快速推荐配置（先跑通再调优）

- 内存/显存紧张：  
  `--batch_size 8 --bank_sample_ratio 0.05 --max_bank_images 128 --faiss_cpu_index`
- 提升伪标签稳健性（你当前场景推荐）：  
  `--use_class_adaptive_threshold --adaptive_threshold_quantile 0.7`
- 参考图更稳（算力允许）：  
  `--num_reference_images_per_class 4`

