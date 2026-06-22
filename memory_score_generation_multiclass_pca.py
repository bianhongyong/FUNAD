"""
Step 1: Memory Score Generation for Multiclass + DINOv3 —— PCA 版本

由 memory_score_generation_multiclass.py（余弦距离 / 最近邻版本）改写而来。

核心区别：
  原版用「随机 memory bank 的余弦距离取 min」作为 per-patch anomaly score；
  本版改用**本项目主训练流水线自带的 PCA 集成重构残差**打分
  （src/train/epoch_precompute.py 的 _build_ensemble_pca / _ensemble_pca_score_patches），
  即对每个 patch 计算其在多组袋装 PCA 子空间上的重构残差能量，ensemble 平均。

其余完全保持一致：
  - 使用 DINOv3 (torch.hub) 提取 patch 特征
  - 使用 per-class layer indices (DINO_CLASS_LAYER_INDICES)
  - 多类别支持：逐类独立处理
  - 每类处理完后自动在测试集上评估（I-AUROC/AP/F1 + P-AUROC/AP/F1/AUPRO）
  - 保存格式不变：dict[cls_name][filename_stem] -> Tensor[P]，下游
    self_train_ad_distillation.py 无需改动即可消费。

Pipeline (per-class)：
  1. 从训练集中筛选该类图像 → 提取 DINOv3 patch features
  2. 用该类全部 patch 构建 PCA 集成 → 对每个 patch 算重构残差 → ensemble 平均
  3. 每张图像的 top-k% 平均异常分数 → 可视化并保存
  4. 用训练特征建 PCA 集成，在测试集上评估 anomaly detection 指标
"""

import argparse
import gc
import math
import os
import random
import sys
import warnings
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.nn import functional as F
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Subset

# Ensure project root is on sys.path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from dataset.feature_extract import (
    _FEATURE_MODEL_CHOICES,
    build_dinov3_feature_extractor,
    extract_dinov3_feature_batch,
)
from dataset.multiclass_feature_dataset import (
    MultiClassFeatureDataset,
    get_all_class_names,
    DINO_CLASS_LAYER_INDICES,
)
import utils.train_utils as common_utils
from src.train.epoch_precompute import (
    _build_ensemble_pca,
    _ensemble_pca_score_patches,
)
from utils.evaluate import (
    _safe_roc_auc,
    _safe_average_precision,
    f1_score_max,
    compute_pro,
)
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_tensor(obj, folder_path: str, file_name: str):
    """保存任意对象（dict / tensor / list）为 .pth 文件。"""
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, str(file_name) + '.pth')
    torch.save(obj, file_path)
    print(f"[save] {file_path}")


def get_gaussian_kernel(kernel_size=5, sigma=4, channels=1):
    """Create a Gaussian blur Conv2d layer (matches MeDS exactly)."""
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = torch.nn.Conv2d(
        in_channels=channels, out_channels=channels,
        kernel_size=kernel_size, groups=channels,
        bias=False, padding=kernel_size // 2,
    )
    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    return gaussian_filter


# ---------------------------------------------------------------------------
# 测试集评估（PCA 重构残差）
# ---------------------------------------------------------------------------

def evaluate_class_test_set(
    train_features: np.ndarray,
    num_train_images: int,
    cls_name: str,
    args,
    feature_extractor: torch.nn.Module,
    device: torch.device,
    print_fn,
) -> Dict[str, float]:
    """在测试集上评估 PCA 重构残差 anomaly scores。

    流程（与原余弦版对齐，仅打分核心换成 PCA）：
      - 提取全部测试特征（原始特征，不做归一化）
      - 用原始训练特征构建 PCA 集成（memory bank）
      - 用 _ensemble_pca_score_patches 给每张测试图打分
      - 每张图：reshape 为 2D → resize 到 256×256 → 高斯平滑 (kernel=5, σ=4)
      - Image-level 分数：top-10% mean（max_ratio=args.memory_sampling_ratio）

    Returns:
        dict with keys: class, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px
    """
    from dataset.dataset_extract import MyDataset

    # Gaussian kernel (kernel_size=5, sigma=4, 同 MeDS evaluation_memory)
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4, channels=1).to(device)

    patches_per_image = train_features.shape[0] // num_train_images

    # 用原始训练特征构建一次 PCA 集成作为 memory bank（不做归一化）
    pca_models = _build_ensemble_pca(
        train_features.astype(np.float32),
        ensemble_size=args.ensemble_size,
        sampling_ratio=args.memory_sampling_ratio,
        pca_ev=args.pseudo_label_pca_ev,
        pca_dim=args.pseudo_label_pca_dim,
        pca_eps=args.pseudo_label_pca_eps,
    )
    print_fn(f"  [{cls_name}] train features: {num_train_images} images × {patches_per_image} patches, "
             f"PCA ensemble={len(pca_models)} models")

    # 加载测试集
    test_set = MyDataset(
        dataset_path=args.data_path,
        dataset=args.dataset,
        class_name=cls_name,
        is_train=False,
        resize=args.image_size,
        cropsize=args.crop_size,
    )
    if len(test_set) == 0:
        print_fn(f"  [{cls_name}] WARNING: test set is empty, skipping evaluation.")
        return None
    test_loader = DataLoader(
        test_set, batch_size=1, shuffle=False, num_workers=args.num_workers, drop_last=False
    )
    print_fn(f"  [{cls_name}] test images: {len(test_set)}")

    cls_layer_indices = DINO_CLASS_LAYER_INDICES.get(cls_name, [-1])

    # ---------------------------------------------------------------
    # Step 1: 提取全部测试特征
    # ---------------------------------------------------------------
    test_feats_list = []
    label_list = []
    gt_list = []  # mask tensors [1, 1, cropsize, cropsize] after MyDataset transforms

    with torch.no_grad():
        for images, y, mask in test_loader:
            images = images.to(device, non_blocking=True)
            feats = extract_dinov3_feature_batch(
                images, feature_extractor,
                dino_layer_indices=cls_layer_indices,
                class_indices=None,
                use_cls_token=args.use_cls_token,
                return_cls_token=False,
            )  # [1, P, C]
            test_feats_list.append(feats.detach().cpu())
            label_list.append(y)
            gt_list.append(mask)  # [1, 1, cropsize, cropsize]

    test_features = torch.cat(test_feats_list, dim=0)  # [N_test, P, C]

    num_test = test_features.size(0)
    P = test_features.size(1)
    patch_side = int(math.isqrt(P))
    print_fn(f"  [{cls_name}] test features: {num_test} images, {P} patches/image")

    # ---------------------------------------------------------------
    # Step 2: PCA 重构残差打分（per image）
    # ---------------------------------------------------------------
    anomaly_maps_test = np.zeros((num_test, P), dtype=np.float32)
    for i in range(num_test):
        q = test_features[i].numpy().astype(np.float32)  # [P, C] 原始特征
        raw = _ensemble_pca_score_patches(q, pca_models)  # [P]
        raw = np.clip(np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
        anomaly_maps_test[i] = raw

    anomaly_maps_test = torch.from_numpy(anomaly_maps_test).to(device)  # [N, P]

    # ---------------------------------------------------------------
    # Step 3: 逐图后处理（reshape → resize=256 → gaussian → top-10% mean）
    # ---------------------------------------------------------------
    seg_maps, img_scores, y_true, mask_gt = [], [], [], []
    max_ratio = args.memory_sampling_ratio  # 0.1, 同 MeDS image_sampling_ratio

    for i in range(num_test):
        # [P] → [1, 1, patch_side, patch_side]
        anomaly_map_2d = anomaly_maps_test[i].view(1, 1, patch_side, patch_side)

        # resize 到 256×256（同 MeDS resize_mask=256）
        anomaly_map_2d = F.interpolate(anomaly_map_2d, size=256, mode='bilinear', align_corners=False)

        # Gaussian 平滑（同 MeDS kernel=5, sigma=4）
        anomaly_map_2d = gaussian_kernel(anomaly_map_2d)

        # image-level score: top-10% mean（同 MeDS max_ratio=0.1）
        flat = anomaly_map_2d.flatten(1)  # [1, 256*256]
        k = max(1, int(flat.shape[1] * max_ratio))
        sp_score = torch.sort(flat, dim=1, descending=True)[0][:, :k].mean(dim=1)

        img_scores.append(sp_score.item())

        # pixel-level: resize anomaly map 到 mask 分辨率以对齐 ground truth
        gt_mask = gt_list[i]  # [1, 1, cropsize, cropsize]
        h, w = int(gt_mask.shape[2]), int(gt_mask.shape[3])
        score_map_resized = F.interpolate(anomaly_map_2d, size=(h, w), mode='bilinear', align_corners=False)

        seg_maps.append(score_map_resized[0, 0].detach().cpu().numpy())  # [H, W]
        y_true.append(int(label_list[i][0].item()))
        mask_gt.append(gt_mask[0, 0].detach().cpu().numpy())  # [H, W]

    # ---- 计算指标 ----
    y_true_arr = np.array(y_true).astype(np.uint8)
    img_scores_arr = np.array(img_scores).astype(np.float32)
    seg_maps_arr = np.stack(seg_maps, axis=0).astype(np.float32)
    mask_gt_arr = np.stack(mask_gt, axis=0).astype(np.uint8)

    auroc_sp = _safe_roc_auc(y_true_arr, img_scores_arr)
    ap_sp = _safe_average_precision(y_true_arr, img_scores_arr)
    f1_sp = f1_score_max(y_true_arr, img_scores_arr)

    y_true_px_flat = mask_gt_arr.ravel()
    y_score_px_flat = seg_maps_arr.ravel()
    auroc_px = _safe_roc_auc(y_true_px_flat, y_score_px_flat)
    ap_px = _safe_average_precision(y_true_px_flat, y_score_px_flat)
    f1_px = f1_score_max(y_true_px_flat, y_score_px_flat)
    aupro_px = compute_pro(mask_gt_arr, seg_maps_arr)

    print_fn(
        f"  [{cls_name}] Test — I-AUROC:{auroc_sp:.4f}, I-AP:{ap_sp:.4f}, I-F1:{f1_sp:.4f} | "
        f"P-AUROC:{auroc_px:.4f}, P-AP:{ap_px:.4f}, P-F1:{f1_px:.4f}, P-AUPRO:{aupro_px:.4f}"
    )

    return {
        "class": cls_name,
        "auroc_sp": auroc_sp, "ap_sp": ap_sp, "f1_sp": f1_sp,
        "auroc_px": auroc_px, "ap_px": ap_px, "f1_px": f1_px, "aupro_px": aupro_px,
    }


# ---------------------------------------------------------------------------
# Per-class 核心流水线
# ---------------------------------------------------------------------------

def extract_class_features(
    loader: DataLoader,
    feature_extractor: torch.nn.Module,
    cls_idx: int,
    cls_layer_indices: List[int],
    use_cls_token: bool,
    device: torch.device,
    print_fn,
) -> Tuple[np.ndarray, List[str], List[int]]:
    """提取单个类别的 DINOv3 patch features。

    Returns:
        features: np.ndarray [N_patches, C]
        filenames: list[str]  每张图的文件名
        binary_flags: list[int] 每张图是否有异常像素 (0/1)
    """
    patch_list = []
    filenames = []
    binary_flags = []

    with torch.no_grad():
        for images, class_idx_tensor, idx, patch_mask in loader:
            images = images.to(device, non_blocking=True)
            # 单类模式，直接传层列表（不传 dict）
            features = extract_dinov3_feature_batch(
                images,
                feature_extractor,
                dino_layer_indices=cls_layer_indices,
                class_indices=None,
                use_cls_token=use_cls_token,
                return_cls_token=False,
            )
            # features: [B, num_patches, C]
            feat_np = features.detach().cpu().numpy().astype(np.float32)

            for b in range(feat_np.shape[0]):
                patch_list.append(feat_np[b])

                # 从 dataset 获取文件名
                # idx 已由 Subset 从原始 MultiClassFeatureDataset 透传过来，直接就是原始索引
                global_idx = int(idx[b].cpu().numpy())
                img_path, _ = loader.dataset.dataset.samples[global_idx]  # type: ignore[union-attr]
                fname = os.path.splitext(os.path.basename(img_path))[0]
                filenames.append(fname)

                pm = patch_mask[b] if patch_mask is not None else None
                if pm is not None and pm.sum().item() > 0.5:
                    binary_flags.append(1)
                else:
                    binary_flags.append(0)

            del features, feat_np

    features = np.concatenate(patch_list, axis=0)  # [N, C]
    print_fn(f"  extracted {features.shape[0]} patches, {len(filenames)} images, "
             f"anomaly_ratio={np.mean(binary_flags):.3f}")
    return features, filenames, binary_flags


def generate_single_class_memory_scores(
    features: np.ndarray,
    filenames: List[str],
    binary_flags: List[int],
    cls_name: str,
    args,
    save_dir_class: str,
    print_fn,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """从单类原始特征生成 PCA 重构残差 anomaly scores。

    与原余弦版的区别：
      - 用该类全部 patch（原始特征，不做归一化）构建本项目的袋装 PCA 集成
        (_build_ensemble_pca)
      - 每个 patch 的异常分 = 在各 PCA 子空间上的重构残差能量，ensemble 平均
        (_ensemble_pca_score_patches)
      - 输出仍为原始（未归一化）分数，下游 distillation 按类 min-max 归一化

    Returns:
        dict[filename] -> 1D tensor of per-patch anomaly scores.
    """
    num_images = len(filenames)
    patches_per_image = features.shape[0] // num_images
    assert features.shape[0] % num_images == 0, \
        f"patches {features.shape[0]} not divisible by images {num_images}"

    # ---------------------------------------------------------------
    # 构建 PCA 集成（memory bank = 该类全部 patch，原始特征不归一化）
    # ---------------------------------------------------------------
    feat_raw = features.astype(np.float32)  # [N, C]

    print_fn(f"  [{cls_name}] generating PCA memory scores: {num_images} images × {patches_per_image} patches, "
             f"ensemble={args.ensemble_size}, sampling_ratio={args.memory_sampling_ratio}, "
             f"pca_ev={args.pseudo_label_pca_ev}, pca_dim={args.pseudo_label_pca_dim}")

    pca_models = _build_ensemble_pca(
        feat_raw,
        ensemble_size=args.ensemble_size,
        sampling_ratio=args.memory_sampling_ratio,
        pca_ev=args.pseudo_label_pca_ev,
        pca_dim=args.pseudo_label_pca_dim,
        pca_eps=args.pseudo_label_pca_eps,
    )
    print_fn(f"  [{cls_name}] built {len(pca_models)} PCA models")

    # ---------------------------------------------------------------
    # 对全量 patch 打分 → reshape [N_img, P]
    # ---------------------------------------------------------------
    raw_scores = _ensemble_pca_score_patches(feat_raw, pca_models)  # [N*P]
    raw_scores = np.clip(
        np.nan_to_num(raw_scores, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None
    ).astype(np.float32)
    anomaly_maps_memory = raw_scores.reshape(num_images, patches_per_image)  # [N, P]

    # 每张图的 per-patch 分数张量
    valid_scores = [
        torch.from_numpy(anomaly_maps_memory[i]).float() for i in range(num_images)
    ]

    # ---------------------------------------------------------------
    # per-image top-k% 统计
    # ---------------------------------------------------------------
    top_1p_means, top_2p_means, top_5p_means = [], [], []
    for valid in valid_scores:
        P = valid.shape[0]
        k1 = max(1, int(math.ceil(P * 0.01)))
        k2 = max(1, int(math.ceil(P * 0.02)))
        k5 = max(1, int(math.ceil(P * 0.05)))
        sorted_vals, _ = torch.sort(valid, descending=True)
        top_1p_means.append(sorted_vals[:k1].mean().item())
        top_2p_means.append(sorted_vals[:k2].mean().item())
        top_5p_means.append(sorted_vals[:k5].mean().item())

    binary_arr = np.array(binary_flags)

    # ---- 可视化 ----
    if not args.no_visualize:
        cls_save_dir = os.path.join(save_dir_class, cls_name)
        os.makedirs(cls_save_dir, exist_ok=True)

        plot_data = {
            'top_1%_memory': (np.array(top_1p_means), 'mean_anomaly_maps_top_1_memory'),
            'top_2%_memory': (np.array(top_2p_means), 'mean_anomaly_maps_top_2_memory'),
            'top_5%_memory': (np.array(top_5p_means), 'mean_anomaly_maps_top_5_memory'),
        }
        for key, (values, plot_title) in plot_data.items():
            plt.figure(figsize=(10, 6))
            defective = values * binary_arr
            plt.plot(values, label=key)
            plt.plot(defective, label='defect/noisy samples', linestyle='', marker='o', markersize=3)
            plt.xlabel('Image Index')
            plt.ylabel('Mean Score')
            plt.title(f'{cls_name} {plot_title}')
            plt.legend()
            plt.grid(True)
            save_path = os.path.join(cls_save_dir, f'{cls_name}_{plot_title}.png')
            plt.savefig(save_path, dpi=100)
            plt.close()
            print_fn(f"  saved plot: {save_path}")

    # ---- 构建返回 dict ----
    cls_memory_scores = {}
    for i, fname in enumerate(filenames):
        cls_memory_scores[fname] = valid_scores[i]

    print_fn(f"  [{cls_name}] done, {len(cls_memory_scores)} images")
    return cls_memory_scores


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        "memory_score_generation_multiclass_pca",
        description="Step 1: Generate PCA-based memory anomaly scores for multiclass + DINOv3."
    )
    parser.add_argument("--data_path", type=str,
                        default="/media/honeywell/D/bhy/dataset/MVTec_overlap/MVTec_noisy10")
    parser.add_argument("--dataset", type=str, default="mvtec", choices=["mvtec", "visa"])
    parser.add_argument("--output_dir", type=str,
                        default="./memory_scores_pca")
    parser.add_argument("--feature_model", type=str, choices=_FEATURE_MODEL_CHOICES,
                        default="dinov3_vitl16")
    parser.add_argument("--dinov3_hub", type=str, default=None)

    # 数据 / 特征参数
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--crop_size", type=int, default=448)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_cls_token", type=common_utils.str2bool, default=False)

    # PCA memory score 核心参数
    parser.add_argument("--ensemble_size", type=int, default=100,
                        help="Number of bagged PCA models in the ensemble.")
    parser.add_argument("--memory_sampling_ratio", type=float, default=0.1,
                        help="Fraction of patches sampled per PCA model (bagging ratio).")
    parser.add_argument("--pseudo_label_pca_dim", type=int, default=0,
                        help="Number of PCA components to retain; 0 = auto by explained variance.")
    parser.add_argument("--pseudo_label_pca_ev", type=float, default=0.99,
                        help="Explained-variance target when pca_dim == 0.")
    parser.add_argument("--pseudo_label_pca_eps", type=float, default=1e-6,
                        help="Numerical stability epsilon for PCA.")

    # 其他
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_visualize", action="store_true",
                        help="Skip plotting per-class visualizations.")
    parser.add_argument("--no_eval", action="store_true",
                        help="Skip test-set evaluation.")
    parser.add_argument("--class_names", type=str, nargs="+", default=None,
                        help="Subset of classes to process. Default: all.")

    args = parser.parse_args()

    # -------------------------------------------------------------------
    # 初始化
    # -------------------------------------------------------------------
    setup_seed(args.seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    save_dir_class = os.path.join(args.output_dir, "plots")
    os.makedirs(save_dir_class, exist_ok=True)

    print(f"device: {device}")
    print(f"feature_model: {args.feature_model}")
    print(f"ensemble_size: {args.ensemble_size}")
    print(f"memory_sampling_ratio: {args.memory_sampling_ratio}")
    print(f"pca_dim: {args.pseudo_label_pca_dim}, pca_ev: {args.pseudo_label_pca_ev}")

    all_class_names = get_all_class_names(args.dataset)
    target_class_names = args.class_names if args.class_names is not None else all_class_names
    unknown = [n for n in target_class_names if n not in all_class_names]
    if unknown:
        raise ValueError(f"Unknown classes: {unknown}. Available: {all_class_names}")
    class_to_idx = {name: idx for idx, name in enumerate(all_class_names)}
    # 只处理 target 类的索引
    target_class_indices = [class_to_idx[n] for n in target_class_names]
    print(f"Classes to process ({len(target_class_names)}): {target_class_names}")

    # -------------------------------------------------------------------
    # Build DINOv3 feature extractor（全局复用一个）
    # -------------------------------------------------------------------
    restore_hub = None
    if args.dinov3_hub:
        old_val = os.environ.get("DINOV3_HUB_DIR")
        os.environ["DINOV3_HUB_DIR"] = args.dinov3_hub
        def restore_hub():
            if old_val is None:
                os.environ.pop("DINOV3_HUB_DIR", None)
            else:
                os.environ["DINOV3_HUB_DIR"] = old_val

    feature_extractor = build_dinov3_feature_extractor(args.feature_model, device)
    if restore_hub is not None:
        restore_hub()

    # -------------------------------------------------------------------
    # 构建一次数据集（包含所有类），后续按类取 Subset
    # -------------------------------------------------------------------
    train_dataset = MultiClassFeatureDataset(
        data_path=args.data_path,
        dataset_name=args.dataset,
        image_size=args.image_size,
        crop_size=args.crop_size,
        seed=args.seed,
        shuffle=False,
    )

    # -------------------------------------------------------------------
    # Per-class 主循环（逐类处理）
    # -------------------------------------------------------------------
    memory_score_dict = {}
    metrics_rows = []

    for cls_idx in target_class_indices:
        cls_name = all_class_names[cls_idx]
        cls_layer_indices = DINO_CLASS_LAYER_INDICES.get(cls_name, [-1])

        print(f"\n{'=' * 60}")
        print(f"Processing class [{cls_idx}]: {cls_name}")
        print(f"  layer_indices: {cls_layer_indices}")
        print(f"{'=' * 60}")

        # ---------------------------------------------------------------
        # Step 1: 提取该类特征
        # ---------------------------------------------------------------
        cls_indices = train_dataset.classwise_global_indices.get(cls_idx, [])
        if not cls_indices:
            print(f"  WARNING: no samples for class {cls_name}, skipping.")
            continue

        cls_subset = Subset(train_dataset, cls_indices)
        cls_loader = DataLoader(
            cls_subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        print(f"\n--- Step 1: Extract DINOv3 features [{cls_name}] ---")
        features, filenames, binary_flags = extract_class_features(
            cls_loader, feature_extractor, cls_idx, cls_layer_indices,
            args.use_cls_token, device, print_fn=print,
        )

        # ---------------------------------------------------------------
        # Step 2: PCA memory score generation（训练集，原始特征 + PCA 残差）
        # ---------------------------------------------------------------
        print(f"\n--- Step 2: Generate PCA memory scores [{cls_name}] ---")
        cls_memory_scores = generate_single_class_memory_scores(
            features, filenames, binary_flags, cls_name,
            args, save_dir_class, print_fn=print, device=device,
        )
        memory_score_dict[cls_name] = cls_memory_scores

        # ---------------------------------------------------------------
        # Step 3: 测试集评估（原始特征 + PCA 残差）
        # ---------------------------------------------------------------
        if not args.no_eval:
            print(f"\n--- Step 3: Test set evaluation [{cls_name}] ---")
            test_metrics = evaluate_class_test_set(
                train_features=features,
                num_train_images=len(filenames),
                cls_name=cls_name,
                args=args,
                feature_extractor=feature_extractor,
                device=device,
                print_fn=print,
            )
            if test_metrics is not None:
                metrics_rows.append(test_metrics)
        else:
            print(f"\n--- Step 3: Test set evaluation [{cls_name}] (skipped, --no_eval) ---")

        # ---------------------------------------------------------------
        # 清理：释放该类所有中间数据
        # ---------------------------------------------------------------
        del features, cls_memory_scores, cls_subset, cls_loader
        gc.collect()
        if use_cuda:
            torch.cuda.empty_cache()

    # 释放特征提取器
    del feature_extractor
    gc.collect()
    if use_cuda:
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------
    # 保存 memory scores + 指标汇总
    # -------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("All classes done. Saving memory scores...")
    save_tensor(memory_score_dict, args.output_dir, "memory_scores")
    print(f"Done. Memory scores saved to {args.output_dir}/memory_scores.pth")
    print(f"Visualizations saved to {save_dir_class}/")

    if metrics_rows:
        import pandas as pd
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_csv_path = os.path.join(args.output_dir, "test_metrics.csv")
        metrics_df.to_csv(metrics_csv_path, index=False)
        print(f"\nTest metrics saved to: {metrics_csv_path}")
        print(metrics_df.to_string(index=False))

        # Mean metrics
        mean_row = {"class": "mean"}
        for col in ["auroc_sp", "ap_sp", "f1_sp", "auroc_px", "ap_px", "f1_px", "aupro_px"]:
            mean_row[col] = metrics_df[col].mean()
        print(f"\nMean: I-AUROC:{mean_row['auroc_sp']:.4f}, I-AP:{mean_row['ap_sp']:.4f}, "
              f"I-F1:{mean_row['f1_sp']:.4f}, P-AUROC:{mean_row['auroc_px']:.4f}, "
              f"P-AP:{mean_row['ap_px']:.4f}, P-F1:{mean_row['f1_px']:.4f}, "
              f"P-AUPRO:{mean_row['aupro_px']:.4f}")


if __name__ == "__main__":
    main()
