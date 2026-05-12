import os
import time
from typing import Optional

import faiss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader, Subset
from utils.memory_bank_stats import print_greedy_memory_bank_anomaly_stats
from utils.sampler import ApproximateGreedyCoresetSampler

from src.train.pseudo_label_normalizers import PseudoLabelNormalizerFactory
from src.train.pseudo_label_scorers import (
    MahalanobisPseudoLabelScorer,
    NNPseudoLabelScorer,
    PCAPseudoLabelScorer,
    PseudoLabelScorer,
    PseudoLabelScorerFactory,
    postprocess_faiss_distances,
)



def _as_1d_float32(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.shape == ():
        x = np.array([x], dtype=np.float32)
    return x.astype(np.float32).reshape(-1)


def _minmax_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    vmin = float(values.min())
    vmax = float(values.max())
    if vmax == vmin:
        return np.zeros_like(values, dtype=np.float32)
    return (values - vmin) / (vmax - vmin)


def _normalize_distance_map_per_class(
    distance_map: np.ndarray, class_stack: np.ndarray, num_classes: int, args
) -> np.ndarray:
    normalizer = PseudoLabelNormalizerFactory.create(args)
    values = np.asarray(distance_map, dtype=np.float32)
    normalized = np.zeros_like(values, dtype=np.float32)
    has_valid_class = False

    for cls in range(num_classes):
        cls_indices = np.where(class_stack == cls)[0]
        if cls_indices.size == 0:
            continue
        cls_values = values[cls_indices]
        normalized[cls_indices] = normalizer.normalize(cls_values)
        has_valid_class = True

    if not has_valid_class:
        return np.zeros_like(values, dtype=np.float32)
    return normalized.astype(np.float32)


def _normalize_distance_map_global(
    distance_map: np.ndarray, args
) -> np.ndarray:
    """Global min-max (or percentile) normalization across ALL patches regardless of class."""
    normalizer = PseudoLabelNormalizerFactory.create(args)
    values = np.asarray(distance_map, dtype=np.float32)
    flat = values.reshape(-1)
    normalized = normalizer.normalize(flat)
    return normalized.reshape(values.shape).astype(np.float32)


def _extract_features_with_optional_cls_token(
    extract_feature_batch_fn, images, feature_extractor, args
):
    try:
        features, cls_token = extract_feature_batch_fn(
            images, feature_extractor, args, return_cls_token=True
        )
        return features, cls_token
    except TypeError:
        features = extract_feature_batch_fn(images, feature_extractor, args)
        return features, None


def _build_patch_class_idx(class_idx: torch.Tensor, patch_features: torch.Tensor):
    if class_idx is None or patch_features is None:
        return None
    token_per_image = int(patch_features.shape[1]) if patch_features.dim() >= 2 else 1
    return (
        class_idx.to(device=patch_features.device, dtype=torch.long)
        .reshape(-1, 1, 1)
        .expand(-1, token_per_image, 1)
        .contiguous()
    )


def _plot_distance_distribution_normal_vs_anomaly(
    args, distance_values, gt_patch_masks, epoch=None
):
    values = np.asarray(distance_values, dtype=np.float32).reshape(-1)
    labels = np.asarray(gt_patch_masks, dtype=np.uint8).reshape(-1) > 0
    if values.size == 0 or labels.size != values.size:
        return

    normal_values = values[~labels]
    anomaly_values = values[labels]
    if normal_values.size == 0 and anomaly_values.size == 0:
        return

    # Keep plotting lightweight even for large datasets.
    max_points = 200000
    if normal_values.size > max_points:
        normal_values = np.random.choice(normal_values, size=max_points, replace=False)
    if anomaly_values.size > max_points:
        anomaly_values = np.random.choice(anomaly_values, size=max_points, replace=False)

    save_dir = os.path.join(args.save_path, args.dataset, args.noise)
    os.makedirs(save_dir, exist_ok=True)
    if epoch is None:
        filename = "distance_distribution_normal_vs_anomaly.png"
    else:
        filename = f"distance_distribution_normal_vs_anomaly_epoch_{int(epoch) + 1:03d}.png"
    save_path = os.path.join(save_dir, filename)

    plt.figure(figsize=(10, 6))
    bins = np.linspace(0.0, 1.0, 101)
    if normal_values.size > 0:
        plt.hist(
            normal_values,
            bins=bins,
            density=True,
            alpha=0.55,
            color="tab:blue",
            label=f"normal ({normal_values.size})",
        )
    if anomaly_values.size > 0:
        plt.hist(
            anomaly_values,
            bins=bins,
            density=True,
            alpha=0.55,
            color="tab:red",
            label=f"anomaly ({anomaly_values.size})",
        )

    plt.xlabel("Normalized distance")
    plt.ylabel("Density")
    plt.title("Distance Distribution: Normal vs Anomaly Patches")
    plt.xlim(0.0, 1.0)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[PseudoLabel-Plot] saved: {save_path}")


def _plot_image_norm_score_distribution_normal_vs_anomaly(
    args, image_norm_scores, gt_patch_masks, epoch=None
):
    scores = np.asarray(image_norm_scores, dtype=np.float32).reshape(-1)
    patch_masks = np.asarray(gt_patch_masks, dtype=np.uint8)
    if scores.size == 0 or patch_masks.ndim != 2 or patch_masks.shape[0] != scores.size:
        return

    # Image is anomaly if any patch is anomaly.
    image_is_anomaly = patch_masks.reshape(patch_masks.shape[0], -1).sum(axis=1) > 0
    normal_scores = scores[~image_is_anomaly]
    anomaly_scores = scores[image_is_anomaly]
    if normal_scores.size == 0 and anomaly_scores.size == 0:
        return

    save_dir = os.path.join(args.save_path, args.dataset, args.noise)
    os.makedirs(save_dir, exist_ok=True)
    if epoch is None:
        filename = "image_norm_scores_distribution_normal_vs_anomaly.png"
    else:
        filename = (
            f"image_norm_scores_distribution_normal_vs_anomaly_epoch_{int(epoch) + 1:03d}.png"
        )
    save_path = os.path.join(save_dir, filename)

    plt.figure(figsize=(10, 6))
    bins = np.linspace(0.0, 1.0, 51)
    if normal_scores.size > 0:
        plt.hist(
            normal_scores,
            bins=bins,
            density=True,
            alpha=0.55,
            color="tab:blue",
            label=f"normal ({normal_scores.size})",
        )
    if anomaly_scores.size > 0:
        plt.hist(
            anomaly_scores,
            bins=bins,
            density=True,
            alpha=0.55,
            color="tab:red",
            label=f"anomaly ({anomaly_scores.size})",
        )

    plt.xlabel("Image normalized score")
    plt.ylabel("Density")
    plt.title("Image Norm Score Distribution: Normal vs Anomaly Images")
    plt.xlim(0.0, 1.0)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[PseudoLabel-Plot] saved: {save_path}")


def _plot_classwise_feature_l2_by_distance_regions(
    args,
    class_names,
    region_gt_means,
    epoch=None,
):
    means = np.asarray(region_gt_means, dtype=np.float32)
    if means.ndim != 3 or means.shape[1:] != (3, 2):
        return
    if means.shape[0] == 0:
        return

    save_dir = os.path.join(args.save_path, args.dataset, args.noise)
    os.makedirs(save_dir, exist_ok=True)
    if epoch is None:
        filename = "classwise_feature_l2_by_distance_regions.png"
    else:
        filename = f"classwise_feature_l2_by_distance_regions_epoch_{int(epoch) + 1:03d}.png"
    save_path = os.path.join(save_dir, filename)

    c = means.shape[0]
    labels = list(class_names) if class_names is not None else []
    if len(labels) != c:
        labels = [f"cls_{i}" for i in range(c)]

    x = np.arange(c, dtype=np.float32)
    width = 0.13
    series = [
        ("d<thr | GT normal", means[:, 0, 0], "tab:blue"),
        ("d<thr | GT anomaly", means[:, 0, 1], "tab:orange"),
        ("thr<=d<=noise_thr | GT normal", means[:, 1, 0], "tab:green"),
        ("thr<=d<=noise_thr | GT anomaly", means[:, 1, 1], "tab:red"),
        ("d>noise_thr | GT normal", means[:, 2, 0], "tab:purple"),
        ("d>noise_thr | GT anomaly", means[:, 2, 1], "tab:brown"),
    ]
    offsets = np.linspace(-2.5 * width, 2.5 * width, num=len(series), dtype=np.float32)

    plt.figure(figsize=(max(12, c * 0.8), 6.5))
    for idx, (name, vals, color) in enumerate(series):
        vals = np.asarray(vals, dtype=np.float32)
        vals_plot = np.where(np.isfinite(vals), vals, np.nan)
        plt.bar(x + offsets[idx], vals_plot, width=width, label=name, color=color, alpha=0.85)

    plt.xlabel("Class")
    plt.ylabel("Mean feature L2 norm")
    plt.title("Class-wise Feature L2 Norm by Distance Region and GT Label")
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()
    print(f"[PseudoLabel-Plot] saved: {save_path}")


def _plot_classwise_abs_distance_box_before_norm(
    args,
    class_stack: np.ndarray,
    distance_map: np.ndarray,
    gt_patch_masks: np.ndarray,
    num_classes: int,
    class_names=None,
    epoch=None,
):
    values = np.abs(np.asarray(distance_map, dtype=np.float32))
    cls_arr = np.asarray(class_stack, dtype=np.int64).reshape(-1)
    gt_mask = np.asarray(gt_patch_masks, dtype=np.uint8)
    if values.ndim != 2 or gt_mask.shape != values.shape:
        return
    if cls_arr.shape[0] != values.shape[0] or int(num_classes) <= 0:
        return

    save_dir = os.path.join(args.save_path, args.dataset, args.noise)
    os.makedirs(save_dir, exist_ok=True)
    if epoch is None:
        filename = "classwise_abs_distance_box_before_norm.png"
    else:
        filename = f"classwise_abs_distance_box_before_norm_epoch_{int(epoch) + 1:03d}.png"
    save_path = os.path.join(save_dir, filename)

    labels = list(class_names) if class_names is not None else []
    if len(labels) != int(num_classes):
        labels = [f"cls_{i}" for i in range(int(num_classes))]

    series = []
    positions = []
    colors = []
    # Keep plotting lightweight for very large patch sets.
    max_points_per_group = 30000

    for cls in range(int(num_classes)):
        cls_idx = np.where(cls_arr == cls)[0]
        if cls_idx.size == 0:
            continue
        cls_vals = values[cls_idx].reshape(-1)
        cls_gt = gt_mask[cls_idx].reshape(-1) > 0

        normal_vals = cls_vals[~cls_gt]
        anomaly_vals = cls_vals[cls_gt]

        if normal_vals.size > max_points_per_group:
            normal_vals = np.random.choice(normal_vals, size=max_points_per_group, replace=False)
        if anomaly_vals.size > max_points_per_group:
            anomaly_vals = np.random.choice(anomaly_vals, size=max_points_per_group, replace=False)

        x = float(cls)
        if normal_vals.size > 0:
            series.append(normal_vals.astype(np.float32))
            positions.append(x - 0.18)
            colors.append("tab:blue")
        if anomaly_vals.size > 0:
            series.append(anomaly_vals.astype(np.float32))
            positions.append(x + 0.18)
            colors.append("tab:red")

    if len(series) == 0:
        return

    fig_w = max(12.0, float(num_classes) * 0.8)
    plt.figure(figsize=(fig_w, 6.8))

    # Density layer (violin) + robust summary layer (boxplot).
    vp = plt.violinplot(
        series,
        positions=positions,
        widths=0.34,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body, c in zip(vp["bodies"], colors):
        body.set_facecolor(c)
        body.set_edgecolor(c)
        body.set_alpha(0.20)

    bp = plt.boxplot(
        series,
        positions=positions,
        widths=0.22,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.2},
        whiskerprops={"linewidth": 1.0},
        capprops={"linewidth": 1.0},
    )
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.5)
        patch.set_edgecolor(c)

    plt.xlabel("Class")
    plt.ylabel("|distance| (before normalization)")
    plt.title("Class-wise |distance_map| Box + Density Before Normalization")
    plt.xticks(np.arange(int(num_classes), dtype=np.float32), labels, rotation=35, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(
        handles=[
            mpatches.Patch(color="tab:blue", alpha=0.5, label="GT normal"),
            mpatches.Patch(color="tab:red", alpha=0.5, label="GT anomaly"),
        ]
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()
    print(f"[PseudoLabel-Plot] saved: {save_path}")
def precompute_pseudo_labels_multiclass_residual(
    args,
    localnet,
    feature_extractor,
    mini_loader,
    num_classes,
    reference_memory_by_class,
    reference_index_by_class,
    device,
    extract_feature_batch_fn,
    compute_residual_feature_batch_fn,
    aggregate_image_scores_fn,
    build_faiss_index_fn,
    update_topk_features_fn,
    print_selected_score_distribution_fn,
    print_selected_clean_ratio_fn,
    print_confusion_matrix_fn,
    epoch=None,
    pseudo_label_scorer: Optional[PseudoLabelScorer] = None,
):
    memory_bank_start = time.perf_counter()

    dataset_size = len(mini_loader.dataset)
    image_scores = np.zeros(dataset_size, dtype=np.float32)
    class_stack = np.zeros(dataset_size, dtype=np.int64)
    gt_patch_masks = np.zeros((dataset_size, 784), dtype=np.uint8)
    global_dim = None

    print("[Phase 1/3] Computing image scores...")
    with torch.no_grad():
        localnet.eval()
        feature_extractor.eval()
        for batch in tqdm.tqdm(mini_loader, desc="Phase 1"):
            if len(batch) == 4:
                images, mini_class_idx, mini_sample_idx, mini_patch_mask = batch
            else:
                images, mini_class_idx, mini_sample_idx = batch
                mini_patch_mask = None
            images = images.to(device)
            image_features, cls_token = _extract_features_with_optional_cls_token(
                extract_feature_batch_fn, images, feature_extractor, args
            )
            residual_features = compute_residual_feature_batch_fn(
                image_features,
                mini_class_idx.to(device),
                reference_memory_by_class,
                reference_index_by_class,
            )
            patch_class_idx = None
            if args.use_moe_discriminator and getattr(args, "moe_hard_class_gate", False):
                patch_class_idx = _build_patch_class_idx(mini_class_idx, residual_features)
            features, score = localnet(
                residual_features, cls_token=cls_token, patch_class_idx=patch_class_idx
            )
            if features.shape[0] > args.batch_size:
                features = features.unsqueeze(0)

            if global_dim is None:
                global_dim = int(features.shape[-1])

            score_np = score.detach().cpu().numpy()
            score_np = _as_1d_float32(
                aggregate_image_scores_fn(
                    score_np,
                    topk_ratio=args.img_score_topk_ratio,
                )
            )

            sample_idx_np = mini_sample_idx.detach().cpu().numpy().astype(np.int64)
            class_np = mini_class_idx.detach().cpu().numpy().astype(np.int64)
            image_scores[sample_idx_np] = score_np.astype(np.float32)
            class_stack[sample_idx_np] = class_np
            if mini_patch_mask is not None:
                mask_np = mini_patch_mask.detach().cpu().numpy()
                gt_patch_masks[sample_idx_np] = (mask_np > 0).astype(np.uint8)
    image_norm_scores = np.zeros(dataset_size, dtype=np.float32)
    normalized_score = np.zeros_like(image_scores, dtype=np.float32)
    #分类归一化
    for cls in range(num_classes):
        cls_mask = class_stack == cls
        if not np.any(cls_mask):
            continue
        cls_scores = image_scores[cls_mask]
        cls_norm = _minmax_normalize(cls_scores)
        normalized_score[cls_mask] = cls_norm.astype(np.float32)

    image_norm_scores[:] = normalized_score.astype(np.float32)
    _plot_image_norm_score_distribution_normal_vs_anomaly(
        args=args,
        image_norm_scores=image_norm_scores,
        gt_patch_masks=gt_patch_masks,
        epoch=epoch,
    )
    
    selected_local_list = []
    print("[Phase 2/3] Building memory bank from normal samples...")
    #下采样选取
    select_mode = str(getattr(args, "normal_sample_selection", "threshold"))
    select_quantile = float(getattr(args, "normal_sample_quantile", 0.3))
    for cls in range(num_classes):
        cls_mask = class_stack == cls
        if not np.any(cls_mask):
            continue
        cls_indices = np.where(cls_mask)[0].astype(np.int64)
        cls_norm_scores = normalized_score[cls_indices]

        if select_mode == "quantile":
            # 取归一化分数最低的前 N% 图片作为正常候选
            q = np.percentile(cls_norm_scores, select_quantile * 100.0)
            cls_normal_local = np.where(cls_norm_scores <= q)[0]
        else:
            # 默认：取分数 < 0.5 的图片作为正常候选
            cls_normal_local = np.where(cls_norm_scores < 0.5)[0]

        if cls_normal_local.shape[0] == 0:
            cls_selected = cls_indices
        elif select_mode == "quantile":
            # quantile 模式下已选出前 N% 图片，无需再随机下采样
            cls_selected = cls_indices[cls_normal_local]
        elif args.random < 1:
            cls_sample_num = max(1, int(cls_normal_local.shape[0] * args.random))
            pick_local = np.random.choice(
                cls_normal_local, size=cls_sample_num, replace=False
            )
            cls_selected = cls_indices[pick_local]
        else:
            cls_selected = cls_indices[cls_normal_local]
        selected_local_list.append(cls_selected)

    if len(selected_local_list) == 0:
        selected_indices = np.arange(dataset_size, dtype=np.int64)
    else:
        selected_indices = np.concatenate(selected_local_list, axis=0).astype(np.int64)
    print_selected_score_distribution_fn(
        selected_indices=selected_indices,
        class_stack=class_stack,
        image_scores=image_scores,
    )
    print_selected_clean_ratio_fn(
        selected_indices=selected_indices,
        dataset=mini_loader.dataset,
    )

    distance_map = np.zeros((dataset_size, 784), dtype=np.float16)
    patch_feature_l2_map = np.zeros((dataset_size, 784), dtype=np.float32)
    confident_feature_bank = torch.zeros((0, global_dim), dtype=torch.float32)
    if selected_indices.shape[0] == 0:
        pseudo_label_time = time.perf_counter() - memory_bank_start
        return distance_map, confident_feature_bank, global_dim, 0.0, pseudo_label_time

    # 按类别记录被选中的图像索引
    selected_images_by_class = {}
    for cls in range(num_classes):
        cls_mask = class_stack[selected_indices] == cls
        selected_images_by_class[cls] = selected_indices[cls_mask]

    # Phase 2: 提取选中样本的 residual 特征，并按类别缓存
    class_feature_buffers = {cls: [] for cls in range(num_classes)}
    with torch.no_grad():
        localnet.eval()
        feature_extractor.eval()
        selected_subset = Subset(mini_loader.dataset, selected_indices.tolist())
        selected_loader = DataLoader(
            selected_subset,
            batch_size=mini_loader.batch_size,
            pin_memory=mini_loader.pin_memory,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
        for batch in tqdm.tqdm(selected_loader, desc="Phase 2"):
            if len(batch) == 4:
                images, mini_class_idx, _mini_sample_idx, _mini_patch_mask = batch
            else:
                images, mini_class_idx, _mini_sample_idx = batch
            images = images.to(device)
            image_features, cls_token = _extract_features_with_optional_cls_token(
                extract_feature_batch_fn, images, feature_extractor, args
            )
            residual_features = compute_residual_feature_batch_fn(
                image_features,
                mini_class_idx.to(device),
                reference_memory_by_class,
                reference_index_by_class,
            )
            patch_class_idx = None
            if args.use_moe_discriminator and getattr(args, "moe_hard_class_gate", False):
                patch_class_idx = _build_patch_class_idx(mini_class_idx, residual_features)
            features, _ = localnet(
                residual_features, cls_token=cls_token, patch_class_idx=patch_class_idx
            )
            features_np = features.detach().cpu().numpy()
            batch_class_np = mini_class_idx.detach().cpu().numpy().astype(np.int64)

            for cls in np.unique(batch_class_np).tolist():
                cls = int(cls)
                cls_mask = batch_class_np == cls
                if not np.any(cls_mask):
                    continue
                cls_features = features_np[cls_mask].reshape(-1, global_dim)
                class_feature_buffers[cls].append(cls_features)

    reduced_features_by_class = {}
    coreset_indices_by_class = {}
    if getattr(args, "global_memory_bank", False):
        # 全局记忆库：将各类特征合并，统一做 GreedyCoreset
        global_features_list = []
        for cls in range(num_classes):
            if len(class_feature_buffers[cls]) == 0:
                continue
            global_features_list.append(np.concatenate(class_feature_buffers[cls], axis=0))
        if global_features_list:
            all_features = np.concatenate(global_features_list, axis=0)
            total_images = max(1, selected_indices.shape[0])
            patches_per_image_global = max(1, all_features.shape[0] // total_images)
            greedy_keep_images = max(1, int(getattr(args, "greedy_keep_images", 2)))
            target_images_global = min(greedy_keep_images, total_images)
            target_features_global = patches_per_image_global * target_images_global

            if 0 < target_features_global < all_features.shape[0]:
                percentage = float(target_features_global) / float(all_features.shape[0])
                sampler = ApproximateGreedyCoresetSampler(
                    percentage=percentage,
                    device=device,
                )
                all_features, core_idx = sampler.run(
                    all_features, return_indices=True
                )
                _ = core_idx  # unused but kept for interface compatibility
            reduced_features_by_class[0] = all_features
        print(
            f"[GlobalMemoryBank] merged {len(global_features_list)} classes → "
            f"{all_features.shape[0] if global_features_list else 0} patches after greedy coreset"
        )
    else:
        # 按类别做 GreedyCoreset，下采样到「约等于 2 张图片的 patch 数」
        for cls in range(num_classes):
            if len(class_feature_buffers[cls]) == 0:
                continue
            normal_features_cls = np.concatenate(class_feature_buffers[cls], axis=0)

            num_selected_images_cls = selected_images_by_class.get(cls, np.array([], dtype=np.int64)).shape[0]
            if num_selected_images_cls > 0 and normal_features_cls.shape[0] > 0:
                patches_per_image = normal_features_cls.shape[0] // num_selected_images_cls
            else:
                patches_per_image = normal_features_cls.shape[0]

            greedy_keep_images = max(1, int(getattr(args, "greedy_keep_images", 2)))
            target_images = (
                min(greedy_keep_images, num_selected_images_cls)
                if num_selected_images_cls > 0
                else 1
            )
            target_features = patches_per_image * target_images

            if 0 < target_features < normal_features_cls.shape[0]:
                percentage = float(target_features) / float(normal_features_cls.shape[0])
                sampler = ApproximateGreedyCoresetSampler(
                    percentage=percentage,
                    device=device,
                )
                normal_features_cls, core_idx = sampler.run(
                    normal_features_cls, return_indices=True
                )
            else:
                core_idx = np.arange(normal_features_cls.shape[0], dtype=np.int64)

            reduced_features_by_class[cls] = normal_features_cls
            coreset_indices_by_class[cls] = core_idx

    if len(reduced_features_by_class) > 0 and not getattr(args, "global_memory_bank", False):
        print_greedy_memory_bank_anomaly_stats(
            num_classes=num_classes,
            selected_images_by_class=selected_images_by_class,
            gt_patch_masks=gt_patch_masks,
            reduced_features_by_class=reduced_features_by_class,
            coreset_indices_by_class=coreset_indices_by_class,
            class_names=getattr(args, "class_names", None),
            epoch=epoch,
        )

    if len(reduced_features_by_class) == 0:
        raise ValueError("No reduced features by class")

    scoring = str(getattr(args, "pseudo_label_scoring", "nn")).lower()
    scorers = PseudoLabelScorerFactory.create(scoring, args, device, build_faiss_index_fn)
    for scorer in scorers:
        for cls, feats in reduced_features_by_class.items():
            if feats is None or feats.shape[0] == 0:
                continue
            scorer.fit_class(int(cls), feats)

    if args.beta:
        top_feat_global = None
        top_dist_global = None

    memory_bank_time = time.perf_counter() - memory_bank_start

    print("[Phase 3/3] Computing distance map for all patches...")
    pseudo_start = time.perf_counter()
    with torch.no_grad():
        localnet.eval()
        feature_extractor.eval()
        for batch in tqdm.tqdm(mini_loader, desc="Phase 3"):
            if len(batch) == 4:
                images, mini_class_idx, mini_sample_idx, _mini_patch_mask = batch
            else:
                images, mini_class_idx, mini_sample_idx = batch
            images = images.to(device)
            sample_idx_np = mini_sample_idx.detach().cpu().numpy().astype(np.int64)

            image_features, cls_token = _extract_features_with_optional_cls_token(
                extract_feature_batch_fn, images, feature_extractor, args
            )
            residual_features = compute_residual_feature_batch_fn(
                image_features,
                mini_class_idx.to(device),
                reference_memory_by_class,
                reference_index_by_class,
            )
            patch_class_idx = None
            if args.use_moe_discriminator and getattr(args, "moe_hard_class_gate", False):
                patch_class_idx = _build_patch_class_idx(mini_class_idx, residual_features)
            features, _ = localnet(
                residual_features, cls_token=cls_token, patch_class_idx=patch_class_idx
            )
            features_np = features.detach().cpu().numpy()
            batch_class_np = mini_class_idx.detach().cpu().numpy().astype(np.int64)
            batch_size = int(batch_class_np.shape[0])
            feature_l2 = np.linalg.norm(
                features_np.reshape(-1, global_dim), ord=2, axis=1
            ).astype(np.float32).reshape(batch_size, 784)

            # 将距离按 batch 原顺序拼回：shape = (B, 784)
            cls_distance_map = np.zeros((batch_size, 784), dtype=np.float32)

            all_scores = []
            for scorer in scorers:
                score_map = np.zeros((batch_size, 784), dtype=np.float32)
                if getattr(args, "global_memory_bank", False):
                    # 全局记忆库：所有 patch 对全局 scorer(class 0) 打分
                    if scorer.has_class(0):
                        features_2d = features_np.reshape(-1, global_dim)
                        global_scores = scorer.score_patches(0, features_2d)
                        score_map = global_scores.reshape(batch_size, 784).astype(np.float32)
                else:
                    for cls in np.unique(batch_class_np).tolist():
                        cls = int(cls)
                        if not scorer.has_class(cls):
                            continue
                        cls_mask = batch_class_np == cls
                        if not np.any(cls_mask):
                            continue
                        cls_features_2d = features_np[cls_mask].reshape(-1, global_dim)
                        cls_score = scorer.score_patches(cls, cls_features_2d)
                        score_map[cls_mask] = cls_score.reshape(-1, 784).astype(np.float32)
                all_scores.append(score_map)
            cls_distance_map = np.mean(all_scores, axis=0).astype(np.float32)

            distance_map[sample_idx_np] = cls_distance_map.astype(np.float32)
            patch_feature_l2_map[sample_idx_np] = feature_l2

            if args.beta:
                high_sample_mask = image_norm_scores[sample_idx_np] > 0.5
                if np.any(high_sample_mask):
                    high_patch_mask = np.repeat(high_sample_mask, 784)
                    features_2d = features_np.reshape(-1, global_dim)
                    cand_feat = features_2d[high_patch_mask]
                    cand_dist = cls_distance_map.reshape(-1)[high_patch_mask]
                    top_feat_global, top_dist_global = update_topk_features_fn(
                        top_feat_global,
                        top_dist_global,
                        cand_feat,
                        cand_dist,
                        args.beta_number,
                    )

    if not getattr(args, "global_memory_bank", False):
        _plot_classwise_abs_distance_box_before_norm(
            args=args,
            class_stack=class_stack,
            distance_map=distance_map,
            gt_patch_masks=gt_patch_masks,
            num_classes=num_classes,
            class_names=getattr(args, "class_names", None),
            epoch=epoch,
        )

    if getattr(args, "global_memory_bank", False):
        distance_map = _normalize_distance_map_global(
            distance_map=distance_map,
            args=args,
        )
    else:
        distance_map = _normalize_distance_map_per_class(
            distance_map=distance_map,
            class_stack=class_stack,
            num_classes=num_classes,
            args=args,
        )

    if args.beta:
        if top_feat_global is None or top_feat_global.shape[0] == 0:
            confident_feature_bank = torch.zeros((0, global_dim), dtype=torch.float32)
        else:
            confident_feature_bank = torch.as_tensor(top_feat_global, dtype=torch.float32)

    # 统计伪标签分配准确率（基于真实 patch mask）
    values = distance_map.astype(np.float32)
    gt_mask_bool = gt_patch_masks.astype(bool)

    pred_normal = values < float(args.threshold)
    pred_anomaly = values > float(args.noise_threshold)
    pred_uncertain = np.logical_not(np.logical_or(pred_normal, pred_anomaly))

    pred_normal_count = int(pred_normal.sum())
    pred_anomaly_count = int(pred_anomaly.sum())
    pred_normal_correct = int(np.logical_and(pred_normal, ~gt_mask_bool).sum())
    pred_anomaly_correct = int(np.logical_and(pred_anomaly, gt_mask_bool).sum())

    normal_precision = (
        float(pred_normal_correct) / float(pred_normal_count)
        if pred_normal_count > 0
        else 0.0
    )
    anomaly_precision = (
        float(pred_anomaly_correct) / float(pred_anomaly_count)
        if pred_anomaly_count > 0
        else 0.0
    )
    print(
        "[PseudoLabel-Stats] "
        f"distance<threshold normal-precision: {normal_precision:.4f} "
        f"({pred_normal_correct}/{pred_normal_count}) | "
        f"distance>noise_threshold anomaly-precision: {anomaly_precision:.4f} "
        f"({pred_anomaly_correct}/{pred_anomaly_count})"
    )

    if not getattr(args, "global_memory_bank", False):
        classwise_region_gt_means = np.full((num_classes, 3, 2), np.nan, dtype=np.float32)
        classwise_region_gt_stds = np.full((num_classes, 3, 2), np.nan, dtype=np.float32)
        classwise_region_gt_medians = np.full((num_classes, 3, 2), np.nan, dtype=np.float32)
        classwise_region_gt_counts = np.zeros((num_classes, 3, 2), dtype=np.int64)
        region_masks = (pred_normal, pred_uncertain, pred_anomaly)
        region_names = ("distance<threshold", "threshold<=distance<=noise_threshold", "distance>noise_threshold")
        gt_names = ("gt_normal", "gt_anomaly")

        for cls in range(num_classes):
            cls_idx = np.where(class_stack == cls)[0]
            if cls_idx.size == 0:
                continue
            cls_norm = patch_feature_l2_map[cls_idx]
            cls_gt = gt_mask_bool[cls_idx]
            for ridx, rmask in enumerate(region_masks):
                cls_region = rmask[cls_idx]
                normal_sel = np.logical_and(cls_region, np.logical_not(cls_gt))
                anomaly_sel = np.logical_and(cls_region, cls_gt)
                for gidx, sel in enumerate((normal_sel, anomaly_sel)):
                    cnt = int(sel.sum())
                    classwise_region_gt_counts[cls, ridx, gidx] = cnt
                    if cnt > 0:
                        selected_vals = cls_norm[sel].astype(np.float32)
                        classwise_region_gt_means[cls, ridx, gidx] = float(selected_vals.mean())
                        classwise_region_gt_stds[cls, ridx, gidx] = float(selected_vals.std())
                        classwise_region_gt_medians[cls, ridx, gidx] = float(
                            np.median(selected_vals)
                        )

        class_names = getattr(args, "class_names", None)
        for cls in range(num_classes):
            cls_name = (
                class_names[cls]
                if isinstance(class_names, (list, tuple)) and cls < len(class_names)
                else f"cls_{cls}"
            )
            stat_parts = []
            for ridx, rname in enumerate(region_names):
                for gidx, gname in enumerate(gt_names):
                    m = classwise_region_gt_means[cls, ridx, gidx]
                    s = classwise_region_gt_stds[cls, ridx, gidx]
                    med = classwise_region_gt_medians[cls, ridx, gidx]
                    n = int(classwise_region_gt_counts[cls, ridx, gidx])
                    if np.isfinite(m):
                        stat_parts.append(
                            f"{rname}/{gname}: mean={m:.4f}, std={s:.4f}, median={med:.4f} (n={n})"
                        )
                    else:
                        stat_parts.append(f"{rname}/{gname}: NA (n=0)")
            print(f"[PseudoLabel-L2Norm] {cls_name} | " + " | ".join(stat_parts))

        _plot_classwise_feature_l2_by_distance_regions(
            args=args,
            class_names=class_names,
            region_gt_means=classwise_region_gt_means,
            epoch=epoch,
        )

    _plot_distance_distribution_normal_vs_anomaly(
        args=args,
        distance_values=values,
        gt_patch_masks=gt_patch_masks,
        epoch=epoch,
    )

    pseudo_label_time = time.perf_counter() - pseudo_start
    return distance_map, confident_feature_bank, global_dim, memory_bank_time, pseudo_label_time
