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

from src.train.pseudo_label_scorers import (
    MahalanobisPseudoLabelScorer,
    NNPseudoLabelScorer,
    PCAPseudoLabelScorer,
    PseudoLabelScorer,
    postprocess_faiss_distances,
)


def _pseudo_label_blend_weights(args) -> tuple:
    w_nn = float(getattr(args, "pseudo_label_blend_nn_weight", 0.5))
    w_m = float(getattr(args, "pseudo_label_blend_maha_weight", 0.5))
    s = w_nn + w_m
    if s <= 0.0:
        return 0.5, 0.5
    return w_nn / s, w_m / s


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


def _indices_lowest_score_quantile(scores: np.ndarray, quantile: float) -> np.ndarray:
    """Indices into ``scores`` (0..n-1) for the lowest ceil(n * quantile) scores (at least one)."""
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    n = int(scores.shape[0])
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    q = float(quantile)
    q = min(max(q, 1e-6), 1.0)
    k = max(1, int(np.ceil(n * q)))
    k = min(k, n)
    order = np.argsort(scores, kind="mergesort")
    return order[:k].astype(np.int64)


def _normalize_distance_map_per_class(
    distance_map: np.ndarray, class_stack: np.ndarray, num_classes: int, args
) -> np.ndarray:
    mode = str(getattr(args, "pseudo_label_distance_norm", "minmax")).strip().lower()
    eps = float(getattr(args, "pseudo_label_distance_norm_eps", 1e-6))
    robust_scale = float(
        getattr(args, "pseudo_label_distance_norm_robust_scale", 1.4826)
    )
    values = np.asarray(distance_map, dtype=np.float32)
    normalized = np.zeros_like(values, dtype=np.float32)
    has_valid_class = False

    for cls in range(num_classes):
        cls_indices = np.where(class_stack == cls)[0]
        if cls_indices.size == 0:
            continue
        cls_values = values[cls_indices]

        if mode == "robust_mad":
            cls_flat = cls_values.reshape(-1)
            cls_med = float(np.median(cls_flat))
            cls_mad = float(np.median(np.abs(cls_flat - cls_med)))
            cls_scale = robust_scale * cls_mad
            if (not np.isfinite(cls_scale)) or (cls_scale <= eps):
                normalized[cls_indices] = 0.0
                continue
            z = (cls_values - cls_med) / (cls_scale + eps)
            normalized[cls_indices] = z.astype(np.float32)
            has_valid_class = True
            continue

        cls_min = float(cls_values.min())
        cls_max = float(cls_values.max())
        if (
            (not np.isfinite(cls_min))
            or (not np.isfinite(cls_max))
            or (cls_max <= cls_min)
        ):
            normalized[cls_indices] = 0.0
            continue
        normalized[cls_indices] = (cls_values - cls_min) / (cls_max - cls_min)
        has_valid_class = True

    if not has_valid_class:
        return np.zeros_like(values, dtype=np.float32)
    return normalized.astype(np.float32)


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


def precompute_pseudo_labels_feature(
    args,
    localnet,
    mini_loader,
    device,
    update_topk_features_fn,
):
    memory_bank_start = time.perf_counter()

    dataset_size = len(mini_loader.dataset)
    score_stack = np.zeros(dataset_size, dtype=np.float32)
    dim = None

    with torch.no_grad():
        localnet.eval()
        for mini_x, sample_idx in mini_loader:
            features, score = localnet(mini_x.to(device))
            if features.shape[0] > args.batch_size:
                features = features.unsqueeze(0)

            if dim is None:
                dim = int(features.shape[-1])

            score = score.detach().cpu().numpy()
            score = _as_1d_float32(score.max(axis=-1))

            sample_idx_np = sample_idx.detach().cpu().numpy().astype(np.int64)
            score_stack[sample_idx_np] = score.astype(np.float32)

    if dim is None:
        raise RuntimeError("未能从 mini_loader 中获取特征维度。")

    normalized_score = _minmax_normalize(score_stack)

    q = float(getattr(args, "memory_bank_score_quantile", 0.1))
    selected_normal_indices = _indices_lowest_score_quantile(normalized_score, q)
    selected_normal_indices = np.asarray(selected_normal_indices, dtype=np.int64)

    normal_features = []
    with torch.no_grad():
        localnet.eval()
        for mini_x, sample_idx in mini_loader:
            features, _ = localnet(mini_x.to(device))
            features = features.detach().cpu().numpy()
            sample_idx_np = sample_idx.detach().cpu().numpy().astype(np.int64)

            keep_mask = np.isin(sample_idx_np, selected_normal_indices)
            if np.any(keep_mask):
                normal_features.append(features[keep_mask].reshape(-1, dim))

    if len(normal_features) == 0:
        raise RuntimeError(
            "构建 memory bank 时未选中任何 normal 特征，请检查 memory_bank_score_quantile 与分数归一化。"
        )

    normal_features = np.concatenate(normal_features, axis=0)

    faiss.omp_set_num_threads(4)
    index = faiss.GpuIndexFlatL2(
        faiss.StandardGpuResources(),
        dim,
        faiss.GpuIndexFlatConfig(),
    )
    index.add(normal_features)
    memory_bank_time = time.perf_counter() - memory_bank_start

    pseudo_start = time.perf_counter()
    distance_map = np.zeros((dataset_size, 784), dtype=np.float32)
    global_min = np.inf
    global_max = -np.inf

    top_feat = None
    top_dist = None
    with torch.no_grad():
        localnet.eval()
        for mini_x, sample_idx in mini_loader:
            features, _ = localnet(mini_x.to(device))
            features = features.detach().cpu().numpy()
            sample_idx_np = sample_idx.detach().cpu().numpy().astype(np.int64)

            batch_features = features.reshape(-1, dim)
            distance, _ = index.search(np.ascontiguousarray(batch_features), k=args.k_number)
            distance = postprocess_faiss_distances(distance, args.k_number)
            if distance.size > 0:
                global_min = min(global_min, float(distance.min()))
                global_max = max(global_max, float(distance.max()))
            distance_map[sample_idx_np] = distance.reshape(-1, 784)

            if args.beta:
                high_sample_mask = normalized_score[sample_idx_np] > 0.5
                if np.any(high_sample_mask):
                    high_patch_mask = np.repeat(high_sample_mask, 784)
                    cand_feat = batch_features[high_patch_mask]
                    cand_dist = distance[high_patch_mask]
                    top_feat, top_dist = update_topk_features_fn(
                        top_feat, top_dist, cand_feat, cand_dist, args.beta_number
                    )

    if not np.isfinite(global_min) or not np.isfinite(global_max) or global_max == global_min:
        distance_map = np.zeros_like(distance_map)
    else:
        distance_map = (distance_map - global_min) / (global_max - global_min)

    pseudo_label_time = time.perf_counter() - pseudo_start

    confident_features = None
    if args.beta:
        if top_feat is None or top_feat.shape[0] == 0:
            confident_features = torch.zeros((0, dim), dtype=torch.float32)
        else:
            confident_features = torch.as_tensor(top_feat, dtype=torch.float32)

    return distance_map, confident_features, dim, memory_bank_time, pseudo_label_time


def precompute_pseudo_labels_multiclass(
    args,
    localnet,
    feature_extractor,
    mini_loader,
    num_classes,
    device,
    extract_feature_batch_fn,
    aggregate_image_scores_fn,
    build_faiss_index_fn,
    update_topk_features_fn,
    print_selected_score_distribution_fn=None,
    print_selected_clean_ratio_fn=None,
    print_confusion_matrix_fn=None,
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
            patch_class_idx = None
            if args.use_moe_discriminator and getattr(args, "moe_hard_class_gate", False):
                patch_class_idx = _build_patch_class_idx(mini_class_idx, image_features)
            features, score = localnet(
                image_features, cls_token=cls_token, patch_class_idx=patch_class_idx
            )
            if features.shape[0] > args.batch_size:
                features = features.unsqueeze(0)

            if global_dim is None:
                global_dim = int(features.shape[-1])

            score = score.detach().cpu().numpy()
            score = _as_1d_float32(
                aggregate_image_scores_fn(score, topk_ratio=args.img_score_topk_ratio)
            )

            sample_idx_np = mini_sample_idx.detach().cpu().numpy().astype(np.int64)
            class_np = mini_class_idx.detach().cpu().numpy().astype(np.int64)
            image_scores[sample_idx_np] = score.astype(np.float32)
            class_stack[sample_idx_np] = class_np
            if mini_patch_mask is not None:
                mask_np = mini_patch_mask.detach().cpu().numpy()
                gt_patch_masks[sample_idx_np] = (mask_np > 0).astype(np.uint8)

    if global_dim is None:
        raise RuntimeError("未能从 mini_loader 获取到特征维度。")

    image_norm_scores = np.zeros(dataset_size, dtype=np.float32)
    normalized_score = np.zeros_like(image_scores, dtype=np.float32)
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
    for cls in range(num_classes):
        cls_mask = class_stack == cls
        if not np.any(cls_mask):
            continue
        cls_indices = np.where(cls_mask)[0].astype(np.int64)
        cls_norm_scores = normalized_score[cls_indices]

        q = float(getattr(args, "memory_bank_score_quantile", 0.1))
        cls_normal_local = _indices_lowest_score_quantile(cls_norm_scores, q)
        cls_selected = cls_indices[cls_normal_local]
        selected_local_list.append(cls_selected)

    if len(selected_local_list) == 0:
        selected_indices = np.arange(dataset_size, dtype=np.int64)
    else:
        selected_indices = np.concatenate(selected_local_list, axis=0).astype(np.int64)

    if print_selected_score_distribution_fn is not None:
        print_selected_score_distribution_fn(
            selected_indices=selected_indices,
            class_stack=class_stack,
            image_scores=image_scores,
        )
    if print_selected_clean_ratio_fn is not None:
        print_selected_clean_ratio_fn(
            selected_indices=selected_indices,
            dataset=mini_loader.dataset,
        )
    if print_confusion_matrix_fn is not None:
        print_confusion_matrix_fn(
            selected_indices=selected_indices,
            class_stack=class_stack,
            gt_patch_masks=gt_patch_masks,
            image_norm_scores=image_norm_scores,
        )

    distance_map = np.zeros((dataset_size, 784), dtype=np.float16)
    confident_feature_bank = torch.zeros((0, global_dim), dtype=torch.float32)
    if selected_indices.shape[0] == 0:
        pseudo_label_time = time.perf_counter() - memory_bank_start
        return distance_map, confident_feature_bank, global_dim, 0.0, pseudo_label_time

    selected_images_by_class = {}
    for cls in range(num_classes):
        cls_mask = class_stack[selected_indices] == cls
        selected_images_by_class[cls] = selected_indices[cls_mask]

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
            num_workers=mini_loader.num_workers,
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
            patch_class_idx = None
            if args.use_moe_discriminator and getattr(args, "moe_hard_class_gate", False):
                patch_class_idx = _build_patch_class_idx(mini_class_idx, image_features)
            features, _ = localnet(
                image_features, cls_token=cls_token, patch_class_idx=patch_class_idx
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
    for cls in range(num_classes):
        if len(class_feature_buffers[cls]) == 0:
            continue
        normal_features_cls = np.concatenate(class_feature_buffers[cls], axis=0)

        num_selected_images_cls = selected_images_by_class.get(
            cls, np.array([], dtype=np.int64)
        ).shape[0]
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

    if len(reduced_features_by_class) > 0:
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
    nn_scorer: Optional[NNPseudoLabelScorer] = None
    maha_scorer: Optional[MahalanobisPseudoLabelScorer] = None
    scorer: Optional[PseudoLabelScorer] = None

    if scoring == "blend":
        nn_scorer = NNPseudoLabelScorer(build_faiss_index_fn, args.k_number)
        maha_scorer = MahalanobisPseudoLabelScorer(
            device,
            gaussian_feature_dim=int(getattr(args, "pseudo_label_mahalanobis_dim", 128)),
        )
        for cls, feats in reduced_features_by_class.items():
            if feats is None or feats.shape[0] == 0:
                continue
            c = int(cls)
            nn_scorer.fit_class(c, feats)
            maha_scorer.fit_class(c, feats)
    elif scoring == "mahalanobis":
        scorer = pseudo_label_scorer or MahalanobisPseudoLabelScorer(
            device,
            gaussian_feature_dim=int(getattr(args, "pseudo_label_mahalanobis_dim", 128)),
        )
        for cls, feats in reduced_features_by_class.items():
            if feats is None or feats.shape[0] == 0:
                continue
            scorer.fit_class(int(cls), feats)
    elif scoring == "pca":
        scorer = pseudo_label_scorer or PCAPseudoLabelScorer(
            pca_dim=int(getattr(args, "pseudo_label_pca_dim", 0)),
            pca_ev=float(getattr(args, "pseudo_label_pca_ev", 0.99)),
            eps=float(getattr(args, "pseudo_label_pca_eps", 1e-6)),
        )
        for cls, feats in reduced_features_by_class.items():
            if feats is None or feats.shape[0] == 0:
                continue
            scorer.fit_class(int(cls), feats)
    else:
        scorer = (
            pseudo_label_scorer
            if pseudo_label_scorer is not None
            else NNPseudoLabelScorer(build_faiss_index_fn, args.k_number)
        )
        for cls, feats in reduced_features_by_class.items():
            if feats is None or feats.shape[0] == 0:
                continue
            scorer.fit_class(int(cls), feats)

    blend_w_nn, blend_w_m = _pseudo_label_blend_weights(args)
    distance_nn_buf: Optional[np.ndarray] = (
        np.zeros((dataset_size, 784), dtype=np.float32) if scoring == "blend" else None
    )
    distance_maha_buf: Optional[np.ndarray] = (
        np.zeros((dataset_size, 784), dtype=np.float32) if scoring == "blend" else None
    )

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
            patch_class_idx = None
            if args.use_moe_discriminator and getattr(args, "moe_hard_class_gate", False):
                patch_class_idx = _build_patch_class_idx(mini_class_idx, image_features)
            features, _ = localnet(
                image_features, cls_token=cls_token, patch_class_idx=patch_class_idx
            )
            features_np = features.detach().cpu().numpy()
            batch_class_np = mini_class_idx.detach().cpu().numpy().astype(np.int64)
            batch_size = int(batch_class_np.shape[0])

            cls_distance_map = np.zeros((batch_size, 784), dtype=np.float32)
            if scoring == "blend":
                assert nn_scorer is not None and maha_scorer is not None
                assert distance_nn_buf is not None and distance_maha_buf is not None
                cls_nn = np.zeros((batch_size, 784), dtype=np.float32)
                cls_maha = np.zeros((batch_size, 784), dtype=np.float32)
                for cls in np.unique(batch_class_np).tolist():
                    cls = int(cls)
                    if not nn_scorer.has_class(cls) or not maha_scorer.has_class(cls):
                        continue
                    cls_mask = batch_class_np == cls
                    if not np.any(cls_mask):
                        continue
                    cls_features_2d = features_np[cls_mask].reshape(-1, global_dim)
                    d_nn = nn_scorer.score_patches(cls, cls_features_2d)
                    d_m = maha_scorer.score_patches(cls, cls_features_2d)
                    cls_nn[cls_mask] = d_nn.reshape(-1, 784).astype(np.float32)
                    cls_maha[cls_mask] = d_m.reshape(-1, 784).astype(np.float32)
                cls_distance_map = blend_w_nn * cls_nn + blend_w_m * cls_maha
                distance_nn_buf[sample_idx_np] = cls_nn.astype(np.float32)
                distance_maha_buf[sample_idx_np] = cls_maha.astype(np.float32)
            else:
                assert scorer is not None
                for cls in np.unique(batch_class_np).tolist():
                    cls = int(cls)
                    if not scorer.has_class(cls):
                        continue
                    cls_mask = batch_class_np == cls
                    if not np.any(cls_mask):
                        continue
                    cls_features_2d = features_np[cls_mask].reshape(-1, global_dim)
                    cls_distance = scorer.score_patches(cls, cls_features_2d)
                    cls_distance_map[cls_mask] = cls_distance.reshape(-1, 784).astype(
                        np.float32
                    )

            distance_map[sample_idx_np] = cls_distance_map.astype(np.float32)

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

    if scoring == "blend":
        assert distance_nn_buf is not None and distance_maha_buf is not None

    _plot_classwise_abs_distance_box_before_norm(
        args=args,
        class_stack=class_stack,
        distance_map=distance_map,
        gt_patch_masks=gt_patch_masks,
        num_classes=num_classes,
        class_names=getattr(args, "class_names", None),
        epoch=epoch,
    )

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

    values = distance_map.astype(np.float32)
    gt_mask_bool = gt_patch_masks.astype(bool)
    pred_normal = values < float(args.threshold)
    pred_anomaly = values > float(args.noise_threshold)
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
    _plot_distance_distribution_normal_vs_anomaly(
        args=args,
        distance_values=values,
        gt_patch_masks=gt_patch_masks,
        epoch=epoch,
    )

    pseudo_label_time = time.perf_counter() - pseudo_start
    return distance_map, confident_feature_bank, global_dim, memory_bank_time, pseudo_label_time


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
    for cls in range(num_classes):
        cls_mask = class_stack == cls
        if not np.any(cls_mask):
            continue
        cls_indices = np.where(cls_mask)[0].astype(np.int64)
        cls_norm_scores = normalized_score[cls_indices]

        q = float(getattr(args, "memory_bank_score_quantile", 0.1))
        cls_normal_local = _indices_lowest_score_quantile(cls_norm_scores, q)
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

    # 按类别做 GreedyCoreset，下采样到「约等于 2 张图片的 patch 数」
    reduced_features_by_class = {}
    coreset_indices_by_class = {}
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

    if len(reduced_features_by_class) > 0:
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
    nn_scorer: Optional[NNPseudoLabelScorer] = None
    maha_scorer: Optional[MahalanobisPseudoLabelScorer] = None
    scorer: Optional[PseudoLabelScorer] = None

    if scoring == "blend":
        nn_scorer = NNPseudoLabelScorer(build_faiss_index_fn, args.k_number)
        maha_scorer = MahalanobisPseudoLabelScorer(
            device,
            gaussian_feature_dim=int(getattr(args, "pseudo_label_mahalanobis_dim", 128)),
        )
        for cls, feats in reduced_features_by_class.items():
            if feats is None or feats.shape[0] == 0:
                continue
            c = int(cls)
            nn_scorer.fit_class(c, feats)
            maha_scorer.fit_class(c, feats)
    elif scoring == "mahalanobis":
        scorer = pseudo_label_scorer or MahalanobisPseudoLabelScorer(
            device,
            gaussian_feature_dim=int(getattr(args, "pseudo_label_mahalanobis_dim", 128)),
        )
        for cls, feats in reduced_features_by_class.items():
            if feats is None or feats.shape[0] == 0:
                continue
            scorer.fit_class(int(cls), feats)
    elif scoring == "pca":
        scorer = pseudo_label_scorer or PCAPseudoLabelScorer(
            pca_dim=int(getattr(args, "pseudo_label_pca_dim", 0)),
            pca_ev=float(getattr(args, "pseudo_label_pca_ev", 0.99)),
            eps=float(getattr(args, "pseudo_label_pca_eps", 1e-6)),
        )
        for cls, feats in reduced_features_by_class.items():
            if feats is None or feats.shape[0] == 0:
                continue
            scorer.fit_class(int(cls), feats)
    else:
        scorer = (
            pseudo_label_scorer
            if pseudo_label_scorer is not None
            else NNPseudoLabelScorer(build_faiss_index_fn, args.k_number)
        )
        for cls, feats in reduced_features_by_class.items():
            if feats is None or feats.shape[0] == 0:
                continue
            scorer.fit_class(int(cls), feats)

    blend_w_nn, blend_w_m = _pseudo_label_blend_weights(args)
    distance_nn_buf: Optional[np.ndarray] = (
        np.zeros((dataset_size, 784), dtype=np.float32) if scoring == "blend" else None
    )
    distance_maha_buf: Optional[np.ndarray] = (
        np.zeros((dataset_size, 784), dtype=np.float32) if scoring == "blend" else None
    )

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

            if scoring == "blend":
                assert nn_scorer is not None and maha_scorer is not None
                assert distance_nn_buf is not None and distance_maha_buf is not None
                cls_nn = np.zeros((batch_size, 784), dtype=np.float32)
                cls_maha = np.zeros((batch_size, 784), dtype=np.float32)
                for cls in np.unique(batch_class_np).tolist():
                    cls = int(cls)
                    if not nn_scorer.has_class(cls) or not maha_scorer.has_class(cls):
                        continue
                    cls_mask = batch_class_np == cls
                    if not np.any(cls_mask):
                        continue
                    cls_features_2d = features_np[cls_mask].reshape(-1, global_dim)
                    d_nn = nn_scorer.score_patches(cls, cls_features_2d)
                    d_m = maha_scorer.score_patches(cls, cls_features_2d)
                    cls_nn[cls_mask] = d_nn.reshape(-1, 784).astype(np.float32)
                    cls_maha[cls_mask] = d_m.reshape(-1, 784).astype(np.float32)
                cls_distance_map = blend_w_nn * cls_nn + blend_w_m * cls_maha
                distance_nn_buf[sample_idx_np] = cls_nn.astype(np.float32)
                distance_maha_buf[sample_idx_np] = cls_maha.astype(np.float32)
            else:
                assert scorer is not None
                for cls in np.unique(batch_class_np).tolist():
                    cls = int(cls)
                    if not scorer.has_class(cls):
                        continue
                    cls_mask = batch_class_np == cls
                    if not np.any(cls_mask):
                        continue

                    cls_features_2d = features_np[cls_mask].reshape(-1, global_dim)
                    cls_distance = scorer.score_patches(cls, cls_features_2d)

                    cls_distance_map[cls_mask] = cls_distance.reshape(-1, 784).astype(
                        np.float32
                    )

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

    if scoring == "blend":
        assert distance_nn_buf is not None and distance_maha_buf is not None

    _plot_classwise_abs_distance_box_before_norm(
        args=args,
        class_stack=class_stack,
        distance_map=distance_map,
        gt_patch_masks=gt_patch_masks,
        num_classes=num_classes,
        class_names=getattr(args, "class_names", None),
        epoch=epoch,
    )

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
