import os
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader, Subset

from src.train.pseudo_label_normalizers import PseudoLabelNormalizerFactory



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


def _top_percent_normalize(values: np.ndarray, top_ratio: float = 0.01) -> np.ndarray:
    """取最高的 top_ratio 比例的分数做 min-max 归一化。

    1. 从 values 中取出最高的 top_ratio 个分数作为子集
    2. 用这个子集的最小值和最大值对所有 values 做 min-max
    3. 低于子集最小值的 clamp 到 0
    """
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    n_top = max(1, int(values.size * top_ratio))
    # 降序排列取前 n_top 个
    top_subset = np.sort(values)[-n_top:]
    vmin = float(top_subset.min())
    vmax = float(top_subset.max())
    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or vmax <= vmin:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0).astype(np.float32)


# ────────────────────────────────────────────────────────────────────────
#  Ensemble PCA helpers (pure NumPy, no PyTorch dependency)
# ────────────────────────────────────────────────────────────────────────

def _select_pca_k(eigvals: np.ndarray, pca_dim: int, pca_ev: float, pca_eps: float) -> int:
    """Select number of principal components to retain.

    Args:
        eigvals: 1-D eigenvalues in descending order.
        pca_dim: If > 0, use this many components directly.
        pca_ev:  Explained-variance target (fraction of total) when pca_dim == 0.
        pca_eps: Numerical stability epsilon.

    Returns:
        Number of components k >= 1.
    """
    d = int(eigvals.shape[0])
    if d <= 0:
        return 1
    if pca_dim > 0:
        return max(1, min(pca_dim, d))
    ev_target = min(max(pca_ev, 0.0), 1.0)
    total = float(eigvals.sum())
    if total <= pca_eps:
        return 1
    cumsum = np.cumsum(eigvals)
    ratio = cumsum / (total + pca_eps)
    idx = int(np.searchsorted(ratio, ev_target))
    return max(1, min(idx + 1, d))


def _build_ensemble_pca(
    features: np.ndarray,
    ensemble_size: int,
    sampling_ratio: float,
    pca_ev: float,
    pca_dim: int,
    pca_eps: float,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Build an ensemble of PCA models from the given features.

    Each iteration: randomly sample a subset → center → SVD → select k components.
    Returns a list of (mean, components) tuples.
    """
    n_total = int(features.shape[0])
    if n_total < 2:
        return []  # degenerate case — cannot build PCA with < 2 points
    sample_size = max(2, int(n_total * sampling_ratio))
    if sample_size >= n_total:
        sample_size = max(2, n_total // 2)

    models: List[Tuple[np.ndarray, np.ndarray]] = []
    for _ in range(ensemble_size):
        idx = np.random.choice(n_total, size=sample_size, replace=False)
        subset = features[idx].astype(np.float64)
        mean = np.mean(subset, axis=0)
        centered = subset - mean
        # Always use covariance eigendecomposition (robust for both N >= D and N < D)
        n = centered.shape[0]
        cov = centered.T.dot(centered) / float(max(n - 1, 1))
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.clip(eigvals, a_min=pca_eps, a_max=None)
        eigvals_desc = eigvals[::-1]
        eigvecs_desc = eigvecs[:, ::-1]

        k = _select_pca_k(eigvals_desc, pca_dim, pca_ev, pca_eps)
        mean_float32 = mean.astype(np.float32)
        components = eigvecs_desc[:, :k].astype(np.float32)
        models.append((mean_float32, components))
    return models


def _ensemble_pca_score_patches(
    queries: np.ndarray,
    ensemble_models: List[Tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Score query patches using an ensemble of PCA models.

    For each PCA model (mean, components):
        score_i = ||(q - mean) - ((q - mean) @ components @ components.T)||^2
    Final score for each query = mean of scores across the ensemble.

    Args:
        queries: [M, D] float32
        ensemble_models: list of (mean, components) tuples

    Returns:
        [M] float32, mean reconstruction error across the ensemble.
    """
    if not ensemble_models:
        return np.zeros(int(queries.shape[0]), dtype=np.float32)

    all_scores = []
    for mean, comp in ensemble_models:
        centered = queries - mean
        proj = centered.dot(comp).dot(comp.T)
        residual = centered - proj
        score = np.sum(residual * residual, axis=1)
        all_scores.append(score)

    return np.mean(all_scores, axis=0).astype(np.float32)


# ────────────────────────────────────────────────────────────────────────
#  Normalization helpers
# ────────────────────────────────────────────────────────────────────────

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
    extract_feature_batch_fn, images, feature_extractor, args, class_indices=None
):
    try:
        features, cls_token = extract_feature_batch_fn(
            images, feature_extractor, args, return_cls_token=True,
            class_indices=class_indices,
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


# ────────────────────────────────────────────────────────────────────────
#  Plotting helpers (unchanged from original)
# ────────────────────────────────────────────────────────────────────────

def _plot_distance_distribution_normal_vs_anomaly(
    args, distance_values, gt_patch_masks, epoch=None,
    class_stack=None, num_classes=None, class_names=None,
    save_dir=None,
):
    values = np.asarray(distance_values, dtype=np.float32).reshape(-1)
    labels = np.asarray(gt_patch_masks, dtype=np.uint8).reshape(-1) > 0
    if values.size == 0 or labels.size != values.size:
        return

    # Keep plotting lightweight even for large datasets.
    max_points = 200000

    def _plot_single_hist(values, labels, save_path, title_suffix=""):
        normal_values = values[~labels]
        anomaly_values = values[labels]
        if normal_values.size == 0 and anomaly_values.size == 0:
            return False

        _normal_vals = normal_values
        _anomaly_vals = anomaly_values
        if _normal_vals.size > max_points:
            _normal_vals = np.random.choice(_normal_vals, size=max_points, replace=False)
        if _anomaly_vals.size > max_points:
            _anomaly_vals = np.random.choice(_anomaly_vals, size=max_points, replace=False)

        plt.figure(figsize=(10, 6))
        bins = np.linspace(0.0, 1.0, 101)
        if _normal_vals.size > 0:
            plt.hist(
                _normal_vals,
                bins=bins,
                density=True,
                alpha=0.55,
                color="tab:blue",
                label=f"normal ({_normal_vals.size})",
            )
        if _anomaly_vals.size > 0:
            plt.hist(
                _anomaly_vals,
                bins=bins,
                density=True,
                alpha=0.55,
                color="tab:red",
                label=f"anomaly ({_anomaly_vals.size})",
            )

        plt.xlabel("Normalized distance")
        plt.ylabel("Density")
        plt.title(f"Distance Distribution: Normal vs Anomaly Patches{title_suffix}")
        plt.xlim(0.0, 1.0)
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"[PseudoLabel-Plot] saved: {save_path}")
        return True

    if save_dir is None:
        save_dir = os.path.join(args.save_path, args.dataset, args.noise)
    # Put epoch plots in a subfolder
    if epoch is not None:
        save_dir = os.path.join(save_dir, f"epoch_{int(epoch) + 1:03d}")
    os.makedirs(save_dir, exist_ok=True)

    if epoch is None:
        filename = "distance_distribution_normal_vs_anomaly.png"
    else:
        filename = "distance_distribution_normal_vs_anomaly.png"
    save_path = os.path.join(save_dir, filename)
    _plot_single_hist(values, labels, save_path)

    # Per-class plots (skipped in global memory bank mode)
    if (
        not getattr(args, "global_memory_bank", False)
        and class_stack is not None
        and num_classes is not None
    ):
        cls_arr = np.asarray(class_stack, dtype=np.int64).reshape(-1)
        if cls_arr.shape[0] == values.shape[0]:
            name_list = list(class_names) if class_names is not None else []
            if len(name_list) != int(num_classes):
                name_list = [f"cls_{i}" for i in range(int(num_classes))]

            for cls in range(int(num_classes)):
                cls_idx = np.where(cls_arr == cls)[0]
                if cls_idx.size == 0:
                    continue
                cls_values = values[cls_idx]
                cls_labels = labels[cls_idx]
                cls_save_path = os.path.join(
                    save_dir, f"distance_distribution_{name_list[cls]}.png"
                )
                _plot_single_hist(
                    cls_values, cls_labels, cls_save_path,
                    title_suffix=f" — {name_list[cls]}",
                )


def _plot_image_raw_score_distribution_normal_vs_anomaly(
    args, image_raw_scores, gt_patch_masks, epoch=None,
    class_stack=None, num_classes=None, class_names=None,
    save_dir=None,
):
    scores = np.asarray(image_raw_scores, dtype=np.float32).reshape(-1)
    patch_masks = np.asarray(gt_patch_masks, dtype=np.uint8)
    if scores.size == 0 or patch_masks.ndim != 2 or patch_masks.shape[0] != scores.size:
        return

    # Image is anomaly if any patch is anomaly.
    image_is_anomaly = patch_masks.reshape(patch_masks.shape[0], -1).sum(axis=1) > 0

    def _plot_single_hist(scores, image_is_anomaly, save_path, title_suffix=""):
        normal_scores = scores[~image_is_anomaly]
        anomaly_scores = scores[image_is_anomaly]
        if normal_scores.size == 0 and anomaly_scores.size == 0:
            return False

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

        plt.xlabel("Image raw score")
        plt.ylabel("Density")
        plt.title(f"Image Raw Score Distribution: Normal vs Anomaly Images{title_suffix}")
        plt.xlim(0.0, 1.0)
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"[PseudoLabel-Plot] saved: {save_path}")
        return True

    if save_dir is None:
        save_dir = os.path.join(args.save_path, args.dataset, args.noise)
    if epoch is not None:
        save_dir = os.path.join(save_dir, f"epoch_{int(epoch) + 1:03d}")
    os.makedirs(save_dir, exist_ok=True)

    if epoch is None:
        filename = "image_raw_scores_distribution_normal_vs_anomaly.png"
    else:
        filename = "image_raw_scores_distribution_normal_vs_anomaly.png"
    save_path = os.path.join(save_dir, filename)
    _plot_single_hist(scores, image_is_anomaly, save_path)

    # Per-class plots (skipped in global memory bank mode)
    if (
        not getattr(args, "global_memory_bank", False)
        and class_stack is not None
        and num_classes is not None
    ):
        cls_arr = np.asarray(class_stack, dtype=np.int64).reshape(-1)
        if cls_arr.shape[0] == scores.shape[0]:
            name_list = list(class_names) if class_names is not None else []
            if len(name_list) != int(num_classes):
                name_list = [f"cls_{i}" for i in range(int(num_classes))]

            for cls in range(int(num_classes)):
                cls_idx = np.where(cls_arr == cls)[0]
                if cls_idx.size == 0:
                    continue
                cls_scores = scores[cls_idx]
                cls_anomaly = image_is_anomaly[cls_idx]
                cls_save_path = os.path.join(
                    save_dir, f"image_raw_scores_distribution_{name_list[cls]}.png"
                )
                _plot_single_hist(
                    cls_scores, cls_anomaly, cls_save_path,
                    title_suffix=f" — {name_list[cls]}",
                )


def _plot_image_norm_score_distribution_normal_vs_anomaly(
    args, image_norm_scores, gt_patch_masks, epoch=None,
    class_stack=None, num_classes=None, class_names=None,
    save_dir=None,
):
    scores = np.asarray(image_norm_scores, dtype=np.float32).reshape(-1)
    patch_masks = np.asarray(gt_patch_masks, dtype=np.uint8)
    if scores.size == 0 or patch_masks.ndim != 2 or patch_masks.shape[0] != scores.size:
        return

    # Image is anomaly if any patch is anomaly.
    image_is_anomaly = patch_masks.reshape(patch_masks.shape[0], -1).sum(axis=1) > 0

    def _plot_single_hist(scores, image_is_anomaly, save_path, title_suffix=""):
        normal_scores = scores[~image_is_anomaly]
        anomaly_scores = scores[image_is_anomaly]
        if normal_scores.size == 0 and anomaly_scores.size == 0:
            return False

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
        plt.title(f"Image Norm Score Distribution: Normal vs Anomaly Images{title_suffix}")
        plt.xlim(0.0, 1.0)
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"[PseudoLabel-Plot] saved: {save_path}")
        return True

    if save_dir is None:
        save_dir = os.path.join(args.save_path, args.dataset, args.noise)
    if epoch is not None:
        save_dir = os.path.join(save_dir, f"epoch_{int(epoch) + 1:03d}")
    os.makedirs(save_dir, exist_ok=True)

    if epoch is None:
        filename = "image_norm_scores_distribution_normal_vs_anomaly.png"
    else:
        filename = "image_norm_scores_distribution_normal_vs_anomaly.png"
    save_path = os.path.join(save_dir, filename)
    _plot_single_hist(scores, image_is_anomaly, save_path)

    # Per-class plots (skipped in global memory bank mode)
    if (
        not getattr(args, "global_memory_bank", False)
        and class_stack is not None
        and num_classes is not None
    ):
        cls_arr = np.asarray(class_stack, dtype=np.int64).reshape(-1)
        if cls_arr.shape[0] == scores.shape[0]:
            name_list = list(class_names) if class_names is not None else []
            if len(name_list) != int(num_classes):
                name_list = [f"cls_{i}" for i in range(int(num_classes))]

            for cls in range(int(num_classes)):
                cls_idx = np.where(cls_arr == cls)[0]
                if cls_idx.size == 0:
                    continue
                cls_scores = scores[cls_idx]
                cls_anomaly = image_is_anomaly[cls_idx]
                cls_save_path = os.path.join(
                    save_dir, f"image_norm_scores_distribution_{name_list[cls]}.png"
                )
                _plot_single_hist(
                    cls_scores, cls_anomaly, cls_save_path,
                    title_suffix=f" — {name_list[cls]}",
                )


def _plot_classwise_feature_l2_by_distance_regions(
    args,
    class_names,
    region_gt_means,
    epoch=None,
    save_dir=None,
):
    means = np.asarray(region_gt_means, dtype=np.float32)
    if means.ndim != 3 or means.shape[1:] != (3, 2):
        return
    if means.shape[0] == 0:
        return

    if save_dir is None:
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


def _parse_mad_k_per_class(
    mad_k_per_class_str: str,
    class_names: list,
    default_k: float,
) -> dict:
    """Parse '--mad_k_per_class' string into {class_idx: k_value} dict.

    Example: 'bottle:4.0,cable:6.0' → {0: 4.0, 3: 6.0} (idx depends on class_names order)
    Classes not listed use default_k.
    """
    k_map = {}
    if not mad_k_per_class_str or not mad_k_per_class_str.strip():
        return k_map
    if class_names is None:
        return k_map
    name_to_idx = {name: idx for idx, name in enumerate(class_names)}
    for part in mad_k_per_class_str.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            print(f"[MAD-Threshold] WARNING: malformed entry '{part}', expected 'class_name:k_value'")
            continue
        cls_name, k_str = part.split(":", 1)
        cls_name = cls_name.strip()
        try:
            k_val = float(k_str.strip())
        except ValueError:
            print(f"[MAD-Threshold] WARNING: invalid k value '{k_str.strip()}' for class '{cls_name}'")
            continue
        if cls_name not in name_to_idx:
            print(f"[MAD-Threshold] WARNING: unknown class '{cls_name}', available: {list(name_to_idx.keys())}")
            continue
        k_map[name_to_idx[cls_name]] = {"k": k_val, "name": cls_name}
    return k_map


def _compute_mad_threshold_global(
    raw_distance_map: np.ndarray,
    k: float = 4.0,
) -> dict:
    """Compute a single MAD-based threshold on all raw distance values."""
    values = np.asarray(raw_distance_map, dtype=np.float32).reshape(-1)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    thr = median + k * 1.4826 * mad
    return {0: {"median": median, "mad": mad, "threshold": thr, "k": float(k)}}


def _compute_mad_thresholds_per_class(
    raw_distance_map: np.ndarray,
    class_stack: np.ndarray,
    num_classes: int,
    k: float = 4.0,
    class_k_map: dict = None,
    class_names: list = None,
):
    """Compute per-class MAD-based threshold on raw distance values.

    For each class, threshold = median + k * 1.4826 * MAD.
    MAD = median(|x_i - median|) — robust to sparse tail outliers.

    If class_k_map is provided, each class uses its own k value; otherwise uses the global k.
    """
    if class_k_map is None:
        class_k_map = {}
    values = np.asarray(raw_distance_map, dtype=np.float32)
    cls_arr = np.asarray(class_stack, dtype=np.int64).reshape(-1)
    thresholds = {}
    for cls in range(int(num_classes)):
        cls_idx = np.where(cls_arr == cls)[0]
        if cls_idx.size == 0:
            continue
        cls_k = float(class_k_map.get(int(cls), {}).get("k", k))
        cls_vals = values[cls_idx].reshape(-1)
        median = float(np.median(cls_vals))
        mad = float(np.median(np.abs(cls_vals - median)))
        thr = median + cls_k * 1.4826 * mad
        thresholds[int(cls)] = {"median": median, "mad": mad, "threshold": thr, "k": cls_k}
    return thresholds


def _plot_classwise_distance_with_mad_threshold(
    args,
    raw_distance_map: np.ndarray,
    class_stack: np.ndarray,
    mad_thresholds: dict,
    num_classes: int,
    class_names=None,
    epoch=None,
    save_dir=None,
):
    """Plot per-class histogram of raw distance values with MAD threshold line."""
    import matplotlib.ticker as mticker

    values = np.asarray(raw_distance_map, dtype=np.float32)
    cls_arr = np.asarray(class_stack, dtype=np.int64).reshape(-1)
    if values.ndim != 2:
        return
    if cls_arr.shape[0] != values.shape[0] or int(num_classes) <= 0:
        return

    if save_dir is None:
        save_dir = os.path.join(args.save_path, args.dataset, args.noise)
    os.makedirs(save_dir, exist_ok=True)
    if epoch is None:
        filename = "classwise_distance_mad_threshold.png"
    else:
        filename = f"classwise_distance_mad_threshold_epoch_{int(epoch) + 1:03d}.png"
    save_path = os.path.join(save_dir, filename)

    labels = list(class_names) if class_names is not None else []
    if len(labels) != int(num_classes):
        labels = [f"cls_{i}" for i in range(int(num_classes))]

    ncols = min(4, int(num_classes))
    nrows = int(np.ceil(int(num_classes) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.2), squeeze=False)
    max_points_per_class = 50000

    for cls in range(int(num_classes)):
        ax = axes[cls // ncols][cls % ncols]
        cls_idx = np.where(cls_arr == cls)[0]
        if cls_idx.size == 0:
            ax.set_title(f"{labels[cls]}\n(no data)", fontsize=9)
            ax.axis("off")
            continue
        cls_vals = values[cls_idx].reshape(-1)
        if cls_vals.size > max_points_per_class:
            cls_vals = np.random.choice(cls_vals, size=max_points_per_class, replace=False)

        ax.hist(cls_vals, bins=80, color="steelblue", edgecolor="white", alpha=0.75,
                density=True, linewidth=0.3)
        ax.set_xlabel("distance (raw)", fontsize=7)
        ax.set_ylabel("density", fontsize=7)
        ax.tick_params(labelsize=6)

        cls_info = mad_thresholds.get(cls)
        if cls_info is not None:
            thr = cls_info["threshold"]
            norm_thr = cls_info.get("norm_threshold", None)
            median = cls_info["median"]
            if norm_thr is not None:
                thr_label = f"MAD raw={thr:.4f}  norm={norm_thr:.4f}"
            else:
                thr_label = f"MAD thr={thr:.4f}"
            ax.axvline(x=thr, color="red", linestyle="--", linewidth=1.5, label=thr_label)
            ax.axvline(x=median, color="gray", linestyle=":", linewidth=1.0,
                       label=f"median={median:.4f}")
            ax.legend(fontsize=5.0, loc="upper right", framealpha=0.7)

        ax.set_title(labels[cls], fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3g"))

    # Hide unused subplots
    for idx in range(int(num_classes), nrows * ncols):
        ax = axes[idx // ncols][idx % ncols]
        ax.axis("off")

    fig.suptitle("Per-Class Raw Distance Distribution + MAD Threshold", fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[MAD-Threshold-Plot] saved: {save_path}")


def _plot_global_distance_with_mad_threshold(
    args,
    raw_distance_map: np.ndarray,
    mad_threshold_info: dict,
    epoch=None,
    save_dir=None,
):
    """Plot global histogram of raw distance values with a single MAD threshold line."""
    import matplotlib.ticker as mticker

    values = np.asarray(raw_distance_map, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return

    if save_dir is None:
        save_dir = os.path.join(args.save_path, args.dataset, args.noise)
    os.makedirs(save_dir, exist_ok=True)
    if epoch is None:
        filename = "global_distance_mad_threshold.png"
    else:
        filename = f"global_distance_mad_threshold_epoch_{int(epoch) + 1:03d}.png"
    save_path = os.path.join(save_dir, filename)

    max_points = 200000
    plot_vals = values
    if plot_vals.size > max_points:
        plot_vals = np.random.choice(plot_vals, size=max_points, replace=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        plot_vals,
        bins=80,
        color="steelblue",
        edgecolor="white",
        alpha=0.75,
        density=True,
        linewidth=0.3,
    )
    thr = mad_threshold_info["threshold"]
    median = mad_threshold_info["median"]
    norm_thr = mad_threshold_info.get("norm_threshold")
    if norm_thr is not None:
        thr_label = f"MAD raw={thr:.4f}  norm={norm_thr:.4f}"
    else:
        thr_label = f"MAD thr={thr:.4f}"
    ax.axvline(x=thr, color="red", linestyle="--", linewidth=1.5, label=thr_label)
    ax.axvline(x=median, color="gray", linestyle=":", linewidth=1.0,
               label=f"median={median:.4f}")
    ax.set_xlabel("distance (raw)")
    ax.set_ylabel("density")
    ax.set_title("Global Raw Distance Distribution + MAD Threshold")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3g"))
    ax.legend(fontsize=8.0, loc="upper right", framealpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[MAD-Threshold-Plot] saved: {save_path}")


def _plot_classwise_abs_distance_box_before_norm(
    args,
    class_stack: np.ndarray,
    distance_map: np.ndarray,
    gt_patch_masks: np.ndarray,
    num_classes: int,
    class_names=None,
    epoch=None,
    save_dir=None,
):
    values = np.abs(np.asarray(distance_map, dtype=np.float32))
    cls_arr = np.asarray(class_stack, dtype=np.int64).reshape(-1)
    gt_mask = np.asarray(gt_patch_masks, dtype=np.uint8)
    if values.ndim != 2 or gt_mask.shape != values.shape:
        return
    if cls_arr.shape[0] != values.shape[0] or int(num_classes) <= 0:
        return

    if save_dir is None:
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


def _write_memory_bank_snapshot(
    memory_bank_snapshot: Dict[str, Any],
    ensemble_pca_by_class: dict,
    class_stack: np.ndarray,
    gt_patch_masks: np.ndarray,
    global_dim: int,
) -> None:
    memory_bank_snapshot.clear()
    memory_bank_snapshot["ensemble_pca_by_class"] = {
        int(k): [(mean.copy(), comp.copy()) for (mean, comp) in models]
        for k, models in ensemble_pca_by_class.items()
    }
    memory_bank_snapshot["class_stack"] = np.asarray(class_stack, dtype=np.int64).copy()
    memory_bank_snapshot["gt_patch_masks"] = np.asarray(gt_patch_masks, dtype=np.uint8).copy()
    memory_bank_snapshot["global_dim"] = int(global_dim)


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
    build_faiss_index_fn,       # kept for backward compatibility, unused
    update_topk_features_fn,
    print_selected_score_distribution_fn,
    print_selected_clean_ratio_fn,
    print_confusion_matrix_fn,
    epoch=None,
    pseudo_label_scorer=None,   # kept for backward compatibility, unused
    memory_bank_snapshot: Optional[Dict[str, Any]] = None,
    threshold: Optional[float] = None,
    save_dir=None,
    memory_score_path: Optional[str] = None,
):
    memory_bank_start = time.perf_counter()

    dataset = mini_loader.dataset
    dataset_size = len(dataset)
    batch_size = mini_loader.batch_size

    # ── Load pre-computed memory scores (static, from memory_score_generation) ──
    # These are used in Phase 1 fusion as the "first stage" score, see 5c-bis.
    _image_memory_scores = None
    if memory_score_path is not None and os.path.isfile(memory_score_path):
        _image_memory_scores = _load_memory_scores_per_image(
            memory_score_path, dataset,
            float(getattr(args, "img_score_topk_ratio", 0.01)),
        )

    # ── classwise_global_indices: must exist on the dataset ──
    if hasattr(dataset, "classwise_global_indices"):
        classwise_global_indices = dataset.classwise_global_indices
    else:
        raise ValueError("Dataset must have classwise_global_indices attribute")

    freeze_start = int(getattr(args, "memory_bank_freeze_start_epoch", -1))
    if freeze_start < 1:
        freeze_start = -1

    # ── Read ensemble PCA args ──
    ensemble_size = int(getattr(args, "ensemble_size", 100))
    memory_sampling_ratio = float(getattr(args, "memory_sampling_ratio", 0.1))
    pca_ev = float(getattr(args, "pseudo_label_pca_ev", 0.99))
    pca_dim = int(getattr(args, "pseudo_label_pca_dim", 0))
    pca_eps = float(getattr(args, "pseudo_label_pca_eps", 1e-6))

    # ── Check frozen mode ──
    frozen_mode = (
        freeze_start >= 1
        and epoch is not None
        and int(epoch) >= freeze_start
        and memory_bank_snapshot is not None
        and memory_bank_snapshot.get("ensemble_pca_by_class")
    )
    if frozen_mode:
        snap_cs = memory_bank_snapshot.get("class_stack")
        if snap_cs is None or len(snap_cs) != dataset_size:
            print(
                "[MemoryBank] snapshot incompatible with current dataset "
                f"(class_stack len {0 if snap_cs is None else len(snap_cs)} "
                f"vs dataset_size {dataset_size}); running full pipeline"
            )
            frozen_mode = False

    # ── Pre-allocate arrays (global index positioning) ──
    distance_map = np.zeros((dataset_size, 784), dtype=np.float16)
    patch_feature_l2_map = np.zeros((dataset_size, 784), dtype=np.float32)
    image_norm_scores = np.zeros(dataset_size, dtype=np.float32)
    class_stack = np.zeros(dataset_size, dtype=np.int64)
    gt_patch_masks = np.zeros((dataset_size, 784), dtype=np.uint8)
    global_dim = None

    # Beta feature collection
    if args.beta:
        top_feat_global = None
        top_dist_global = None

    # ────────────────────────────────────────────────────────────────────
    #  Frozen mode — load PCA models from snapshot, skip the main loop
    # ────────────────────────────────────────────────────────────────────
    ensemble_pca_by_class = {}
    if frozen_mode:
        print(
            f"[MemoryBank] frozen from epoch {freeze_start} "
            f"(current epoch index {int(epoch)}): skipping per-class memory bank build"
        )
        ensemble_pca_by_class_raw = memory_bank_snapshot["ensemble_pca_by_class"]
        ensemble_pca_by_class = {
            int(k): [(mean.copy(), comp.copy()) for (mean, comp) in models]
            for k, models in ensemble_pca_by_class_raw.items()
        }
        class_stack = np.asarray(memory_bank_snapshot["class_stack"], dtype=np.int64).copy()
        gt_patch_masks = np.asarray(memory_bank_snapshot["gt_patch_masks"], dtype=np.uint8).copy()
        global_dim = int(memory_bank_snapshot["global_dim"])
        confident_feature_bank = torch.zeros((0, global_dim), dtype=torch.float32)

        # ── Still need to score all patches — same per-class loop, skip memory bank build ──
        print("[MemoryBank frozen] Computing distance map for all patches...")
        pseudo_start = time.perf_counter()
        with torch.no_grad():
            localnet.eval()
            feature_extractor.eval()
            for batch in tqdm.tqdm(mini_loader, desc="Scoring"):
                if len(batch) == 4:
                    images, mini_class_idx, mini_sample_idx, _mini_patch_mask = batch
                else:
                    images, mini_class_idx, mini_sample_idx = batch
                images = images.to(device)
                image_features, cls_token = _extract_features_with_optional_cls_token(
                    extract_feature_batch_fn, images, feature_extractor, args,
                    class_indices=mini_class_idx,
                )
                residual_features = compute_residual_feature_batch_fn(
                    image_features,
                    mini_class_idx.to(device),
                    reference_memory_by_class,
                    reference_index_by_class,
                )
                patch_class_idx = None
                if getattr(args, "use_moe_discriminator", False) and getattr(args, "moe_hard_class_gate", False):
                    patch_class_idx = _build_patch_class_idx(mini_class_idx, residual_features)
                features, _ = localnet(
                    residual_features, cls_token=cls_token, patch_class_idx=patch_class_idx,
                    class_idx=mini_class_idx.to(device),
                )
                features_np = features.detach().cpu().numpy()
                sample_idx_np = mini_sample_idx.detach().cpu().numpy().astype(np.int64)
                batch_class_np = mini_class_idx.detach().cpu().numpy().astype(np.int64)
                batch_size_local = int(batch_class_np.shape[0])
                feature_l2 = np.linalg.norm(
                    features_np.reshape(-1, global_dim), ord=2, axis=1
                ).astype(np.float32).reshape(batch_size_local, 784)

                cls_distance_map = np.zeros((batch_size_local, 784), dtype=np.float32)
                for cls in np.unique(batch_class_np).tolist():
                    cls = int(cls)
                    if cls not in ensemble_pca_by_class:
                        continue
                    cls_mask = batch_class_np == cls
                    if not np.any(cls_mask):
                        continue
                    cls_features_2d = features_np[cls_mask].reshape(-1, global_dim)
                    cls_scores = _ensemble_pca_score_patches(
                        cls_features_2d.astype(np.float32),
                        ensemble_pca_by_class[cls],
                    )
                    cls_scores = np.nan_to_num(cls_scores, nan=0.0, posinf=0.0, neginf=0.0)
                    cls_scores = np.clip(cls_scores, 0.0, None)
                    score_map_2d = cls_scores.reshape(-1, 784).astype(np.float32)
                    cls_distance_map[cls_mask] = score_map_2d

                distance_map[sample_idx_np] = cls_distance_map.astype(np.float32)
                patch_feature_l2_map[sample_idx_np] = feature_l2

                if args.beta:
                    # In frozen mode, we don't have fresh image_norm_scores — skip beta collection
                    pass

        # Skip to post-processing (jump over the main loop)
        # We still need image_norm_scores for post-processing, so fill with zeros
        # after the scoring loop
        image_norm_scores = np.zeros(dataset_size, dtype=np.float32)

        use_mad = bool(getattr(args, "use_mad_threshold", False))
        mad_threshold_map = None
        if use_mad:
            _mad_raw = np.asarray(distance_map, dtype=np.float32).copy()
        # fall through to the post-processing block
        print("[MemoryBank frozen] Scoring complete, proceeding to post-processing.")
        pseudo_time_this = time.perf_counter() - pseudo_start
        memory_bank_time = time.perf_counter() - memory_bank_start
        # return early — frozen mode handled separately below post-processing
        # We'll refactor to share post-processing.
        # For now we reuse the post-processing logic at the end of this function.
        _frozen_result = _finish_pseudo_labels(
            args, distance_map, patch_feature_l2_map, image_norm_scores, class_stack,
            gt_patch_masks, global_dim, num_classes, epoch, save_dir, threshold,
            use_mad, _mad_raw if use_mad else None,
            None if args.beta else None,  # confident_feature_bank already set
            memory_bank_time, memory_bank_start, ensemble_pca_by_class,
            ensemble_pca_by_class if True else ensemble_pca_by_class,
        )
        return _frozen_result

    # ════════════════════════════════════════════════════════════════════
    #  Main per-class loop: scoring → memory bank → distance map
    # ════════════════════════════════════════════════════════════════════

    print(f"[EnsemblePCA] per-class memory bank (ensemble_size={ensemble_size}, "
          f"sampling_ratio={memory_sampling_ratio}, pca_ev={pca_ev})")

    global_bank = bool(getattr(args, "global_memory_bank", False))

    # For global memory bank: collect selected features across classes
    global_selected_features_list = []
    global_selected_class_info = []  # list of (class_idx, n_patches)

    for cls in range(num_classes):
        cls_indices = classwise_global_indices.get(cls, [])
        if not cls_indices:
            print(f"[Class {cls}] No indices found, skipping.")
            continue

        print(f"[Class {cls}] Processing {len(cls_indices)} images...")

        # ── 5a: Build Subset DataLoader (inherits original dataset, preserves global indices) ──
        cls_dataset = Subset(dataset, cls_indices)
        cls_loader = DataLoader(
            cls_dataset,
            batch_size=batch_size,
            shuffle=False,         # key: no shuffle → stable iteration order
            drop_last=False,
            num_workers=min(4, int(batch_size)),  # 避免 num_workers=0 串行加载
            prefetch_factor=2,
        )

        # ── 5b: Single pass: extract features, compute scores, store features ──
        cls_scores = []
        cls_features_by_img = []
        cls_sample_indices = []

        with torch.no_grad():
            localnet.eval()
            feature_extractor.eval()
            for batch in cls_loader:
                if len(batch) == 4:
                    images, mini_class_idx, mini_sample_idx, mini_patch_mask = batch
                else:
                    images, mini_class_idx, mini_sample_idx = batch
                    mini_patch_mask = None

                images = images.to(device)
                image_features, cls_token = _extract_features_with_optional_cls_token(
                    extract_feature_batch_fn, images, feature_extractor, args,
                    class_indices=mini_class_idx,
                )
                residual_features = compute_residual_feature_batch_fn(
                    image_features, mini_class_idx.to(device),
                    reference_memory_by_class, reference_index_by_class,
                )
                patch_class_idx = None
                if getattr(args, "use_moe_discriminator", False) and getattr(args, "moe_hard_class_gate", False):
                    patch_class_idx = _build_patch_class_idx(mini_class_idx, residual_features)

                features, score = localnet(
                    residual_features, cls_token=cls_token,
                    patch_class_idx=patch_class_idx,
                    class_idx=mini_class_idx.to(device),
                )
                if features.shape[0] > args.batch_size:
                    features = features.unsqueeze(0)

                if global_dim is None:
                    global_dim = int(features.shape[-1])

                score_np = score.detach().cpu().numpy()
                score_np = _as_1d_float32(
                    aggregate_image_scores_fn(score_np, topk_ratio=args.img_score_topk_ratio)
                )

                features_np = features.detach().cpu().numpy().astype(np.float16)
                batch_global_indices = mini_sample_idx.detach().cpu().numpy()

                for i in range(features_np.shape[0]):
                    cls_features_by_img.append(features_np[i])
                    cls_scores.append(score_np[i])
                    cls_sample_indices.append(batch_global_indices[i])

                # Update class_stack and gt_patch_masks via global indices
                sample_idx_np = batch_global_indices.astype(np.int64)
                class_stack[sample_idx_np] = cls  # all same class
                if mini_patch_mask is not None:
                    mask_np = mini_patch_mask.detach().cpu().numpy()
                    gt_patch_masks[sample_idx_np] = (mask_np > 0).astype(np.uint8)

        # ── 5c: Normalize scores for this class ──
        cls_scores_arr = np.array(cls_scores, dtype=np.float32)
        cls_norm_scores = _minmax_normalize(cls_scores_arr)

        # ── 5c-bis: Fuse with static pre-computed memory scores ──
        # Early epochs rely more on the stable memory bank distance; later
        # epochs trust the trained model's own predictions.
        # With static memory_scores.pth, epoch 0 also benefits from fusion
        # (alpha=1.0 → pure memory score, bypassing unreliable untrained model).
        _mem_beta = float(getattr(args, "memory_score_beta_end", 0.5))
        if (
            _image_memory_scores is not None
            and epoch is not None
            and _mem_beta > 0
        ):
            _total_epochs = max(1, int(getattr(args, "epoch", 200)))
            _alpha = max(1.0 - float(epoch) / (_total_epochs * _mem_beta), 0.0)
            # Per-class normalize stored memory scores
            _cls_mem = _image_memory_scores[np.array(cls_sample_indices, dtype=np.int64)]
            _cls_mem_norm = _minmax_normalize(_cls_mem)
            # Fuse: alpha * memory + (1-alpha) * model
            cls_norm_scores = _alpha * _cls_mem_norm + (1.0 - _alpha) * cls_norm_scores

        for gidx, norm_score in zip(cls_sample_indices, cls_norm_scores):
            image_norm_scores[gidx] = norm_score

        # ── 5d: Select normal samples ──
        normal_mask = cls_norm_scores < 0.5

        print(f"  [Class {cls}] {normal_mask.sum()}/{len(cls_norm_scores)} images selected as normal")

        # ── 5e: Build ensemble PCA memory bank ──
        if normal_mask.sum() == 0:
            print(f"  [Class {cls}] No normal samples, skipping memory bank for this class")
            if not global_bank:
                # For per-class mode, we skip distance map for this class (stays zero)
                pass
            continue

        selected_features_list = [
            cls_features_by_img[i].reshape(-1, global_dim).astype(np.float32)
            for i in np.where(normal_mask)[0]
        ]
        cls_selected_features = np.concatenate(selected_features_list, axis=0)

        if global_bank:
            # Collect for global memory bank
            global_selected_features_list.append(cls_selected_features)
            global_selected_class_info.append((cls, cls_selected_features.shape[0]))
        else:
            # Build per-class ensemble PCA
            cls_ensemble_pca = _build_ensemble_pca(
                cls_selected_features, ensemble_size, memory_sampling_ratio,
                pca_ev, pca_dim, pca_eps,
            )
            ensemble_pca_by_class[cls] = cls_ensemble_pca

            # ── 5f: Score all patches for this class → write to distance_map ──
            for i, gidx in enumerate(cls_sample_indices):
                img_features_2d = cls_features_by_img[i].reshape(-1, global_dim).astype(np.float32)
                patch_scores = _ensemble_pca_score_patches(img_features_2d, cls_ensemble_pca)
                patch_scores = np.nan_to_num(patch_scores, nan=0.0, posinf=0.0, neginf=0.0)
                patch_scores = np.clip(patch_scores, 0.0, None)
                distance_map[gidx] = patch_scores.reshape(784).astype(np.float16)
                patch_feature_l2_map[gidx] = np.linalg.norm(
                    img_features_2d, ord=2, axis=1
                ).astype(np.float32)

        # ── 5g: Beta feature collection (cross-class accumulation) ──
        if args.beta:
            high_mask = cls_norm_scores > 0.5
            if np.any(high_mask):
                for i in np.where(high_mask)[0]:
                    feat_2d = cls_features_by_img[i].reshape(-1, global_dim).astype(np.float32)
                    scores_1d = distance_map[cls_sample_indices[i]].reshape(-1)
                    top_feat_global, top_dist_global = update_topk_features_fn(
                        top_feat_global, top_dist_global,
                        feat_2d, scores_1d, args.beta_number,
                    )

        # Release per-class memory
        del cls_features_by_img, cls_selected_features, selected_features_list

    # ── Global memory bank: build a single ensemble PCA on merged features ──
    if global_bank and global_selected_features_list:
        all_features = np.concatenate(global_selected_features_list, axis=0)
        print(f"[GlobalMemoryBank] merged {len(global_selected_features_list)} class sets "
              f"→ {all_features.shape[0]} patches")
        global_ensemble_pca = _build_ensemble_pca(
            all_features, ensemble_size, memory_sampling_ratio,
            pca_ev, pca_dim, pca_eps,
        )
        ensemble_pca_by_class[0] = global_ensemble_pca

        # Re-score all patches with global bank
        # We need to re-traverse all data. Since we already scored per-class,
        # we build a fresh traversal.
        print("[GlobalMemoryBank] Re-scoring all patches with global ensemble PCA...")
        with torch.no_grad():
            localnet.eval()
            feature_extractor.eval()
            for batch in tqdm.tqdm(mini_loader, desc="Global re-scoring"):
                if len(batch) == 4:
                    images, mini_class_idx, mini_sample_idx, _ = batch
                else:
                    images, mini_class_idx, mini_sample_idx = batch
                images = images.to(device)
                image_features, cls_token = _extract_features_with_optional_cls_token(
                    extract_feature_batch_fn, images, feature_extractor, args,
                    class_indices=mini_class_idx,
                )
                residual_features = compute_residual_feature_batch_fn(
                    image_features, mini_class_idx.to(device),
                    reference_memory_by_class, reference_index_by_class,
                )
                patch_class_idx = None
                if getattr(args, "use_moe_discriminator", False) and getattr(args, "moe_hard_class_gate", False):
                    patch_class_idx = _build_patch_class_idx(mini_class_idx, residual_features)
                features, _ = localnet(
                    residual_features, cls_token=cls_token, patch_class_idx=patch_class_idx,
                    class_idx=mini_class_idx.to(device),
                )
                features_np = features.detach().cpu().numpy()
                sample_idx_np = mini_sample_idx.detach().cpu().numpy().astype(np.int64)
                batch_sz = int(sample_idx_np.shape[0])

                features_2d = features_np.reshape(-1, global_dim).astype(np.float32)
                patch_scores = _ensemble_pca_score_patches(features_2d, global_ensemble_pca)
                patch_scores = np.nan_to_num(patch_scores, nan=0.0, posinf=0.0, neginf=0.0)
                patch_scores = np.clip(patch_scores, 0.0, None)

                distance_map[sample_idx_np] = patch_scores.reshape(batch_sz, 784).astype(np.float16)
                patch_feature_l2_map[sample_idx_np] = np.linalg.norm(
                    features_np.reshape(-1, global_dim), ord=2, axis=1
                ).astype(np.float32).reshape(batch_sz, 784)

        del global_selected_features_list, global_ensemble_pca

    # ── Plot image norm score & raw score distributions ──
    _plot_image_norm_score_distribution_normal_vs_anomaly(
        args=args,
        image_norm_scores=image_norm_scores,
        gt_patch_masks=gt_patch_masks,
        epoch=epoch,
        class_stack=class_stack,
        num_classes=num_classes,
        class_names=getattr(args, "class_names", None),
        save_dir=save_dir,
    )
    _plot_image_raw_score_distribution_normal_vs_anomaly(
        args=args,
        image_raw_scores=image_norm_scores,  # use norm scores for display
        gt_patch_masks=gt_patch_masks,
        epoch=epoch,
        class_stack=class_stack,
        num_classes=num_classes,
        class_names=getattr(args, "class_names", None),
        save_dir=save_dir,
    )

    # ── Snapshot for freeze ──
    if (
        freeze_start >= 1
        and epoch is not None
        and int(epoch) == freeze_start - 1
        and memory_bank_snapshot is not None
    ):
        _write_memory_bank_snapshot(
            memory_bank_snapshot,
            ensemble_pca_by_class,
            class_stack,
            gt_patch_masks,
            global_dim,
        )
        print(
            f"[MemoryBank] snapshot saved at epoch {int(epoch)} "
            f"(frozen Phase 1/2 from epoch index {freeze_start} onward)"
        )

    # ── Beta feature bank ──
    if args.beta:
        if top_feat_global is None or top_feat_global.shape[0] == 0:
            confident_feature_bank = torch.zeros((0, global_dim), dtype=torch.float32)
        else:
            confident_feature_bank = torch.as_tensor(top_feat_global, dtype=torch.float32)
    else:
        confident_feature_bank = torch.zeros((0, global_dim), dtype=torch.float32)

    # ════════════════════════════════════════════════════════════════════
    #  Post-processing (shared with frozen mode)
    # ════════════════════════════════════════════════════════════════════

    memory_bank_time = time.perf_counter() - memory_bank_start
    _main_result = _finish_pseudo_labels(
        args, distance_map, patch_feature_l2_map, image_norm_scores, class_stack,
        gt_patch_masks, global_dim, num_classes, epoch, save_dir, threshold,
        bool(getattr(args, "use_mad_threshold", False)),
        np.asarray(distance_map, dtype=np.float32).copy() if getattr(args, "use_mad_threshold", False) else None,
        confident_feature_bank,
        memory_bank_time, memory_bank_start, ensemble_pca_by_class,
        ensemble_pca_by_class,
    )
    return _main_result


def _load_memory_scores_per_image(
    memory_score_path: str,
    dataset,
    topk_ratio: float = 0.01,
) -> np.ndarray:
    """Load memory_scores.pth → per-image top-k% mean → per-class min-max normalize.

    Structure of memory_scores.pth:
        {class_name: {filename_stem: Tensor[P]}}   — P = num_patches (raw cosine distances)

    Pipeline:
        1. 每张图内: P 个 patch → top-k% 取均值 → 图像级原始分数
        2. 按类: 对这些图像级分数做标准 min-max 归一化到 [0,1]

    Returns:
        ndarray [dataset_size] — per-image memory scores in [0, 1].
        Images not found in the file get score 0.
    """
    raw = torch.load(memory_score_path, map_location="cpu")
    n_patches = dataset.patch_mask_size ** 2
    k = max(1, int(n_patches * topk_ratio))

    # ── Step 1: 每张图内 top-k patch 均值 → 图像级原始分数 ──
    # class_img_raw[cls_idx][fname] = raw_image_score (float)
    class_img_raw: dict = {}
    for cls_name, cls_dict in raw.items():
        if cls_name not in dataset.class_to_idx:
            continue
        cls_idx = dataset.class_to_idx[cls_name]
        class_img_raw.setdefault(cls_idx, {})
        for fname, patch_scores in cls_dict.items():
            topk_vals, _ = patch_scores.topk(k)
            class_img_raw[cls_idx][fname] = float(topk_vals.mean().item())

    # ── Step 2: 按类对图像级分数做标准 min-max 归一化 ──
    norm_lookup: dict = {}
    for cls_idx, img_dict in class_img_raw.items():
        scores = np.array(list(img_dict.values()), dtype=np.float32)
        vmin = float(scores.min())
        vmax = float(scores.max())
        if vmax > vmin:
            normed = (scores - vmin) / (vmax - vmin)
        else:
            normed = np.zeros_like(scores)
        norm_lookup[cls_idx] = dict(zip(img_dict.keys(), normed))

    # ── Step 3: 映射回 dataset 索引 ──
    result = np.zeros(len(dataset), dtype=np.float32)
    for global_idx, (img_path, class_idx) in enumerate(dataset.samples):
        fname = os.path.splitext(os.path.basename(img_path))[0]
        cls_lookup = norm_lookup.get(class_idx)
        if cls_lookup is not None:
            result[global_idx] = cls_lookup.get(fname, 0.0)
    return result


def _finish_pseudo_labels(
    args,
    distance_map,
    patch_feature_l2_map,
    image_norm_scores,
    class_stack,
    gt_patch_masks,
    global_dim,
    num_classes,
    epoch,
    save_dir,
    threshold,
    use_mad,
    _mad_raw,
    confident_feature_bank,
    memory_bank_time,
    memory_bank_start,
    ensemble_pca_by_class,
    ensemble_pca_by_class_for_norm,  # for snapshot, unused here
):
    """Post-processing: normalization, MAD threshold, stats, plots, returns."""

    pseudo_start = time.perf_counter()

    if not getattr(args, "global_memory_bank", False):
        _plot_classwise_abs_distance_box_before_norm(
            args=args,
            class_stack=class_stack,
            distance_map=distance_map,
            gt_patch_masks=gt_patch_masks,
            num_classes=num_classes,
            class_names=getattr(args, "class_names", None),
            epoch=epoch,
            save_dir=save_dir,
        )

    # ── MAD-based threshold ──
    mad_threshold_map = None
    mad_thresholds = None
    if use_mad and _mad_raw is not None:
        mad_k = float(getattr(args, "mad_k", 4.0))
        global_bank = bool(getattr(args, "global_memory_bank", False))
        if global_bank:
            mad_thresholds = _compute_mad_threshold_global(
                raw_distance_map=_mad_raw,
                k=mad_k,
            )
        else:
            mad_k_per_class_str = str(getattr(args, "mad_k_per_class", ""))
            class_names_for_k = getattr(args, "class_names", None)
            class_k_map = _parse_mad_k_per_class(mad_k_per_class_str, class_names_for_k, mad_k)
            mad_thresholds = _compute_mad_thresholds_per_class(
                raw_distance_map=_mad_raw,
                class_stack=class_stack,
                num_classes=num_classes,
                k=mad_k,
                class_k_map=class_k_map,
                class_names=class_names_for_k,
            )

    # ── Normalize distance map ──
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

    # Build MAD threshold_map in normalized space
    if use_mad and mad_thresholds:
        normalized = np.asarray(distance_map, dtype=np.float32)
        mad_threshold_map = np.zeros_like(normalized, dtype=np.float32)
        global_bank = bool(getattr(args, "global_memory_bank", False))
        if global_bank:
            info = mad_thresholds[0]
            raw_thr = info["threshold"]
            raw_all = _mad_raw.reshape(-1)
            vmin = float(raw_all.min())
            vmax = float(raw_all.max())
            if vmax > vmin:
                norm_thr = (raw_thr - vmin) / (vmax - vmin)
            else:
                norm_thr = 0.0
            norm_thr = min(norm_thr, 0.65)
            info["norm_threshold"] = float(norm_thr)
            mad_threshold_map[:] = norm_thr

            print("[MAD-Threshold] Global threshold (k={:.1f}):".format(mad_k))
            print(
                "  k={:.1f}  median(raw)={:.6f}  MAD(raw)={:.6f}  "
                "thr(raw)={:.6f}  thr(norm)={:.6f}".format(
                    info["k"],
                    info["median"],
                    info["mad"],
                    info["threshold"],
                    info["norm_threshold"],
                )
            )
            _plot_global_distance_with_mad_threshold(
                args=args,
                raw_distance_map=_mad_raw,
                mad_threshold_info=info,
                epoch=epoch,
                save_dir=save_dir,
            )
        else:
            cls_arr = np.asarray(class_stack, dtype=np.int64).reshape(-1)
            for cls, info in mad_thresholds.items():
                cls_mask_flat = cls_arr == int(cls)
                if not np.any(cls_mask_flat):
                    continue
                raw_thr = info["threshold"]
                cls_raw = _mad_raw[cls_mask_flat]
                vmin = float(cls_raw.min())
                vmax = float(cls_raw.max())
                if vmax > vmin:
                    norm_thr = (raw_thr - vmin) / (vmax - vmin)
                else:
                    norm_thr = 0.0
                norm_thr = min(norm_thr, 0.65)
                info["norm_threshold"] = float(norm_thr)
                mad_threshold_map[cls_mask_flat] = norm_thr

            mad_k_default = float(getattr(args, "mad_k", 4.0))
            print("[MAD-Threshold] Per-class thresholds (default k={:.1f}):".format(mad_k_default))
            print("  {:20s}  {:>6s}  {:>10s}  {:>10s}  {:>10s}  {:>10s}".format(
                "class", "k", "median(raw)", "MAD(raw)", "thr(raw)", "thr(norm)"))
            class_names_mad = getattr(args, "class_names", None)
            for cls in sorted(mad_thresholds.keys()):
                info = mad_thresholds[cls]
                cls_k = info.get("k", mad_k_default)
                custom_mark = "*" if abs(cls_k - mad_k_default) > 1e-6 else " "
                cls_name = (
                    class_names_mad[cls]
                    if isinstance(class_names_mad, (list, tuple)) and cls < len(class_names_mad)
                    else f"cls_{cls}"
                )
                print(
                    "  {:20s}  {:>5.1f}{}  {:10.6f}  {:10.6f}  {:10.6f}  {:10.6f}".format(
                        cls_name,
                        cls_k,
                        custom_mark,
                        info["median"],
                        info["mad"],
                        info["threshold"],
                        info["norm_threshold"],
                    )
                )

            _plot_classwise_distance_with_mad_threshold(
                args=args,
                raw_distance_map=_mad_raw,
                class_stack=class_stack,
                mad_thresholds=mad_thresholds,
                num_classes=num_classes,
                class_names=getattr(args, "class_names", None),
                epoch=epoch,
                save_dir=save_dir,
            )

    # ── Statistics ──
    values = distance_map.astype(np.float32)
    gt_mask_bool = gt_patch_masks.astype(bool)
    noise_thr = float(args.noise_threshold)

    if use_mad and mad_threshold_map is not None:
        pred_normal = values < mad_threshold_map
        pred_anomaly = values > noise_thr
        pred_uncertain = np.logical_not(np.logical_or(pred_normal, pred_anomaly))
        effective_threshold = None
    else:
        effective_threshold = float(threshold if threshold is not None else getattr(args, 'threshold', 0.5))
        pred_normal = values < effective_threshold
        pred_anomaly = values > noise_thr
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
    if use_mad and mad_threshold_map is not None:
        thr_label = "MAD_thr"
    else:
        thr_label = f"thr={effective_threshold:.3f}"
    print(
        "[PseudoLabel-Stats] "
        f"distance<{thr_label} normal-precision: {normal_precision:.4f} "
        f"({pred_normal_correct}/{pred_normal_count}) | "
        f"distance>noise_threshold anomaly-precision: {anomaly_precision:.4f} "
        f"({pred_anomaly_correct}/{pred_anomaly_count})"
    )

    if getattr(args, "global_memory_bank", False):
        region_masks = (pred_normal, pred_uncertain, pred_anomaly)
        region_names = (
            f"distance<{thr_label}",
            f"{thr_label}<=distance<=noise_threshold",
            "distance>noise_threshold",
        )
        gt_names = ("gt_normal", "gt_anomaly")
        stat_parts = []
        for ridx, rmask in enumerate(region_masks):
            for gidx, gt_sel in enumerate((np.logical_not(gt_mask_bool), gt_mask_bool)):
                sel = np.logical_and(rmask, gt_sel)
                cnt = int(sel.sum())
                if cnt > 0:
                    selected_vals = patch_feature_l2_map[sel].astype(np.float32)
                    stat_parts.append(
                        f"{region_names[ridx]}/{gt_names[gidx]}: "
                        f"mean={float(selected_vals.mean()):.4f}, "
                        f"std={float(selected_vals.std()):.4f}, "
                        f"median={float(np.median(selected_vals)):.4f} (n={cnt})"
                    )
                else:
                    stat_parts.append(f"{region_names[ridx]}/{gt_names[gidx]}: NA (n=0)")
        print("[PseudoLabel-L2Norm] global | " + " | ".join(stat_parts))
    else:
        classwise_region_gt_means = np.full((num_classes, 3, 2), np.nan, dtype=np.float32)
        classwise_region_gt_stds = np.full((num_classes, 3, 2), np.nan, dtype=np.float32)
        classwise_region_gt_medians = np.full((num_classes, 3, 2), np.nan, dtype=np.float32)
        classwise_region_gt_counts = np.zeros((num_classes, 3, 2), dtype=np.int64)
        region_masks = (pred_normal, pred_uncertain, pred_anomaly)
        region_names = (
            f"distance<{thr_label}",
            f"{thr_label}<=distance<=noise_threshold",
            "distance>noise_threshold",
        )
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
            save_dir=save_dir,
        )

    _plot_distance_distribution_normal_vs_anomaly(
        args=args,
        distance_values=values,
        gt_patch_masks=gt_patch_masks,
        epoch=epoch,
        class_stack=class_stack,
        num_classes=num_classes,
        class_names=getattr(args, "class_names", None),
        save_dir=save_dir,
    )

    pseudo_label_time = time.perf_counter() - pseudo_start
    return distance_map, confident_feature_bank, global_dim, memory_bank_time, pseudo_label_time, mad_threshold_map
