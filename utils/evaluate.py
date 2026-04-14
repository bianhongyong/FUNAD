import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from skimage import measure
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

from utils.train_utils import aggregate_image_scores


def _cv2_resize_dsize_from_mask(mask_i):
    """(W, H) for cv2.resize so seg matches GT mask spatial size."""
    mi = np.asarray(mask_i)
    if mi.ndim < 2:
        raise ValueError(f"mask must be at least 2D, got shape {mi.shape}")
    h, w = int(mi.shape[-2]), int(mi.shape[-1])
    return (w, h)


def _build_patch_class_idx(class_idx: torch.Tensor, patch_features: torch.Tensor):
    token_per_image = int(patch_features.shape[1]) if patch_features.dim() >= 2 else 1
    return (
        class_idx.to(device=patch_features.device, dtype=torch.long)
        .reshape(-1, 1, 1)
        .expand(-1, token_per_image, 1)
        .contiguous()
    )


def f1_score_max(y_true, y_score):
    y_true = np.asarray(y_true).astype(np.uint8).ravel()
    y_score = np.asarray(y_score).astype(np.float32).ravel()
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return 0.0
    precisions, recalls, _ = precision_recall_curve(y_true, y_score)
    f1s = 2.0 * precisions * recalls / (precisions + recalls + 1e-7)
    if f1s.size <= 1:
        return 0.0
    return float(np.max(f1s[:-1]))


def compute_pro(masks, amaps, num_th=200):
    masks = np.asarray(masks).astype(np.uint8)
    amaps = np.asarray(amaps).astype(np.float32)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if amaps.ndim == 4 and amaps.shape[1] == 1:
        amaps = amaps[:, 0]
    if masks.shape != amaps.shape or masks.ndim != 3:
        raise ValueError("masks and amaps must be [N,H,W] with the same shape")
    if num_th <= 0:
        return 0.0

    min_th = float(amaps.min())
    max_th = float(amaps.max())
    if max_th <= min_th:
        return 0.0

    thresholds = np.linspace(min_th, max_th, num=num_th, endpoint=False, dtype=np.float32)
    inverse_masks = 1 - masks
    inverse_sum = float(inverse_masks.sum())
    if inverse_sum <= 0:
        return 0.0

    pro_list = []
    fpr_list = []
    for th in thresholds:
        binary_amaps = (amaps > th).astype(np.uint8)
        per_region_overlaps = []
        for pred, mask in zip(binary_amaps, masks):
            labeled = measure.label(mask, connectivity=1)
            for region in measure.regionprops(labeled):
                coords = region.coords
                tp_pixels = pred[coords[:, 0], coords[:, 1]].sum()
                per_region_overlaps.append(float(tp_pixels) / float(region.area))
        if not per_region_overlaps:
            continue
        fpr = float(np.logical_and(inverse_masks, binary_amaps).sum()) / inverse_sum
        pro_list.append(float(np.mean(per_region_overlaps)))
        fpr_list.append(fpr)

    if not pro_list:
        return 0.0

    fpr_arr = np.asarray(fpr_list, dtype=np.float32)
    pro_arr = np.asarray(pro_list, dtype=np.float32)
    valid = np.isfinite(fpr_arr) & np.isfinite(pro_arr) & (fpr_arr < 0.3)
    if not np.any(valid):
        return 0.0
    fpr_arr = fpr_arr[valid]
    pro_arr = pro_arr[valid]
    if fpr_arr.size < 2:
        return 0.0
    order = np.argsort(fpr_arr)
    fpr_arr = fpr_arr[order]
    pro_arr = pro_arr[order]

    fpr_max = float(fpr_arr.max())
    if fpr_max <= 0:
        return 0.0
    fpr_arr = fpr_arr / fpr_max
    return float(auc(fpr_arr, pro_arr))


def _safe_roc_auc(y_true, y_score):
    if np.asarray(y_true).size == 0 or np.unique(y_true).size < 2:
        return 0.0
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return 0.0


def _safe_average_precision(y_true, y_score):
    if np.asarray(y_true).size == 0 or np.unique(y_true).size < 2:
        return 0.0
    try:
        return float(average_precision_score(y_true, y_score))
    except ValueError:
        return 0.0


def _finalize_metrics(label_gt, img_map, mask_gt, seg_map):
    y_true_img = np.asarray(label_gt).astype(np.uint8)
    y_score_img = np.asarray(img_map).astype(np.float32)

    gt_px = np.asarray(mask_gt).astype(np.uint8)
    pr_px = np.asarray(seg_map).astype(np.float32)
    if gt_px.ndim == 4 and gt_px.shape[1] == 1:
        gt_px = gt_px[:, 0]
    if pr_px.ndim == 4 and pr_px.shape[1] == 1:
        pr_px = pr_px[:, 0]

    y_true_px_flat = gt_px.ravel()
    y_score_px_flat = pr_px.ravel()

    auroc_sp = _safe_roc_auc(y_true_img, y_score_img)
    ap_sp = _safe_average_precision(y_true_img, y_score_img)
    f1_sp = f1_score_max(y_true_img, y_score_img)

    auroc_px = _safe_roc_auc(y_true_px_flat, y_score_px_flat)
    ap_px = _safe_average_precision(y_true_px_flat, y_score_px_flat)
    f1_px = f1_score_max(y_true_px_flat, y_score_px_flat)
    aupro_px = compute_pro(gt_px, pr_px)

    return auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px


def evaluate_multiclass_epoch(
    localnet,
    feature_extractor,
    test_loader,
    args,
    device,
    extract_feature_batch_fn,
):
    seg_map, img_map, label_gt, mask_gt = [], [], [], []
    for images, y, mask in test_loader:
        with torch.no_grad():
            localnet.eval()
            feature_extractor.eval()
            images = images.to(device)
            y = y.detach().numpy()
            mask = mask.detach().numpy()
            features = extract_feature_batch_fn(images, feature_extractor, args)
            _, score = localnet(features)
            score = score.detach().cpu().numpy()
            img_map.append(aggregate_image_scores(score, topk_ratio=args.img_score_topk_ratio))
            score = score.reshape(-1, 28, 28)
            for i in range(score.shape[0]):
                _map = cv2.resize(score[i], _cv2_resize_dsize_from_mask(mask[i]))
                _map = gaussian_filter(_map, sigma=4)
                seg_map.append(_map)
                label_gt.append(y[i])
                mask_gt.append(mask[i])
    img_map = np.concatenate(img_map, axis=0)
    label_gt = np.array(label_gt)
    seg_map = np.stack(seg_map, axis=0)
    mask_gt = np.stack(mask_gt, axis=0)
    return _finalize_metrics(label_gt, img_map, mask_gt, seg_map)


def evaluate_residual_multiclass_epoch(
    localnet,
    feature_extractor,
    test_loader,
    args,
    class_idx_eval,
    reference_memory_by_class,
    reference_index_by_class,
    device,
    extract_feature_batch_fn,
    compute_residual_feature_batch_fn,
):
    seg_map, img_map, label_gt, mask_gt = [], [], [], []
    for images, y, mask in test_loader:
        with torch.no_grad():
            localnet.eval()
            feature_extractor.eval()
            images = images.to(device)
            y = y.detach().numpy()
            mask = mask.detach().numpy()
            features = extract_feature_batch_fn(images, feature_extractor, args)
            class_idx_batch = torch.full(
                (features.shape[0],),
                int(class_idx_eval),
                dtype=torch.long,
                device=features.device,
            )
            residual_features = compute_residual_feature_batch_fn(
                features,
                class_idx_batch,
                reference_memory_by_class,
                reference_index_by_class,
            )
            patch_class_idx = None
            if args.use_moe_discriminator and getattr(args, "moe_hard_class_gate", False):
                patch_class_idx = _build_patch_class_idx(class_idx_batch, residual_features)
            _, score = localnet(residual_features, patch_class_idx=patch_class_idx)
            score = score.detach().cpu().numpy()
            img_map.append(aggregate_image_scores(score, topk_ratio=args.img_score_topk_ratio))
            score = score.reshape(-1, 28, 28)
            for i in range(score.shape[0]):
                _map = cv2.resize(score[i], _cv2_resize_dsize_from_mask(mask[i]))
                _map = gaussian_filter(_map, sigma=4)
                seg_map.append(_map)
                label_gt.append(y[i])
                mask_gt.append(mask[i])
    img_map = np.concatenate(img_map, axis=0)
    label_gt = np.array(label_gt)
    seg_map = np.stack(seg_map, axis=0)
    mask_gt = np.stack(mask_gt, axis=0)
    return _finalize_metrics(label_gt, img_map, mask_gt, seg_map)


def evaluate_feature_epoch(localnet, test_loader, device):
    seg_map, img_map, label_gt, mask_gt = [], [], [], []
    for x, y, mask in test_loader:
        with torch.no_grad():
            localnet.eval()
            x = x.to(device)
            y = y.detach().numpy()
            mask = mask.detach().numpy()
            _, score = localnet(x)
            score = score.detach().cpu().numpy()
            img_map.append(score.max(axis=1))
            score = score.reshape(-1, 28, 28)
            for i in range(score.shape[0]):
                _map = cv2.resize(score[i], _cv2_resize_dsize_from_mask(mask[i]))
                _map = gaussian_filter(_map, sigma=4)
                seg_map.append(_map)
                label_gt.append(y[i])
                mask_gt.append(mask[i])
    img_map = np.concatenate(img_map, axis=0)
    label_gt = np.array(label_gt)
    seg_map = np.stack(seg_map, axis=0)
    mask_gt = np.stack(mask_gt, axis=0)
    return _finalize_metrics(label_gt, img_map, mask_gt, seg_map)
