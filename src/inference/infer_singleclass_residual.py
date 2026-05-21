import argparse
import glob
import os
import sys

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter
from skimage import measure
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from dataset import dataset_extract
from dataset.feature_extract import (
    _FEATURE_MODEL_CHOICES,
    build_dinov3_feature_extractor,
    extract_dinov3_feature_batch,
    infer_dinov3_feature_dim,
    resolve_dino_block_indices,
)
from dataset.multiclass_feature_dataset import DINO_CLASS_LAYER_INDICES, get_all_class_names
from src.model import model
import utils.train_utils as common_utils
from utils.evaluate import _cv2_resize_dsize_from_mask


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def str2bool(value):
    return common_utils.str2bool(value)


def parse_args():
    parser = argparse.ArgumentParser("infer_singleclass_residual")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/media/honeywell/D/bhy/dataset/MVTec_overlap/MVTec_noisy10",
    )
    parser.add_argument("--dataset", type=str, default="mvtec", choices=["mvtec", "visa"])
    parser.add_argument("--class_name", type=str, default="all",
                        help="类别名称，默认 'all' 表示推理全部类别。")
    parser.add_argument("--checkpoint_path", type=str, default="/media/honeywell/E/bhy/FUNAD/save_results/singleclass_residual_dinov3vitl16/mvtec/10%",
                        help="训练权重根目录，每类权重在 {checkpoint_path}/{class_name}/*_localnet.pt")
    parser.add_argument("--output_dir", type=str, default="./output_single")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_images_per_class", type=int, default=-1)
    parser.add_argument("--img_score_topk_ratio", type=float, default=0.01)
    parser.add_argument("--gaussian_sigma", type=float, default=4.0)
    parser.add_argument("--heatmap_score_min", type=float, default=0.0)
    parser.add_argument("--heatmap_score_max", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_overlay", action="store_true")
    parser.add_argument("--overlay_alpha", type=float, default=0.45)
    parser.add_argument("--faiss_cpu_index", action="store_true")
    parser.add_argument("--faiss_gpu_temp_mem_mb", type=int, default=256)
    parser.add_argument("--feature_model", type=str, choices=_FEATURE_MODEL_CHOICES,
                        default="dinov3_vitl16")
    parser.add_argument("--dinov3_hub", type=str, default=None)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--crop_size", type=int, default=448)
    return parser.parse_args()


def _set_dinov3_hub_env(hub_path: str):
    if not hub_path:
        return None
    old_val = os.environ.get("DINOV3_HUB_DIR")
    os.environ["DINOV3_HUB_DIR"] = hub_path
    def restore():
        if old_val is None:
            os.environ.pop("DINOV3_HUB_DIR", None)
        else:
            os.environ["DINOV3_HUB_DIR"] = old_val
    return restore


def build_feature_extractor(args, device):
    restore_hub = _set_dinov3_hub_env(args.dinov3_hub)
    try:
        return build_dinov3_feature_extractor(args.feature_model, device)
    finally:
        if restore_hub is not None:
            restore_hub()


def denormalize_imagenet(img_chw: np.ndarray) -> np.ndarray:
    img = np.transpose(img_chw, (1, 2, 0))
    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)
    img = (img * std) + mean
    img = np.clip(img, 0.0, 1.0)
    return (img * 255.0).astype(np.uint8)


def colorize_heatmap(score_224: np.ndarray, score_min: float, score_max: float) -> np.ndarray:
    score = score_224.astype(np.float32)
    denom = float(score_max - score_min)
    if denom <= 1e-12:
        score_u8 = np.zeros_like(score, dtype=np.uint8)
    else:
        score = (score - float(score_min)) / denom
        score = np.clip(score, 0.0, 1.0)
        score_u8 = (score * 255.0).astype(np.uint8)
    return cv2.applyColorMap(score_u8, cv2.COLORMAP_JET)


def add_title(img_bgr: np.ndarray, title: str) -> np.ndarray:
    out = img_bgr.copy()
    cv2.putText(out, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def make_triplet_panel(rgb_u8: np.ndarray, mask_1hw: np.ndarray,
                       heatmap_bgr: np.ndarray) -> np.ndarray:
    orig_bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    mask_u8 = (mask_1hw.squeeze(0).astype(np.uint8) * 255)
    mask_bgr = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)
    orig_bgr = add_title(orig_bgr, "Image")
    mask_bgr = add_title(mask_bgr, "Mask")
    heatmap_bgr = add_title(heatmap_bgr, "Pred Heatmap")
    return np.concatenate([orig_bgr, mask_bgr, heatmap_bgr], axis=1)


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
        return 0.0
    if num_th <= 0:
        return 0.0
    min_th = float(amaps.min())
    max_th = float(amaps.max())
    if max_th <= min_th:
        return 0.0
    thresholds = np.linspace(min_th, max_th, num=num_th, endpoint=True, dtype=np.float32)
    inverse_masks = 1 - masks
    inverse_sum = float(inverse_masks.sum())
    if inverse_sum <= 0:
        return 0.0
    pro_list, fpr_list = [], []
    for th in thresholds:
        binary_amaps = (amaps > th).astype(np.uint8)
        per_region_overlaps = []
        for pred, mask in zip(binary_amaps, masks):
            labeled = measure.label(mask, connectivity=2)
            for region in measure.regionprops(labeled):
                coords = region.coords
                tp_pixels = pred[coords[:, 0], coords[:, 1]].sum()
                per_region_overlaps.append(float(tp_pixels) / float(region.area))
        pro_list.append(float(np.mean(per_region_overlaps)) if per_region_overlaps else 0.0)
        fpr_list.append(float(np.logical_and(inverse_masks, binary_amaps).sum()) / inverse_sum)
    fpr_arr = np.asarray(fpr_list, dtype=np.float32)
    pro_arr = np.asarray(pro_list, dtype=np.float32)
    valid = np.isfinite(fpr_arr) & np.isfinite(pro_arr) & (fpr_arr <= 0.3)
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
    return float(auc(fpr_arr / fpr_max, pro_arr))


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


def finalize_metrics(y_true_img, y_score_img, gt_px, pr_px):
    y_true_img = np.asarray(y_true_img).astype(np.uint8)
    y_score_img = np.asarray(y_score_img).astype(np.float32)
    gt_px = np.asarray(gt_px).astype(np.uint8)
    pr_px = np.asarray(pr_px).astype(np.float32)
    if gt_px.ndim == 4 and gt_px.shape[1] == 1:
        gt_px = gt_px[:, 0]
    if pr_px.ndim == 4 and pr_px.shape[1] == 1:
        pr_px = pr_px[:, 0]
    return {
        "auroc_sp": _safe_roc_auc(y_true_img, y_score_img),
        "ap_sp": _safe_average_precision(y_true_img, y_score_img),
        "f1_sp": f1_score_max(y_true_img, y_score_img),
        "auroc_px": _safe_roc_auc(gt_px.ravel(), pr_px.ravel()),
        "ap_px": _safe_average_precision(gt_px.ravel(), pr_px.ravel()),
        "f1_px": f1_score_max(gt_px.ravel(), pr_px.ravel()),
        "aupro_px": compute_pro(gt_px, pr_px),
    }


def build_reference_index(reference_memory: np.ndarray, use_cuda: bool,
                          use_cpu_index: bool, gpu_temp_mem_mb: int):
    mem_np = np.ascontiguousarray(np.asarray(reference_memory, dtype=np.float32))
    if mem_np.size == 0:
        raise RuntimeError("reference_memory 为空。")
    index = common_utils.build_faiss_index(
        mem_np,
        use_cuda=use_cuda,
        use_cpu_index=use_cpu_index,
        gpu_temp_mem_mb=gpu_temp_mem_mb,
    )
    return mem_np, index


def compute_residual_faiss(features: torch.Tensor, reference_memory: np.ndarray,
                           reference_index, num_patches: int, device: torch.device):
    feat_np = features.detach().cpu().numpy().astype(np.float32, copy=False)
    dim = int(features.shape[-1])
    orig_shape = feat_np.shape
    feat_2d = feat_np.reshape(-1, dim)
    _, nearest_id = reference_index.search(np.ascontiguousarray(feat_2d), k=1)
    nearest_feat = reference_memory[nearest_id.reshape(-1)]
    residual = (feat_2d - nearest_feat).reshape(orig_shape)
    return torch.as_tensor(residual, dtype=features.dtype, device=device)


def infer_single_class(args, class_name, feature_extractor, device, use_cuda):
    """推理单个类别，返回指标 dict。"""
    # ── 3. 拼接 checkpoint 路径 ─────────────────────────────
    class_ckpt_dir = os.path.join(args.checkpoint_path, class_name)
    pt_files = sorted(glob.glob(os.path.join(class_ckpt_dir, "*_localnet.pt")))
    if not pt_files:
        raise FileNotFoundError(f"在 {class_ckpt_dir} 下未找到 *_localnet.pt 文件。")
    checkpoint_path = pt_files[0]
    print(f"\n[{class_name}] checkpoint: {checkpoint_path}")

    # ── 4. 加载 checkpoint ──────────────────────────────────
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    # ── 5. 确定层索引 & 特征维度 & patch 数 ────────────────
    class_layer_indices = DINO_CLASS_LAYER_INDICES.get(class_name, [-1])
    all_layer_values = sorted(set(class_layer_indices))

    test_set_for_dim = dataset_extract.MyDataset(
        dataset_path=args.data_path, dataset=args.dataset,
        class_name=class_name, is_train=False,
        resize=args.image_size, cropsize=args.crop_size,
    )
    dummy_loader = DataLoader(test_set_for_dim, batch_size=1, shuffle=False, num_workers=0)
    feature_dim = infer_dinov3_feature_dim(
        feature_extractor, dummy_loader, all_layer_values, False, device,
    )
    img_tensor, _, _ = test_set_for_dim[0]
    features = extract_dinov3_feature_batch(
        img_tensor.unsqueeze(0).to(device), feature_extractor, all_layer_values,
    )
    num_patches = int(features.shape[1])
    patch_side = int(np.sqrt(num_patches))
    if patch_side * patch_side != num_patches:
        raise RuntimeError(f"[{class_name}] patch 数 {num_patches} 不是完全平方数。")

    _selected_blocks = resolve_dino_block_indices(feature_extractor, all_layer_values)
    print(f"[{class_name}] feature_dim={feature_dim}, num_patches={num_patches}, "
          f"patch_side={patch_side}, layers={_selected_blocks}")

    # ── 6. 构建模型 ─────────────────────────────────────────
    localnet_model = model.localnet(
        len_feature=feature_dim,
        use_moe_discriminator=False,
        num_classes=1,
        class_conditioned_adapter=False,
    ).to(device)
    localnet_model.load_state_dict(checkpoint["net"], strict=True)
    localnet_model.eval()

    # ── 7. 构建 FAISS 索引（参考记忆库）────────────────────
    ref_mem_dict = checkpoint["reference_memory_by_class"]
    ref_mem = next(iter(ref_mem_dict.values()))
    ref_mem = np.asarray(ref_mem, dtype=np.float32)
    print(f"[{class_name}] reference_memory shape: {ref_mem.shape}")

    reference_memory, reference_index = build_reference_index(
        ref_mem,
        use_cuda=use_cuda,
        use_cpu_index=bool(args.faiss_cpu_index),
        gpu_temp_mem_mb=int(args.faiss_gpu_temp_mem_mb),
    )

    # ── 8. 测试集 DataLoader ──────────────────────────────
    test_set = dataset_extract.MyDataset(
        dataset_path=args.data_path, dataset=args.dataset,
        class_name=class_name, is_train=False,
        resize=args.image_size, cropsize=args.crop_size,
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, pin_memory=False,
        shuffle=False, num_workers=args.num_workers, drop_last=False,
    )
    print(f"[{class_name}] test images: {len(test_set)}")

    # ── 9. 推理循环 ─────────────────────────────────────────
    class_out_dir = os.path.join(args.output_dir, class_name)
    os.makedirs(class_out_dir, exist_ok=True)

    saved_count = 0
    img_scores = []
    y_true_list = []
    seg_maps = []
    mask_gt_list = []

    with torch.no_grad():
        sample_offset = 0
        for images, y, mask in test_loader:
            images = images.to(device)

            features = extract_dinov3_feature_batch(
                images, feature_extractor, all_layer_values,
            )
            residual_features = compute_residual_faiss(
                features, reference_memory, reference_index, num_patches, device,
            )
            _, score = localnet_model(residual_features)
            score_np = score.detach().cpu().numpy().reshape(-1, patch_side, patch_side)
            score_flat = score_np.reshape(score_np.shape[0], -1)

            image_np = images.detach().cpu().numpy()
            y_np = y.detach().cpu().numpy().astype(np.int64)
            mask_np = mask.detach().cpu().numpy().astype(np.uint8)

            for i in range(score_np.shape[0]):
                one_score = cv2.resize(
                    score_np[i],
                    _cv2_resize_dsize_from_mask(mask_np[i]),
                    interpolation=cv2.INTER_LINEAR,
                )
                one_score = gaussian_filter(one_score, sigma=args.gaussian_sigma)
                seg_maps.append(one_score)
                mask_gt_list.append(mask_np[i])

                heatmap_bgr = colorize_heatmap(
                    one_score,
                    score_min=args.heatmap_score_min,
                    score_max=args.heatmap_score_max,
                )
                rgb_u8 = denormalize_imagenet(image_np[i])

                if args.max_images_per_class <= 0 or saved_count < args.max_images_per_class:
                    panel = make_triplet_panel(rgb_u8, mask_np[i], heatmap_bgr)
                    src_path = test_set.x[sample_offset + i]
                    stem = os.path.splitext(os.path.basename(src_path))[0]
                    tag = "anomaly" if int(y_np[i]) == 1 else "good"
                    save_name = f"{class_name}_{sample_offset + i:05d}_{tag}_{stem}.png"
                    cv2.imwrite(os.path.join(class_out_dir, save_name), panel)

                    if args.save_overlay:
                        orig_bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
                        overlay = cv2.addWeighted(
                            orig_bgr, 1.0 - float(args.overlay_alpha),
                            heatmap_bgr, float(args.overlay_alpha), 0.0,
                        )
                        cv2.imwrite(
                            os.path.join(class_out_dir, save_name.replace(".png", "_overlay.png")),
                            overlay,
                        )
                    saved_count += 1

            img_scores.append(
                common_utils.aggregate_image_scores(score_flat, topk_ratio=args.img_score_topk_ratio)
            )
            y_true_list.append(y_np)
            sample_offset += score_np.shape[0]

    # ── 10. 指标汇总 ────────────────────────────────────────
    y_true_img = np.concatenate(y_true_list, axis=0)
    y_score_img = np.concatenate(img_scores, axis=0)
    gt_px = np.stack(mask_gt_list, axis=0)
    pr_px = np.stack(seg_maps, axis=0)

    metrics = finalize_metrics(y_true_img, y_score_img, gt_px, pr_px)
    metrics["class"] = class_name

    print(
        f"[{class_name}] metrics | auroc_sp: {metrics['auroc_sp']:.5f}, "
        f"ap_sp: {metrics['ap_sp']:.5f}, f1_sp: {metrics['f1_sp']:.5f}, "
        f"auroc_px: {metrics['auroc_px']:.5f}, ap_px: {metrics['ap_px']:.5f}, "
        f"f1_px: {metrics['f1_px']:.5f}, aupro_px: {metrics['aupro_px']:.5f}"
    )
    print(f"[{class_name}] saved panels: {saved_count}")
    return metrics


def main():
    args = parse_args()
    common_utils.fix_seed(args.seed)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. 确定推理类别列表 ─────────────────────────────────
    all_names = get_all_class_names(args.dataset)
    if args.class_name == "all":
        target_classes = all_names
    else:
        target_classes = [args.class_name]

    unknown = [n for n in target_classes if n not in all_names]
    if unknown:
        raise ValueError(f"未知类别: {unknown}; 可选: {all_names}")

    print(f"Classes to infer ({len(target_classes)}): {target_classes}")

    # ── 2. 构建共享 feature extractor ───────────────────────
    feature_extractor = build_feature_extractor(args, device)

    # ── 3. 逐类推理 ─────────────────────────────────────────
    all_metrics = []
    for class_name in target_classes:
        try:
            metrics = infer_single_class(args, class_name, feature_extractor, device, use_cuda)
            all_metrics.append(metrics)
        except Exception as e:
            print(f"[{class_name}] ERROR: {e}")
            import traceback
            traceback.print_exc()

    # ── 4. 汇总并保存 ───────────────────────────────────────
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        mean_row = {"class": "mean"}
        for metric_name in ["auroc_sp", "ap_sp", "f1_sp", "auroc_px", "ap_px", "f1_px", "aupro_px"]:
            mean_row[metric_name] = float(metrics_df[metric_name].mean())
        metrics_df = pd.concat([metrics_df, pd.DataFrame([mean_row])], ignore_index=True)

        csv_path = os.path.join(args.output_dir, "class_metrics.csv")
        metrics_df.to_csv(csv_path, index=False)
        print("\nFinal metrics saved:", csv_path)
        print(metrics_df.to_string(index=False))
    else:
        print("没有成功推理任何类别。")

    print("Done. Output dir:", args.output_dir)


if __name__ == "__main__":
    main()
