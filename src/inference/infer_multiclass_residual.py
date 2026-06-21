import argparse
import os
import sys
from typing import Dict, List, Tuple

# Ensure project root is on sys.path (script lives at src/inference/infer_multiclass_residual.py)
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
    DINOV3_FEATURE_MODEL_REGISTRY,
    build_dinov3_feature_extractor,
    extract_dinov3_feature_batch,
    infer_dinov3_feature_dim,
)
from src.model import model
import utils.train_utils as common_utils
from utils.evaluate import _cv2_resize_dsize_from_mask
from dataset.multiclass_feature_dataset import get_all_class_names, DINO_CLASS_LAYER_INDICES


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# Per-class DINOv3 layer indices, shared via multiclass_feature_dataset.
# Defined in dataset/multiclass_feature_dataset.py: DINO_CLASS_LAYER_INDICES.


def str2bool(value):
    return common_utils.str2bool(value)


def parse_args():
    parser = argparse.ArgumentParser("infer_multiclass_residual")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/media/honeywell/D/bhy/dataset/MVTec_overlap/MVTec_noisy10",
    )
    parser.add_argument("--dataset", type=str, default="mvtec", choices=["mvtec", "visa"])
    parser.add_argument("--checkpoint_path", type=str, default="/media/honeywell/E/bhy/FUNAD/save_results/ablation_study/residual+cross_attention+multi_memorybank+moe_hard_gate/mvtec/10%/gaussian_True_noise_10%_balancing_True_oto_True_weight_2.5_multiclass_residual_localnet.pt")
    parser.add_argument("--output_dir", type=str, default="./output_10")
    parser.add_argument(
        "--class_names",
        type=str,
        nargs="+",
        default=None,
        help="推理类别列表，默认全部类别。",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_images_per_class", type=int, default=-1)
    parser.add_argument(
        "--img_score_topk_ratio",
        type=float,
        default=0.01,
        help="图像级分数采用 top-k patch 分数均值，k=ceil(num_patches*ratio)。",
    )
    parser.add_argument("--gaussian_sigma", type=float, default=4.0)
    parser.add_argument(
        "--heatmap_score_min",
        type=float,
        default=0.0,
        help="热力图固定映射下限（不做单图拉伸）。",
    )
    parser.add_argument(
        "--heatmap_score_max",
        type=float,
        default=1.0,
        help="热力图固定映射上限（不做单图拉伸）。",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_plot", action="store_true",
                        help="禁用所有可视化输出（heatmap/panel/overlay 绘图和保存），仅计算指标。")
    parser.add_argument("--save_overlay", action="store_true")
    parser.add_argument(
        "--overlay_alpha",
        type=float,
        default=0.45,
        help="原图与热力图叠加时的热力图权重。",
    )

    parser.add_argument("--faiss_cpu_index", action="store_true")
    parser.add_argument("--faiss_gpu_temp_mem_mb", type=int, default=256)

    parser.add_argument("--use_cls_token", type=str2bool, default=False)
    parser.add_argument(
        "--moe_use_cls_token",
        type=str2bool,
        default=False,
        help="Whether MoE discriminator gate uses cls_token as routing input.",
    )
    parser.add_argument(
        "--feature_model",
        type=str,
        choices=_FEATURE_MODEL_CHOICES,
        default="dinov3_vitl16",
        help="DINOv3 variant (torch.hub entry), e.g. dinov3_vitb16.",
    )
    parser.add_argument(
        "--dinov3_hub",
        type=str,
        default=None,
        help="本地 dinov3 torch hub 目录（默认: <torch.hub 目录>/facebookresearch_dinov3_main）。",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
        help="与 DINOv3 训练一致：Resize 边长。",
    )
    parser.add_argument(
        "--crop_size",
        type=int,
        default=448,
        help="与 DINOv3 训练一致：CenterCrop 边长。",
    )
    parser.add_argument(
        "--residual",
        type=str2bool,
        default=None,
        help="Enable residual features (feat - nearest_class_reference). "
        "Default: auto-detect from checkpoint (non-empty reference_memory or _no_residual tag).",
    )
    # MoE discriminator args (mirror training script)
    parser.add_argument(
        "--use_moe_discriminator",
        type=str2bool,
        default=None,
        help="Use MoE discriminator. Default: auto-detect from checkpoint.",
    )
    parser.add_argument(
        "--moe_num_expert",
        type=int,
        default=None,
        help="Number of experts. Default: auto-detect from checkpoint.",
    )
    parser.add_argument(
        "--moe_top_k",
        type=int,
        default=None,
        help="Top-k experts per token. Default: auto-detect (1 for hard_class_gate, else 2).",
    )
    parser.add_argument(
        "--moe_hard_class_gate",
        type=str2bool,
        default=None,
        help="Use hard class->expert routing. Default: auto-detect from checkpoint.",
    )

    return parser.parse_args()


def infer_model_params_from_checkpoint(state_dict_keys: set, args):
    """Detect MoE / adapter params from checkpoint state_dict, with CLI overrides.

    Returns a dict with keys: use_moe_discriminator, moe_num_expert, moe_top_k,
    moe_hard_class_gate, class_conditioned_adapter.
    """
    import re

    has_class_adapter = any(
        k.startswith("adaptor.class_embed") or k.startswith("adaptor.cross_attn")
        for k in state_dict_keys
    )

    use_moe = any(k.startswith("discriminator.moe_layer.") for k in state_dict_keys)
    if args.use_moe_discriminator is not None:
        use_moe = bool(args.use_moe_discriminator)

    hard_class = False
    num_expert = 4
    top_k = 2

    if use_moe:
        # Count experts from state dict
        expert_indices = set()
        for k in state_dict_keys:
            m = re.search(r'discriminator\.moe_layer\.experts\.(\d+)', k)
            if m:
                expert_indices.add(int(m.group(1)))
        inferred_num_expert = max(expert_indices) + 1 if expert_indices else 4

        # Detect hard class gate: MoE present but no gate.weight → ClassHardGate
        has_gate_weight = any(
            k.startswith("discriminator.moe_layer.gate.") and "weight" in k
            for k in state_dict_keys
        )
        inferred_hard = not has_gate_weight

        hard_class = args.moe_hard_class_gate if args.moe_hard_class_gate is not None else inferred_hard
        num_expert = args.moe_num_expert if args.moe_num_expert is not None else inferred_num_expert
        top_k = args.moe_top_k if args.moe_top_k is not None else (1 if hard_class else 2)

    return {
        "use_moe_discriminator": use_moe,
        "moe_num_expert": num_expert,
        "moe_top_k": top_k,
        "moe_hard_class_gate": hard_class,
        "class_conditioned_adapter": has_class_adapter,
    }


def infer_use_residual(checkpoint: dict, checkpoint_path: str, args) -> bool:
    """Detect whether to compute residual features, with CLI override."""
    if args.residual is not None:
        return bool(args.residual)

    ref_raw = checkpoint.get("reference_memory_by_class", {})
    has_ref = any(np.asarray(mem).size > 0 for mem in ref_raw.values())
    if has_ref:
        return True

    basename = os.path.basename(checkpoint_path)
    if "_no_residual_" in basename or basename.endswith("_no_residual_localnet.pt"):
        return False
    return False


def _set_dinov3_hub_env(hub_path: str):
    """Temporarily override DINOV3_HUB_DIR env var; returns a cleanup callable."""
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


def extract_feature_batch(input_tensor, feature_extractor, args, return_cls_token=False, class_indices=None, class_layer_indices=None):
    return extract_dinov3_feature_batch(
        input_tensor, feature_extractor,
        dino_layer_indices=class_layer_indices,
        class_indices=class_indices,
        use_cls_token=args.use_cls_token,
        return_cls_token=return_cls_token,
    )


def compute_residual_feature_batch(
    features,
    class_idx_batch,
    reference_memory_by_class: Dict[int, np.ndarray],
    reference_index_by_class,
    num_patches: int,
):
    feat_np = features.detach().cpu().numpy().astype(np.float32, copy=False)
    class_np = class_idx_batch.detach().cpu().numpy().astype(np.int64)
    dim = int(features.shape[-1])

    for cls in np.unique(class_np).tolist():
        cls = int(cls)
        cls_mask = class_np == cls
        cls_feat = feat_np[cls_mask].reshape(-1, dim)
        if cls not in reference_index_by_class:
            continue

        _, nearest_id = reference_index_by_class[cls].search(
            np.ascontiguousarray(cls_feat), k=1
        )
        nearest_feat = reference_memory_by_class[cls][nearest_id.reshape(-1)]
        cls_residual = (cls_feat - nearest_feat).reshape(-1, num_patches, dim)
        feat_np[cls_mask] = cls_residual

    return torch.as_tensor(feat_np, dtype=features.dtype, device=features.device)


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
    cv2.putText(
        out,
        title,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def make_triplet_panel(
    rgb_u8: np.ndarray,
    mask_1hw: np.ndarray,
    heatmap_bgr: np.ndarray,
) -> np.ndarray:
    orig_bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    mask_u8 = (mask_1hw.squeeze(0).astype(np.uint8) * 255)
    mask_bgr = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)

    orig_bgr = add_title(orig_bgr, "Image")
    mask_bgr = add_title(mask_bgr, "Mask")
    heatmap_bgr = add_title(heatmap_bgr, "Pred Heatmap")
    return np.concatenate([orig_bgr, mask_bgr, heatmap_bgr], axis=1)


def find_feature_dim_and_patches(
    args, feature_extractor, device, class_names: List[str], class_to_idx: dict,
    class_layer_indices=None,
) -> Tuple[int, int]:
    # Compute union of all layer indices (flat list) — same approach as training
    _all_layer_values = sorted(set(
        idx for lst in class_layer_indices.values() for idx in lst
    ))
    for class_name in class_names:
        test_set = dataset_extract.MyDataset(
            dataset_path=args.data_path,
            dataset=args.dataset,
            class_name=class_name,
            is_train=False,
            resize=args.image_size,
            cropsize=args.crop_size,
        )
        if len(test_set) == 0:
            continue
        from torch.utils.data import DataLoader
        dummy_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
        # Pass flat layer list; infer_dinov3_feature_dim doesn't need per-class routing.
        feature_dim = infer_dinov3_feature_dim(
            feature_extractor, dummy_loader, _all_layer_values,
            args.use_cls_token, device,
        )
        from PIL import Image
        img_tensor, _, _ = test_set[0]
        image = img_tensor.unsqueeze(0).to(device, non_blocking=True)
        # Need class_indices when class_layer_indices is a dict
        class_idx_tensor = torch.full((1,), int(class_to_idx[class_name]), dtype=torch.long, device=device)
        features = extract_dinov3_feature_batch(
            image, feature_extractor, class_layer_indices,
            class_indices=class_idx_tensor,
            use_cls_token=args.use_cls_token,
        )
        return feature_dim, int(features.shape[1])
    raise RuntimeError("测试集为空，无法推断特征维度与 patch 数。")


def build_reference_index(
    reference_memory_by_class_raw: Dict[int, np.ndarray],
    use_cuda: bool,
    use_cpu_index: bool,
    gpu_temp_mem_mb: int,
    *,
    allow_empty: bool = False,
) -> Tuple[Dict[int, np.ndarray], Dict[int, object]]:
    memory_np_by_class = {}
    index_by_class = {}
    for class_idx, mem in reference_memory_by_class_raw.items():
        cls = int(class_idx)
        mem_np = np.ascontiguousarray(np.asarray(mem, dtype=np.float32))
        if mem_np.size == 0:
            continue
        memory_np_by_class[cls] = mem_np
        index_by_class[cls] = common_utils.build_faiss_index(
            mem_np,
            use_cuda=use_cuda,
            use_cpu_index=use_cpu_index,
            gpu_temp_mem_mb=gpu_temp_mem_mb,
        )
    if len(index_by_class) == 0 and not allow_empty:
        raise RuntimeError("checkpoint 中未找到可用 reference_memory_by_class。")
    return memory_np_by_class, index_by_class


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


import numpy as np
from skimage import measure
from sklearn.metrics import auc

def compute_pro(masks, amaps, num_th=200):
    """
    计算MVTec AD官方标准的PRO-AUC（FPR 0-0.3范围内的曲线下面积）
    
    Args:
        masks: 真实掩码，形状为[N, H, W]或[N, 1, H, W]，0=正常，1=异常
        amaps: 异常分数图，形状与masks相同，值越高越异常
        num_th: 阈值数量
    
    Returns:
        pro_auc: PRO-AUC值
    """
    masks = np.asarray(masks).astype(np.uint8)
    amaps = np.asarray(amaps).astype(np.float32)
    
    # 处理4维输入
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if amaps.ndim == 4 and amaps.shape[1] == 1:
        amaps = amaps[:, 0]
    
    # 输入验证
    if masks.shape != amaps.shape or masks.ndim != 3:
        raise ValueError(f"masks和amaps形状必须相同且为[N,H,W]，当前masks={masks.shape}, amaps={amaps.shape}")
    if num_th <= 0:
        return 0.0
    
    min_th = float(amaps.min())
    max_th = float(amaps.max())
    if max_th <= min_th:
        return 0.0
    
    # 生成阈值（修正：包含最大值）
    thresholds = np.linspace(min_th, max_th, num=num_th, endpoint=True, dtype=np.float32)
    
    inverse_masks = 1 - masks
    inverse_sum = float(inverse_masks.sum())
    if inverse_sum <= 0:
        return 0.0  # 所有样本都是异常，无法计算FPR
    
    pro_list = []
    fpr_list = []
    
    for th in thresholds:
        binary_amaps = (amaps > th).astype(np.uint8)
        per_region_overlaps = []
        
        for pred, mask in zip(binary_amaps, masks):
            # 修正：使用8连通（MVTec官方标准）
            labeled = measure.label(mask, connectivity=2)
            for region in measure.regionprops(labeled):
                coords = region.coords
                tp_pixels = pred[coords[:, 0], coords[:, 1]].sum()
                overlap = float(tp_pixels) / float(region.area)
                per_region_overlaps.append(overlap)
        
        # 处理预测全黑的情况
        if not per_region_overlaps:
            pro = 0.0
        else:
            pro = float(np.mean(per_region_overlaps))
        
        fpr = float(np.logical_and(inverse_masks, binary_amaps).sum()) / inverse_sum
        
        pro_list.append(pro)
        fpr_list.append(fpr)
    
    # 转换为数组
    fpr_arr = np.asarray(fpr_list, dtype=np.float32)
    pro_arr = np.asarray(pro_list, dtype=np.float32)
    
    # 只保留FPR < 0.3的有效点（工业标准）
    valid = np.isfinite(fpr_arr) & np.isfinite(pro_arr) & (fpr_arr <= 0.3)
    if not np.any(valid):
        return 0.0
    
    fpr_arr = fpr_arr[valid]
    pro_arr = pro_arr[valid]
    
    if fpr_arr.size < 2:
        return 0.0
    
    # 按FPR排序
    order = np.argsort(fpr_arr)
    fpr_arr = fpr_arr[order]
    pro_arr = pro_arr[order]
    
    # 计算FPR 0-0.3范围内的AUC
    fpr_max = float(fpr_arr.max())
    if fpr_max <= 0:
        return 0.0
    
    # 归一化FPR到[0,1]区间
    fpr_normalized = fpr_arr / fpr_max
    
    return float(auc(fpr_normalized, pro_arr))


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

    y_true_px_flat = gt_px.ravel()
    y_score_px_flat = pr_px.ravel()

    auroc_sp = _safe_roc_auc(y_true_img, y_score_img)
    ap_sp = _safe_average_precision(y_true_img, y_score_img)
    f1_sp = f1_score_max(y_true_img, y_score_img)

    auroc_px = _safe_roc_auc(y_true_px_flat, y_score_px_flat)
    ap_px = _safe_average_precision(y_true_px_flat, y_score_px_flat)
    f1_px = f1_score_max(y_true_px_flat, y_score_px_flat)
    aupro_px = compute_pro(gt_px, pr_px)

    return {
        "auroc_sp": auroc_sp,
        "ap_sp": ap_sp,
        "f1_sp": f1_sp,
        "auroc_px": auroc_px,
        "ap_px": ap_px,
        "f1_px": f1_px,
        "aupro_px": aupro_px,
    }


def main():
    args = parse_args()
    common_utils.fix_seed(args.seed)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    all_class_names = get_all_class_names(args.dataset)
    target_class_names = args.class_names if args.class_names is not None else all_class_names
    unknown = [name for name in target_class_names if name not in all_class_names]
    if unknown:
        raise ValueError(f"未知类别: {unknown}; 可选: {all_class_names}")
    class_to_idx = {name: idx for idx, name in enumerate(all_class_names)}

    print("Loading checkpoint:", args.checkpoint_path)
    try:
        checkpoint = torch.load(
            args.checkpoint_path, map_location=device, weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
    if "net" not in checkpoint:
        raise KeyError("checkpoint 缺少 'net' 键。")
    if "reference_memory_by_class" not in checkpoint:
        raise KeyError("checkpoint 缺少 'reference_memory_by_class' 键。")

    use_residual = infer_use_residual(checkpoint, args.checkpoint_path, args)
    print(f"[model] use_residual={use_residual}")

    feature_extractor = build_feature_extractor(args, device)

    # Build per-class layer indices (matches training DINO_CLASS_LAYER_INDICES)
    # Must happen before find_feature_dim_and_patches which needs class_layer_indices.
    args.class_layer_indices = {
        i: DINO_CLASS_LAYER_INDICES.get(name, [-1])
        for i, name in enumerate(all_class_names)
    }

    feature_dim, num_patches = find_feature_dim_and_patches(
        args, feature_extractor, device, target_class_names, class_to_idx,
        class_layer_indices=args.class_layer_indices,
    )
    patch_side = int(np.sqrt(num_patches))
    if patch_side * patch_side != num_patches:
        raise RuntimeError(
            f"patch 数 {num_patches} 不是完全平方数，无法形成热力图网格。"
        )

    state_dict_keys = set(checkpoint["net"].keys())

    # Infer model params from checkpoint state dict (with CLI overrides)
    model_params = infer_model_params_from_checkpoint(state_dict_keys, args)
    has_class_adapter = model_params["class_conditioned_adapter"]
    use_moe = model_params["use_moe_discriminator"]
    moe_num_expert = model_params["moe_num_expert"]
    moe_top_k = model_params["moe_top_k"]
    moe_hard_class_gate = model_params["moe_hard_class_gate"]

    print(f"[model] class_conditioned_adapter={has_class_adapter}, "
          f"use_moe_discriminator={use_moe}")
    if use_moe:
        print(f"[model]   moe: num_expert={moe_num_expert}, top_k={moe_top_k}, "
              f"hard_class_gate={moe_hard_class_gate}")

    localnet = model.localnet(
        len_feature=feature_dim,
        use_moe_discriminator=use_moe,
        moe_num_expert=moe_num_expert,
        moe_top_k=moe_top_k,
        moe_use_cls_token=args.moe_use_cls_token,
        moe_hard_class_gate=moe_hard_class_gate,
        num_classes=len(all_class_names) if has_class_adapter else None,
        class_conditioned_adapter=has_class_adapter,
    ).to(device)
    localnet.load_state_dict(checkpoint["net"], strict=True)
    localnet.eval()
    feature_extractor.eval()

    reference_memory_by_class, reference_index_by_class = build_reference_index(
        reference_memory_by_class_raw=checkpoint["reference_memory_by_class"],
        use_cuda=use_cuda,
        use_cpu_index=bool(args.faiss_cpu_index),
        gpu_temp_mem_mb=int(args.faiss_gpu_temp_mem_mb),
        allow_empty=not use_residual,
    )
    if use_residual and len(reference_index_by_class) == 0:
        raise RuntimeError(
            "use_residual=True 但 checkpoint 中 reference_memory_by_class 为空。"
        )

    total_saved = 0
    metrics_rows = []
    for class_name in target_class_names:
        class_idx = class_to_idx[class_name]
        class_out_dir = os.path.join(args.output_dir, class_name)
        os.makedirs(class_out_dir, exist_ok=True)

        test_set = dataset_extract.MyDataset(
            dataset_path=args.data_path,
            dataset=args.dataset,
            class_name=class_name,
            is_train=False,
            resize=args.image_size,
            cropsize=args.crop_size,
        )
        test_loader = DataLoader(
            test_set,
            batch_size=args.batch_size,
            pin_memory=False,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
        )

        print(f"[{class_name}] test images: {len(test_set)}")
        sample_offset = 0
        class_saved = 0
        class_img_scores = []
        class_y_true = []
        class_seg_maps = []
        class_mask_gt = []
        with torch.no_grad():
            for images, y, mask in test_loader:
                images = images.to(device)
                class_idx_batch = torch.full(
                    (images.shape[0],),
                    int(class_idx),
                    dtype=torch.long,
                    device=device,
                )
                features, cls_token = extract_feature_batch(
                    images, feature_extractor, args, return_cls_token=True,
                    class_indices=class_idx_batch,
                    class_layer_indices=args.class_layer_indices,
                )
                if use_residual:
                    model_features = compute_residual_feature_batch(
                        features,
                        class_idx_batch,
                        reference_memory_by_class,
                        reference_index_by_class,
                        num_patches=num_patches,
                    )
                else:
                    model_features = features
                patch_class_idx = None
                if use_moe and moe_hard_class_gate:
                    token_per_image = (
                        int(model_features.shape[1]) if model_features.dim() >= 2 else 1
                    )
                    patch_class_idx = (
                        class_idx_batch.to(device=model_features.device, dtype=torch.long)
                        .reshape(-1, 1, 1)
                        .expand(-1, token_per_image, 1)
                        .contiguous()
                    )
                _, score = localnet(
                    model_features,
                    cls_token=cls_token,
                    patch_class_idx=patch_class_idx,
                    class_idx=class_idx_batch,
                )
                score_np = score.detach().cpu().numpy().reshape(-1, patch_side, patch_side)
                score_flat = score.detach().cpu().numpy().reshape(score_np.shape[0], -1)
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
                    class_seg_maps.append(one_score)
                    class_mask_gt.append(mask_np[i])
                    if not args.no_plot:
                        heatmap_bgr = colorize_heatmap(
                            one_score,
                            score_min=args.heatmap_score_min,
                            score_max=args.heatmap_score_max,
                        )

                        rgb_u8 = denormalize_imagenet(image_np[i])
                        if args.max_images_per_class <= 0 or class_saved < args.max_images_per_class:
                            panel = make_triplet_panel(rgb_u8, mask_np[i], heatmap_bgr)

                            src_path = test_set.x[sample_offset + i]
                            stem = os.path.splitext(os.path.basename(src_path))[0]
                            tag = "anomaly" if int(y_np[i]) == 1 else "good"
                            save_name = f"{class_name}_{sample_offset + i:05d}_{tag}_{stem}.png"
                            save_path = os.path.join(class_out_dir, save_name)
                            cv2.imwrite(save_path, panel)

                            if args.save_overlay:
                                orig_bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
                                overlay = cv2.addWeighted(
                                    orig_bgr,
                                    1.0 - float(args.overlay_alpha),
                                    heatmap_bgr,
                                    float(args.overlay_alpha),
                                    0.0,
                                )
                                overlay_name = save_name.replace(".png", "_overlay.png")
                                cv2.imwrite(os.path.join(class_out_dir, overlay_name), overlay)

                            class_saved += 1
                            total_saved += 1

                class_img_scores.append(
                    common_utils.aggregate_image_scores(
                        score_flat, topk_ratio=args.img_score_topk_ratio
                    )
                )
                class_y_true.append(y_np)

                sample_offset += score_np.shape[0]

        y_true_img = np.concatenate(class_y_true, axis=0)
        y_score_img = np.concatenate(class_img_scores, axis=0)
        gt_px = np.stack(class_mask_gt, axis=0)
        pr_px = np.stack(class_seg_maps, axis=0)
        class_metrics = finalize_metrics(y_true_img, y_score_img, gt_px, pr_px)
        metrics_rows.append({"class": class_name, **class_metrics})

        print(
            (
                f"[{class_name}] metrics | auroc_sp: {class_metrics['auroc_sp']:.5f}, "
                f"ap_sp: {class_metrics['ap_sp']:.5f}, f1_sp: {class_metrics['f1_sp']:.5f}, "
                f"auroc_px: {class_metrics['auroc_px']:.5f}, ap_px: {class_metrics['ap_px']:.5f}, "
                f"f1_px: {class_metrics['f1_px']:.5f}, aupro_px: {class_metrics['aupro_px']:.5f}"
            )
        )
        print(f"[{class_name}] saved: {class_saved}")

    metrics_df = pd.DataFrame(
        metrics_rows,
        columns=["class", "auroc_sp", "ap_sp", "f1_sp", "auroc_px", "ap_px", "f1_px", "aupro_px"],
    )
    if len(metrics_df) > 0:
        mean_row = {"class": "mean"}
        for metric_name in ["auroc_sp", "ap_sp", "f1_sp", "auroc_px", "ap_px", "f1_px", "aupro_px"]:
            mean_row[metric_name] = float(metrics_df[metric_name].mean())
        metrics_df = pd.concat([metrics_df, pd.DataFrame([mean_row])], ignore_index=True)

    metrics_csv_path = os.path.join(args.output_dir, "class_metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)

    print("Metrics saved:", metrics_csv_path)
    if len(metrics_df) > 0:
        print(metrics_df.to_string(index=False))
    if args.no_plot:
        print("Done. (plotting disabled, no panels saved).")
    else:
        print("Done. Total saved panels:", total_saved)
    print("Output dir:", args.output_dir)


if __name__ == "__main__":
    main()
