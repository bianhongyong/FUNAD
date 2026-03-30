import argparse
import datetime
import os
import random
import sys
import time
import warnings

import cv2
import faiss
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import tqdm
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset

import AnomalyCLIP_lib
import dataset_extract
import model
from multiclass_feature_dataset import MultiClassFeatureDataset, get_all_class_names

warnings.filterwarnings("ignore")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
_FAISS_GPU_RESOURCES = None
_FAISS_USE_CPU_INDEX = False
_FAISS_GPU_TEMP_MEM_MB = 256
_LOG_STREAM_HOLDER = []


def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y", "t"}:
        return True
    if lowered in {"false", "0", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected, e.g. true/false")


class _TeeStream:
    """Mirror writes to terminal and log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def _enable_print_logging(log_path: str):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_f = open(log_path, "a", encoding="utf-8")
    _LOG_STREAM_HOLDER.append(log_f)
    sys.stdout = _TeeStream(sys.__stdout__, log_f)
    sys.stderr = _TeeStream(sys.__stderr__, log_f)
    print(f"[log] mirrored stdout/stderr to: {log_path}")


def parse_args():
    parser = argparse.ArgumentParser("self-train_ad_multiclass")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/media/honeywell/D/bhy/dataset/MVTec_overlap/MVTec_noisy10",
    )
    parser.add_argument(
        "--save_path", type=str, default="/media/honeywell/E/bhy/FUNAD/save_results/muti_class_residual"
    )
    parser.add_argument("--kl", action="store_false")
    parser.add_argument("--beta", action="store_true")
    parser.add_argument("--gaussian", action="store_false")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--hist", action="store_true")
    parser.add_argument(
        "--dataset", type=str, default="mvtec", choices=["mvtec", "visa"]
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-l", "--lr", type=float, default=2e-5)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("-b", "--batch_size", type=int, default=16)
    parser.add_argument("-r", "--random", type=float, default=0.1)
    parser.add_argument("-t", "--threshold", type=float, default=0.5)
    parser.add_argument("-n", "--noise_threshold", type=float, default=0.9)
    parser.add_argument(
        "--noise",
        type=str,
        default="10%",
        choices=["0%", "1%", "2%", "3%", "5%", "10%", "20%"],
    )
    parser.add_argument("--std", type=float, default=None)
    parser.add_argument("--k_number", type=int, default=2)
    parser.add_argument("--llambda", type=float, default=1)
    parser.add_argument("--weight", type=float, default=0)
    parser.add_argument("--iter", type=int, default=0)
    parser.add_argument("--beta_number", type=int, default=15)
    parser.add_argument("--alternative", action="store_true")
    parser.add_argument("--balancing", action="store_false")
    parser.add_argument(
        "--oto_loss", type=str, choices=["kl", "mae", "mse"], default="mae"
    )
    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument("--save_log", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    # parser.add_argument("--max_bank_images", type=int, default=128)
    parser.add_argument("--faiss_cpu_index", action="store_true")
    parser.add_argument("--faiss_gpu_temp_mem_mb", type=int, default=256)
    # parser.add_argument(
    #     "--bank_sample_ratio",
    #     type=float,
    #     default=0.25,
    #     choices=[0.05, 0.1],
    #     help="Random sampling ratio used before memory bank construction (5% or 10%).",
    # )
    parser.add_argument("--num_reference_images_per_class", type=int, default=4)
    parser.add_argument("--strict_clean_reference", action="store_true")
    parser.add_argument("--use_class_adaptive_threshold", action="store_true")
    parser.add_argument("--adaptive_threshold_quantile", type=float, default=0.7)
    parser.add_argument("--use_origin_regularizer", action="store_true")
    parser.add_argument("--origin_normal_weight", type=float, default=0.001)
    parser.add_argument("--origin_anomaly_weight", type=float, default=0.001)
    parser.add_argument(
        "--img_score_topk_ratio",
        type=float,
        default=0.01,
        help="Image-level score uses mean of top-k patch scores, k=ceil(num_patches*ratio).",
    )

    parser.add_argument("--use_cls_token", type=str2bool, default=True)
    parser.add_argument("--feature_model", type=str, choices=["dino", "clip"], default="dino")
    parser.add_argument("--clip_model_name", type=str, default="ViT-L/14@336px")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24])
    parser.add_argument("--dpam_layer", type=int, default=24)
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--n_ctx", type=int, default=12)
    parser.add_argument("--t_n_ctx", type=int, default=4)

    return parser.parse_args()


def fix_seed(number):
    np.random.seed(number)
    random.seed(number)
    torch.manual_seed(number)
    torch.cuda.manual_seed(number)
    torch.cuda.manual_seed_all(number)
    cudnn.benchmark = False
    cudnn.deterministic = True


def find_matching(id_array):
    matching = [[], []]
    for i in range(id_array.shape[0]):
        if i in matching[1]:
            continue
        if i == id_array[id_array[i]]:
            matching[0].append(i)
            matching[1].append(id_array[i])
    return matching


def _convert_imagenet_norm_to_clip_norm(input_tensor):
    mean_imagenet = torch.tensor(IMAGENET_MEAN, device=input_tensor.device, dtype=input_tensor.dtype).view(1, 3, 1, 1)
    std_imagenet = torch.tensor(IMAGENET_STD, device=input_tensor.device, dtype=input_tensor.dtype).view(1, 3, 1, 1)
    mean_clip = torch.tensor(CLIP_MEAN, device=input_tensor.device, dtype=input_tensor.dtype).view(1, 3, 1, 1)
    std_clip = torch.tensor(CLIP_STD, device=input_tensor.device, dtype=input_tensor.dtype).view(1, 3, 1, 1)

    rgb_01 = input_tensor * std_imagenet + mean_imagenet
    rgb_01 = torch.clamp(rgb_01, 0.0, 1.0)
    clip_tensor = (rgb_01 - mean_clip) / std_clip
    return clip_tensor


def build_feature_extractor(args):
    if args.feature_model == "clip":
        anomalyclip_parameters = {
            "Prompt_length": args.n_ctx,
            "learnabel_text_embedding_depth": args.depth,
            "learnabel_text_embedding_length": args.t_n_ctx,
        }
        feature_extractor, _ = AnomalyCLIP_lib.load(
            args.clip_model_name,
            device=device,
            design_details=anomalyclip_parameters,
        )
        feature_extractor.eval()
        feature_extractor.visual.DAPM_replace(DPAM_layer=args.dpam_layer)
        return feature_extractor

    feature_extractor = torch.hub.load("facebookresearch/dino:main", "dino_vitb8")
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()
    return feature_extractor


def extract_feature_batch(input_tensor, feature_extractor, args):
    with torch.no_grad():
        feature_extractor.eval()
        if args.feature_model == "clip":
            clip_input = _convert_imagenet_norm_to_clip_norm(input_tensor)
            image_features, _, _, patch_projections = feature_extractor.encode_image(
                clip_input,
                args.features_list,
                DPAM_layer=args.dpam_layer,
            )
            x_norm = image_features
            x_prenorm = patch_projections[-1]
        else:
            feature = feature_extractor.get_intermediate_layers(input_tensor)[0]
            x_norm = feature[:, 0, :]
            x_prenorm = feature[:, 1:, :]

    if args.use_cls_token:
        x_norm = torch.repeat_interleave(x_norm.unsqueeze(1), x_prenorm.shape[1], dim=1)
        x_prenorm = torch.cat([x_norm, x_prenorm], dim=-1)

    return x_prenorm


def infer_feature_dim(feature_extractor, train_loader, args):
    for images, _, _ in train_loader:
        images = images.to(device)
        features = extract_feature_batch(images, feature_extractor, args)
        return int(features.shape[-1])
    raise RuntimeError("训练集为空，无法推断特征维度。")


def select_reference_indices_by_class(train_dataset, num_classes, args):
    rng = np.random.default_rng(args.seed)
    reference_indices_by_class = {class_idx: [] for class_idx in range(num_classes)}

    for class_idx in range(num_classes):
        cls_candidates = []
        for idx, (path, cls) in enumerate(train_dataset.samples):
            if int(cls) != int(class_idx):
                continue
            filename = os.path.basename(path).lower()
            if filename.startswith("noisy"):
                continue
            cls_candidates.append(idx)

        if len(cls_candidates) == 0:
            msg = f"class {class_idx} 没有可用 clean 参考图（已过滤 noisy*）。"
            if args.strict_clean_reference:
                raise RuntimeError(msg)
            warnings.warn(msg)
            continue

        sample_n = min(max(1, args.num_reference_images_per_class), len(cls_candidates))
        selected = rng.choice(np.array(cls_candidates), size=sample_n, replace=False)
        reference_indices_by_class[class_idx] = sorted(selected.tolist())

    return reference_indices_by_class


def build_reference_memory_bank(
    train_dataset, reference_indices_by_class, feature_extractor, args
):
    all_indices = []
    for class_idx in sorted(reference_indices_by_class.keys()):
        all_indices.extend(reference_indices_by_class[class_idx])

    if len(all_indices) == 0:
        raise RuntimeError("没有参考图可用于构建残差记忆库。")

    reference_subset = Subset(train_dataset, all_indices)
    reference_loader = DataLoader(
        reference_subset,
        batch_size=max(1, min(args.batch_size, 16)),
        pin_memory=False,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    memory_by_class = {class_idx: [] for class_idx in reference_indices_by_class.keys()}
    with torch.no_grad():
        feature_extractor.eval()
        for images, class_idx, _ in reference_loader:
            images = images.to(device)
            base_features = extract_feature_batch(images, feature_extractor, args)
            feat_np = base_features.detach().cpu().numpy()
            class_np = class_idx.detach().cpu().numpy().astype(np.int64)
            dim = int(base_features.shape[-1])

            for cls in np.unique(class_np).tolist():
                cls = int(cls)
                cls_mask = class_np == cls
                cls_feat = feat_np[cls_mask].reshape(-1, dim)
                memory_by_class.setdefault(cls, []).append(cls_feat)

    memory_np_by_class = {}
    index_by_class = {}
    for class_idx, feat_list in memory_by_class.items():
        if len(feat_list) == 0:
            continue
        cls_memory = np.concatenate(feat_list, axis=0).astype(np.float32)
        memory_np_by_class[int(class_idx)] = cls_memory
        index_by_class[int(class_idx)] = _build_faiss_index(cls_memory)

    if len(index_by_class) == 0:
        raise RuntimeError("参考图特征提取失败，无法构建残差记忆库。")

    return memory_np_by_class, index_by_class


def compute_residual_feature_batch(
    features, class_idx_batch, reference_memory_by_class, reference_index_by_class
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
        cls_residual = (cls_feat - nearest_feat).reshape(-1, 784, dim)
        feat_np[cls_mask] = cls_residual

    return torch.as_tensor(feat_np, dtype=features.dtype, device=features.device)


def remove_reference_samples_from_dataset(train_dataset, reference_indices_by_class):
    remove_ids = set()
    for class_idx in reference_indices_by_class.keys():
        remove_ids.update(reference_indices_by_class[class_idx])

    if len(remove_ids) == 0:
        return

    keep_samples = [
        sample for idx, sample in enumerate(train_dataset.samples) if idx not in remove_ids
    ]
    train_dataset.samples = keep_samples
    train_dataset.classwise_global_indices = {
        idx: [] for idx in range(len(train_dataset.class_names))
    }
    for idx, (_, class_idx) in enumerate(train_dataset.samples):
        train_dataset.classwise_global_indices[int(class_idx)].append(idx)


def compute_distance(feature):
    faiss.omp_set_num_threads(4)
    embedding = np.ascontiguousarray(feature.astype(np.float32))
    dim = int(embedding.shape[-1])

    if use_cuda and (not _FAISS_USE_CPU_INDEX):
        try:
            resources = _get_faiss_gpu_resources()
            index = faiss.GpuIndexFlatL2(resources, dim, faiss.GpuIndexFlatConfig())
        except RuntimeError:
            index = faiss.IndexFlatL2(dim)
    else:
        index = faiss.IndexFlatL2(dim)

    index.add(embedding)
    distance, id_array = index.search(embedding, k=2)
    distance = distance.T[-1]
    id_array = id_array.T[-1]
    return np.expand_dims(distance, axis=-1), id_array


def build_test_loader(args, class_name):
    test_set = dataset_extract.MyDataset(
        dataset_path=args.data_path,
        dataset=args.dataset,
        class_name=class_name,
        is_train=False,
    )
    return DataLoader(
        test_set,
        batch_size=16,
        pin_memory=False,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )


def get_oto_loss(args):
    if args.oto_loss == "mae":
        return nn.L1Loss().to(device)
    if args.oto_loss == "mse":
        return nn.MSELoss().to(device)
    return nn.KLDivLoss(reduction="batchmean").to(device)


def evaluate_epoch(
    localnet,
    feature_extractor,
    test_loader,
    args,
    class_idx_eval,
    reference_memory_by_class,
    reference_index_by_class,
):
    seg_map = []
    img_map = []
    label_gt = []
    mask_gt = []

    for images, y, mask in test_loader:
        with torch.no_grad():
            localnet.eval()
            feature_extractor.eval()
            images = images.to(device)
            y = y.detach().numpy()
            mask = mask.detach().numpy()

            features = extract_feature_batch(images, feature_extractor, args)
            class_idx_batch = torch.full(
                (features.shape[0],),
                int(class_idx_eval),
                dtype=torch.long,
                device=features.device,
            )
            residual_features = compute_residual_feature_batch(
                features,
                class_idx_batch,
                reference_memory_by_class,
                reference_index_by_class,
            )
            _, score = localnet(residual_features)
            score = score.detach().cpu().numpy()

            img_score = aggregate_image_scores(
                score,
                topk_ratio=args.img_score_topk_ratio,
            )
            img_map.append(img_score)

            score = score.reshape(-1, 28, 28)
            for i in range(score.shape[0]):
                _map = cv2.resize(score[i], (224, 224))
                _map = gaussian_filter(_map, sigma=4)
                seg_map.append(_map)
                label_gt.append(y[i])
                mask_gt.append(mask[i])

    img_map = np.concatenate(img_map, axis=0)
    label_gt = np.array(label_gt)
    seg_map = np.stack(seg_map, axis=0)
    mask_gt = np.stack(mask_gt, axis=0)

    auroc = roc_auc_score(label_gt, img_map)
    pixel_auroc = roc_auc_score(mask_gt.ravel(), seg_map.ravel())
    return auroc, pixel_auroc


def _build_faiss_index(feature_np):
    faiss.omp_set_num_threads(4)
    dim = int(feature_np.shape[-1])
    feat = np.ascontiguousarray(feature_np.astype(np.float32))

    if use_cuda and (not _FAISS_USE_CPU_INDEX):
        try:
            resources = _get_faiss_gpu_resources()
            index = faiss.GpuIndexFlatL2(
                resources,
                dim,
                faiss.GpuIndexFlatConfig(),
            )
        except RuntimeError:
            index = faiss.IndexFlatL2(dim)
    else:
        index = faiss.IndexFlatL2(dim)

    index.add(feat)
    return index


def aggregate_image_scores(score_2d, topk_ratio):
    if score_2d.ndim != 2:
        raise ValueError(f"Expected 2D score array, got shape={score_2d.shape}")

    patch_count = int(score_2d.shape[1])
    safe_ratio = float(np.clip(topk_ratio, 0.0, 1.0))
    k = max(1, int(np.ceil(patch_count * safe_ratio)))
    k = min(k, patch_count)

    # Fast top-k selection without full sort.
    topk = np.partition(score_2d, patch_count - k, axis=1)[:, -k:]
    return topk.mean(axis=1)


def _get_faiss_gpu_resources(temp_mem_mb=None):
    global _FAISS_GPU_RESOURCES
    if temp_mem_mb is None:
        temp_mem_mb = _FAISS_GPU_TEMP_MEM_MB
    if _FAISS_GPU_RESOURCES is None:
        _FAISS_GPU_RESOURCES = faiss.StandardGpuResources()
        if temp_mem_mb is not None and temp_mem_mb > 0:
            _FAISS_GPU_RESOURCES.setTempMemory(int(temp_mem_mb) * 1024 * 1024)
    return _FAISS_GPU_RESOURCES


def _update_topk_features(top_feat, top_dist, cand_feat, cand_dist, k):
    if cand_feat is None or cand_feat.shape[0] == 0 or k <= 0:
        return top_feat, top_dist

    if top_feat is None or top_dist is None or top_feat.shape[0] == 0:
        if cand_feat.shape[0] <= k:
            return cand_feat, cand_dist
        pick = np.argpartition(cand_dist, -k)[-k:]
        return cand_feat[pick], cand_dist[pick]

    merged_feat = np.concatenate([top_feat, cand_feat], axis=0)
    merged_dist = np.concatenate([top_dist, cand_dist], axis=0)
    if merged_feat.shape[0] <= k:
        return merged_feat, merged_dist

    pick = np.argpartition(merged_dist, -k)[-k:]
    return merged_feat[pick], merged_dist[pick]


def _print_selected_score_distribution_by_class(
    selected_indices: np.ndarray, class_stack: np.ndarray, image_scores: np.ndarray
):
    if selected_indices.shape[0] == 0:
        print("[Phase 2/3] selected score distribution (raw): no selected samples.")
        return

    print("[Phase 2/3] selected score distribution by class (raw, before normalization):")
    selected_classes = np.unique(class_stack[selected_indices]).tolist()
    for class_idx in selected_classes:
        cls = int(class_idx)
        cls_selected = selected_indices[class_stack[selected_indices] == cls]
        cls_scores = image_scores[cls_selected]
        if cls_scores.size == 0:
            continue
        q25, q50, q75 = np.quantile(cls_scores, [0.25, 0.5, 0.75])
        print(
            "  class %d | n=%d | min=%.6f | q25=%.6f | median=%.6f | q75=%.6f | max=%.6f | mean=%.6f | std=%.6f"
            % (
                cls,
                cls_scores.size,
                float(cls_scores.min()),
                float(q25),
                float(q50),
                float(q75),
                float(cls_scores.max()),
                float(cls_scores.mean()),
                float(cls_scores.std()),
            )
        )


def _print_selected_clean_ratio(selected_indices: np.ndarray, dataset):
    if selected_indices.shape[0] == 0:
        print("[Phase 2/3] selected image clean ratio: no selected samples.")
        return

    if not hasattr(dataset, "samples"):
        print("[Phase 2/3] selected image clean ratio: dataset has no samples metadata.")
        return

    selected_noisy = 0
    selected_total = 0
    samples = dataset.samples
    max_index = len(samples) - 1

    for idx in selected_indices.tolist():
        idx_int = int(idx)
        if idx_int < 0 or idx_int > max_index:
            continue
        path, _class_idx = samples[idx_int]
        filename = os.path.basename(path).lower()
        if "noisy" in filename:
            selected_noisy += 1
        selected_total += 1

    if selected_total == 0:
        print("[Phase 2/3] selected image clean ratio: no valid sample metadata found.")
        return

    selected_clean = selected_total - selected_noisy
    clean_ratio = selected_clean / selected_total
    noisy_ratio = selected_noisy / selected_total
    print(
        "[Phase 2/3] selected image clean ratio | clean=%d/%d (%.2f%%) | noisy=%d/%d (%.2f%%)"
        % (
            selected_clean,
            selected_total,
            clean_ratio * 100.0,
            selected_noisy,
            selected_total,
            noisy_ratio * 100.0,
        )
    )


def _print_full_dataset_confusion_matrix_by_raw_score(
    dataset_size: int,
    image_scores: np.ndarray,
    class_stack: np.ndarray,
    dataset,
    num_classes: int,
    threshold: float = 0.5,
):
    """All training images after Phase 1; raw scores, no normalization / bank filtering."""
    if not hasattr(dataset, "samples"):
        print(
            "[Phase 1] confusion matrix (raw score, full set): dataset has no samples metadata."
        )
        return

    samples = dataset.samples
    n = min(int(dataset_size), len(samples), int(image_scores.shape[0]))
    class_names = (
        list(dataset.class_names)
        if hasattr(dataset, "class_names") and len(dataset.class_names) >= num_classes
        else [f"class_{i}" for i in range(num_classes)]
    )

    def _init_counter():
        return {"tp": 0, "fp": 0, "tn": 0, "fn": 0}

    per_class = {i: _init_counter() for i in range(num_classes)}
    overall = _init_counter()

    for idx_int in range(n):
        path, _class_idx = samples[idx_int]
        cls = int(class_stack[idx_int])
        if cls < 0 or cls >= num_classes:
            continue

        filename = os.path.basename(path).lower()
        actual_anomaly = "noisy" in filename
        score = float(image_scores[idx_int])
        pred_anomaly = score > float(threshold)

        bucket = per_class[cls]
        if pred_anomaly and actual_anomaly:
            bucket["tp"] += 1
            overall["tp"] += 1
        elif pred_anomaly and (not actual_anomaly):
            bucket["fp"] += 1
            overall["fp"] += 1
        elif (not pred_anomaly) and (not actual_anomaly):
            bucket["tn"] += 1
            overall["tn"] += 1
        else:
            bucket["fn"] += 1
            overall["fn"] += 1

    def _safe_div(num, den):
        return float(num) / float(den) if den > 0 else 0.0

    def _metrics(c):
        tp, fp, tn, fn = c["tp"], c["fp"], c["tn"], c["fn"]
        support = tp + fp + tn + fn
        acc = _safe_div(tp + tn, support)
        prec = _safe_div(tp, tp + fp)
        rec = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * prec * rec, prec + rec) if (prec + rec) > 0 else 0.0
        return support, acc, prec, rec, f1

    print(
        "[Phase 1] confusion matrix by class (raw score, full set, before bank filter; "
        "threshold=%.3f, pred: score>thr => anomaly)"
        % float(threshold)
    )
    print(
        "CM_BY_CLASS|class_idx|class_name|support|tp|fp|tn|fn|accuracy|precision|recall|f1"
    )
    for cls in range(num_classes):
        c = per_class[cls]
        support, acc, prec, rec, f1 = _metrics(c)
        print(
            "CM_BY_CLASS|%d|%s|%d|%d|%d|%d|%d|%.6f|%.6f|%.6f|%.6f"
            % (
                cls,
                str(class_names[cls]),
                support,
                c["tp"],
                c["fp"],
                c["tn"],
                c["fn"],
                acc,
                prec,
                rec,
                f1,
            )
        )

    support, acc, prec, rec, f1 = _metrics(overall)
    print(
        "CM_OVERALL|all|all|%d|%d|%d|%d|%d|%.6f|%.6f|%.6f|%.6f"
        % (
            support,
            overall["tp"],
            overall["fp"],
            overall["tn"],
            overall["fn"],
            acc,
            prec,
            rec,
            f1,
        )
    )


def precompute_pseudo_labels_multiclass(
    args,
    localnet,
    feature_extractor,
    mini_loader,
    num_classes,
    reference_memory_by_class,
    reference_index_by_class,
):
    memory_bank_start = time.perf_counter()

    dataset_size = len(mini_loader.dataset)
    image_scores = np.zeros(dataset_size, dtype=np.float32)
    class_stack = np.zeros(dataset_size, dtype=np.int64)
    global_dim = None

    print("[Phase 1/3] Computing image scores...")
    with torch.no_grad():
        localnet.eval()
        feature_extractor.eval()
        for batch in tqdm.tqdm(mini_loader, desc="Phase 1"):
            images, mini_class_idx, mini_sample_idx = batch
            images = images.to(device)
            image_features = extract_feature_batch(images, feature_extractor, args)
            residual_features = compute_residual_feature_batch(
                image_features,
                mini_class_idx.to(device),
                reference_memory_by_class,
                reference_index_by_class,
            )

            features, score = localnet(residual_features)
            if features.shape[0] > args.batch_size:
                features = features.unsqueeze(0)

            if global_dim is None:
                global_dim = int(features.shape[-1])

            score_np = score.detach().cpu().numpy()
            score_np = aggregate_image_scores(
                score_np,
                topk_ratio=args.img_score_topk_ratio,
            )

            if score_np.shape == ():
                score_np = np.array([score_np], dtype=np.float32)

            sample_idx_np = mini_sample_idx.detach().cpu().numpy().astype(np.int64)
            class_np = mini_class_idx.detach().cpu().numpy().astype(np.int64)
            image_scores[sample_idx_np] = score_np.astype(np.float32)
            class_stack[sample_idx_np] = class_np

    if global_dim is None:
        raise RuntimeError("未能从 mini_loader 获取到特征维度。")

    _print_full_dataset_confusion_matrix_by_raw_score(
        dataset_size=dataset_size,
        image_scores=image_scores,
        class_stack=class_stack,
        dataset=mini_loader.dataset,
        num_classes=num_classes,
        threshold=0.5,
    )

    print("[Phase 2/3] Building memory bank from normal samples...")
    image_norm_scores = np.zeros(dataset_size, dtype=np.float32)
    # 不再使用 args.bank_sample_ratio，直接在整个数据集上按类别归一化并筛选
    sampled_indices = np.arange(dataset_size, dtype=np.int64)

    # 1）按类别归一化 image score（在整个数据集上做 per-class min-max）
    sampled_scores = image_scores[sampled_indices]
    sampled_classes = class_stack[sampled_indices]
    normalized_score = np.zeros_like(sampled_scores, dtype=np.float32)
    for cls in range(num_classes):
        cls_mask = sampled_classes == cls
        if not np.any(cls_mask):
            continue
        cls_scores = sampled_scores[cls_mask]
        cls_min = cls_scores.min()
        cls_max = cls_scores.max()
        if cls_max == cls_min:
            cls_norm = np.zeros_like(cls_scores, dtype=np.float32)
        else:
            cls_norm = (cls_scores - cls_min) / (cls_max - cls_min)
        normalized_score[cls_mask] = cls_norm.astype(np.float32)

    image_norm_scores[sampled_indices] = normalized_score.astype(np.float32)

    # 2）在“按类别归一化”后的分数上，再按类别从低分样本中采样
    selected_local_list = []
    for cls in range(num_classes):
        cls_mask = sampled_classes == cls
        if not np.any(cls_mask):
            continue
        cls_indices_local = np.where(cls_mask)[0]  # 在 sampled_indices 里的位置
        cls_norm_scores = normalized_score[cls_indices_local]

        # 先选出该类别中“正常”的（低分）样本
        cls_normal_local = np.where(cls_norm_scores < 0.5)[0]
        if cls_normal_local.shape[0] == 0:
            cls_selected_local = cls_indices_local
        elif args.random < 1:
            cls_sample_num = max(1, int(cls_normal_local.shape[0] * args.random))
            pick_local = np.random.choice(cls_normal_local, size=cls_sample_num, replace=False)
            cls_selected_local = cls_indices_local[pick_local]
        else:
            cls_selected_local = cls_indices_local[cls_normal_local]

        selected_local_list.append(cls_selected_local)

    if len(selected_local_list) == 0:
        selected_local = np.arange(sampled_indices.shape[0])
    else:
        selected_local = np.concatenate(selected_local_list, axis=0)

    selected_indices = sampled_indices[selected_local].astype(np.int64)
    _print_selected_score_distribution_by_class(
        selected_indices=selected_indices,
        class_stack=class_stack,
        image_scores=image_scores,
    )
    _print_selected_clean_ratio(
        selected_indices=selected_indices,
        dataset=mini_loader.dataset,
    )

    distance_map = np.zeros((dataset_size, 784), dtype=np.float16)
    confident_feature_bank = torch.zeros((0, global_dim), dtype=torch.float32)
    if selected_indices.shape[0] == 0:
        pseudo_label_time = time.perf_counter() - memory_bank_start
        return distance_map, confident_feature_bank, global_dim, 0.0, pseudo_label_time

    selected_features_buffer = []
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
            images, mini_class_idx, _mini_sample_idx = batch
            images = images.to(device)
            image_features = extract_feature_batch(images, feature_extractor, args)
            residual_features = compute_residual_feature_batch(
                image_features,
                mini_class_idx.to(device),
                reference_memory_by_class,
                reference_index_by_class,
            )
            features, _ = localnet(residual_features)
            features_np = features.detach().cpu().numpy()
            selected_features_buffer.append(features_np.reshape(-1, global_dim))

    if len(selected_features_buffer) == 0:
        pseudo_label_time = time.perf_counter() - memory_bank_start
        return distance_map, confident_feature_bank, global_dim, 0.0, pseudo_label_time

    global_normal_features = np.concatenate(selected_features_buffer, axis=0)
    memory_bank = _build_faiss_index(global_normal_features)

    if args.beta:
        top_feat_global = None
        top_dist_global = None

    # 按类别维护距离的最小值和最大值
    class_min_distance = np.full(num_classes, np.inf, dtype=np.float32)
    class_max_distance = np.full(num_classes, -np.inf, dtype=np.float32)

    memory_bank_time = time.perf_counter() - memory_bank_start

    print("[Phase 3/3] Computing distance map for all patches...")
    pseudo_start = time.perf_counter()
    with torch.no_grad():
        localnet.eval()
        feature_extractor.eval()
        for batch in tqdm.tqdm(mini_loader, desc="Phase 3"):
            images, mini_class_idx, mini_sample_idx = batch
            images = images.to(device)
            sample_idx_np = mini_sample_idx.detach().cpu().numpy().astype(np.int64)

            image_features = extract_feature_batch(images, feature_extractor, args)
            residual_features = compute_residual_feature_batch(
                image_features,
                mini_class_idx.to(device),
                reference_memory_by_class,
                reference_index_by_class,
            )
            features, _ = localnet(residual_features)
            features_np = features.detach().cpu().numpy()
            features_2d = features_np.reshape(-1, global_dim)

            cls_distance, _ = memory_bank.search(
                np.ascontiguousarray(features_2d), k=args.k_number
            )

            cls_distance[cls_distance < 1e-2] = 0
            if args.k_number == 2:
                same_feature = cls_distance[:, 0] == 0
                cls_distance[same_feature, 0] = cls_distance[same_feature, 1]
            cls_distance = cls_distance[:, 0]

            # 先按样本 reshape，方便做每个类别的统计
            cls_distance_map = cls_distance.reshape(-1, 784)

            # 更新每个类别的全局 min / max（按图像维度聚合）
            batch_class_np = mini_class_idx.detach().cpu().numpy().astype(np.int64)
            row_min = cls_distance_map.min(axis=1)
            row_max = cls_distance_map.max(axis=1)
            for cls in np.unique(batch_class_np):
                cls_mask = batch_class_np == cls
                if not np.any(cls_mask):
                    continue
                cls_row_min = float(row_min[cls_mask].min())
                cls_row_max = float(row_max[cls_mask].max())
                class_min_distance[cls] = min(class_min_distance[cls], cls_row_min)
                class_max_distance[cls] = max(class_max_distance[cls], cls_row_max)

            distance_map[sample_idx_np] = cls_distance_map.astype(np.float16)

            if args.beta:
                high_sample_mask = image_norm_scores[sample_idx_np] > 0.5
                if np.any(high_sample_mask):
                    high_patch_mask = np.repeat(high_sample_mask, 784)
                    cand_feat = features_2d[high_patch_mask]
                    cand_dist = cls_distance[high_patch_mask]
                    top_feat_global, top_dist_global = _update_topk_features(
                        top_feat_global,
                        top_dist_global,
                        cand_feat,
                        cand_dist,
                        args.beta_number,
                    )

    # 按类别归一化 distance_map
    finite_min_mask = np.isfinite(class_min_distance)
    finite_max_mask = np.isfinite(class_max_distance)
    valid_cls_mask = finite_min_mask & finite_max_mask & (
        class_max_distance > class_min_distance
    )

    if not np.any(valid_cls_mask):
        distance_map[:] = 0
    else:
        values = distance_map.astype(np.float32)
        for cls in range(num_classes):
            cls_indices = np.where(class_stack == cls)[0]
            if cls_indices.size == 0:
                continue

            if not (
                np.isfinite(class_min_distance[cls])
                and np.isfinite(class_max_distance[cls])
                and class_max_distance[cls] > class_min_distance[cls]
            ):
                # 若该类统计无效，则直接置零
                values[cls_indices] = 0.0
                continue

            cls_min = class_min_distance[cls]
            cls_max = class_max_distance[cls]
            cls_values = values[cls_indices]
            cls_values = (cls_values - cls_min) / (cls_max - cls_min)
            values[cls_indices] = cls_values

        distance_map = values.astype(np.float16)

    if args.beta:
        if top_feat_global is None or top_feat_global.shape[0] == 0:
            confident_feature_bank = torch.zeros((0, global_dim), dtype=torch.float32)
        else:
            confident_feature_bank = torch.as_tensor(top_feat_global, dtype=torch.float32)

    pseudo_label_time = time.perf_counter() - pseudo_start

    return distance_map, confident_feature_bank, global_dim, memory_bank_time, pseudo_label_time


def _sample_beta_anomaly(confident_feature_bank, dim):
    if confident_feature_bank is None or confident_feature_bank.shape[0] < 2:
        return None, None

    pool = confident_feature_bank
    pool_size = pool.shape[0]

    first_indices = torch.randint(low=0, high=pool_size, size=(784,))
    second_indices = torch.randint(low=0, high=pool_size, size=(784,))
    same_positions = first_indices == second_indices
    while same_positions.any():
        second_indices[same_positions] = torch.randint(
            low=0, high=pool_size, size=(int(same_positions.sum().item()),)
        )
        same_positions = first_indices == second_indices

    first_vector = pool[first_indices]
    second_vector = pool[second_indices]

    mix_ratio = random.random()
    syn_anomaly = mix_ratio * first_vector + (1 - mix_ratio) * second_vector
    syn_anomaly = syn_anomaly.reshape(1, 784, dim)
    syn_class = torch.tensor([0], dtype=torch.long)
    return syn_anomaly, syn_class


def build_adaptive_threshold_map(distance, class_idx_np, default_threshold, quantile):
    threshold_map = np.full_like(distance, fill_value=default_threshold, dtype=np.float32)
    safe_q = min(max(float(quantile), 0.0), 1.0)

    for cls in np.unique(class_idx_np).tolist():
        cls_mask = class_idx_np == int(cls)
        if not np.any(cls_mask):
            continue
        cls_values = distance[cls_mask].reshape(-1)
        if cls_values.size == 0:
            continue
        cls_thr = float(np.quantile(cls_values, safe_q))
        threshold_map[cls_mask] = cls_thr

    return threshold_map


def _compute_oto_loss_multiclass(args, batch_feature, local_pred, class_idx, l_loss):
    if args.oto_loss == "kl":
        transform_fn = lambda a, b: 0.5 * (
            l_loss(a.log(), (a + b) / 2) + l_loss(b.log(), (a + b) / 2)
        )
    else:
        transform_fn = lambda a, b: 0.5 * (
            l_loss(a, (a + b) / 2) + l_loss(b, (a + b) / 2)
        )

    unique_cls = torch.unique(class_idx)
    loss_list = []

    for cls in unique_cls:
        cls_mask = class_idx == cls
        if cls_mask.sum().item() < 2:
            continue

        cls_feature = batch_feature[cls_mask]
        cls_score = local_pred[cls_mask]

        feature_np = cls_feature.detach().cpu().numpy().reshape(-1, cls_feature.shape[-1])
        _, id_array = compute_distance(feature_np)
        matched = find_matching(id_array)
        if len(matched[0]) == 0:
            continue

        target = cls_score.reshape(-1)[matched[0]]
        input_score = cls_score.reshape(-1)[matched[1]]
        loss_list.append(transform_fn(input_score, target))

    if len(loss_list) == 0:
        return torch.tensor(0.0, device=local_pred.device)

    return torch.stack(loss_list).mean()


def _compute_origin_regularizer(
    args, input_feature, output_feature, normal_mask, anomaly_mask
):
    input_2d = input_feature.reshape(-1, input_feature.shape[-1])
    output_2d = output_feature.reshape(-1, output_feature.shape[-1])

    normal_loss = torch.tensor(0.0, device=output_feature.device)
    if normal_mask.any().item():
        normal_feat = output_2d[normal_mask]
        # Pull high-confidence normal outputs toward origin.
        normal_loss = (normal_feat.pow(2).sum(dim=-1)).mean()

    anomaly_loss = torch.tensor(0.0, device=output_feature.device)
    if anomaly_mask.any().item():
        anomaly_in = input_2d[anomaly_mask]
        anomaly_out = output_2d[anomaly_mask]
        # Identity learning on high-confidence anomalies.
        anomaly_loss = torch.mean((anomaly_out - anomaly_in) ** 2)

    return (
        args.origin_normal_weight * normal_loss
        + args.origin_anomaly_weight * anomaly_loss
    )


def train_one_epoch(
    args,
    epoch,
    localnet,
    feature_extractor,
    localnet_optimizer,
    onetoone_optimizer,
    localnet_criterion,
    l_loss,
    local_loader,
    mini_loader,
    iteration,
    num_classes,
    reference_memory_by_class,
    reference_index_by_class,
):
    total_batch = len(local_loader)
    threshold = args.threshold

    local_loss = 0
    oto_loss = 0
    bce_loss = 0
    origin_loss = 0
    memory_bank_time = 0.0
    pseudo_label_time = 0.0
    kl_loss_time = 0.0

    distance_map = None
    confident_feature_bank = None
    global_dim = None

    if threshold <= 1:
        distance_map, confident_feature_bank, global_dim, mb_time, pl_time = precompute_pseudo_labels_multiclass(
            args,
            localnet,
            feature_extractor,
            mini_loader,
            num_classes,
            reference_memory_by_class,
            reference_index_by_class,
        )
        memory_bank_time += mb_time
        pseudo_label_time += pl_time

    for images, class_idx, sample_idx in tqdm.tqdm(
        local_loader, f"| run | train | {epoch + 1} |"
    ):
        class_idx = class_idx.to(device)
        class_idx_np = class_idx.detach().cpu().numpy().astype(np.int64)
        sample_idx = sample_idx.detach().cpu().numpy()
        batch = images.shape[0]

        if batch != args.batch_size:
            continue

        images = images.to(device)
        x = extract_feature_batch(images, feature_extractor, args)
        x = compute_residual_feature_batch(
            x,
            class_idx,
            reference_memory_by_class,
            reference_index_by_class,
        )
        dim = int(x.shape[-1]) if global_dim is None else int(global_dim)

        distance = np.zeros((batch, 784), dtype=np.float32)
        if threshold <= 1:
            distance = distance_map[sample_idx]
        threshold_map = np.full_like(distance, fill_value=threshold, dtype=np.float32)
        if (threshold <= 1) and args.use_class_adaptive_threshold:
            threshold_map = build_adaptive_threshold_map(
                distance=distance,
                class_idx_np=class_idx_np,
                default_threshold=threshold,
                quantile=args.adaptive_threshold_quantile,
            )
        _copy = None
        if (threshold <= 1) and args.gaussian:
            uncertain_mask = (distance > threshold_map) & (distance < args.noise_threshold)
            uncertain_mask = uncertain_mask.reshape(-1)
            x_std = x.detach().std(dim=0).max(dim=0)[0]
            _copy = x.detach().clone().reshape(-1, dim)
            n_uncertain = int(uncertain_mask.sum())

            if n_uncertain > 0:
                if args.std is None:
                    std = x_std
                    std = std.repeat_interleave(n_uncertain).reshape(-1, n_uncertain).T
                    _copy[uncertain_mask] += torch.normal(mean=0, std=std)
                else:
                    _copy[uncertain_mask] += torch.normal(
                        mean=0, std=args.std, size=_copy[uncertain_mask].shape
                    )

            _copy = _copy.reshape(-1, 784, dim)

        pseudo_label_assign_start = time.perf_counter()
        class_idx_for_oto = class_idx
        if args.beta:
            local_label = torch.zeros((args.batch_size + 1, 784))
            distance_bin = np.zeros_like(distance)
            if args.threshold <= 1:
                distance_bin[distance > threshold_map] = 1
            local_label[:-1] = torch.as_tensor(distance_bin, dtype=torch.float32)

            syn_anomaly, syn_class = _sample_beta_anomaly(
                confident_feature_bank=confident_feature_bank,
                dim=dim,
            )

            if syn_anomaly is not None:
                local_label[-1] = 1
                if (args.threshold <= 1) and args.gaussian:
                    _copy = torch.cat([_copy, syn_anomaly], dim=0)
                else:
                    x = torch.cat([x, syn_anomaly.to(device)], dim=0)
                    class_idx_for_oto = torch.cat([class_idx_for_oto, syn_class.to(device)], dim=0)
            else:
                local_label = local_label[:-1]
        else:
            local_label = torch.zeros((args.batch_size, 784))
            if args.threshold <= 1:
                distance_mask = torch.as_tensor(
                    distance > threshold_map, dtype=torch.bool
                )
                local_label[distance_mask] = 1
        pseudo_label_time += time.perf_counter() - pseudo_label_assign_start

        localnet.train()
        localnet_optimizer.zero_grad()
        if args.alternative and onetoone_optimizer is not None:
            onetoone_optimizer.zero_grad()

        x = x.to(device)
        local_label = local_label.to(device)

        batch_feature, local_pred = localnet(x)
        origin_input_feature = x
        origin_output_feature = batch_feature

        pred_for_loss = local_pred
        if (threshold <= 1) and args.gaussian:
            _copy = _copy.to(device)
            gaussian_feature, gaussian_pred = localnet(_copy)
            pred_for_loss = gaussian_pred
            
        if args.balancing:
            pos_mask = local_label == 1
            neg_mask = local_label == 0

            if pos_mask.any().item():
                _a_loss = localnet_criterion(pred_for_loss[pos_mask], local_label[pos_mask])
            else:
                _a_loss = torch.tensor(0.0, device=local_label.device)

            if neg_mask.any().item():
                _n_loss = localnet_criterion(pred_for_loss[neg_mask], local_label[neg_mask])
            else:
                _n_loss = torch.tensor(0.0, device=local_label.device)
            _loss = _a_loss + _n_loss
        else:
            _loss = localnet_criterion(pred_for_loss, local_label)

        if (iteration >= args.iter) and args.kl:
            kl_start = time.perf_counter()
            _l_loss = _compute_oto_loss_multiclass(
                args=args,
                batch_feature=batch_feature,
                local_pred=local_pred,
                class_idx=class_idx_for_oto,
                l_loss=l_loss,
            )
            kl_loss_time += time.perf_counter() - kl_start
        else:
            _l_loss = torch.tensor(0.0, device=local_pred.device)

        if args.use_origin_regularizer:
            # Use only high-confidence extremes; overlap region is ignored.
            normal_mask_np = distance < threshold
            anomaly_mask_np = distance > args.noise_threshold
            normal_mask_t = torch.as_tensor(
                normal_mask_np.reshape(-1),
                dtype=torch.bool,
                device=origin_output_feature.device,
            )
            anomaly_mask_t = torch.as_tensor(
                anomaly_mask_np.reshape(-1),
                dtype=torch.bool,
                device=origin_output_feature.device,
            )
            # Ignore synthetic sample if beta branch appends one sample.
            origin_input_main = origin_input_feature[:batch]
            origin_output_main = origin_output_feature[:batch]
            _origin_loss = _compute_origin_regularizer(
                args=args,
                input_feature=origin_input_main,
                output_feature=origin_output_main,
                normal_mask=normal_mask_t,
                anomaly_mask=anomaly_mask_t,
            )
        else:
            _origin_loss = torch.tensor(0.0, device=local_pred.device)

        _local_loss = _loss if args.alternative else (_loss + args.weight * _l_loss)
        _local_loss = _local_loss + _origin_loss

        _local_loss.backward()
        localnet_optimizer.step()

        if args.alternative and onetoone_optimizer is not None and _l_loss.requires_grad:
            _l_loss.backward()
            onetoone_optimizer.step()

        local_loss += _local_loss / total_batch
        bce_loss += _loss / total_batch
        oto_loss += _l_loss / total_batch
        origin_loss += _origin_loss / total_batch
        iteration += 1

    local_loss_value = (
        local_loss.item() if torch.is_tensor(local_loss) else float(local_loss)
    )
    bce_loss_value = bce_loss.item() if torch.is_tensor(bce_loss) else float(bce_loss)
    oto_loss_value = oto_loss.item() if torch.is_tensor(oto_loss) else float(oto_loss)
    origin_loss_value = (
        origin_loss.item() if torch.is_tensor(origin_loss) else float(origin_loss)
    )

    return (
        local_loss_value,
        bce_loss_value,
        oto_loss_value,
        origin_loss_value,
        iteration,
        memory_bank_time,
        pseudo_label_time,
        kl_loss_time,
    )


def main():
    torch.autograd.set_detect_anomaly(True)
    args = parse_args()
    global _FAISS_USE_CPU_INDEX, _FAISS_GPU_TEMP_MEM_MB
    _FAISS_USE_CPU_INDEX = bool(args.faiss_cpu_index)
    _FAISS_GPU_TEMP_MEM_MB = int(args.faiss_gpu_temp_mem_mb)
    fix_seed(args.seed)

    saved_dir = os.path.join(args.save_path, args.dataset, args.noise)
    os.makedirs(saved_dir, exist_ok=True)
    _enable_print_logging(os.path.join(saved_dir, "run_stdout.log"))

    if args.synthetic:
        raise ValueError("当前多类脚本暂不支持 --synthetic。")

    class_names = get_all_class_names(args.dataset)
    train_dataset = MultiClassFeatureDataset(
        data_path=args.data_path,
        dataset_name=args.dataset,
        seed=args.seed,
        shuffle=True,
    )

    reference_indices_by_class = select_reference_indices_by_class(
        train_dataset, len(class_names), args
    )
    for class_idx, ref_ids in reference_indices_by_class.items():
        print(
            f"class {class_idx} | selected clean references: {len(ref_ids)} "
            f"(target={args.num_reference_images_per_class})"
        )

    feature_extractor = build_feature_extractor(args)
    reference_memory_by_class, reference_index_by_class = build_reference_memory_bank(
        train_dataset, reference_indices_by_class, feature_extractor, args
    )
    remove_reference_samples_from_dataset(train_dataset, reference_indices_by_class)
    print(f"train samples after removing references: {len(train_dataset)}")

    local_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        pin_memory=False,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
    )
    mini_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        pin_memory=False,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    feature_dim = infer_feature_dim(feature_extractor, mini_loader, args)

    localnet = model.localnet(len_feature=feature_dim).to(device)
    localnet_optimizer = optim.RMSprop(localnet.parameters(), lr=args.lr, momentum=0.2)
    onetoone_optimizer = optim.Adam(localnet.parameters(), lr=args.lr) if args.alternative else None
    localnet_criterion = nn.BCELoss().to(device)
    l_loss = get_oto_loss(args)

    run_name = (
        "gaussian_"
        + str(args.gaussian)
        + "_noise_"
        + args.noise
        + "_balancing_"
        + str(args.balancing)
        + "_oto_"
        + str(args.kl)
        + "_weight_"
        + str(args.weight)
        + "_multiclass_residual"
    )

    iteration = 0
    best_mean = -1
    best_result_by_class = {}

    for epoch in range(args.epoch):
        (
            local_loss_value,
            bce_loss_value,
            oto_loss_value,
            origin_loss_value,
            iteration,
            memory_bank_time,
            pseudo_label_time,
            kl_loss_time,
        ) = train_one_epoch(
            args=args,
            epoch=epoch,
            localnet=localnet,
            feature_extractor=feature_extractor,
            localnet_optimizer=localnet_optimizer,
            onetoone_optimizer=onetoone_optimizer,
            localnet_criterion=localnet_criterion,
            l_loss=l_loss,
            local_loader=local_loader,
            mini_loader=mini_loader,
            iteration=iteration,
            num_classes=len(class_names),
            reference_memory_by_class=reference_memory_by_class,
            reference_index_by_class=reference_index_by_class,
        )

        print(
            "epoch %d | loss: %.6f, bce loss: %.6f, one-to-one loss: %.6f, origin loss: %.6f"
            % (
                epoch + 1,
                local_loss_value,
                bce_loss_value,
                oto_loss_value,
                origin_loss_value,
            )
        )
        print(
            "epoch %d | memory bank: %.4fs, pseudo label: %.4fs, kl: %.4fs"
            % (epoch + 1, memory_bank_time, pseudo_label_time, kl_loss_time)
        )

        if epoch % args.eval_interval == 0:
            eval_rows = []
            mean_img_list = []
            mean_pixel_list = []
            for class_idx_eval, class_name in enumerate(class_names):
                test_loader = build_test_loader(args, class_name)
                auroc, pixel_auroc = evaluate_epoch(
                    localnet,
                    feature_extractor,
                    test_loader,
                    args,
                    class_idx_eval=class_idx_eval,
                    reference_memory_by_class=reference_memory_by_class,
                    reference_index_by_class=reference_index_by_class,
                )
                eval_rows.append([class_name, auroc, pixel_auroc])
                mean_img_list.append(auroc)
                mean_pixel_list.append(pixel_auroc)
                print(
                    f"epoch {epoch + 1} | {class_name} | auroc: {auroc:.5f}, pixel auroc: {pixel_auroc:.5f}"
                )
                del test_loader

            epoch_mean_img = float(np.mean(mean_img_list))
            epoch_mean_pixel = float(np.mean(mean_pixel_list))
            epoch_mean = (epoch_mean_img + epoch_mean_pixel) / 2
            print(f"epoch {epoch + 1} | multiclass img mean: {epoch_mean_img:.5f}")
            print(f"epoch {epoch + 1} | multiclass pixel mean: {epoch_mean_pixel:.5f}")

            if epoch_mean > best_mean:
                best_mean = epoch_mean
                best_result_by_class = {row[0]: (row[1], row[2]) for row in eval_rows}
                torch.save(
                    {
                        "net": localnet.state_dict(),
                        "reference_memory_by_class": reference_memory_by_class,
                        "reference_indices_by_class": reference_indices_by_class,
                    },
                    os.path.join(saved_dir, run_name + "_localnet.pt"),
                )

            if args.save_log:
                with open(os.path.join(saved_dir, "log.txt"), "a") as file:
                    file.write(
                        f"epoch {epoch + 1} | total loss: {local_loss_value:.6f} | bce loss: {bce_loss_value:.6f} | one-to-one loss: {oto_loss_value:.6f} | multiclass img mean: {epoch_mean_img:.5f} | multiclass pixel mean: {epoch_mean_pixel:.5f}\n"
                    )

    if len(best_result_by_class) == 0:
        raise RuntimeError("训练未产生可用评估结果，请检查数据与参数。")

    results = []
    for class_name in class_names:
        auroc, pixel_auroc = best_result_by_class[class_name]
        results.append([class_name, auroc, pixel_auroc])
        print(f"best | {class_name} | img_auroc: {auroc:.5f} | pixel_auroc: {pixel_auroc:.5f}")

    df = pd.DataFrame(results, columns=["class", "auroc", "pixel_auroc"])
    result_path = os.path.join(
        saved_dir,
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    os.makedirs(result_path, exist_ok=True)
    df.to_excel(os.path.join(result_path, run_name + "_result.xlsx"), index=False)


if __name__ == "__main__":
    main()
