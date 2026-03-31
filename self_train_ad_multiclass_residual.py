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
from epoch_precompute import precompute_pseudo_labels_multiclass_residual
import evaluate as eval_utils
from loss import (
    build_adaptive_threshold_map,
    compute_balanced_bce_loss,
    compute_origin_regularizer,
    compute_oto_loss_multiclass,
)
import model
from multiclass_feature_dataset import MultiClassFeatureDataset, get_all_class_names
from print import (
    print_epoch_losses,
    print_epoch_times,
    print_full_dataset_confusion_matrix_by_raw_score,
    print_selected_clean_ratio,
    print_selected_score_distribution_by_class,
)
import utils as common_utils

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
    return common_utils.str2bool(value)


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
    parser.add_argument("-r", "--random", type=float, default=0.15)
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
    parser.add_argument("--faiss_cpu_index", action="store_true")
    parser.add_argument("--faiss_gpu_temp_mem_mb", type=int, default=256)
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
    parser.add_argument("--use_cls_token", type=str2bool, default="False")
    parser.add_argument("--feature_model", type=str, choices=["dino", "clip"], default="dino")
    parser.add_argument("--clip_model_name", type=str, default="ViT-L/14@336px")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24])
    parser.add_argument("--dpam_layer", type=int, default=24)
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--n_ctx", type=int, default=12)
    parser.add_argument("--t_n_ctx", type=int, default=4)

    return parser.parse_args()


def fix_seed(number):
    common_utils.fix_seed(number)


def find_matching(id_array):
    return common_utils.find_matching(id_array)


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

    # DINO imports `trunc_normal_` via `from utils import ...`.
    # This project also has `utils.py`, so temporarily unshadow it.
    local_utils_module = sys.modules.get("utils")
    should_restore_utils = (
        local_utils_module is not None
        and os.path.abspath(getattr(local_utils_module, "__file__", "")).endswith(
            os.path.join("FUNAD", "utils.py")
        )
    )
    if should_restore_utils:
        del sys.modules["utils"]
    try:
        feature_extractor = torch.hub.load("facebookresearch/dino:main", "dino_vitb8")
    finally:
        if should_restore_utils:
            sys.modules["utils"] = local_utils_module
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
    return common_utils.compute_distance(
        feature,
        use_cuda=use_cuda,
        use_cpu_index=_FAISS_USE_CPU_INDEX,
        gpu_temp_mem_mb=_FAISS_GPU_TEMP_MEM_MB,
    )


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
    return common_utils.get_oto_loss(args.oto_loss, device)


def evaluate_epoch(
    localnet,
    feature_extractor,
    test_loader,
    args,
    class_idx_eval,
    reference_memory_by_class,
    reference_index_by_class,
):
    return eval_utils.evaluate_residual_multiclass_epoch(
        localnet=localnet,
        feature_extractor=feature_extractor,
        test_loader=test_loader,
        args=args,
        class_idx_eval=class_idx_eval,
        reference_memory_by_class=reference_memory_by_class,
        reference_index_by_class=reference_index_by_class,
        device=device,
        extract_feature_batch_fn=extract_feature_batch,
        compute_residual_feature_batch_fn=compute_residual_feature_batch,
    )


def _build_faiss_index(feature_np):
    return common_utils.build_faiss_index(
        feature_np,
        use_cuda=use_cuda,
        use_cpu_index=_FAISS_USE_CPU_INDEX,
        gpu_temp_mem_mb=_FAISS_GPU_TEMP_MEM_MB,
    )


def aggregate_image_scores(score_2d, topk_ratio):
    return common_utils.aggregate_image_scores(score_2d, topk_ratio)


def _get_faiss_gpu_resources(temp_mem_mb=None):
    if temp_mem_mb is None:
        temp_mem_mb = _FAISS_GPU_TEMP_MEM_MB
    return common_utils._get_faiss_gpu_resources(temp_mem_mb)


def _update_topk_features(top_feat, top_dist, cand_feat, cand_dist, k):
    return common_utils.update_topk_features(top_feat, top_dist, cand_feat, cand_dist, k)


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
        distance_map, confident_feature_bank, global_dim, mb_time, pl_time = precompute_pseudo_labels_multiclass_residual(
            args=args,
            localnet=localnet,
            feature_extractor=feature_extractor,
            mini_loader=mini_loader,
            num_classes=num_classes,
            reference_memory_by_class=reference_memory_by_class,
            reference_index_by_class=reference_index_by_class,
            device=device,
            extract_feature_batch_fn=extract_feature_batch,
            compute_residual_feature_batch_fn=compute_residual_feature_batch,
            aggregate_image_scores_fn=aggregate_image_scores,
            build_faiss_index_fn=_build_faiss_index,
            update_topk_features_fn=_update_topk_features,
            print_selected_score_distribution_fn=print_selected_score_distribution_by_class,
            print_selected_clean_ratio_fn=print_selected_clean_ratio,
            print_confusion_matrix_fn=print_full_dataset_confusion_matrix_by_raw_score,
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
            _loss = compute_balanced_bce_loss(
                localnet_criterion=localnet_criterion,
                pred=pred_for_loss,
                target=local_label,
            )
        else:
            _loss = localnet_criterion(pred_for_loss, local_label)

        if (iteration >= args.iter) and args.kl:
            kl_start = time.perf_counter()
            _l_loss = compute_oto_loss_multiclass(
                args=args,
                batch_feature=batch_feature,
                local_pred=local_pred,
                class_idx=class_idx_for_oto,
                l_loss=l_loss,
                compute_distance_fn=compute_distance,
                find_matching_fn=find_matching,
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
            _origin_loss = compute_origin_regularizer(
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

        print_epoch_losses(
            epoch,
            local_loss_value,
            bce_loss_value,
            oto_loss_value,
            origin_loss=origin_loss_value,
        )
        print_epoch_times(epoch, memory_bank_time, pseudo_label_time, kl_loss_time)

        if epoch % args.eval_interval == 0:
            eval_rows = []
            mean_img_list = []
            mean_pixel_list = []
            mean_ap_sp_list = []
            mean_f1_sp_list = []
            mean_ap_px_list = []
            mean_f1_px_list = []
            mean_aupro_px_list = []
            for class_idx_eval, class_name in enumerate(class_names):
                test_loader = build_test_loader(args, class_name)
                (
                    auroc,
                    ap_sp,
                    f1_sp,
                    pixel_auroc,
                    ap_px,
                    f1_px,
                    aupro_px,
                ) = evaluate_epoch(
                    localnet,
                    feature_extractor,
                    test_loader,
                    args,
                    class_idx_eval=class_idx_eval,
                    reference_memory_by_class=reference_memory_by_class,
                    reference_index_by_class=reference_index_by_class,
                )
                eval_rows.append(
                    [class_name, auroc, ap_sp, f1_sp, pixel_auroc, ap_px, f1_px, aupro_px]
                )
                mean_img_list.append(auroc)
                mean_pixel_list.append(pixel_auroc)
                mean_ap_sp_list.append(ap_sp)
                mean_f1_sp_list.append(f1_sp)
                mean_ap_px_list.append(ap_px)
                mean_f1_px_list.append(f1_px)
                mean_aupro_px_list.append(aupro_px)
                print(
                    (
                        f"epoch {epoch + 1} | {class_name} | auroc: {auroc:.5f}, ap_sp: {ap_sp:.5f}, "
                        f"f1_sp: {f1_sp:.5f}, pixel auroc: {pixel_auroc:.5f}, ap_px: {ap_px:.5f}, "
                        f"f1_px: {f1_px:.5f}, aupro_px: {aupro_px:.5f}"
                    )
                )
                del test_loader

            epoch_mean_img = float(np.mean(mean_img_list))
            epoch_mean_pixel = float(np.mean(mean_pixel_list))
            epoch_mean_ap_sp = float(np.mean(mean_ap_sp_list))
            epoch_mean_f1_sp = float(np.mean(mean_f1_sp_list))
            epoch_mean_ap_px = float(np.mean(mean_ap_px_list))
            epoch_mean_f1_px = float(np.mean(mean_f1_px_list))
            epoch_mean_aupro_px = float(np.mean(mean_aupro_px_list))
            epoch_mean = (epoch_mean_img + epoch_mean_pixel) / 2
            print(f"epoch {epoch + 1} | multiclass img mean: {epoch_mean_img:.5f}")
            print(f"epoch {epoch + 1} | multiclass pixel mean: {epoch_mean_pixel:.5f}")
            print(f"epoch {epoch + 1} | multiclass ap_sp mean: {epoch_mean_ap_sp:.5f}")
            print(f"epoch {epoch + 1} | multiclass f1_sp mean: {epoch_mean_f1_sp:.5f}")
            print(f"epoch {epoch + 1} | multiclass ap_px mean: {epoch_mean_ap_px:.5f}")
            print(f"epoch {epoch + 1} | multiclass f1_px mean: {epoch_mean_f1_px:.5f}")
            print(f"epoch {epoch + 1} | multiclass aupro_px mean: {epoch_mean_aupro_px:.5f}")

            if epoch_mean > best_mean:
                best_mean = epoch_mean
                best_result_by_class = {row[0]: tuple(row[1:]) for row in eval_rows}
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
                        f"epoch {epoch + 1} | total loss: {local_loss_value:.6f} | bce loss: {bce_loss_value:.6f} | one-to-one loss: {oto_loss_value:.6f} | multiclass img mean: {epoch_mean_img:.5f} | multiclass pixel mean: {epoch_mean_pixel:.5f} | multiclass ap_sp mean: {epoch_mean_ap_sp:.5f} | multiclass f1_sp mean: {epoch_mean_f1_sp:.5f} | multiclass ap_px mean: {epoch_mean_ap_px:.5f} | multiclass f1_px mean: {epoch_mean_f1_px:.5f} | multiclass aupro_px mean: {epoch_mean_aupro_px:.5f}\n"
                    )

    if len(best_result_by_class) == 0:
        raise RuntimeError("训练未产生可用评估结果，请检查数据与参数。")

    results = []
    for class_name in class_names:
        auroc, ap_sp, f1_sp, pixel_auroc, ap_px, f1_px, aupro_px = best_result_by_class[class_name]
        results.append([class_name, auroc, ap_sp, f1_sp, pixel_auroc, ap_px, f1_px, aupro_px])
        print(
            (
                f"best | {class_name} | img_auroc: {auroc:.5f} | ap_sp: {ap_sp:.5f} "
                f"| f1_sp: {f1_sp:.5f} | pixel_auroc: {pixel_auroc:.5f} "
                f"| ap_px: {ap_px:.5f} | f1_px: {f1_px:.5f} | aupro_px: {aupro_px:.5f}"
            )
        )

    df = pd.DataFrame(
        results,
        columns=[
            "class",
            "auroc_sp",
            "ap_sp",
            "f1_sp",
            "auroc_px",
            "ap_px",
            "f1_px",
            "aupro_px",
        ],
    )
    result_path = os.path.join(
        saved_dir,
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    os.makedirs(result_path, exist_ok=True)
    df.to_excel(os.path.join(result_path, run_name + "_result.xlsx"), index=False)


if __name__ == "__main__":
    main()
