import argparse
import datetime
import os
import random
import sys
import time
import warnings

import cv2
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as torch_mp
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader

from dataset import dataset_extract, multiclass_feature_dataset
from dataset.feature_extract import (
    _FEATURE_MODEL_CHOICES,
    build_dinov3_feature_extractor,
    resolve_dino_block_indices,
    extract_dinov3_feature_batch,
    infer_dinov3_feature_dim,
    infer_dinov3_patch_mask_size,
)
from dataset.multiclass_feature_dataset import MultiClassFeatureDataset, get_all_class_names
from src.model import model
from src.train.epoch_precompute import precompute_pseudo_labels_multiclass_residual
from utils import evaluate as eval_utils
from utils.logging import enable_print_logging
from utils.loss import (
    build_adaptive_threshold_map,
    compute_balanced_bce_loss,
    compute_origin_regularizer,
    compute_oto_loss_multiclass,
)
from utils.print import (
    print_epoch_losses,
    print_epoch_times,
    print_full_dataset_confusion_matrix_by_raw_score,
    print_selected_clean_ratio,
    print_selected_score_distribution_by_class,
)
import utils.train_utils as common_utils
from utils.visualize import save_moe_expert_visualizations

warnings.filterwarnings("ignore")

try:
    # Avoid exhausting file descriptors when DataLoader workers share CPU tensors.
    torch_mp.set_sharing_strategy("file_system")
except (AttributeError, RuntimeError):
    warnings.warn("Failed to set torch multiprocessing sharing strategy.")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
_FAISS_GPU_RESOURCES = None
_FAISS_USE_CPU_INDEX = False
_FAISS_GPU_TEMP_MEM_MB = 256


def _adapt_extract_fn(images, feature_extractor, args, return_cls_token=False):
    """Adapter: epoch_precompute passes args (Namespace), extract_dinov3_feature_batch needs dino_layer_indices."""
    return extract_dinov3_feature_batch(
        images, feature_extractor, args.dino_layer_indices,
        use_cls_token=args.use_cls_token, return_cls_token=return_cls_token,
    )


def str2bool(value):
    return common_utils.str2bool(value)


def parse_args():
    parser = argparse.ArgumentParser("self-train_ad_multiclass")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/media/honeywell/D/bhy/dataset/MVTec_overlap/MVTec_noisy10",
    )
    parser.add_argument(
        "--save_path", type=str, default="/media/honeywell/E/bhy/test"
    )
    parser.add_argument("--kl", action="store_true")
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
    parser.add_argument("-n", "--noise_threshold", type=float, default=0.995)
    parser.add_argument(
        "--noise",
        type=str,
        default="10%",
        choices=["0%", "1%", "2%", "3%", "5%","15%", "10%", "20%"],
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
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--crop_size", type=int, default=448)
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
        "--gate_aux_weight",
        type=float,
        default=0.05,
        help="Weight for FMoE gate auxiliary loss from discriminator.",
    )
    parser.add_argument(
        "--use_moe_discriminator",
        action="store_true",
        help="Whether to use MoE discriminator for localnet."
    )
    parser.add_argument(
        "--moe_num_expert",
        type=int,
        default=4,
        help="Number of experts used when MoE discriminator is enabled.",
    )
    parser.add_argument(
        "--moe_top_k",
        type=int,
        default=2,
        help="Top-k experts per token when MoE discriminator is enabled.",
    )
    parser.add_argument(
        "--moe_use_cls_token",
        action="store_true",
        help="Whether MoE discriminator gate uses cls_token as routing input.",
    )
    parser.add_argument(
        "--moe_hard_class_gate",
        action="store_true",
        help="Use fixed class->expert routing gate (expert count follows class count).",
    )
    parser.add_argument(
        "--moe_expert_vis_enable",
        action="store_true",
        help="Enable MoE class-to-expert routing visualization.",
    )
    parser.add_argument(
        "--moe_expert_vis_interval",
        type=int,
        default=1,
        help="Save MoE class-to-expert visualization every N epochs.",
    )
    parser.add_argument(
        "--moe_expert_vis_dirname",
        type=str,
        default="moe_expert_vis",
        help="Sub-directory name for MoE class-to-expert visualization outputs.",
    )
    parser.add_argument(
        "--img_score_topk_ratio",
        type=float,
        default=0.01,
        help="Image-level score uses mean of top-k patch scores, k=ceil(num_patches*ratio).",
    )
    parser.add_argument("--use_cls_token", type=str2bool, default="False")
    parser.add_argument(
        "--feature_model",
        type=str,
        choices=_FEATURE_MODEL_CHOICES,
        default="dinov3_vitb16",
        help="DINOv3 variant (torch.hub entry); see DINOV3_FEATURE_MODEL_REGISTRY for hub_entry / hub_repo_dir.",
    )
    parser.add_argument(
        "--dino_layer_indices",
        type=int,
        nargs="+",
        default=[-1],
        help="DINOv3 layer ids (1-based) to aggregate by mean pooling. -1 means last layer.",
    )
    parser.add_argument(
        "--pseudo_label_scoring",
        type=str,
        default="nn",
        help="Scorer(s) separated by '+', e.g. 'nn', 'nn+mahalanobis', 'nn+mahalanobis+pca'. "
        "Available: nn, mahalanobis, pca.",
    )
    parser.add_argument(
        "--pseudo_label_mahalanobis_dim",
        type=int,
        default=128,
        help="Mahalanobis mode: project patch features to this dim (bias-free Linear) before "
        "Gaussian fit and scoring when feature dim is larger; set 0 to disable projection.",
    )
    parser.add_argument(
        "--pseudo_label_pca_dim",
        type=int,
        default=0,
        help="PCA mode: retained principal components; set 0 to auto-select by explained variance.",
    )
    parser.add_argument(
        "--pseudo_label_pca_ev",
        type=float,
        default=0.99,
        help="PCA mode: explained variance target used when --pseudo_label_pca_dim=0.",
    )
    parser.add_argument(
        "--pseudo_label_pca_eps",
        type=float,
        default=1e-6,
        help="PCA mode: numerical stability epsilon for eigenvalue clamping and ratio computation.",
    )
    parser.add_argument(
        "--pseudo_label_distance_norm",
        type=str,
        choices=["minmax", "percentile"],
        default="minmax",
        help="Per-class patch-distance normalization: minmax or percentile (robust to outliers).",
    )
    parser.add_argument(
        "--pseudo_label_distance_norm_eps",
        type=float,
        default=1e-6,
        help="Numerical stability epsilon for patch-distance normalization.",
    )
    parser.add_argument(
        "--pseudo_label_distance_norm_percentile",
        type=float,
        default=99.0,
        help="Percentile threshold (0-100) for percentile normalization. "
        "Values outside this percentile range are clipped before min-max.",
    )
    parser.add_argument(
        "--greedy_keep_images",
        type=int,
        default=2,
        help="Greedy coreset keeps feature points equivalent to this many images per class.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Path to a training checkpoint (.pt) saved by this script; empty means start from scratch.",
    )
    parser.add_argument(
        "--residual",
        type=str2bool,
        default="True",
        help="Enable residual feature computation (feat - nearest_class_reference). "
        "Set to False to use raw DINOv3 features directly.",
    )
    parser.add_argument(
        "--global_memory_bank",
        action="store_true",
        help="Merge all classes into a single global memory bank (instead of per-class). "
        "GreedyCoreset is applied on the merged features. Scoring uses one global "
        "scorer for all patches, which tests whether per-class memory bank is beneficial.",
    )
    parser.add_argument(
        "--normal_sample_selection",
        type=str,
        choices=["threshold", "quantile"],
        default="threshold",
        help="How to select 'normal' images in Phase 2 of pseudo-label pipeline. "
        "'threshold' (default): select images with norm_score < 0.5. "
        "'quantile': select images with norm_score below the top-N quantile (e.g. top 30%).",
    )
    parser.add_argument(
        "--normal_sample_quantile",
        type=float,
        default=0.3,
        help="Quantile threshold for 'quantile' selection mode. "
        "E.g., 0.3 means top 30% lowest-scoring images are considered normal candidates. "
        "Only used when --normal_sample_selection=quantile.",
    )

    return parser.parse_args()


def fix_seed(number):
    common_utils.fix_seed(number)


def find_matching(id_array):
    return common_utils.find_matching(id_array)





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
        resize=args.image_size,
        cropsize=args.crop_size,
    )
    # Eval loader is created/destroyed per class; keep workers small and
    # non-persistent to avoid accumulating processes/resources.
    eval_num_workers = max(1, min(int(args.num_workers), 2))
    loader_kwargs = common_utils.build_loader_kwargs(
        eval_num_workers,
        pin_memory=False,
        prefetch_factor=1,
        persistent_workers=False,
    )
    return DataLoader(
        test_set,
        batch_size=16,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
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
    compute_residual_fn=common_utils.compute_residual_feature_batch,
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
        extract_feature_batch_fn=_adapt_extract_fn,
        compute_residual_feature_batch_fn=compute_residual_fn,
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
    moe_expert_vis_ctx=None,
    compute_residual_fn=common_utils.compute_residual_feature_batch,
):
    total_batch = len(local_loader)
    threshold = args.threshold

    local_loss = 0
    oto_loss = 0
    bce_loss = 0
    origin_loss = 0
    gate_aux_loss = 0
    memory_bank_time = 0.0
    pseudo_label_time = 0.0
    kl_loss_time = 0.0
    pseudo_normal_correct = 0
    pseudo_normal_total = 0
    pseudo_anomaly_correct = 0
    pseudo_anomaly_total = 0

    distance_map = None
    confident_feature_bank = None
    global_dim = None
    moe_vis_enabled = bool(
        moe_expert_vis_ctx is not None
        and moe_expert_vis_ctx.get("enabled", False)
        and args.use_moe_discriminator
    )
    class_expert_count = None
    if moe_vis_enabled:
        num_expert = int(moe_expert_vis_ctx["num_expert"])
        class_expert_count = np.zeros((num_classes, num_expert), dtype=np.float64)

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
            extract_feature_batch_fn=_adapt_extract_fn,
            compute_residual_feature_batch_fn=compute_residual_fn,
            aggregate_image_scores_fn=aggregate_image_scores,
            build_faiss_index_fn=_build_faiss_index,
            update_topk_features_fn=_update_topk_features,
            print_selected_score_distribution_fn=print_selected_score_distribution_by_class,
            print_selected_clean_ratio_fn=print_selected_clean_ratio,
            print_confusion_matrix_fn=print_full_dataset_confusion_matrix_by_raw_score,
            epoch=epoch,
            pseudo_label_scorer=None,
        )
        memory_bank_time += mb_time
        pseudo_label_time += pl_time

    for batch_data in tqdm.tqdm(local_loader, f"| run | train | {epoch + 1} |"):
        images, class_idx, sample_idx = batch_data[0], batch_data[1], batch_data[2]
        class_idx = class_idx.to(device, non_blocking=True)
        class_idx_np = class_idx.detach().cpu().numpy().astype(np.int64)
        sample_idx = sample_idx.detach().cpu().numpy()
        batch = images.shape[0]

        if batch != args.batch_size:
            continue

        images = images.to(device, non_blocking=True)
        x, cls_token = extract_dinov3_feature_batch(
            images, feature_extractor, args.dino_layer_indices, use_cls_token=args.use_cls_token, return_cls_token=True
        )
        x = compute_residual_fn(
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
                    x = torch.cat([x, syn_anomaly.to(device, non_blocking=True)], dim=0)
                    class_idx_for_oto = torch.cat([class_idx_for_oto, syn_class.to(device, non_blocking=True)], dim=0)
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

        x = x.to(device, non_blocking=True)
        local_label = local_label.to(device, non_blocking=True)

        gate_cls_token = cls_token if cls_token.shape[0] == x.shape[0] else None
        patch_class_idx = None
        if args.use_moe_discriminator and args.moe_hard_class_gate:
            token_per_image = int(x.shape[1]) if x.dim() >= 2 else 1
            patch_class_idx = (
                class_idx_for_oto.to(device=x.device, dtype=torch.long)
                .reshape(-1, 1, 1)
                .expand(-1, token_per_image, 1)
                .contiguous()
            )
        batch_feature, local_pred = localnet(
            x,
            cls_token=gate_cls_token,
            patch_class_idx=patch_class_idx,
        )
        origin_input_feature = x
        origin_output_feature = batch_feature
        if moe_vis_enabled:
            discriminator = getattr(localnet, "discriminator", None)
            get_stats = getattr(discriminator, "get_latest_gate_stats", None)
            gate_stats = get_stats() if callable(get_stats) else None
            if gate_stats is not None:
                gate_top_k_idx, _gate_score = gate_stats
                token_per_image = int(batch_feature.shape[1]) if batch_feature.dim() >= 2 else 1
                class_for_gate = class_idx_for_oto
                if class_for_gate.shape[0] == batch_feature.shape[0]:
                    token_labels = (
                        class_for_gate.detach().cpu().numpy().astype(np.int64).repeat(token_per_image)
                    )
                    gate_idx = gate_top_k_idx.detach().reshape(-1).cpu().numpy().astype(np.int64)
                    if token_labels.size > 0 and gate_idx.size % token_labels.size == 0:
                        rep = gate_idx.size // token_labels.size
                        if rep > 1:
                            token_labels = np.repeat(token_labels, rep)
                        valid = (
                            (token_labels >= 0)
                            & (token_labels < num_classes)
                            & (gate_idx >= 0)
                            & (gate_idx < class_expert_count.shape[1])
                        )
                        if np.any(valid):
                            np.add.at(
                                class_expert_count,
                                (token_labels[valid], gate_idx[valid]),
                                1.0,
                            )

        pred_for_loss = local_pred
        if (threshold <= 1) and args.gaussian:
            _copy = _copy.to(device, non_blocking=True)
            gaussian_cls_token = (
                cls_token if cls_token.shape[0] == _copy.shape[0] else None
            )
            gaussian_patch_class_idx = None
            if args.use_moe_discriminator and args.moe_hard_class_gate:
                token_per_image = int(_copy.shape[1]) if _copy.dim() >= 2 else 1
                gaussian_patch_class_idx = class_idx_for_oto.to(
                    device=_copy.device, dtype=torch.long
                ).reshape(-1, 1, 1).expand(
                    -1, token_per_image, 1
                )
                gaussian_patch_class_idx = gaussian_patch_class_idx.contiguous()
            gaussian_feature, gaussian_pred = localnet(
                _copy,
                cls_token=gaussian_cls_token,
                patch_class_idx=gaussian_patch_class_idx,
            )
            pred_for_loss = gaussian_pred
            
        if args.balancing:
            _loss = compute_balanced_bce_loss(
                localnet_criterion=localnet_criterion,
                pred=pred_for_loss,
                target=local_label,
            )
        else:
            _loss = localnet_criterion(pred_for_loss, local_label)

        with torch.no_grad():
            pred_bin = pred_for_loss.detach() >= 0.5
            target_bin = local_label >= 0.5
            normal_mask = ~target_bin
            anomaly_mask = target_bin
            if normal_mask.any().item():
                pseudo_normal_total += int(normal_mask.sum().item())
                pseudo_normal_correct += int((~pred_bin[normal_mask]).sum().item())
            if anomaly_mask.any().item():
                pseudo_anomaly_total += int(anomaly_mask.sum().item())
                pseudo_anomaly_correct += int(pred_bin[anomaly_mask].sum().item())

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

        if args.use_moe_discriminator:
            discriminator = getattr(localnet, "discriminator", None)
            if discriminator is not None and hasattr(discriminator, "get_loss"):
                _gate_aux_loss = discriminator.get_loss(
                    clear=True, reduction="mean", default=None
                )
                if _gate_aux_loss is None:
                    _gate_aux_loss = torch.tensor(0.0, device=local_pred.device)
            else:
                _gate_aux_loss = torch.tensor(0.0, device=local_pred.device)
        else:
            _gate_aux_loss = torch.tensor(0.0, device=local_pred.device)

        _local_loss = _loss if args.alternative else (_loss + args.weight * _l_loss)
        _local_loss = _local_loss + _origin_loss
        if args.use_moe_discriminator:
            _local_loss = _local_loss + args.gate_aux_weight * _gate_aux_loss

        _local_loss.backward()
        localnet_optimizer.step()

        if args.alternative and onetoone_optimizer is not None and _l_loss.requires_grad:
            _l_loss.backward()
            onetoone_optimizer.step()

        local_loss += _local_loss / total_batch
        bce_loss += _loss / total_batch
        oto_loss += _l_loss / total_batch
        origin_loss += _origin_loss / total_batch
        gate_aux_loss += _gate_aux_loss / total_batch
        iteration += 1

    if moe_vis_enabled:
        save_interval = max(1, int(moe_expert_vis_ctx.get("save_interval", 1)))
        if (epoch + 1) % save_interval == 0:
            save_moe_expert_visualizations(
                epoch=epoch,
                class_expert_count=class_expert_count,
                class_names=moe_expert_vis_ctx.get("class_names", []),
                save_dir=moe_expert_vis_ctx["save_dir"],
            )

    local_loss_value = (
        local_loss.item() if torch.is_tensor(local_loss) else float(local_loss)
    )
    bce_loss_value = bce_loss.item() if torch.is_tensor(bce_loss) else float(bce_loss)
    oto_loss_value = oto_loss.item() if torch.is_tensor(oto_loss) else float(oto_loss)
    origin_loss_value = (
        origin_loss.item() if torch.is_tensor(origin_loss) else float(origin_loss)
    )
    gate_aux_loss_value = (
        gate_aux_loss.item() if torch.is_tensor(gate_aux_loss) else float(gate_aux_loss)
    )
    pseudo_normal_acc = (
        float(pseudo_normal_correct) / float(pseudo_normal_total)
        if pseudo_normal_total > 0
        else 0.0
    )
    pseudo_anomaly_acc = (
        float(pseudo_anomaly_correct) / float(pseudo_anomaly_total)
        if pseudo_anomaly_total > 0
        else 0.0
    )

    return (
        local_loss_value,
        bce_loss_value,
        oto_loss_value,
        origin_loss_value,
        gate_aux_loss_value,
        iteration,
        memory_bank_time,
        pseudo_label_time,
        kl_loss_time,
        pseudo_normal_acc,
        pseudo_normal_correct,
        pseudo_normal_total,
        pseudo_anomaly_acc,
        pseudo_anomaly_correct,
        pseudo_anomaly_total,
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
    enable_print_logging(os.path.join(saved_dir, "run_stdout.log"))

    # 将 CLI 参数保存为 CSV
    import csv
    args_csv_path = os.path.join(saved_dir, "cli_args.csv")
    with open(args_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        for key, value in sorted(vars(args).items()):
            writer.writerow([key, value])
    print(f"[CLI Args] saved -> {args_csv_path}")

    if args.synthetic:
        raise ValueError("当前多类脚本暂不支持 --synthetic。")

    class_names = get_all_class_names(args.dataset)
    args.class_names = class_names
    num_classes_runtime = len(class_names)
    if args.use_moe_discriminator and args.moe_hard_class_gate and args.beta:
        raise ValueError("Hard class gate does not support --beta. Please disable beta.")
    train_dataset = MultiClassFeatureDataset(
        data_path=args.data_path,
        dataset_name=args.dataset,
        image_size=args.image_size,
        crop_size=args.crop_size,
        seed=args.seed,
        shuffle=True,
    )

    feature_extractor = build_dinov3_feature_extractor(args.feature_model, device)
    inferred_patch_mask_size = infer_dinov3_patch_mask_size(train_dataset, feature_extractor, args.dino_layer_indices, args.use_cls_token, device)
    train_dataset.patch_mask_size = inferred_patch_mask_size
    print(f"inferred patch_mask_size from feature extractor: {inferred_patch_mask_size}")
    _selected_blocks = resolve_dino_block_indices(feature_extractor, args.dino_layer_indices)
    _display = [b + 1 for b in _selected_blocks]
    print(f"[DINOv3] aggregating layers {_display} (0-based blocks {_selected_blocks})")

    if args.residual:
        reference_indices_by_class = common_utils.select_reference_indices_by_class(
            train_dataset, len(class_names),
            num_reference_images_per_class=args.num_reference_images_per_class,
            seed=args.seed,
            strict_clean_reference=args.strict_clean_reference,
        )
        for class_idx, ref_ids in reference_indices_by_class.items():
            print(
                f"class {class_idx} | selected clean references: {len(ref_ids)} "
                f"(target={args.num_reference_images_per_class})"
            )
        reference_memory_by_class_cpu, reference_index_by_class = common_utils.build_reference_memory_bank(
            train_dataset, reference_indices_by_class,
            extract_feature_fn=lambda images: extract_dinov3_feature_batch(
                images, feature_extractor, args.dino_layer_indices, use_cls_token=args.use_cls_token,
            ),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        reference_memory_by_class = common_utils.move_reference_memory_to_gpu(reference_memory_by_class_cpu, device)
        common_utils.remove_reference_samples_from_dataset(train_dataset, reference_indices_by_class)
        print(f"train samples after removing references: {len(train_dataset)}")
        compute_residual_fn = common_utils.compute_residual_feature_batch
    else:
        reference_indices_by_class = {cls: [] for cls in range(len(class_names))}
        reference_memory_by_class_cpu = {}
        reference_memory_by_class = {}
        reference_index_by_class = {}
        compute_residual_fn = common_utils.identity_residual

    loader_kwargs = common_utils.build_loader_kwargs(args.num_workers, pin_memory=True, prefetch_factor=1,persistent_workers=False)
    local_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        **loader_kwargs,
    )
    mini_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    feature_dim = infer_dinov3_feature_dim(feature_extractor, mini_loader, args.dino_layer_indices, args.use_cls_token, device)

    moe_num_expert_runtime = (
        num_classes_runtime if args.moe_hard_class_gate else int(args.moe_num_expert)
    )
    moe_top_k_runtime = 1 if args.moe_hard_class_gate else int(args.moe_top_k)
    localnet = model.localnet(
        len_feature=feature_dim,
        use_moe_discriminator=args.use_moe_discriminator,
        moe_num_expert=moe_num_expert_runtime,
        moe_top_k=moe_top_k_runtime,
        moe_use_cls_token=args.moe_use_cls_token,
        moe_hard_class_gate=args.moe_hard_class_gate,
    ).to(device)
    if args.use_moe_discriminator:
        discriminator = getattr(localnet, "discriminator", None)
        if discriminator is not None and hasattr(discriminator, "enable_gate_stats"):
            discriminator.enable_gate_stats(bool(args.moe_expert_vis_enable))
    localnet_optimizer = optim.RMSprop(localnet.parameters(), lr=args.lr, momentum=0.2)
    onetoone_optimizer = optim.Adam(localnet.parameters(), lr=args.lr) if args.alternative else None
    localnet_criterion = nn.BCELoss().to(device)
    l_loss = get_oto_loss(args)

    residual_tag = "" if args.residual else "_no_residual"
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
        + residual_tag
    )
    moe_expert_vis_ctx = {
        "enabled": bool(args.moe_expert_vis_enable),
        "save_interval": max(1, int(args.moe_expert_vis_interval)),
        "save_dir": os.path.join(saved_dir, args.moe_expert_vis_dirname, run_name),
        "class_names": class_names,
        "num_expert": int(moe_num_expert_runtime),
    }
    if moe_expert_vis_ctx["enabled"] and args.use_moe_discriminator:
        os.makedirs(moe_expert_vis_ctx["save_dir"], exist_ok=True)

    resume_path = (args.resume or "").strip()

    start_epoch = 0
    iteration = 0
    best_mean = -1.0
    best_result_by_class = {}

    if resume_path:
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"--resume path not found: {resume_path}")
        print(f"[resume] loading training checkpoint: {resume_path}")
        last_done, iteration, best_mean, best_result_by_class = common_utils.load_train_checkpoint(
            resume_path,
            localnet,
            localnet_optimizer,
            onetoone_optimizer,
        )
        start_epoch = last_done + 1
        if start_epoch >= args.epoch:
            print(
                f"[resume] checkpoint epoch {last_done} finished; "
                f"--epoch {args.epoch} means no further epochs to run."
            )
        else:
            print(
                f"[resume] will continue from epoch {start_epoch + 1} "
                f"(completed through epoch {last_done + 1}), global_iteration={iteration}"
            )

    for epoch in range(start_epoch, args.epoch):
        (
            local_loss_value,
            bce_loss_value,
            oto_loss_value,
            origin_loss_value,
            gate_aux_loss_value,
            iteration,
            memory_bank_time,
            pseudo_label_time,
            kl_loss_time,
            pseudo_normal_acc,
            pseudo_normal_correct,
            pseudo_normal_total,
            pseudo_anomaly_acc,
            pseudo_anomaly_correct,
            pseudo_anomaly_total,
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
            moe_expert_vis_ctx=moe_expert_vis_ctx,
            compute_residual_fn=compute_residual_fn,
        )

        print_epoch_losses(
            epoch,
            local_loss_value,
            bce_loss_value,
            oto_loss_value,
            origin_loss=origin_loss_value,
            gate_aux_loss=gate_aux_loss_value,
        )
        print_epoch_times(epoch, memory_bank_time, pseudo_label_time, kl_loss_time)
        print(
            f"epoch {epoch + 1} | pseudo normal->normal acc@0.5: {pseudo_normal_acc:.4f} "
            f"({pseudo_normal_correct}/{pseudo_normal_total})"
        )
        print(
            f"epoch {epoch + 1} | pseudo anomaly->anomaly acc@0.5: {pseudo_anomaly_acc:.4f} "
            f"({pseudo_anomaly_correct}/{pseudo_anomaly_total})"
        )

        if (epoch + 1) % args.eval_interval == 0:
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
                    compute_residual_fn=compute_residual_fn,
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
                        "reference_memory_by_class": reference_memory_by_class_cpu,
                        "reference_indices_by_class": reference_indices_by_class,
                    },
                    os.path.join(saved_dir, run_name + "_localnet.pt"),
                )

            if args.save_log:
                with open(os.path.join(saved_dir, "log.txt"), "a") as file:
                    file.write(
                        f"epoch {epoch + 1} | total loss: {local_loss_value:.6f} | bce loss: {bce_loss_value:.6f} | one-to-one loss: {oto_loss_value:.6f} | gate aux loss: {gate_aux_loss_value:.6f} | gate aux weight: {args.gate_aux_weight:.6f} | multiclass img mean: {epoch_mean_img:.5f} | multiclass pixel mean: {epoch_mean_pixel:.5f} | multiclass ap_sp mean: {epoch_mean_ap_sp:.5f} | multiclass f1_sp mean: {epoch_mean_f1_sp:.5f} | multiclass ap_px mean: {epoch_mean_ap_px:.5f} | multiclass f1_px mean: {epoch_mean_f1_px:.5f} | multiclass aupro_px mean: {epoch_mean_aupro_px:.5f}\n"
                    )

        ckpt_path = common_utils.default_train_checkpoint_path(saved_dir, run_name)
        common_utils.save_train_checkpoint(
            ckpt_path,
            epoch_completed=epoch,
            iteration=iteration,
            localnet=localnet,
            localnet_optimizer=localnet_optimizer,
            onetoone_optimizer=onetoone_optimizer,
            best_mean=best_mean,
            best_result_by_class=best_result_by_class,
        )
        print(f"[checkpoint] saved training state -> {ckpt_path}")

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
