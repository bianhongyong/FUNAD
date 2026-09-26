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
from dataset.multiclass_feature_dataset import (
    MultiClassFeatureDataset,
    get_all_class_names,
    DINO_CLASS_LAYER_INDICES,
)
from src.model import model
from src.train.epoch_precompute import precompute_pseudo_labels_multiclass_residual
from utils import evaluate as eval_utils
from utils.logging import enable_print_logging
from utils.loss import (
    build_adaptive_threshold_map,
    compute_balanced_bce_loss,
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

warnings.filterwarnings("ignore")

try:
    torch_mp.set_sharing_strategy("file_system")
except (AttributeError, RuntimeError):
    warnings.warn("Failed to set torch multiprocessing sharing strategy.")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
_FAISS_GPU_RESOURCES = None
_FAISS_USE_CPU_INDEX = False
_FAISS_GPU_TEMP_MEM_MB = 256


def _adapt_extract_fn(images, feature_extractor, args, return_cls_token=False, class_indices=None):
    return extract_dinov3_feature_batch(
        images, feature_extractor, args.class_layer_indices,
        class_indices=class_indices,
        use_cls_token=False, return_cls_token=return_cls_token,
    )


def str2bool(value):
    return common_utils.str2bool(value)


def parse_args():
    parser = argparse.ArgumentParser("self-train_ad_singleclass")
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
    parser.add_argument("--threshold_stage1", type=float, default=0.5)
    parser.add_argument("--threshold_stage2", type=float, default=0.5)
    parser.add_argument("--threshold_epoch_split", type=int, default=1)
    parser.add_argument("-n", "--noise_threshold", type=float, default=0.8)
    parser.add_argument("--use_mad_threshold", action="store_true")
    parser.add_argument("--mad_k", type=float, default=5)
    parser.add_argument("--mad_k_per_class", type=str, default="screw:20.0")
    parser.add_argument(
        "--noise",
        type=str,
        default="10%",
        choices=["0%", "1%", "2%", "3%", "5%", "15%", "10%", "20%"],
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
    parser.add_argument("--num_reference_images_per_class", type=int, default=8)
    parser.add_argument("--strict_clean_reference", action="store_true")
    parser.add_argument("--use_class_adaptive_threshold", action="store_true")
    parser.add_argument("--adaptive_threshold_quantile", type=float, default=0.7)
    parser.add_argument(
        "--img_score_topk_ratio",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--feature_model",
        type=str,
        choices=_FEATURE_MODEL_CHOICES,
        default="dinov3_vitb16",
    )
    parser.add_argument(
        "--pseudo_label_scoring",
        type=str,
        default="pca",
    )
    parser.add_argument("--pseudo_label_mahalanobis_dim", type=int, default=128)
    parser.add_argument("--pseudo_label_pca_dim", type=int, default=0)
    parser.add_argument("--pseudo_label_pca_ev", type=float, default=0.99)
    parser.add_argument("--pseudo_label_pca_eps", type=float, default=1e-6)
    parser.add_argument(
        "--pseudo_label_distance_norm",
        type=str,
        choices=["minmax", "percentile"],
        default="minmax",
    )
    parser.add_argument("--pseudo_label_distance_norm_eps", type=float, default=1e-6)
    parser.add_argument("--pseudo_label_distance_norm_percentile", type=float, default=99.0)
    parser.add_argument("--greedy_keep_images", type=int, default=2)
    parser.add_argument(
        "--residual",
        type=str2bool,
        default="True",
    )
    parser.add_argument("--use_moe_discriminator", action="store_true")
    parser.add_argument("--moe_num_expert", type=str, default="16")
    parser.add_argument("--moe_top_k", type=str, default="1")
    parser.add_argument("--moe_hard_class_gate", action="store_true")
    parser.add_argument("--gate_aux_weight", type=float, default=0.5)
    parser.add_argument("--global_memory_bank", action="store_true")
    parser.add_argument(
        "--normal_sample_selection",
        type=str,
        choices=["threshold", "quantile"],
        default="threshold",
    )
    parser.add_argument("--normal_sample_quantile", type=float, default=0.3)
    parser.add_argument("--memory_bank_freeze_start_epoch", type=int, default=10)
    parser.add_argument(
        "--class_names",
        type=str,
        nargs="*",
        default=[],
        help="Specific class names to train. If not provided, trains all classes.",
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
    reference_memory_by_class,
    reference_index_by_class,
    compute_residual_fn=common_utils.compute_residual_feature_batch,
):
    return eval_utils.evaluate_residual_multiclass_epoch(
        localnet=localnet,
        feature_extractor=feature_extractor,
        test_loader=test_loader,
        args=args,
        class_idx_eval=0,
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
    reference_memory_by_class,
    reference_index_by_class,
    memory_bank_snapshot,
    compute_residual_fn=common_utils.compute_residual_feature_batch,
    class_save_dir=None,
):
    total_batch = len(local_loader)
    threshold = args.threshold_stage1 if epoch < args.threshold_epoch_split else args.threshold_stage2
    use_pseudo_label = threshold <= 1 or args.use_mad_threshold

    local_loss = 0
    oto_loss = 0
    bce_loss = 0
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
    mad_threshold_map = None

    if use_pseudo_label:
        distance_map, confident_feature_bank, global_dim, mb_time, pl_time, mad_threshold_map = precompute_pseudo_labels_multiclass_residual(
            args=args,
            localnet=localnet,
            feature_extractor=feature_extractor,
            mini_loader=mini_loader,
            num_classes=1,
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
            memory_bank_snapshot=memory_bank_snapshot,
            threshold=threshold,
            save_dir=class_save_dir,
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
            images, feature_extractor, args.class_layer_indices,
            class_indices=class_idx,
            use_cls_token=False, return_cls_token=True,
        )
        x = compute_residual_fn(
            x,
            class_idx,
            reference_memory_by_class,
            reference_index_by_class,
        )
        dim = int(x.shape[-1]) if global_dim is None else int(global_dim)

        distance = np.zeros((batch, 784), dtype=np.float32)
        if use_pseudo_label:
            distance = distance_map[sample_idx]
        if args.use_mad_threshold and mad_threshold_map is not None:
            threshold_map = mad_threshold_map[sample_idx]
        else:
            threshold_map = np.full_like(distance, fill_value=threshold, dtype=np.float32)
            if (
                use_pseudo_label
                and args.use_class_adaptive_threshold
                and not args.global_memory_bank
            ):
                threshold_map = build_adaptive_threshold_map(
                    distance=distance,
                    class_idx_np=class_idx_np,
                    default_threshold=threshold,
                    quantile=args.adaptive_threshold_quantile,
                )
        _copy = None
        if use_pseudo_label and args.gaussian:
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
            if use_pseudo_label:
                distance_bin[distance > threshold_map] = 1
            local_label[:-1] = torch.as_tensor(distance_bin, dtype=torch.float32)

            syn_anomaly, syn_class = _sample_beta_anomaly(
                confident_feature_bank=confident_feature_bank,
                dim=dim,
            )

            if syn_anomaly is not None:
                local_label[-1] = 1
                if use_pseudo_label and args.gaussian:
                    _copy = torch.cat([_copy, syn_anomaly], dim=0)
                else:
                    x = torch.cat([x, syn_anomaly.to(device, non_blocking=True)], dim=0)
                    class_idx_for_oto = torch.cat([class_idx_for_oto, syn_class.to(device, non_blocking=True)], dim=0)
            else:
                local_label = local_label[:-1]
        else:
            local_label = torch.zeros((args.batch_size, 784))
            if use_pseudo_label:
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

        batch_feature, local_pred = localnet(x)

        pred_for_loss = local_pred
        if use_pseudo_label and args.gaussian:
            _copy = _copy.to(device, non_blocking=True)
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

        _local_loss = _loss if args.alternative else (_loss + args.weight * _l_loss)

        _local_loss.backward()
        localnet_optimizer.step()

        if args.alternative and onetoone_optimizer is not None and _l_loss.requires_grad:
            _l_loss.backward()
            onetoone_optimizer.step()

        local_loss += _local_loss / total_batch
        bce_loss += _loss / total_batch
        oto_loss += _l_loss / total_batch
        iteration += 1

    local_loss_value = (
        local_loss.item() if torch.is_tensor(local_loss) else float(local_loss)
    )
    bce_loss_value = bce_loss.item() if torch.is_tensor(bce_loss) else float(bce_loss)
    oto_loss_value = oto_loss.item() if torch.is_tensor(oto_loss) else float(oto_loss)
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


def train_single_class(args, class_name, feature_extractor):
    """Train one independent model for a single class."""
    print(f"\n{'='*60}")
    print(f"  Training class: {class_name}")
    print(f"{'='*60}")

    fix_seed(args.seed)

    class_layer_indices = DINO_CLASS_LAYER_INDICES.get(class_name, [-1])
    args.class_layer_indices = {0: class_layer_indices}
    _all_layer_values = sorted(set(class_layer_indices))

    train_dataset = MultiClassFeatureDataset(
        data_path=args.data_path,
        dataset_name=args.dataset,
        image_size=args.image_size,
        crop_size=args.crop_size,
        seed=args.seed,
        shuffle=True,
        class_names_override=[class_name],
    )

    inferred_patch_mask_size = infer_dinov3_patch_mask_size(
        train_dataset, feature_extractor, _all_layer_values, False, device
    )
    train_dataset.patch_mask_size = inferred_patch_mask_size
    print(f"inferred patch_mask_size: {inferred_patch_mask_size}")

    _selected_blocks = resolve_dino_block_indices(feature_extractor, _all_layer_values)
    print(f"[DINOv3] aggregating layers (0-based blocks {_selected_blocks})")

    if args.residual:
        reference_indices_by_class = common_utils.select_reference_indices_by_class(
            train_dataset, 1,
            num_reference_images_per_class=args.num_reference_images_per_class,
            seed=args.seed,
            strict_clean_reference=args.strict_clean_reference,
        )
        for class_idx, ref_ids in reference_indices_by_class.items():
            print(f"class {class_idx} ({class_name}) | selected clean references: {len(ref_ids)}")

        reference_memory_by_class_cpu, reference_index_by_class = common_utils.build_reference_memory_bank(
            train_dataset, reference_indices_by_class,
            extract_feature_fn=lambda images, class_indices=None: extract_dinov3_feature_batch(
                images, feature_extractor, args.class_layer_indices,
                class_indices=class_indices,
                use_cls_token=False,
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
        reference_indices_by_class = {0: []}
        reference_memory_by_class_cpu = {}
        reference_memory_by_class = {}
        reference_index_by_class = {}
        compute_residual_fn = common_utils.identity_residual

    loader_kwargs = common_utils.build_loader_kwargs(
        args.num_workers, pin_memory=True, prefetch_factor=1, persistent_workers=False
    )
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
    feature_dim = infer_dinov3_feature_dim(
        feature_extractor, mini_loader, _all_layer_values, False, device
    )

    localnet_model = model.localnet(
        len_feature=feature_dim,
        use_moe_discriminator=False,
        num_classes=1,
        class_conditioned_adapter=False,
    ).to(device)

    localnet_optimizer = optim.RMSprop(localnet_model.parameters(), lr=args.lr, momentum=0.2)
    onetoone_optimizer = optim.Adam(localnet_model.parameters(), lr=args.lr) if args.alternative else None
    localnet_criterion = nn.BCELoss().to(device)
    l_loss = get_oto_loss(args)

    residual_tag = "" if args.residual else "_no_residual"
    run_name = (
        "gaussian_" + str(args.gaussian)
        + "_noise_" + args.noise
        + "_balancing_" + str(args.balancing)
        + "_oto_" + str(args.kl)
        + "_weight_" + str(args.weight)
        + "_singleclass" + residual_tag
    )

    class_save_dir = os.path.join(args.save_path, args.dataset, args.noise, class_name)
    os.makedirs(class_save_dir, exist_ok=True)

    # 修复: 单类训练时 class_idx 总是 0, 从 mad_k_per_class 中查找当前类的 k 值覆盖默认 mad_k
    if getattr(args, "mad_k_per_class", None) and args.class_names:
        for entry in str(args.mad_k_per_class).split(","):
            entry = entry.strip()
            if ":" not in entry:
                continue
            cls_name, k_str = entry.split(":", 1)
            if cls_name.strip() == class_name:
                try:
                    override_k = float(k_str.strip())
                    args.mad_k = override_k
                    print(f"[MAD-Threshold] overriding mad_k for class '{class_name}' "
                          f"from --mad_k_per_class: {args.mad_k} -> {override_k}")
                except ValueError:
                    print(f"[MAD-Threshold] WARNING: invalid k value '{k_str}' for class '{cls_name}'")
                break

    iteration = 0
    best_mean = -1.0
    best_auroc = -1.0
    best_pixel_auroc = -1.0
    memory_bank_snapshot: dict = {}

    for epoch in range(args.epoch):
        (
            local_loss_value,
            bce_loss_value,
            oto_loss_value,
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
            localnet=localnet_model,
            feature_extractor=feature_extractor,
            localnet_optimizer=localnet_optimizer,
            onetoone_optimizer=onetoone_optimizer,
            localnet_criterion=localnet_criterion,
            l_loss=l_loss,
            local_loader=local_loader,
            mini_loader=mini_loader,
            iteration=iteration,
            class_save_dir=class_save_dir,
            reference_memory_by_class=reference_memory_by_class,
            reference_index_by_class=reference_index_by_class,
            memory_bank_snapshot=memory_bank_snapshot,
            compute_residual_fn=compute_residual_fn,
        )

        print_epoch_losses(
            epoch,
            local_loss_value,
            bce_loss_value,
            oto_loss_value,
        )
        print_epoch_times(epoch, memory_bank_time, pseudo_label_time, kl_loss_time)
        print(
            f"epoch {epoch + 1} | pseudo normal acc@0.5: {pseudo_normal_acc:.4f} "
            f"({pseudo_normal_correct}/{pseudo_normal_total})"
        )
        print(
            f"epoch {epoch + 1} | pseudo anomaly acc@0.5: {pseudo_anomaly_acc:.4f} "
            f"({pseudo_anomaly_correct}/{pseudo_anomaly_total})"
        )

        if (epoch + 1) % args.eval_interval == 0:
            test_loader = build_test_loader(args, class_name)
            auroc, pixel_auroc = evaluate_epoch(
                localnet_model,
                feature_extractor,
                test_loader,
                args,
                reference_memory_by_class=reference_memory_by_class,
                reference_index_by_class=reference_index_by_class,
                compute_residual_fn=compute_residual_fn,
            )
            del test_loader

            epoch_mean = (auroc + pixel_auroc) / 2
            print(
                f"epoch {epoch + 1} | {class_name} | I-AUROC: {auroc:.5f}, P-AUROC: {pixel_auroc:.5f}"
            )

            if epoch_mean > best_mean:
                best_mean = epoch_mean
                best_auroc = auroc
                best_pixel_auroc = pixel_auroc
                torch.save(
                    {
                        "net": localnet_model.state_dict(),
                        "reference_memory_by_class": reference_memory_by_class_cpu,
                        "reference_indices_by_class": reference_indices_by_class,
                    },
                    os.path.join(class_save_dir, run_name + "_localnet.pt"),
                )

            if args.save_log:
                with open(os.path.join(class_save_dir, "log.txt"), "a") as file:
                    file.write(
                        f"epoch {epoch + 1} | total loss: {local_loss_value:.6f} | "
                        f"bce loss: {bce_loss_value:.6f} | oto loss: {oto_loss_value:.6f} | "
                        f"I-AUROC: {auroc:.5f} | P-AUROC: {pixel_auroc:.5f}\n"
                    )

    print(f"\n[{class_name}] best I-AUROC: {best_auroc:.5f}, P-AUROC: {best_pixel_auroc:.5f}")
    return class_name, best_auroc, best_pixel_auroc


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

    import csv
    args_csv_path = os.path.join(saved_dir, "cli_args.csv")
    with open(args_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        for key, value in sorted(vars(args).items()):
            writer.writerow([key, value])
    print(f"[CLI Args] saved -> {args_csv_path}")

    if args.synthetic:
        raise ValueError("当前脚本暂不支持 --synthetic。")

    all_class_names = get_all_class_names(args.dataset)
    if args.class_names:
        class_names_to_train = [c for c in args.class_names if c in all_class_names]
        invalid = [c for c in args.class_names if c not in all_class_names]
        if invalid:
            print(f"[WARNING] unknown class names (skipped): {invalid}")
    else:
        class_names_to_train = list(all_class_names)

    print(f"Classes to train ({len(class_names_to_train)}): {class_names_to_train}")

    # 修复: 将 args.class_names 设为完整类别列表, 供 _parse_mad_k_per_class 等下游逻辑使用
    args.class_names = list(all_class_names)

    feature_extractor = build_dinov3_feature_extractor(args.feature_model, device)

    results = []
    for class_name in class_names_to_train:
        class_name, auroc, pixel_auroc = train_single_class(args, class_name, feature_extractor)
        results.append([class_name, auroc, pixel_auroc])

    print(f"\n{'='*60}")
    print("  Final Results Summary")
    print(f"{'='*60}")
    for row in results:
        print(f"  {row[0]:15s} | I-AUROC: {row[1]:.5f} | P-AUROC: {row[2]:.5f}")

    mean_img = float(np.mean([r[1] for r in results]))
    mean_pix = float(np.mean([r[2] for r in results]))
    print(f"  {'MEAN':15s} | I-AUROC: {mean_img:.5f} | P-AUROC: {mean_pix:.5f}")

    df = pd.DataFrame(results, columns=["class", "auroc_sp", "auroc_px"])
    result_path = os.path.join(
        saved_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    os.makedirs(result_path, exist_ok=True)
    df.to_excel(os.path.join(result_path, "singleclass_result.xlsx"), index=False)
    print(f"[Results] saved -> {result_path}")


if __name__ == "__main__":
    main()
