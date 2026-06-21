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
from utils import evaluate as eval_utils
from utils.logging import enable_print_logging
from utils.loss import compute_balanced_bce_loss
from utils.visualize import save_moe_expert_visualizations
import utils.train_utils as common_utils

warnings.filterwarnings("ignore")

try:
    torch_mp.set_sharing_strategy("file_system")
except (AttributeError, RuntimeError):
    warnings.warn("Failed to set torch multiprocessing sharing strategy.")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def _adapt_extract_fn(images, feature_extractor, args, return_cls_token=False, class_indices=None):
    return extract_dinov3_feature_batch(
        images, feature_extractor, args.class_layer_indices,
        class_indices=class_indices,
        use_cls_token=args.use_cls_token, return_cls_token=return_cls_token,
    )


def str2bool(value):
    return common_utils.str2bool(value)


def parse_args():
    parser = argparse.ArgumentParser("self-train_ad_multiclass_clean")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/media/honeywell/D/bhy/dataset/MVTec_overlap/MVTec_noisy10",
    )
    parser.add_argument(
        "--save_path", type=str, default="/media/honeywell/E/bhy/test"
    )
    parser.add_argument(
        "--dataset", type=str, default="mvtec", choices=["mvtec", "visa"]
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-l", "--lr", type=float, default=2e-5)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("-b", "--batch_size", type=int, default=16)
    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument("--save_log", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--crop_size", type=int, default=448)
    parser.add_argument("--num_reference_images_per_class", type=int, default=8)
    parser.add_argument("--strict_clean_reference", action="store_true")
    parser.add_argument(
        "--gate_aux_weight",
        type=float,
        default=0.05,
        help="Weight for FMoE gate auxiliary loss from discriminator.",
    )
    parser.add_argument(
        "--use_moe_discriminator",
        action="store_true",
        help="Whether to use MoE discriminator for localnet.",
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
        help="DINOv3 variant (torch.hub entry).",
    )
    parser.add_argument(
        "--residual",
        type=str2bool,
        default="True",
        help="Enable residual feature computation (feat - nearest_class_reference).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Path to a training checkpoint (.pt) saved by this script; empty means start from scratch.",
    )
    parser.add_argument(
        "--balancing",
        action="store_true",
        help="Use balanced BCE loss (separate pos/neg normalization).",
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=100,
        help="Number of PCA models in the ensemble for memory bank scoring. "
        "(Not used in clean training, kept for parameter consistency.)",
    )
    parser.add_argument(
        "--memory_sampling_ratio",
        type=float,
        default=0.1,
        help="Fraction of selected normal patches sampled per ensemble PCA iteration. "
        "(Not used in clean training, kept for parameter consistency.)",
    )
    parser.add_argument(
        "--memory_score_beta_end",
        type=float,
        default=0.5,
        help="Controls image-level score fusion decay. "
        "(Not used in clean training, kept for parameter consistency.)",
    )

    return parser.parse_args()


def fix_seed(number):
    common_utils.fix_seed(number)


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


def train_one_epoch(
    args,
    epoch,
    localnet,
    feature_extractor,
    localnet_optimizer,
    localnet_criterion,
    local_loader,
    iteration,
    num_classes,
    reference_memory_by_class,
    reference_index_by_class,
    moe_expert_vis_ctx=None,
    compute_residual_fn=common_utils.compute_residual_feature_batch,
):
    total_batch = len(local_loader)

    local_loss = 0
    bce_loss = 0
    gate_aux_loss = 0

    moe_vis_enabled = bool(
        moe_expert_vis_ctx is not None
        and moe_expert_vis_ctx.get("enabled", False)
        and args.use_moe_discriminator
    )
    class_expert_count = None
    if moe_vis_enabled:
        num_expert = int(moe_expert_vis_ctx["num_expert"])
        class_expert_count = np.zeros((num_classes, num_expert), dtype=np.float64)

    for batch_data in tqdm.tqdm(local_loader, f"| run | train | {epoch + 1} |"):
        images, class_idx, sample_idx = batch_data[0], batch_data[1], batch_data[2]
        class_idx = class_idx.to(device, non_blocking=True)
        batch = images.shape[0]

        if batch != args.batch_size:
            continue

        images = images.to(device, non_blocking=True)
        x, cls_token = extract_dinov3_feature_batch(
            images, feature_extractor, args.class_layer_indices,
            class_indices=class_idx,
            use_cls_token=args.use_cls_token, return_cls_token=True,
        )
        x = compute_residual_fn(
            x,
            class_idx,
            reference_memory_by_class,
            reference_index_by_class,
        )

        # All labels are 0: training data contains only normal samples.
        local_label = torch.zeros((batch, 784), device=device)

        localnet.train()
        localnet_optimizer.zero_grad()

        gate_cls_token = cls_token if cls_token.shape[0] == x.shape[0] else None
        patch_class_idx = None
        if args.use_moe_discriminator and args.moe_hard_class_gate:
            token_per_image = int(x.shape[1]) if x.dim() >= 2 else 1
            patch_class_idx = (
                class_idx.to(device=x.device, dtype=torch.long)
                .reshape(-1, 1, 1)
                .expand(-1, token_per_image, 1)
                .contiguous()
            )
        batch_feature, local_pred = localnet(
            x,
            cls_token=gate_cls_token,
            patch_class_idx=patch_class_idx,
            class_idx=class_idx,
        )

        # MoE expert visualization tracking
        if moe_vis_enabled:
            discriminator = getattr(localnet, "discriminator", None)
            get_stats = getattr(discriminator, "get_latest_gate_stats", None)
            gate_stats = get_stats() if callable(get_stats) else None
            if gate_stats is not None:
                gate_top_k_idx, _gate_score = gate_stats
                token_per_image = int(batch_feature.shape[1]) if batch_feature.dim() >= 2 else 1
                class_for_gate = class_idx
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

        # BCE loss
        if args.balancing:
            _loss = compute_balanced_bce_loss(
                localnet_criterion=localnet_criterion,
                pred=local_pred,
                target=local_label,
            )
        else:
            _loss = localnet_criterion(local_pred, local_label)

        # MoE gate auxiliary loss
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

        _local_loss = _loss
        if args.use_moe_discriminator:
            _local_loss = _local_loss + args.gate_aux_weight * _gate_aux_loss

        _local_loss.backward()
        localnet_optimizer.step()

        local_loss += _local_loss / total_batch
        bce_loss += _loss / total_batch
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
    gate_aux_loss_value = (
        gate_aux_loss.item() if torch.is_tensor(gate_aux_loss) else float(gate_aux_loss)
    )

    return (
        local_loss_value,
        bce_loss_value,
        gate_aux_loss_value,
        iteration,
    )


def main():
    torch.autograd.set_detect_anomaly(True)
    args = parse_args()
    fix_seed(args.seed)

    saved_dir = os.path.join(args.save_path, args.dataset, "clean")
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

    class_names = get_all_class_names(args.dataset)
    args.class_names = class_names
    num_classes_runtime = len(class_names)

    args.class_layer_indices = {
        i: DINO_CLASS_LAYER_INDICES.get(name, [-1])
        for i, name in enumerate(class_names)
    }
    _all_layer_values = sorted(set(
        idx for lst in args.class_layer_indices.values() for idx in lst
    ))

    if args.use_moe_discriminator and args.moe_hard_class_gate:
        raise ValueError("Hard class gate is not supported in clean-only training mode.")

    train_dataset = MultiClassFeatureDataset(
        data_path=args.data_path,
        dataset_name=args.dataset,
        image_size=args.image_size,
        crop_size=args.crop_size,
        seed=args.seed,
        shuffle=True,
    )

    feature_extractor = build_dinov3_feature_extractor(args.feature_model, device)
    inferred_patch_mask_size = infer_dinov3_patch_mask_size(
        train_dataset, feature_extractor, _all_layer_values, args.use_cls_token, device
    )
    train_dataset.patch_mask_size = inferred_patch_mask_size
    print(f"inferred patch_mask_size from feature extractor: {inferred_patch_mask_size}")
    _selected_blocks = resolve_dino_block_indices(feature_extractor, _all_layer_values)
    _display = [b + 1 for b in _selected_blocks]
    print(f"[DINOv3] aggregating layers {_display} (0-based blocks {_selected_blocks})")
    for cls_name, cls_layers in DINO_CLASS_LAYER_INDICES.items():
        if cls_name in class_names:
            _cls_blocks = resolve_dino_block_indices(feature_extractor, cls_layers)
            if set(_cls_blocks) != set(_selected_blocks):
                print(f"  class {cls_name}: layers {cls_layers} -> blocks {_cls_blocks}")

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
            extract_feature_fn=lambda images, class_indices=None: extract_dinov3_feature_batch(
                images, feature_extractor, args.class_layer_indices,
                class_indices=class_indices,
                use_cls_token=args.use_cls_token,
            ),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        reference_memory_by_class = common_utils.move_reference_memory_to_gpu(
            reference_memory_by_class_cpu, device
        )
        common_utils.remove_reference_samples_from_dataset(train_dataset, reference_indices_by_class)
        print(f"train samples after removing references: {len(train_dataset)}")
        compute_residual_fn = common_utils.compute_residual_feature_batch
    else:
        reference_indices_by_class = {cls: [] for cls in range(len(class_names))}
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
        feature_extractor, mini_loader, _all_layer_values, args.use_cls_token, device
    )

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
        num_classes=len(class_names),
        class_conditioned_adapter=True,
    ).to(device)
    if args.use_moe_discriminator:
        discriminator = getattr(localnet, "discriminator", None)
        if discriminator is not None and hasattr(discriminator, "enable_gate_stats"):
            discriminator.enable_gate_stats(bool(args.moe_expert_vis_enable))

    localnet_optimizer = optim.RMSprop(localnet.parameters(), lr=args.lr, momentum=0.2)
    localnet_criterion = nn.BCELoss().to(device)

    residual_tag = "" if args.residual else "_no_residual"
    run_name = (
        "clean_"
        + "_balancing_"
        + str(args.balancing)
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
        (
            last_done,
            iteration,
            best_mean,
            best_result_by_class,
            _ckpt_mb_freeze,
            _ckpt_mb_snapshot,
        ) = common_utils.load_train_checkpoint(
            resume_path,
            localnet,
            localnet_optimizer,
            None,  # onetoone_optimizer
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
            gate_aux_loss_value,
            iteration,
        ) = train_one_epoch(
            args=args,
            epoch=epoch,
            localnet=localnet,
            feature_extractor=feature_extractor,
            localnet_optimizer=localnet_optimizer,
            localnet_criterion=localnet_criterion,
            local_loader=local_loader,
            iteration=iteration,
            num_classes=len(class_names),
            reference_memory_by_class=reference_memory_by_class,
            reference_index_by_class=reference_index_by_class,
            moe_expert_vis_ctx=moe_expert_vis_ctx,
            compute_residual_fn=compute_residual_fn,
        )

        print(
            "epoch %d | loss: %.6f, bce loss: %.6f, gate aux loss: %.6f"
            % (epoch + 1, local_loss_value, bce_loss_value, gate_aux_loss_value)
        )

        if (epoch + 1) % args.eval_interval == 0:
            eval_rows = []
            mean_img_list = []
            mean_pixel_list = []
            for class_idx_eval, class_name in enumerate(class_names):
                test_loader = build_test_loader(args, class_name)
                (
                    auroc, _, _, pixel_auroc, _, _, _,
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
                    [class_name, auroc, pixel_auroc]
                )
                mean_img_list.append(auroc)
                mean_pixel_list.append(pixel_auroc)
                print(
                    f"epoch {epoch + 1} | {class_name} | I-AUROC: {auroc:.5f}, P-AUROC: {pixel_auroc:.5f}"
                )
                del test_loader

            epoch_mean_img = float(np.mean(mean_img_list))
            epoch_mean_pixel = float(np.mean(mean_pixel_list))
            epoch_mean = (epoch_mean_img + epoch_mean_pixel) / 2
            print(f"epoch {epoch + 1} | multiclass I-AUROC mean: {epoch_mean_img:.5f}")
            print(f"epoch {epoch + 1} | multiclass P-AUROC mean: {epoch_mean_pixel:.5f}")

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
                        f"epoch {epoch + 1} | total loss: {local_loss_value:.6f} | bce loss: {bce_loss_value:.6f} | gate aux loss: {gate_aux_loss_value:.6f} | gate aux weight: {args.gate_aux_weight:.6f} | multiclass I-AUROC mean: {epoch_mean_img:.5f} | multiclass P-AUROC mean: {epoch_mean_pixel:.5f}\n"
                    )

        ckpt_path = common_utils.default_train_checkpoint_path(saved_dir, run_name)
        common_utils.save_train_checkpoint(
            ckpt_path,
            epoch_completed=epoch,
            iteration=iteration,
            localnet=localnet,
            localnet_optimizer=localnet_optimizer,
            onetoone_optimizer=None,
            best_mean=best_mean,
            best_result_by_class=best_result_by_class,
            memory_bank_freeze_start_epoch=-1,
            memory_bank_snapshot=None,
        )
        print(f"[checkpoint] saved training state -> {ckpt_path}")

    if len(best_result_by_class) == 0:
        raise RuntimeError("训练未产生可用评估结果，请检查数据与参数。")

    results = []
    for class_name in class_names:
        auroc, pixel_auroc = best_result_by_class[class_name]
        results.append([class_name, auroc, pixel_auroc])
        print(
            f"best | {class_name} | I-AUROC: {auroc:.5f} | P-AUROC: {pixel_auroc:.5f}"
        )

    df = pd.DataFrame(
        results,
        columns=[
            "class",
            "auroc_sp",
            "auroc_px",
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
