"""Chapter 1: clean multiclass anomaly detection training entry.

This entry implements ``docs/chapter1_clean_multiclass_ad/implementation_plan.md``:

* frozen DINOv3 multi-layer features + ``class_patch`` fusion in
  ``prior_top1`` reference-matching mode (one unique reference ``j*`` per
  query patch, selected by the class prior);
* feature-level anomaly synthesis in the reference-residual space, with
  per-class LOIO-calibrated Gaussian scales and structured spatial masks;
* the original legacy MLP discriminator + class-conditioned adapter by
  default, with an optional hard-class MoE discriminator
  (``--use_moe_discriminator --moe_hard_class_gate``);
* Balanced BCE only -- no OTO, no IRMB / pseudo labels, no noisy
  self-training.

Training data are clean normal images only.  Validation / test / independent
inference never synthesize anomalies; they compute the same class-patch
residual used at train time.

The class-patch runtime, fusion config builder and validation helpers are
imported from the noisy entry so the residual definition cannot drift between
chapters.
"""

import argparse
import csv
import datetime
import math
import os
import shutil
import time
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as torch_mp
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader

from dataset import dataset_extract
from dataset.feature_extract import (
    _FEATURE_MODEL_CHOICES,
    build_dinov3_feature_extractor,
    extract_dinov3_multilayer_feature_batch,
    resolve_dino_block_indices,
)
from dataset.multiclass_feature_dataset import (
    MultiClassFeatureDataset,
    get_all_class_names,
    get_fusion_candidate_layers,
    normalize_candidate_layers,
)
from src.model import model
from src.model.class_patch_fusion_adapter import ClassPatchFusionAdapter
from src.model.layer_fusion import ClassConditionedLayerFusion
from src.train.feature_anomaly_synthesis import (
    FeatureAnomalySynthesizer,
    build_synthetic_training_batch,
    calibrate_class_sigmas_loio,
)
from utils import evaluate as eval_utils
from utils.logging import enable_print_logging
from utils.loss import compute_balanced_bce_components
import utils.train_utils as common_utils

# Reuse the exact runtime + fusion metadata code from the noisy entry.
from self_train_ad_multiclass_dinov3 import (
    ClassPatchFusionRuntime,
    LayerFusionRuntime,
    _build_fusion_config,
    _fusion_preprocess_metadata,
    _parse_no_layer_fusion_classes,
    _validate_resume_fusion_config,
)

warnings.filterwarnings("ignore")

try:
    torch_mp.set_sharing_strategy("file_system")
except (AttributeError, RuntimeError):
    warnings.warn("Failed to set torch multiprocessing sharing strategy.")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def str2bool(value):
    return common_utils.str2bool(value)


def parse_args():
    parser = argparse.ArgumentParser("self_train_ad_multiclass_dinov3_clean")

    # ------------------------------------------------------------------
    # Data / optimisation
    # ------------------------------------------------------------------
    parser.add_argument(
        "--data_path",
        type=str,
        default="/media/honeywell/D/bhy/dataset/MVTec",
        help="Clean multiclass dataset root (only train/good is read).",
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
    parser.add_argument(
        "--reference_extract_batch_size",
        type=int,
        default=0,
        help="Images per backbone forward when building the raw reference "
        "bank. 0 means min(--batch_size, 8). Lower this if host RAM is tight.",
    )
    parser.add_argument("--strict_clean_reference", action="store_true")
    parser.add_argument(
        "--img_score_topk_ratio",
        type=float,
        default=0.01,
        help="Image-level score uses mean of top-k patch scores, k=ceil(P*ratio).",
    )
    parser.add_argument("--use_cls_token", type=str2bool, default="False")
    parser.add_argument(
        "--feature_model",
        type=str,
        choices=_FEATURE_MODEL_CHOICES,
        default="dinov3_vitl16",
        help="DINOv3 variant. ViT-L/16 matches the chapter-2 noisy entry.",
    )
    parser.add_argument(
        "--residual",
        type=str2bool,
        default="True",
        help="Chapter 1 requires the reference-residual path.",
    )
    parser.add_argument(
        "--balancing",
        type=str2bool,
        default="True",
        help="Balanced BCE (separate pos/neg means); chapter 1 uses True.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Path to a training checkpoint saved by this script.",
    )
    parser.add_argument(
        "--model_selection",
        type=str,
        choices=["last", "best"],
        default="last",
        help="'last' (default) follows the chapter-1 anti-leak rule: test "
        "metrics are never used to pick the checkpoint. 'best' reproduces the "
        "legacy monitor-based selection.",
    )
    parser.add_argument(
        "--eval_compute_pro",
        type=str2bool,
        default="False",
        help="Also compute AUPRO each evaluation (expensive).",
    )
    # ------------------------------------------------------------------
    # Optional MoE discriminator (hard class routing)
    # ------------------------------------------------------------------
    parser.add_argument(
        "--use_moe_discriminator",
        action="store_true",
        help="Replace the legacy MLP discriminator with an MoE discriminator. "
        "The clean entry currently supports hard class routing only.",
    )
    parser.add_argument(
        "--moe_hard_class_gate",
        action="store_true",
        help="Route every patch token to the expert with the same class index. "
        "Requires --use_moe_discriminator; expert count follows num_classes "
        "and top_k is forced to 1.",
    )
    parser.add_argument(
        "--moe_num_expert",
        type=int,
        default=4,
        help="Soft-MoE expert count. Ignored with --moe_hard_class_gate "
        "(experts = number of classes).",
    )
    parser.add_argument(
        "--moe_top_k",
        type=int,
        default=2,
        help="Soft-MoE top-k. Ignored with --moe_hard_class_gate (forced 1).",
    )
    parser.add_argument(
        "--moe_use_cls_token",
        action="store_true",
        help="Soft-MoE gate uses the layer-mean CLS token. No effect with "
        "--moe_hard_class_gate.",
    )
    parser.add_argument(
        "--gate_aux_weight",
        type=float,
        default=0.05,
        help="Weight for soft-MoE gate load-balancing loss. Hard class routing "
        "has no auxiliary loss, so this is a no-op in hard mode.",
    )

    # ------------------------------------------------------------------
    # class_patch fusion (prior_top1)
    # ------------------------------------------------------------------
    parser.add_argument(
        "--fusion_mode",
        type=str,
        choices=["class_patch", "learned_class", "learned_shared"],
        default="class_patch",
        help="Chapter 1 main model is class_patch. learned_class / "
        "learned_shared are the E1 ablations (class-conditioned vs shared "
        "softmax layer weights). Freeze class_patch with --fusion_freeze for "
        "the E0 fixed-mean ablation.",
    )
    parser.add_argument(
        "--fusion_layers",
        type=int,
        nargs="+",
        default=None,
        help="Shared 1-based candidate DINOv3 layers. Default depends on "
        "--feature_model (DINO_FUSION_CANDIDATE_LAYERS).",
    )
    parser.add_argument("--fusion_temperature", type=float, default=1.0)
    parser.add_argument(
        "--fusion_norm",
        type=str,
        choices=["layer_norm", "none"],
        default="none",
    )
    parser.add_argument("--fusion_norm_affine", type=str2bool, default="False")
    parser.add_argument(
        "--fusion_query_chunk",
        type=int,
        default=256,
        help="Query patches per reference-gather/fusion chunk.",
    )
    parser.add_argument(
        "--fusion_ref_chunk",
        type=int,
        default=1024,
        help="Rows per chunk when re-fusing selected raw references.",
    )
    parser.add_argument("--fusion_nn_query_chunk", type=int, default=2048)
    parser.add_argument("--fusion_nn_ref_chunk", type=int, default=4096)
    parser.add_argument(
        "--reference_refresh_unit",
        type=str,
        choices=["epoch", "step"],
        default="epoch",
    )
    parser.add_argument("--reference_refresh_interval", type=int, default=1)
    parser.add_argument(
        "--reference_storage_dtype",
        type=str,
        choices=["float16", "float32"],
        default="float16",
    )
    parser.add_argument("--gate_embedding_dim", type=int, default=32)
    parser.add_argument("--gate_hidden_dim", type=int, default=64)
    parser.add_argument(
        "--gate_norm", type=str, choices=["layer_norm", "none"], default="layer_norm"
    )
    parser.add_argument("--gate_lambda_init", type=float, default=0.1)
    parser.add_argument("--gate_use_context", type=str2bool, default="False")
    parser.add_argument("--gate_demean_delta", type=str2bool, default="True")
    parser.add_argument(
        "--fusion_no_layer_fusion_classes",
        type=str,
        default="",
        help="Comma-separated classes pinned to the last candidate layer. "
        "Empty by default in chapter 1.",
    )
    parser.add_argument("--fusion_last_layer_prior_logit", type=float, default=8.0)
    parser.add_argument(
        "--feature_adapter_mode",
        type=str,
        choices=["legacy"],
        default="legacy",
        help="Chapter 1 keeps the existing localnet class adapter.",
    )
    parser.add_argument("--adapter_hidden_dim", type=int, default=128)
    parser.add_argument("--adapter_beta_init", type=float, default=0.15)
    parser.add_argument(
        "--fusion_lr",
        type=float,
        default=None,
        help="Optional separate LR for fusion parameters.",
    )
    parser.add_argument(
        "--fusion_freeze",
        type=str2bool,
        default="False",
        help="Freeze theta / patch gate / lambda at initialisation. With the "
        "zero-initialised gate this is the E0 fixed-mean ablation.",
    )
    parser.add_argument(
        "--fusion_profile", type=str2bool, default="False"
    )
    parser.add_argument(
        "--fusion_stale_probe_interval", type=int, default=0
    )

    # ------------------------------------------------------------------
    # Feature-level anomaly synthesis
    # ------------------------------------------------------------------
    parser.add_argument(
        "--enable_feature_synthesis",
        type=str2bool,
        default="True",
        help="Chapter 1 requires True; False is a debug-only negative control.",
    )
    parser.add_argument(
        "--synthesis_mask_type",
        type=str,
        choices=["mixed", "blob", "line", "spot", "irregular", "independent"],
        default="mixed",
        help="'independent' is the no-spatial-structure ablation.",
    )
    parser.add_argument("--synthesis_area_ratio_min", type=float, default=0.005)
    parser.add_argument("--synthesis_area_ratio_max", type=float, default=0.10)
    parser.add_argument("--synthesis_gamma_min", type=float, default=0.5)
    parser.add_argument("--synthesis_gamma_max", type=float, default=1.5)
    parser.add_argument(
        "--synthesis_sigma_mode",
        type=str,
        choices=["class_loio", "global"],
        default="class_loio",
        help="class_loio: per-class Q75 LOIO residual norm / sqrt(C). "
        "global: one shared scale (mean of the class scales unless "
        "--synthesis_global_sigma is set).",
    )
    parser.add_argument("--synthesis_sigma_quantile", type=float, default=0.75)
    parser.add_argument(
        "--synthesis_global_sigma",
        type=float,
        default=None,
        help="Explicit fixed sigma shared by all classes (overrides LOIO).",
    )
    parser.add_argument("--synthesis_sigma_scale", type=float, default=1.0)
    parser.add_argument("--synthesis_sigma_min", type=float, default=1e-6)

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


def infer_patch_side(train_dataset, feature_extractor, candidate_layers, args):
    """Infer the square patch-grid side from one clean image."""
    image_path, _ = train_dataset.samples[0]
    image = Image.open(image_path).convert("RGB")
    image = train_dataset.transform_x(image).unsqueeze(0).to(device)
    features = extract_dinov3_multilayer_feature_batch(
        image,
        feature_extractor,
        candidate_layers,
        use_cls_token=args.use_cls_token,
    )
    num_patches = int(features.shape[1])
    side = int(round(math.sqrt(num_patches)))
    if side * side != num_patches:
        raise RuntimeError(
            f"patch 数 {num_patches} 不是完全平方数，无法构造方形空间 mask。"
        )
    return side


def reconstruct_reference_image_ids(raw_memory_by_class, reference_indices_by_class):
    """Rebuild the patch -> image mapping from the bank shape + ref indices.

    The bank builder consumes each class's reference images in exactly the
    order of ``reference_indices_by_class[cls]`` and contributes ``P`` patch
    rows per image, so the mapping is recoverable without persisting it in the
    sidecar.  This keeps LOIO recalibration possible when resuming an older
    checkpoint that predates ``class_sigmas``.
    """
    image_ids = {}
    for class_idx, raw in raw_memory_by_class.items():
        refs = list(reference_indices_by_class.get(int(class_idx), []))
        if len(refs) == 0:
            raise RuntimeError(
                f"class {class_idx}: no reference image indices to rebuild "
                "the patch->image mapping."
            )
        num_rows = int(np.asarray(raw).shape[0])
        if num_rows % len(refs) != 0:
            raise RuntimeError(
                f"class {class_idx}: {num_rows} bank rows are not divisible by "
                f"{len(refs)} reference images."
            )
        patches_per_image = num_rows // len(refs)
        image_ids[int(class_idx)] = np.repeat(
            np.asarray(refs, dtype=np.int64), patches_per_image
        )
    return image_ids


def evaluate_epoch(
    localnet,
    feature_extractor,
    test_loader,
    args,
    class_idx_eval,
    fusion_runtime,
):
    return eval_utils.evaluate_residual_multiclass_epoch(
        localnet=localnet,
        feature_extractor=feature_extractor,
        test_loader=test_loader,
        args=args,
        class_idx_eval=class_idx_eval,
        reference_memory_by_class=None,
        reference_index_by_class=None,
        device=device,
        extract_feature_batch_fn=(
            lambda images, fe, a, class_indices=None: fusion_runtime.extract(
                images, fe, a, class_indices=class_indices
            )
        ),
        compute_residual_feature_batch_fn=fusion_runtime.residual,
        return_dict=True,
        compute_pro_metric=args.eval_compute_pro,
    )


def _accumulate_synthesis_stats(accum, stats, steps):
    if not stats:
        return
    for key, value in stats.items():
        if key.startswith("mask_type_") and key.endswith("_count"):
            accum[key] += int(value)
        else:
            accum[key] += float(value)
    accum["_count"] += 1


def _summarise_synthesis_stats(accum):
    count = max(1, int(accum.get("_count", 0)))
    summary = {}
    for key, value in accum.items():
        if key == "_count":
            continue
        if key.startswith("mask_type_") and key.endswith("_count"):
            summary[key] = int(value)
        else:
            summary[key] = float(value) / count
    return summary


def train_one_epoch(
    args,
    epoch,
    localnet,
    feature_extractor,
    optimizer,
    criterion,
    loader,
    iteration,
    runtime,
    synthesizer,
):
    localnet.train()
    runtime.begin_epoch(iteration, epoch)

    total_steps = 0
    loss_sum = pos_sum = neg_sum = gate_sum = 0.0
    synth_accum = defaultdict(float)
    synth_accum["_count"] = 0

    for batch_data in tqdm.tqdm(loader, f"| run | clean train | {epoch + 1} |"):
        images, class_idx = batch_data[0], batch_data[1]
        images = images.to(device, non_blocking=True)
        class_idx = class_idx.to(device, non_blocking=True)

        raw_features = runtime.extract(images, feature_extractor, args)
        residual = runtime.residual(raw_features, class_idx)

        if synthesizer is not None:
            features, labels, classes, _mask, synth_stats = (
                build_synthetic_training_batch(
                    residual, class_idx, synthesizer
                )
            )
            _accumulate_synthesis_stats(synth_accum, synth_stats, total_steps)
        else:
            batch, patches, _ = residual.shape
            features = residual
            labels = torch.zeros(
                (batch, patches), dtype=torch.float32, device=residual.device
            )
            classes = class_idx

        # Hard class routing needs the class id of every patch token.  The
        # synthetic half is concatenated after the normal half, so ``classes``
        # already has the matching 2B rows.
        patch_class_idx = None
        if args.use_moe_discriminator and args.moe_hard_class_gate:
            token_per_image = (
                int(features.shape[1]) if features.dim() >= 2 else 1
            )
            patch_class_idx = (
                classes.to(device=features.device, dtype=torch.long)
                .reshape(-1, 1, 1)
                .expand(-1, token_per_image, 1)
                .contiguous()
            )

        optimizer.zero_grad(set_to_none=True)
        _, pred = localnet(
            features,
            patch_class_idx=patch_class_idx,
            class_idx=classes,
        )
        pos_loss, neg_loss = compute_balanced_bce_components(
            criterion, pred, labels
        )
        gate_aux_loss = torch.tensor(0.0, device=residual.device)
        if args.use_moe_discriminator:
            discriminator = getattr(localnet, "discriminator", None)
            if discriminator is not None and hasattr(discriminator, "get_loss"):
                maybe_gate_loss = discriminator.get_loss(
                    clear=True, reduction="mean", default=None
                )
                if maybe_gate_loss is not None:
                    gate_aux_loss = maybe_gate_loss
        loss = (
            pos_loss
            + neg_loss
            + float(args.gate_aux_weight) * gate_aux_loss
        )
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"non-finite clean training loss at epoch {epoch + 1} "
                f"iteration {iteration}"
            )
        loss.backward()
        optimizer.step()

        if hasattr(runtime, "maybe_probe_stale"):
            runtime.maybe_probe_stale(raw_features, class_idx, iteration)
        runtime.after_step(iteration, epoch)

        loss_sum += float(loss.detach().item())
        pos_sum += float(pos_loss.detach().item())
        neg_sum += float(neg_loss.detach().item())
        gate_sum += float(gate_aux_loss.detach().item())
        total_steps += 1
        iteration += 1

    if total_steps == 0:
        raise RuntimeError(
            "训练 DataLoader 没有产生任何 batch；请检查数据路径与 batch_size。"
        )

    return (
        loss_sum / total_steps,
        pos_sum / total_steps,
        neg_sum / total_steps,
        gate_sum / total_steps,
        iteration,
        _summarise_synthesis_stats(synth_accum),
    )


def _mean_metrics(per_class, key):
    values = [
        row[key] for row in per_class.values()
        if row.get(key) is not None
    ]
    return float(np.mean(values)) if values else float("nan")


def main():
    args = parse_args()
    fix_seed(args.seed)

    if args.moe_hard_class_gate and not args.use_moe_discriminator:
        raise ValueError(
            "--moe_hard_class_gate requires --use_moe_discriminator."
        )
    if args.use_moe_discriminator and not args.moe_hard_class_gate:
        raise ValueError(
            "The clean chapter-1 entry currently supports MoE hard-class "
            "routing only (pass --moe_hard_class_gate). Soft MoE needs a "
            "CLS-token-aware evaluation path that this entry does not wire up."
        )
    if args.use_moe_discriminator:
        if int(args.moe_num_expert) < 1:
            raise ValueError("--moe_num_expert must be >= 1.")
        if int(args.moe_top_k) < 1:
            raise ValueError("--moe_top_k must be >= 1.")
        if float(args.gate_aux_weight) < 0:
            raise ValueError("--gate_aux_weight must be >= 0.")
        if args.moe_use_cls_token:
            warnings.warn(
                "--moe_use_cls_token has no effect with hard class routing."
            )
    if not bool(args.residual):
        raise ValueError("Chapter 1 requires --residual True.")
    if not bool(args.balancing):
        raise ValueError("Chapter 1 requires Balanced BCE (--balancing True).")
    if args.synthesis_area_ratio_min <= 0 or (
        args.synthesis_area_ratio_min > args.synthesis_area_ratio_max
    ):
        raise ValueError(
            "--synthesis_area_ratio_min/max must satisfy 0 < min <= max."
        )
    if args.synthesis_gamma_min < 0 or (
        args.synthesis_gamma_min > args.synthesis_gamma_max
    ):
        raise ValueError("--synthesis_gamma_min/max must satisfy 0 <= min <= max.")
    if args.synthesis_sigma_scale <= 0:
        raise ValueError("--synthesis_sigma_scale must be > 0.")
    if args.synthesis_global_sigma is not None and args.synthesis_global_sigma <= 0:
        raise ValueError("--synthesis_global_sigma must be > 0.")

    saved_dir = os.path.join(args.save_path, args.dataset, "clean_class_patch")
    os.makedirs(saved_dir, exist_ok=True)
    enable_print_logging(os.path.join(saved_dir, "run_stdout.log"))

    args_csv_path = os.path.join(saved_dir, "cli_args.csv")
    with open(args_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        for key, value in sorted(vars(args).items()):
            writer.writerow([key, value])
    print(f"[CLI Args] saved -> {args_csv_path}")

    class_names = get_all_class_names(args.dataset)
    args.class_names = class_names
    num_classes = len(class_names)
    args.fusion_no_layer_fusion_class_names = _parse_no_layer_fusion_classes(
        args.fusion_no_layer_fusion_classes, class_names
    )
    no_layer_fusion_classes = [
        class_names.index(name)
        for name in args.fusion_no_layer_fusion_class_names
    ]

    # Chapter-1 hard config: class_patch + prior_top1 + legacy adapter.
    args.fusion_match_mode = "prior_top1"
    args.fusion_topk = 1
    args.fusion_candidate_chunk = 8
    args.feature_adapter_mode = "legacy"

    resume_path = (args.resume or "").strip()
    resume_meta = None
    if resume_path:
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"--resume path not found: {resume_path}")
        resume_meta = common_utils.inspect_train_checkpoint(resume_path)
        if not resume_meta.get("has_fusion_state"):
            raise RuntimeError(
                "resume checkpoint has no fusion_state; chapter-1 clean "
                "training requires a class_patch checkpoint."
            )

    ckpt_cfg = (resume_meta or {}).get("fusion_config")
    if args.fusion_layers is None:
        if ckpt_cfg and ckpt_cfg.get("candidate_layers"):
            candidate_layers = ckpt_cfg["candidate_layers"]
            print(f"[Fusion] adopting candidate layers from checkpoint: {candidate_layers}")
        else:
            candidate_layers = get_fusion_candidate_layers(args.feature_model)
    else:
        candidate_layers = args.fusion_layers
    candidate_layers = normalize_candidate_layers(
        candidate_layers, args.feature_model
    )
    args.fusion_layers = list(candidate_layers)

    if ckpt_cfg:
        _validate_resume_fusion_config(
            ckpt_cfg, args, class_names, candidate_layers
        )
        if bool(ckpt_cfg.get("fusion_freeze", False)) != bool(args.fusion_freeze):
            raise RuntimeError(
                "resume fusion_freeze mismatch: checkpoint="
                f"{ckpt_cfg.get('fusion_freeze')} current={args.fusion_freeze}"
            )
        ckpt_synth_cfg = (resume_meta or {}).get("synthesis_config")
        if ckpt_synth_cfg:
            if bool(ckpt_synth_cfg.get("enabled", True)) != bool(
                args.enable_feature_synthesis
            ):
                raise RuntimeError(
                    "resume enable_feature_synthesis mismatch: checkpoint="
                    f"{ckpt_synth_cfg.get('enabled')} "
                    f"current={args.enable_feature_synthesis}"
                )
            if str(ckpt_synth_cfg.get("mask_type", "")) != str(
                args.synthesis_mask_type
            ):
                warnings.warn(
                    "resume changes the synthesis mask type from "
                    f"{ckpt_synth_cfg.get('mask_type')} to "
                    f"{args.synthesis_mask_type}; class sigmas stay fixed from "
                    "the checkpoint."
                )

    print(
        f"[Chapter1] backbone={args.feature_model} | layers={candidate_layers} | "
        f"fusion={args.fusion_mode}"
        + ("/prior_top1" if args.fusion_mode == "class_patch" else "")
        + " | adapter=legacy | loss=balanced_bce"
    )

    train_dataset = MultiClassFeatureDataset(
        data_path=args.data_path,
        dataset_name=args.dataset,
        image_size=args.image_size,
        crop_size=args.crop_size,
        seed=args.seed,
        shuffle=True,
    )
    if len(train_dataset) == 0:
        raise RuntimeError(
            "Clean training set is empty; expected <data_path>/<class>/train/good."
        )

    feature_extractor = build_dinov3_feature_extractor(args.feature_model, device)
    patch_side = infer_patch_side(
        train_dataset, feature_extractor, candidate_layers, args
    )
    train_dataset.patch_mask_size = patch_side
    selected_blocks = resolve_dino_block_indices(feature_extractor, candidate_layers)
    print(
        f"[DINOv3] chapter-1 candidate layers {[b + 1 for b in selected_blocks]} "
        f"(0-based blocks {selected_blocks}); patch grid {patch_side}x{patch_side}"
    )

    # ------------------------------------------------------------------
    # Clean references + raw multilayer bank
    # ------------------------------------------------------------------
    reference_indices_by_class = common_utils.select_reference_indices_by_class(
        train_dataset,
        num_classes,
        num_reference_images_per_class=args.num_reference_images_per_class,
        seed=args.seed,
        strict_clean_reference=args.strict_clean_reference,
    )
    for class_idx, ref_ids in reference_indices_by_class.items():
        print(
            f"class {class_idx} | selected clean references: {len(ref_ids)} "
            f"(target={args.num_reference_images_per_class})"
        )

    raw_metadata = {
        "feature_model": str(args.feature_model),
        "candidate_layers": [int(v) for v in candidate_layers],
        "class_names": list(class_names),
        "use_cls_token": bool(args.use_cls_token),
        "image_size": int(args.image_size),
        "crop_size": int(args.crop_size),
        "dataset": str(args.dataset),
        "preprocess": _fusion_preprocess_metadata(args),
        "reference_indices_by_class": {
            int(k): [int(v) for v in vals]
            for k, vals in reference_indices_by_class.items()
        },
        "num_reference_images_per_class": int(
            args.num_reference_images_per_class
        ),
        "strict_clean_reference": bool(args.strict_clean_reference),
        "reference_seed": int(args.seed),
        "storage_dtype": str(args.reference_storage_dtype),
    }

    ckpt_sigmas = (resume_meta or {}).get("class_sigmas")
    need_loio = bool(args.enable_feature_synthesis) and not ckpt_sigmas
    if args.synthesis_global_sigma is not None:
        need_loio = False

    raw_reference_memory_by_class = None
    reference_image_ids_by_class = None
    loaded_sidecar_path = None
    ckpt_manifest = (resume_meta or {}).get("raw_reference_manifest")
    if ckpt_manifest:
        resume_dir = os.path.dirname(os.path.abspath(resume_path))
        try:
            loaded_raw, loaded_meta = common_utils.load_raw_reference_bank(
                ckpt_manifest, base_dir=resume_dir
            )
        except (FileNotFoundError, ValueError, KeyError, RuntimeError) as exc:
            print(
                "[Fusion] WARNING: could not reuse checkpoint raw reference "
                f"bank ({exc}); re-extracting it from the training set."
            )
        else:
            mismatched = [
                key for key, value in raw_metadata.items()
                if key in loaded_meta and loaded_meta.get(key) != value
            ]
            if mismatched:
                raise RuntimeError(
                    "checkpoint raw reference bank metadata mismatch: "
                    f"{mismatched}"
                )
            raw_reference_memory_by_class = loaded_raw
            loaded_sidecar_path = os.path.abspath(
                os.path.join(resume_dir, ckpt_manifest["path"])
            )
            if need_loio:
                reference_image_ids_by_class = reconstruct_reference_image_ids(
                    raw_reference_memory_by_class, reference_indices_by_class
                )
            print(
                "[Fusion] loaded raw reference bank from checkpoint sidecar: "
                f"{ckpt_manifest.get('path')}"
            )

    if raw_reference_memory_by_class is None:
        if need_loio:
            (
                raw_reference_memory_by_class,
                raw_info,
                reference_image_ids_by_class,
            ) = common_utils.build_multilayer_reference_memory_bank(
                train_dataset,
                reference_indices_by_class,
                extract_multilayer_fn=lambda images, class_indices=None: (
                    extract_dinov3_multilayer_feature_batch(
                        images,
                        feature_extractor,
                        candidate_layers,
                        use_cls_token=args.use_cls_token,
                    )
                ),
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
                storage_dtype=args.reference_storage_dtype,
                return_image_ids=True,
                extract_batch_size=(
                    args.reference_extract_batch_size or None
                ),
            )
        else:
            raw_reference_memory_by_class, raw_info = (
                common_utils.build_multilayer_reference_memory_bank(
                    train_dataset,
                    reference_indices_by_class,
                    extract_multilayer_fn=lambda images, class_indices=None: (
                        extract_dinov3_multilayer_feature_batch(
                            images,
                            feature_extractor,
                            candidate_layers,
                            use_cls_token=args.use_cls_token,
                        )
                    ),
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    device=device,
                    storage_dtype=args.reference_storage_dtype,
                    extract_batch_size=(
                        args.reference_extract_batch_size or None
                    ),
                )
            )
        raw_metadata.update(raw_info)

    first_raw = next(iter(raw_reference_memory_by_class.values()))
    num_layers = int(first_raw.shape[1])
    feature_dim = int(first_raw.shape[2])
    raw_metadata["num_layers"] = num_layers
    raw_metadata["feature_dim"] = feature_dim
    if num_layers != len(candidate_layers):
        raise RuntimeError(
            f"Raw bank has {num_layers} layers but {len(candidate_layers)} "
            "candidate layers were requested."
        )

    if loaded_sidecar_path is None:
        raw_reference_manifest = common_utils.save_raw_reference_bank(
            raw_reference_memory_by_class,
            raw_metadata,
            save_dir=saved_dir,
            prefix=(
                f"raw_reference_bank_{args.dataset}_clean_{args.feature_model}"
            ),
        )
    elif os.path.dirname(loaded_sidecar_path) == os.path.abspath(saved_dir):
        # Same directory: keep the checkpoint's content hash identity instead
        # of re-serialising a semantically identical metadata dict.
        raw_reference_manifest = dict(ckpt_manifest)
    else:
        # Resume into a new save directory: byte-copy the verified sidecar so
        # the relative manifest path stays portable and the sha256 is stable.
        dest = os.path.join(
            saved_dir, os.path.basename(loaded_sidecar_path)
        )
        if os.path.isfile(dest):
            if common_utils._sha256_file(dest) != ckpt_manifest.get("sha256"):
                raise RuntimeError(
                    f"Existing sidecar {dest} has a different content hash."
                )
        else:
            shutil.copyfile(loaded_sidecar_path, dest)
        raw_reference_manifest = dict(ckpt_manifest)
        raw_reference_manifest["path"] = os.path.basename(dest)
    print(
        "[Fusion] raw reference bank sidecar: "
        f"{os.path.join(saved_dir, raw_reference_manifest['path'])} "
        f"(sha256={raw_reference_manifest['sha256'][:16]}...)"
    )

    common_utils.remove_reference_samples_from_dataset(
        train_dataset, reference_indices_by_class
    )
    print(f"train samples after removing references: {len(train_dataset)}")

    # ------------------------------------------------------------------
    # Fusion + fused reference cache + prior_top1 runtime
    # ------------------------------------------------------------------
    if args.fusion_mode == "class_patch":
        fusion = ClassPatchFusionAdapter(
            num_classes=num_classes,
            num_layers=num_layers,
            feature_dim=feature_dim,
            temperature=args.fusion_temperature,
            norm_type=args.fusion_norm,
            norm_affine=args.fusion_norm_affine,
            class_embedding_dim=args.gate_embedding_dim,
            gate_hidden_dim=args.gate_hidden_dim,
            gate_norm=args.gate_norm,
            gate_lambda_init=args.gate_lambda_init,
            gate_use_context=args.gate_use_context,
            gate_demean_delta=args.gate_demean_delta,
            enable_adapter=False,
            adapter_hidden_dim=args.adapter_hidden_dim,
            adapter_beta_init=args.adapter_beta_init,
            no_layer_fusion_classes=no_layer_fusion_classes,
            last_layer_prior_logit=args.fusion_last_layer_prior_logit,
        ).to(device)
    else:
        fusion = ClassConditionedLayerFusion(
            num_classes=num_classes,
            num_layers=num_layers,
            feature_dim=feature_dim,
            temperature=args.fusion_temperature,
            norm_type=args.fusion_norm,
            norm_affine=args.fusion_norm_affine,
            shared_weights=bool(args.fusion_mode == "learned_shared"),
        ).to(device)
    if bool(args.fusion_freeze):
        for parameter in fusion.parameters():
            parameter.requires_grad_(False)
        print(
            "[Chapter1] fusion frozen at init (zero theta + zero-init gate) "
            "-> E0 fixed-mean / uniform-residual ablation."
        )

    fusion_cache = common_utils.FusedReferenceCache(
        raw_reference_memory_by_class,
        fusion,
        device,
        expected_num_layers=num_layers,
        expected_feature_dim=feature_dim,
        storage_dtype=args.reference_storage_dtype,
        use_faiss=False,
        query_chunk_size=int(args.fusion_nn_query_chunk),
        ref_chunk_size=int(args.fusion_nn_ref_chunk),
        fuse_chunk_size=int(args.fusion_ref_chunk),
    )
    fusion_cache.refresh(step=0, epoch=0)
    if args.fusion_mode == "class_patch":
        fusion_runtime = ClassPatchFusionRuntime(
            fusion, fusion_cache, candidate_layers, args
        )
    else:
        fusion_runtime = LayerFusionRuntime(
            fusion, fusion_cache, candidate_layers, args
        )

    # ------------------------------------------------------------------
    # Per-class synthesis scales (fixed before training)
    # ------------------------------------------------------------------
    if not bool(args.enable_feature_synthesis):
        print(
            "[Synthesis] DISABLED: only normal patches are supervised. "
            "This is a debug-only negative control, not a chapter-1 baseline."
        )
        class_sigmas = {}
        sigma_stats = {}
        synthesizer = None
    elif ckpt_sigmas:
        class_sigmas = {int(k): float(v) for k, v in ckpt_sigmas.items()}
        if len(class_sigmas) != num_classes:
            raise RuntimeError(
                f"checkpoint class_sigmas has {len(class_sigmas)} entries, "
                f"expected {num_classes}."
            )
        sigma_stats = {"source": "checkpoint"}
        print("[Synthesis] restored fixed class sigmas from checkpoint.")
        for cls in range(num_classes):
            print(
                f"[Synthesis sigma] {class_names[cls]}: {class_sigmas[cls]:.6f}"
            )
        synthesizer = FeatureAnomalySynthesizer(
            class_sigmas=class_sigmas,
            num_classes=num_classes,
            mask_type=args.synthesis_mask_type,
            area_ratio_range=(
                args.synthesis_area_ratio_min,
                args.synthesis_area_ratio_max,
            ),
            gamma_range=(args.synthesis_gamma_min, args.synthesis_gamma_max),
            sigma_scale=args.synthesis_sigma_scale,
        )
    elif args.synthesis_global_sigma is not None:
        class_sigmas = {
            cls: float(args.synthesis_global_sigma) for cls in range(num_classes)
        }
        sigma_stats = {"source": "explicit_global"}
        print(
            f"[Synthesis] explicit global sigma={args.synthesis_global_sigma:.6f}"
        )
        synthesizer = FeatureAnomalySynthesizer(
            class_sigmas=class_sigmas,
            num_classes=num_classes,
            mask_type=args.synthesis_mask_type,
            area_ratio_range=(
                args.synthesis_area_ratio_min,
                args.synthesis_area_ratio_max,
            ),
            gamma_range=(args.synthesis_gamma_min, args.synthesis_gamma_max),
            sigma_scale=args.synthesis_sigma_scale,
        )
    else:
        class_sigmas, sigma_stats = calibrate_class_sigmas_loio(
            raw_reference_memory_by_class,
            reference_image_ids_by_class,
            fusion,
            device,
            quantile=args.synthesis_sigma_quantile,
            sigma_min=args.synthesis_sigma_min,
        )
        sigma_stats["source"] = "loio"
        for cls in range(num_classes):
            stats = sigma_stats[cls]
            print(
                f"[Synthesis sigma] {class_names[cls]}: {class_sigmas[cls]:.6f} "
                f"(Q{args.synthesis_sigma_quantile:.2f}="
                f"{stats['residual_norm_quantile']:.6f}, "
                f"images={stats['num_images']}, patches={stats['num_patches']})"
            )
        if args.synthesis_sigma_mode == "global":
            global_sigma = float(np.mean(list(class_sigmas.values())))
            class_sigmas = {
                cls: global_sigma for cls in range(num_classes)
            }
            sigma_stats = {
                "source": "loio_mean",
                "global_sigma": global_sigma,
            }
            print(
                f"[Synthesis] global sigma mode: shared sigma={global_sigma:.6f}"
            )
        synthesizer = FeatureAnomalySynthesizer(
            class_sigmas=class_sigmas,
            num_classes=num_classes,
            mask_type=args.synthesis_mask_type,
            area_ratio_range=(
                args.synthesis_area_ratio_min,
                args.synthesis_area_ratio_max,
            ),
            gamma_range=(args.synthesis_gamma_min, args.synthesis_gamma_max),
            sigma_scale=args.synthesis_sigma_scale,
        )

    synthesis_config = {
        "enabled": bool(args.enable_feature_synthesis),
        "space": "class_patch_residual",
        "mask_type": str(args.synthesis_mask_type),
        "area_ratio_min": float(args.synthesis_area_ratio_min),
        "area_ratio_max": float(args.synthesis_area_ratio_max),
        "gamma_min": float(args.synthesis_gamma_min),
        "gamma_max": float(args.synthesis_gamma_max),
        "sigma_mode": str(args.synthesis_sigma_mode),
        "sigma_quantile": float(args.synthesis_sigma_quantile),
        "sigma_scale": float(args.synthesis_sigma_scale),
        "sigma_min": float(args.synthesis_sigma_min),
        "training_only": True,
    }

    # ------------------------------------------------------------------
    # Model / optimizer / loaders
    # ------------------------------------------------------------------
    loader_kwargs = common_utils.build_loader_kwargs(
        args.num_workers,
        pin_memory=True,
        prefetch_factor=1,
        persistent_workers=False,
    )
    local_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        **loader_kwargs,
    )

    moe_num_expert_runtime = (
        num_classes
        if args.moe_hard_class_gate
        else int(args.moe_num_expert)
    )
    moe_top_k_runtime = 1 if args.moe_hard_class_gate else int(args.moe_top_k)
    localnet = model.localnet(
        len_feature=feature_dim,
        use_moe_discriminator=bool(args.use_moe_discriminator),
        moe_num_expert=moe_num_expert_runtime,
        moe_top_k=moe_top_k_runtime,
        moe_use_cls_token=bool(args.moe_use_cls_token),
        moe_hard_class_gate=bool(args.moe_hard_class_gate),
        num_classes=num_classes,
        class_conditioned_adapter=True,
        feature_adapter_mode="legacy",
    ).to(device)
    if args.use_moe_discriminator:
        print(
            "[MoE] discriminator enabled | "
            f"hard_class_gate={args.moe_hard_class_gate} | "
            f"experts={moe_num_expert_runtime} | top_k={moe_top_k_runtime}"
        )

    localnet_params = list(localnet.parameters())
    fusion_params = [p for p in fusion.parameters() if p.requires_grad]
    trainable_params = localnet_params + fusion_params
    if args.fusion_lr is not None and float(args.fusion_lr) <= 0:
        raise ValueError("--fusion_lr must be > 0.")

    def build_param_groups():
        if args.fusion_lr is not None and fusion_params:
            return [
                {"params": localnet_params, "lr": float(args.lr)},
                {"params": fusion_params, "lr": float(args.fusion_lr)},
            ]
        return [{"params": trainable_params, "lr": float(args.lr)}]

    localnet_optimizer = optim.RMSprop(
        build_param_groups(), lr=args.lr, momentum=0.2
    )
    localnet_criterion = nn.BCELoss().to(device)

    run_name = (
        "clean_"
        + str(args.fusion_mode)
        + "_synth_"
        + str(args.synthesis_mask_type)
        + "_sigma_"
        + str(args.synthesis_sigma_mode)
        + "_seed_"
        + str(args.seed)
        + "_"
        + str(args.feature_model)
        + "_layers_"
        + "-".join(str(v) for v in candidate_layers)
    )
    if bool(args.fusion_freeze):
        run_name += "_frozen_fusion"
    if not bool(args.enable_feature_synthesis):
        run_name += "_no_synth"
    if args.use_moe_discriminator:
        run_name += "_moe_hard_class"

    fusion_config = _build_fusion_config(
        args, class_names, candidate_layers, feature_dim
    )
    fusion_config["clean_chapter"] = 1
    fusion_config["fusion_freeze"] = bool(args.fusion_freeze)
    fusion_config["task"] = "chapter1_clean_multiclass_ad"

    start_epoch = 0
    iteration = 0
    best_mean = -1.0
    best_result_by_class = {}
    last_result_by_class = {}

    if resume_path:
        print(f"[resume] loading training checkpoint: {resume_path}")
        (
            last_done,
            iteration,
            best_mean,
            best_result_by_class,
            _ckpt_mb_freeze,
            _ckpt_mb_snapshot,
            resume_checkpoint_meta,
        ) = common_utils.load_train_checkpoint(
            resume_path,
            localnet,
            localnet_optimizer,
            None,
            fusion=fusion,
            return_metadata=True,
        )
        _validate_resume_fusion_config(
            resume_checkpoint_meta.get("fusion_config"),
            args,
            class_names,
            candidate_layers,
        )
        fusion_cache.refresh(step=iteration, epoch=last_done + 1)
        start_epoch = last_done + 1
        if start_epoch >= args.epoch:
            print(
                f"[resume] checkpoint epoch {last_done} finished; "
                f"--epoch {args.epoch} means no further epochs to run."
            )
        else:
            print(
                f"[resume] will continue from epoch {start_epoch + 1} "
                f"(completed through epoch {last_done + 1}), "
                f"global_iteration={iteration}"
            )

    def checkpoint_extras():
        return {
            "fusion_state": fusion.state_dict(),
            "fusion_config": fusion_config,
            "raw_reference_manifest": raw_reference_manifest,
            "fusion_cache_metadata": fusion_cache.metadata(),
            "reference_memory_format": "multilayer_raw_v1",
            "class_sigmas": {
                int(k): float(v) for k, v in class_sigmas.items()
            },
            "sigma_stats": sigma_stats,
            "synthesis_config": synthesis_config,
            "reference_indices_by_class": reference_indices_by_class,
            "chapter1_clean": True,
            "moe_config": (
                {
                    "use_moe_discriminator": True,
                    "hard_class_gate": bool(args.moe_hard_class_gate),
                    "num_expert": int(moe_num_expert_runtime),
                    "top_k": int(moe_top_k_runtime),
                    "gate_aux_weight": float(args.gate_aux_weight),
                }
                if args.use_moe_discriminator
                else None
            ),
        }

    def save_inference_checkpoint(path):
        payload = {
            "net": localnet.state_dict(),
            "reference_indices_by_class": reference_indices_by_class,
        }
        payload.update(checkpoint_extras())
        torch.save(payload, path)

    for epoch in range(start_epoch, args.epoch):
        epoch_start = time.perf_counter()
        (
            local_loss_value,
            pos_loss_value,
            neg_loss_value,
            gate_aux_loss_value,
            iteration,
            synth_summary,
        ) = train_one_epoch(
            args=args,
            epoch=epoch,
            localnet=localnet,
            feature_extractor=feature_extractor,
            optimizer=localnet_optimizer,
            criterion=localnet_criterion,
            loader=local_loader,
            iteration=iteration,
            runtime=fusion_runtime,
            synthesizer=synthesizer,
        )

        train_seconds = time.perf_counter() - epoch_start
        fusion_runtime.log_epoch(epoch, class_names, save_dir=saved_dir)
        print(
            f"epoch {epoch + 1} | total loss: {local_loss_value:.6f} | "
            f"normal BCE: {neg_loss_value:.6f} | synthetic BCE: "
            f"{pos_loss_value:.6f} | train_time: {train_seconds:.2f}s"
            + (
                f" | gate aux: {gate_aux_loss_value:.6f}"
                if args.use_moe_discriminator
                else ""
            )
        )
        if class_sigmas:
            print(
                "[Synthesis sigma] epoch "
                f"{epoch + 1} | "
                + " ".join(
                    f"{class_names[cls]}={class_sigmas[cls]:.6f}"
                    for cls in range(num_classes)
                    if cls in class_sigmas
                )
            )
        if synth_summary:
            type_counts = {
                key.replace("mask_type_", "").replace("_count", ""): value
                for key, value in synth_summary.items()
                if key.startswith("mask_type_")
            }
            print(
                f"epoch {epoch + 1} | synthesis | mask={args.synthesis_mask_type} "
                f"types={type_counts} | area={synth_summary['area_ratio_mean']:.4f} "
                f"| gamma={synth_summary['gamma_mean']:.3f}"
                f"±{synth_summary['gamma_std']:.3f} "
                f"[{synth_summary['gamma_min']:.3f},"
                f"{synth_summary['gamma_max']:.3f}] "
                f"| sigma={synth_summary['sigma_mean']:.4f}"
            )

        last_epoch = epoch == args.epoch - 1
        evaluate_now = ((epoch + 1) % args.eval_interval == 0) or last_epoch
        if evaluate_now:
            eval_start = time.perf_counter()
            fusion_runtime.before_eval(iteration, epoch)
            per_class = {}
            for class_idx_eval, class_name in enumerate(class_names):
                test_loader = build_test_loader(args, class_name)
                metrics = evaluate_epoch(
                    localnet,
                    feature_extractor,
                    test_loader,
                    args,
                    class_idx_eval=class_idx_eval,
                    fusion_runtime=fusion_runtime,
                )
                per_class[class_name] = metrics
                print(
                    f"epoch {epoch + 1} | {class_name} | "
                    f"I-AUROC: {metrics['image_auroc']:.5f}, "
                    f"P-AUROC: {metrics['pixel_auroc']:.5f}, "
                    f"I-AP: {metrics['image_ap']:.5f}, "
                    f"P-AP: {metrics['pixel_ap']:.5f}, "
                    f"P-F1max: {metrics['pixel_f1_max']:.5f}"
                    + (
                        f", AUPRO: {metrics['pixel_aupro']:.5f}"
                        if metrics.get("pixel_aupro") is not None
                        else ""
                    )
                )
                del test_loader

            last_result_by_class = per_class
            mean_img = _mean_metrics(per_class, "image_auroc")
            mean_px = _mean_metrics(per_class, "pixel_auroc")
            mean_p_ap = _mean_metrics(per_class, "pixel_ap")
            eval_seconds = time.perf_counter() - eval_start
            print(
                f"epoch {epoch + 1} | multiclass means | "
                f"I-AUROC: {mean_img:.5f} | P-AUROC: {mean_px:.5f} | "
                f"P-AP: {mean_p_ap:.5f} | eval_time: {eval_seconds:.2f}s"
            )

            epoch_mean = (mean_img + mean_px) / 2.0
            if epoch_mean > best_mean:
                best_mean = epoch_mean
                best_result_by_class = dict(per_class)
                if args.model_selection == "best":
                    save_inference_checkpoint(
                        os.path.join(saved_dir, run_name + "_localnet.pt")
                    )

            if args.save_log:
                with open(os.path.join(saved_dir, "log.txt"), "a") as file:
                    file.write(
                        f"epoch {epoch + 1} | total loss: {local_loss_value:.6f} "
                        f"| normal BCE: {neg_loss_value:.6f} "
                        f"| synthetic BCE: {pos_loss_value:.6f} "
                        f"| gate aux: {gate_aux_loss_value:.6f} "
                        f"| I-AUROC mean: {mean_img:.5f} "
                        f"| P-AUROC mean: {mean_px:.5f} "
                        f"| P-AP mean: {mean_p_ap:.5f}\n"
                    )

        ckpt_path = common_utils.default_train_checkpoint_path(
            saved_dir, run_name
        )
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
            extra=checkpoint_extras(),
        )
        print(f"[checkpoint] saved training state -> {ckpt_path}")

    if len(last_result_by_class) == 0:
        raise RuntimeError("训练未产生可用评估结果，请检查数据与参数。")

    if args.model_selection == "best":
        selected = best_result_by_class
        selection_note = "best monitor epoch (legacy behaviour)"
        # ``_localnet.pt`` was written when the best monitor epoch improved;
        # do not overwrite it with the last-epoch weights.
    else:
        selected = last_result_by_class
        selection_note = "last epoch (chapter-1 anti-leak default)"
        save_inference_checkpoint(
            os.path.join(saved_dir, run_name + "_localnet.pt")
        )
    print(f"[result] reporting {selection_note}")

    metric_keys = [
        "image_auroc",
        "pixel_auroc",
        "image_ap",
        "pixel_ap",
        "pixel_f1_max",
        "pixel_aupro",
    ]
    rows = []
    for class_name in class_names:
        metrics = selected[class_name]
        row = {"class": class_name}
        for key in metric_keys:
            row[key] = metrics.get(key)
        rows.append(row)
        print(
            f"selected | {class_name} | "
            f"I-AUROC: {metrics['image_auroc']:.5f} | "
            f"P-AUROC: {metrics['pixel_auroc']:.5f} | "
            f"I-AP: {metrics['image_ap']:.5f} | "
            f"P-AP: {metrics['pixel_ap']:.5f} | "
            f"P-F1max: {metrics['pixel_f1_max']:.5f}"
            + (
                f" | AUPRO: {metrics['pixel_aupro']:.5f}"
                if metrics.get("pixel_aupro") is not None
                else ""
            )
        )

    means = {key: _mean_metrics(selected, key) for key in metric_keys}
    print(
        "[selected means] "
        + " | ".join(
            f"{key}: {value:.5f}" if value is not None else f"{key}: n/a"
            for key, value in means.items()
        )
    )

    df = pd.DataFrame(rows)
    result_path = os.path.join(
        saved_dir,
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    os.makedirs(result_path, exist_ok=True)
    df.to_excel(os.path.join(result_path, run_name + "_result.xlsx"), index=False)


if __name__ == "__main__":
    main()
