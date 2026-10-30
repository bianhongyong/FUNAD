import argparse
import datetime
import os
import random
import shutil
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
    extract_dinov3_multilayer_feature_batch,
    infer_dinov3_feature_dim,
    infer_dinov3_multilayer_num_patches,
    infer_dinov3_patch_mask_size,
)
from dataset.multiclass_feature_dataset import (
    MultiClassFeatureDataset,
    get_all_class_names,
    get_class_layer_indices,
    get_fusion_candidate_layers,
    normalize_candidate_layers,
)
from src.model import model
from src.model.class_patch_fusion_adapter import ClassPatchFusionAdapter
from src.model.layer_fusion import ClassConditionedLayerFusion
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

# Per-class DINOv3 layer indices, shared via multiclass_feature_dataset.
# Per-backbone tables live in dataset/multiclass_feature_dataset.py;
# look them up with get_class_layer_indices(feature_model).


def _adapt_extract_fn(images, feature_extractor, args, return_cls_token=False, class_indices=None):
    """Adapter: epoch_precompute passes args (Namespace), extract_dinov3_feature_batch needs class_layer_indices."""
    return extract_dinov3_feature_batch(
        images, feature_extractor, args.class_layer_indices,
        class_indices=class_indices,
        use_cls_token=args.use_cls_token, return_cls_token=return_cls_token,
    )


def _adapt_extract_fn_multilayer(images, feature_extractor, args, return_cls_token=False, class_indices=None):
    """Extraction hook for learned fusion: raw [B,P,L,C], no layer mean."""
    return extract_dinov3_multilayer_feature_batch(
        images,
        feature_extractor,
        args.fusion_layers,
        use_cls_token=args.use_cls_token,
        return_cls_token=return_cls_token,
    )


def _fusion_preprocess_metadata(args):
    return {
        "resize": [int(args.image_size), int(args.image_size)],
        "center_crop": [int(args.crop_size), int(args.crop_size)],
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }


def _parse_no_layer_fusion_classes(value, class_names):
    """Parse ``screw,foo`` into a class-ordered list of valid class names."""
    raw = str(value or "").strip()
    if not raw:
        return []
    requested = {item.strip() for item in raw.split(",") if item.strip()}
    unknown = sorted(requested - set(class_names))
    if unknown:
        # The CLI default is "screw" (MVTec-specific).  Do not fail a VisA run
        # just because that default class does not exist; an explicit typo for
        # any other name still raises.
        if raw == "screw":
            print(
                "[Fusion] --fusion_no_layer_fusion_classes='screw' ignored: "
                f"this dataset has no 'screw' class ({list(class_names)})."
            )
            return []
        raise ValueError(
            "--fusion_no_layer_fusion_classes has unknown class name(s): "
            f"{unknown}; known classes are {list(class_names)}."
        )
    # Keep the dataset class order so indices and logging are deterministic.
    return [name for name in class_names if name in requested]


def _build_fusion_config(args, class_names, candidate_layers, feature_dim):
    config = {
        "schema_version": 1,
        "fusion_mode": str(args.fusion_mode),
        "num_classes": int(len(class_names)),
        "num_layers": int(len(candidate_layers)),
        "feature_dim": int(feature_dim),
        "temperature": float(args.fusion_temperature),
        "norm_type": str(args.fusion_norm),
        "norm_affine": bool(args.fusion_norm_affine),
        "shared_weights": bool(args.fusion_mode == "learned_shared"),
        "candidate_layers": [int(v) for v in candidate_layers],
        "feature_model": str(args.feature_model),
        "class_names": list(class_names),
        "use_cls_token": bool(args.use_cls_token),
        "dataset": str(args.dataset),
        "image_size": int(args.image_size),
        "crop_size": int(args.crop_size),
        "preprocess": _fusion_preprocess_metadata(args),
        "reference_refresh_unit": str(args.reference_refresh_unit),
        "reference_refresh_interval": int(args.reference_refresh_interval),
        "reference_storage_dtype": str(args.reference_storage_dtype),
        "num_reference_images_per_class": int(args.num_reference_images_per_class),
        "strict_clean_reference": bool(args.strict_clean_reference),
        "reference_seed": int(args.seed),
    }
    if args.fusion_mode == "class_patch":
        config.update(
            {
                # schema 3: explicit reference-match mode.  schema 2
                # checkpoints predate ``match_mode`` and mean dynamic_topk.
                "schema_version": 3,
                "match_mode": str(args.fusion_match_mode),
                "feature_adapter_mode": str(args.feature_adapter_mode),
                "fusion_query_chunk": int(args.fusion_query_chunk),
                "gate_embedding_dim": int(args.gate_embedding_dim),
                "gate_hidden_dim": int(args.gate_hidden_dim),
                "gate_norm": str(args.gate_norm),
                "gate_lambda_init": float(args.gate_lambda_init),
                "gate_use_context": bool(args.gate_use_context),
                "gate_demean_delta": bool(args.gate_demean_delta),
                "adapter_hidden_dim": int(args.adapter_hidden_dim),
                "adapter_beta_init": float(args.adapter_beta_init),
                # Per-class pinned last-layer prior (e.g. screw): no learned
                # inter-layer weights, patch gate may still deviate.
                "no_layer_fusion_classes": list(
                    getattr(args, "fusion_no_layer_fusion_class_names", []) or []
                ),
                "last_layer_prior_logit": float(
                    getattr(args, "fusion_last_layer_prior_logit", 8.0)
                ),
            }
        )
        if args.fusion_match_mode == "dynamic_topk":
            config["fusion_topk"] = int(args.fusion_topk)
            config["fusion_candidate_chunk"] = int(args.fusion_candidate_chunk)
    return config


def _validate_resume_fusion_config(
    ckpt_cfg, args, class_names, candidate_layers, ignore_match_mode=False
):
    """Refuse silent train/resume metadata mismatches.

    ``ignore_match_mode`` is only for explicit weights-only initialization:
    the network/fusion parameters are reusable across prior_top1 and
    dynamic_topk, but every architecture/preprocessing field must still match.
    """
    if not ckpt_cfg:
        raise RuntimeError(
            "resume checkpoint has no fusion_config; cannot resume a learned-fusion run."
        )
    errors = []
    checks = [
        ("fusion_mode", str(ckpt_cfg.get("fusion_mode"))),
        ("feature_model", str(ckpt_cfg.get("feature_model"))),
        ("norm_type", str(ckpt_cfg.get("norm_type"))),
        ("norm_affine", bool(ckpt_cfg.get("norm_affine"))),
        ("shared_weights", bool(ckpt_cfg.get("shared_weights"))),
        ("use_cls_token", bool(ckpt_cfg.get("use_cls_token"))),
        ("dataset", str(ckpt_cfg.get("dataset"))),
        ("image_size", int(ckpt_cfg.get("image_size", -1))),
        ("crop_size", int(ckpt_cfg.get("crop_size", -1))),
        ("num_classes", int(ckpt_cfg.get("num_classes", -1))),
        ("num_reference_images_per_class",
         int(ckpt_cfg.get("num_reference_images_per_class", -1))),
        ("strict_clean_reference", bool(ckpt_cfg.get("strict_clean_reference"))),
        ("reference_seed", int(ckpt_cfg.get("reference_seed", -1))),
    ]
    current = {
        "fusion_mode": str(args.fusion_mode),
        "feature_model": str(args.feature_model),
        "norm_type": str(args.fusion_norm),
        "norm_affine": bool(args.fusion_norm_affine),
        "shared_weights": bool(args.fusion_mode == "learned_shared"),
        "use_cls_token": bool(args.use_cls_token),
        "dataset": str(args.dataset),
        "image_size": int(args.image_size),
        "crop_size": int(args.crop_size),
        "num_classes": int(len(class_names)),
        "num_reference_images_per_class": int(args.num_reference_images_per_class),
        "strict_clean_reference": bool(args.strict_clean_reference),
        "reference_seed": int(args.seed),
    }
    if str(args.fusion_mode) == "class_patch":
        schema_version = int(ckpt_cfg.get("schema_version", 0))
        ckpt_match_mode = str(
            ckpt_cfg.get("match_mode")
            or ("prior_top1" if schema_version >= 3 else "dynamic_topk")
        )
        v2_checks = [
            ("feature_adapter_mode", str(ckpt_cfg.get("feature_adapter_mode"))),
            ("gate_embedding_dim", int(ckpt_cfg.get("gate_embedding_dim", -1))),
            ("gate_hidden_dim", int(ckpt_cfg.get("gate_hidden_dim", -1))),
            ("gate_norm", str(ckpt_cfg.get("gate_norm"))),
            ("gate_use_context", bool(ckpt_cfg.get("gate_use_context"))),
            ("gate_demean_delta", bool(ckpt_cfg.get("gate_demean_delta"))),
            ("adapter_hidden_dim", int(ckpt_cfg.get("adapter_hidden_dim", -1))),
        ]
        current_v2 = {
            "match_mode": str(args.fusion_match_mode),
            "feature_adapter_mode": str(args.feature_adapter_mode),
            "gate_embedding_dim": int(args.gate_embedding_dim),
            "gate_hidden_dim": int(args.gate_hidden_dim),
            "gate_norm": str(args.gate_norm),
            "gate_use_context": bool(args.gate_use_context),
            "gate_demean_delta": bool(args.gate_demean_delta),
            "adapter_hidden_dim": int(args.adapter_hidden_dim),
        }
        # Missing in pre-existing schema-3 checkpoints == no pinned classes.
        ckpt_no_layer = sorted(
            str(name) for name in (ckpt_cfg.get("no_layer_fusion_classes") or [])
        )
        current_no_layer = sorted(
            getattr(args, "fusion_no_layer_fusion_class_names", []) or []
        )
        v2_checks.append(("no_layer_fusion_classes", ckpt_no_layer))
        current_v2["no_layer_fusion_classes"] = current_no_layer
        if ckpt_no_layer or current_no_layer:
            v2_checks.append(
                (
                    "last_layer_prior_logit",
                    float(ckpt_cfg.get("last_layer_prior_logit", -1.0)),
                )
            )
            current_v2["last_layer_prior_logit"] = float(
                getattr(args, "fusion_last_layer_prior_logit", 8.0)
            )
        if ignore_match_mode:
            # Weights-only init may switch match modes; reference search is
            # rebuilt from the current prior and holds no learned parameters.
            pass
        else:
            v2_checks.append(("match_mode", ckpt_match_mode))
            if args.fusion_match_mode == "dynamic_topk":
                v2_checks.extend(
                    [
                        ("fusion_topk", int(ckpt_cfg.get("fusion_topk", -1))),
                        (
                            "fusion_candidate_chunk",
                            int(ckpt_cfg.get("fusion_candidate_chunk", -1)),
                        ),
                    ]
                )
                current_v2["fusion_topk"] = int(args.fusion_topk)
                current_v2["fusion_candidate_chunk"] = int(
                    args.fusion_candidate_chunk
                )
            if (
                ckpt_match_mode != str(args.fusion_match_mode)
                and schema_version < 3
            ):
                errors.append(
                    "checkpoint schema_version=2 predates match_mode and is "
                    "interpreted as dynamic_topk; pass "
                    "--fusion_match_mode dynamic_topk to resume it."
                )
        for key, expected in v2_checks:
            if current_v2[key] != expected:
                errors.append(
                    f"{key}: checkpoint={expected!r}, current={current_v2[key]!r}"
                )
    for key, expected in checks:
        if current[key] != expected:
            errors.append(f"{key}: checkpoint={expected!r}, current={current[key]!r}")
    if list(ckpt_cfg.get("class_names") or []) != list(class_names):
        errors.append("class_names/class order differs from checkpoint")
    if [int(v) for v in ckpt_cfg.get("candidate_layers") or []] != [
        int(v) for v in candidate_layers
    ]:
        errors.append(
            f"candidate_layers: checkpoint={ckpt_cfg.get('candidate_layers')}, "
            f"current={list(candidate_layers)}"
        )
    if errors:
        raise RuntimeError(
            "Resume checkpoint fusion_config mismatch:\n  - " + "\n  - ".join(errors)
        )
    if float(ckpt_cfg.get("temperature", 1.0)) != float(args.fusion_temperature):
        print(
            "[resume] fusion temperature changes from "
            f"{ckpt_cfg.get('temperature')} to {args.fusion_temperature}"
        )


class LayerFusionRuntime:
    """Owns fusion state, derived cache refresh scheduling and epoch logging."""

    def __init__(self, fusion, cache, candidate_layers, args):
        self.fusion = fusion
        self.cache = cache
        self.candidate_layers = list(candidate_layers)
        self.args = args
        self._last_neighbor_ids = None
        self.begin_epoch_stats()

    # -- hooks used by train_one_epoch / epoch_precompute / evaluate --------
    def extract(self, images, feature_extractor, args, return_cls_token=False, class_indices=None):
        return _adapt_extract_fn_multilayer(
            images, feature_extractor, args,
            return_cls_token=return_cls_token, class_indices=class_indices,
        )

    def residual(self, raw_features, class_idx, reference_memory_by_class=None,
                 reference_index_by_class=None):
        residual, neighbor_ids = common_utils.compute_fused_residual_batch(
            raw_features,
            class_idx,
            self.cache,
            self.fusion,
            fuse_chunk_size=int(self.args.fusion_ref_chunk),
            return_neighbor_ids=True,
        )
        self._last_neighbor_ids = neighbor_ids
        return residual

    # -- lifecycle ---------------------------------------------------------
    def begin_epoch_stats(self):
        self.epoch_theta_grad_norm = 0.0
        self.epoch_theta_grad_steps = 0
        self.epoch_bce_theta_grad_norm = 0.0
        self.epoch_bce_theta_grad_steps = 0
        self.epoch_oto_theta_grad_norm = 0.0
        self.epoch_oto_theta_grad_steps = 0
        self.epoch_bce_oto_cos = []
        self.epoch_refresh_time = 0.0
        self.epoch_refresh_count = 0
        self.epoch_stale_rates = []

    def begin_epoch(self, step, epoch):
        self.fusion.train()
        self.begin_epoch_stats()
        if common_utils.should_refresh_reference_cache(
            self.args.reference_refresh_unit,
            self.args.reference_refresh_interval,
            step=step,
            epoch=epoch,
        ):
            self._refresh(step, epoch)

    def after_step(self, step, epoch):
        if common_utils.should_refresh_reference_cache(
            self.args.reference_refresh_unit,
            self.args.reference_refresh_interval,
            step=step,
            epoch=None,
        ):
            self._refresh(step, epoch)

    def before_eval(self, step, epoch):
        # Final/eval numbers must use the current checkpoint's theta exactly.
        # Fusion is frozen (eval + no_grad) during evaluation.
        self.fusion.eval()
        self._refresh(step, epoch)

    def _refresh(self, step, epoch):
        info = self.cache.refresh(step=step, epoch=epoch)
        self.epoch_refresh_time += float(info["elapsed"])
        self.epoch_refresh_count += 1
        print(
            f"[Fusion] cache refreshed v{info['version']} "
            f"(step={info['refresh_step']}, epoch={info['refresh_epoch']}, "
            f"{info['elapsed']:.2f}s, mean drift={info['mean_drift']:.4f})"
        )
        return info

    def snapshot_theta_grad(self):
        theta = getattr(self.fusion, "theta", None)
        if theta is None or theta.grad is None:
            return None
        return theta.grad.detach().clone()

    def record_theta_grad(self, before=None, after_bce=None):
        """Record total / BCE / OTO theta-gradient norms and their cosine.

        ``before`` is the (normally absent) theta grad at step start;
        ``after_bce`` is the theta grad captured after the BCE backward and
        before the OTO backward.  Without ``after_bce`` (single fused
        backward) only the total is available.
        """
        theta = getattr(self.fusion, "theta", None)
        if theta is None or theta.grad is None:
            return
        total = theta.grad.detach()
        total_norm = float(total.norm().item())
        self.epoch_theta_grad_norm += total_norm
        if total_norm > 0:
            self.epoch_theta_grad_steps += 1

        zero = torch.zeros_like(total)
        base = zero if before is None else before.to(total.device, total.dtype)
        if after_bce is not None:
            after_bce = after_bce.to(total.device, total.dtype)
            bce_delta = after_bce - base
            oto_delta = total - after_bce
        else:
            bce_delta = total - base
            oto_delta = None

        bce_norm = float(bce_delta.norm().item())
        self.epoch_bce_theta_grad_norm += bce_norm
        if bce_norm > 0:
            self.epoch_bce_theta_grad_steps += 1
        if oto_delta is not None:
            oto_norm = float(oto_delta.norm().item())
            self.epoch_oto_theta_grad_norm += oto_norm
            if oto_norm > 0:
                self.epoch_oto_theta_grad_steps += 1
            if bce_norm > 0 and oto_norm > 0:
                cos = float(
                    torch.dot(bce_delta.flatten(), oto_delta.flatten()).item()
                    / (bce_norm * oto_norm)
                )
                self.epoch_bce_oto_cos.append(cos)

    def maybe_probe_stale(self, raw_features, class_idx, step):
        interval = int(getattr(self.args, "fusion_stale_probe_interval", 0))
        if interval <= 0 or step % interval != 0:
            return
        with torch.no_grad():
            fused_query = self.fusion(raw_features, class_idx)
            rate = self.cache.nn_mismatch_rate(fused_query, class_idx)
        self.epoch_stale_rates.append(rate)
        print(
            f"[Fusion] stale-NN probe @step {step}: mismatch={rate:.4f} "
            f"(cache v{self.cache.version}, theta lag={step - self.cache.refresh_step} steps)"
        )

    def log_epoch(self, epoch, class_names, save_dir=None):
        alpha = self.fusion.get_layer_weights().detach().cpu().numpy()
        entropy = self.fusion.weight_entropy().detach().cpu().numpy()
        shared = bool(getattr(self.fusion, "shared_weights", False))
        print(
            f"[Fusion] epoch {epoch + 1} | cache v{self.cache.version} | "
            f"refresh_count={self.epoch_refresh_count} "
            f"({self.epoch_refresh_time:.2f}s) | "
            f"theta_grad_mean={self.epoch_theta_grad_norm / max(1, self.epoch_theta_grad_steps):.3e} | "
            f"bce_grad_mean={self.epoch_bce_theta_grad_norm / max(1, self.epoch_bce_theta_grad_steps):.3e} | "
            f"oto_grad_mean={self.epoch_oto_theta_grad_norm / max(1, self.epoch_oto_theta_grad_steps):.3e} | "
            f"bce_oto_cos={float(np.mean(self.epoch_bce_oto_cos)) if self.epoch_bce_oto_cos else float('nan'):.4f}"
        )
        for row in range(alpha.shape[0]):
            name = "shared" if shared else class_names[row]
            weights = " ".join(f"{v:.4f}" for v in alpha[row])
            print(
                f"[Fusion alpha] {name}: [{weights}] | entropy={float(entropy[row]):.4f}"
            )
        if self.epoch_stale_rates:
            print(
                "[Fusion] stale-NN mismatch mean: "
                f"{float(np.mean(self.epoch_stale_rates)):.4f}"
            )

        if save_dir and not shared:
            os.makedirs(save_dir, exist_ok=True)
            csv_path = os.path.join(save_dir, f"fusion_alpha_epoch_{epoch + 1:04d}.csv")
            header = "class," + ",".join(f"layer_{l}" for l in self.candidate_layers) + ",entropy\n"
            with open(csv_path, "w") as handle:
                handle.write(header)
                for row, name in enumerate(class_names):
                    values = ",".join(f"{v:.8f}" for v in alpha[row])
                    handle.write(f"{name},{values},{float(entropy[row]):.8f}\n")
        return {
            "alpha": alpha,
            "entropy": entropy,
            "cache_version": self.cache.version,
            "refresh_count": self.epoch_refresh_count,
            "refresh_time": self.epoch_refresh_time,
            "theta_grad_mean": self.epoch_theta_grad_norm / max(1, self.epoch_theta_grad_steps),
            "bce_theta_grad_mean": (
                self.epoch_bce_theta_grad_norm / max(1, self.epoch_bce_theta_grad_steps)
            ),
            "oto_theta_grad_mean": (
                self.epoch_oto_theta_grad_norm / max(1, self.epoch_oto_theta_grad_steps)
            ),
            "bce_oto_grad_cos": (
                float(np.mean(self.epoch_bce_oto_cos))
                if self.epoch_bce_oto_cos else None
            ),
            "stale_nn_rate": float(np.mean(self.epoch_stale_rates)) if self.epoch_stale_rates else None,
        }


class ClassPatchFusionRuntime(LayerFusionRuntime):
    """V2 runtime: class-prior matching + patch-alpha diagnostics.

    ``fusion_match_mode`` selects how the single reference ``j*`` is chosen:

    * ``prior_top1`` (default): class prior only, no patch gate in matching;
      the patch alpha is used only for the final query/reference fusion.
    * ``dynamic_topk``: historical V2 coarse Top-K + query-conditioned rerank.
    """

    def __init__(self, fusion, cache, candidate_layers, args):
        super().__init__(fusion, cache, candidate_layers, args)
        self.match_mode = str(
            getattr(args, "fusion_match_mode", "prior_top1")
        )
        if self.match_mode not in ("prior_top1", "dynamic_topk"):
            raise ValueError(
                f"Unknown --fusion_match_mode {self.match_mode!r}."
            )
        if self.match_mode == "dynamic_topk":
            self.topk = int(args.fusion_topk)
            if self.topk < 1:
                raise ValueError("--fusion_topk must be >= 1.")
        else:
            self.topk = None
        self._last_match_meta = None
        self._last_op_timing = None

    def residual(self, raw_features, class_idx, reference_memory_by_class=None,
                 reference_index_by_class=None):
        if self.match_mode == "prior_top1":
            timing = (
                {}
                if bool(getattr(self.args, "fusion_profile", False))
                else None
            )
            (
                residual,
                neighbor_ids,
                diagnostics,
            ) = common_utils.compute_class_patch_prior_top1_residual_batch(
                raw_features,
                class_idx,
                self.cache,
                self.fusion,
                query_chunk_size=int(self.args.fusion_query_chunk),
                return_neighbor_ids=True,
                return_diagnostics=True,
                timing=timing,
            )
            self._last_op_timing = timing
        else:
            (
                residual,
                neighbor_ids,
                diagnostics,
            ) = common_utils.compute_class_patch_fused_residual_batch(
                raw_features,
                class_idx,
                self.cache,
                self.fusion,
                topk=self.topk,
                query_chunk_size=int(self.args.fusion_query_chunk),
                candidate_chunk_size=int(self.args.fusion_candidate_chunk),
                return_neighbor_ids=True,
                return_diagnostics=True,
            )
            self._last_op_timing = None
        self._last_neighbor_ids = neighbor_ids
        self._last_match_meta = diagnostics
        self._accumulate_patch_stats(diagnostics)
        self._accumulate_op_timing(self._last_op_timing)
        return residual

    # -- diagnostics -------------------------------------------------------
    def begin_epoch(self, step, epoch):
        if getattr(self.fusion, "enable_adapter", False):
            self.fusion.reset_adapter_stats()
        super().begin_epoch(step, epoch)

    def begin_epoch_stats(self):
        super().begin_epoch_stats()
        num_classes = int(self.fusion.num_classes)
        num_layers = int(self.fusion.num_layers)
        self._patch_alpha_sum = np.zeros((num_classes, num_layers), dtype=np.float64)
        self._patch_alpha_sq_sum = np.zeros((num_classes, num_layers), dtype=np.float64)
        self._patch_alpha_count = np.zeros(num_classes, dtype=np.int64)
        self._patch_l1_dev_sum = np.zeros(num_classes, dtype=np.float64)
        self._patch_entropy_sum = np.zeros(num_classes, dtype=np.float64)
        self._patch_nn_switch_sum = 0.0
        self._patch_nn_switch_count = 0
        self.epoch_topk_recall = []
        self._op_timing_sum = {}
        self._op_timing_count = 0

    def _accumulate_op_timing(self, timing):
        if not timing:
            return
        for key, value in timing.items():
            self._op_timing_sum[key] = (
                self._op_timing_sum.get(key, 0.0) + float(value)
            )
        self._op_timing_count += 1

    def _accumulate_patch_stats(self, diagnostics):
        if diagnostics is None:
            return
        alpha = diagnostics["alpha"].detach().float().cpu().numpy()
        class_np = diagnostics["class_idx"].detach().cpu().numpy().astype(np.int64)
        # ``nn_switch_rate`` only exists for dynamic_topk (coarse Top-K vs
        # reranked j*).  prior_top1 matches with the class prior only, so the
        # switch rate is identically zero and is not a meaningful metric.
        best_ids = diagnostics.get("best_ids")
        coarse_ids = diagnostics.get("coarse_ids")
        if best_ids is not None and coarse_ids is not None:
            best_np = best_ids.detach().cpu().numpy()
            coarse_first = coarse_ids[..., 0].detach().cpu().numpy()
            self._patch_nn_switch_sum += float((best_np != coarse_first).mean())
            self._patch_nn_switch_count += 1

        num_layers = alpha.shape[-1]
        for cls in np.unique(class_np).tolist():
            cls = int(cls)
            mask = class_np == cls
            if not np.any(mask):
                continue
            patch_alpha = alpha[mask].reshape(-1, num_layers).astype(np.float64)
            prior = (
                self.fusion.get_layer_weights(cls)[0].detach().float().cpu().numpy()
            )
            self._patch_alpha_sum[cls] += patch_alpha.sum(axis=0)
            self._patch_alpha_sq_sum[cls] += (patch_alpha ** 2).sum(axis=0)
            self._patch_alpha_count[cls] += patch_alpha.shape[0]
            self._patch_l1_dev_sum[cls] += float(
                np.abs(patch_alpha - prior).sum(axis=-1).sum()
            )
            self._patch_entropy_sum[cls] += float(
                (-(patch_alpha * np.log(np.clip(patch_alpha, 1e-12, None)))
                 .sum(axis=-1)).sum()
            )

    def maybe_probe_topk_recall(self, raw_features, class_idx, step):
        if self.match_mode != "dynamic_topk":
            # Top-K recall is only defined for the rerank path; prior_top1
            # explicitly does not consult the patch gate when matching.
            return
        interval = int(getattr(self.args, "fusion_topk_recall_probe_interval", 0))
        if interval <= 0 or step % interval != 0:
            return
        with torch.no_grad():
            result = common_utils.topk_recall_vs_exact(
                raw_features,
                class_idx,
                self.cache,
                self.fusion,
                topk=self.topk,
                sample_size=int(getattr(self.args, "fusion_topk_recall_sample_size", 32)),
            )
        self.epoch_topk_recall.append(result)
        print(
            f"[Fusion topk] recall@{result['topk']} @step {step}: "
            f"{result['recall']:.4f} (n={result['num_samples']})"
        )

    def log_epoch(self, epoch, class_names, save_dir=None):
        base = super().log_epoch(epoch, class_names, save_dir=save_dir)
        counts = self._patch_alpha_count
        mean = np.zeros_like(self._patch_alpha_sum)
        var = np.zeros_like(self._patch_alpha_sum)
        valid = counts > 0
        mean[valid] = self._patch_alpha_sum[valid] / counts[valid, None]
        var[valid] = (
            self._patch_alpha_sq_sum[valid] / counts[valid, None] - mean[valid] ** 2
        )
        std = np.sqrt(np.clip(var, 0.0, None))
        l1_dev = np.zeros(self.fusion.num_classes, dtype=np.float64)
        entropy = np.zeros(self.fusion.num_classes, dtype=np.float64)
        l1_dev[valid] = self._patch_l1_dev_sum[valid] / counts[valid]
        entropy[valid] = self._patch_entropy_sum[valid] / counts[valid]
        lam = self.fusion.get_patch_lambda().detach().float().cpu().numpy()

        if self.match_mode == "dynamic_topk":
            switch_rate = (
                self._patch_nn_switch_sum / max(1, self._patch_nn_switch_count)
            )
            print(
                f"[Fusion patch] epoch {epoch + 1} | "
                f"match_mode={self.match_mode} | nn_switch_rate={switch_rate:.4f}"
            )
        else:
            # prior_top1 pins j* to the class-prior NN; the old switch rate is
            # structurally zero, so do not report it as an architectural signal.
            switch_rate = None
            print(
                f"[Fusion patch] epoch {epoch + 1} | "
                f"match_mode={self.match_mode} (reference fixed by class prior)"
            )

        op_timing = None
        if self._op_timing_count > 0:
            op_timing = {
                key: value / self._op_timing_count
                for key, value in self._op_timing_sum.items()
            }
            print(
                "[Fusion timing] mean per batch ms: "
                + " ".join(f"{k}={v:.2f}" for k, v in sorted(op_timing.items()))
            )
        for row in range(self.fusion.num_classes):
            name = class_names[row] if row < len(class_names) else f"cls_{row}"
            weights = " ".join(f"{v:.4f}" for v in mean[row])
            stds = " ".join(f"{v:.4f}" for v in std[row])
            print(
                f"[Fusion patch alpha] {name}: mean=[{weights}] "
                f"std=[{stds}] | l1_to_prior={l1_dev[row]:.4f} "
                f"entropy={entropy[row]:.4f} | lambda={lam[row]:.4f}"
            )

        beta = None
        rho = None
        rho_count = None
        if getattr(self.fusion, "enable_adapter", False):
            beta = self.fusion.get_beta().detach().float().cpu().numpy()
            rho, rho_count = self.fusion.adapter_rho()
            print(
                "[Fusion adapter] beta=["
                + " ".join(f"{v:.4f}" for v in beta)
                + "] rho=["
                + " ".join(f"{v:.4f}" for v in rho.numpy())
                + "]"
            )

        if save_dir and self._patch_alpha_count.sum() > 0:
            os.makedirs(save_dir, exist_ok=True)
            csv_path = os.path.join(
                save_dir, f"fusion_patch_alpha_epoch_{epoch + 1:04d}.csv"
            )
            header = (
                "class,"
                + ",".join(f"mean_layer_{l}" for l in self.candidate_layers)
                + ","
                + ",".join(f"std_layer_{l}" for l in self.candidate_layers)
                + ",l1_to_prior,entropy,lambda,beta,rho\n"
            )
            with open(csv_path, "w") as handle:
                handle.write(header)
                for row in range(self.fusion.num_classes):
                    name = class_names[row] if row < len(class_names) else f"cls_{row}"
                    values = ",".join(f"{v:.8f}" for v in mean[row])
                    stds = ",".join(f"{v:.8f}" for v in std[row])
                    beta_v = float(beta[row]) if beta is not None else float("nan")
                    rho_v = float(rho[row]) if rho is not None else float("nan")
                    handle.write(
                        f"{name},{values},{stds},{l1_dev[row]:.8f},"
                        f"{entropy[row]:.8f},{lam[row]:.8f},{beta_v:.8f},{rho_v:.8f}\n"
                    )

        base.update(
            {
                "patch_alpha_mean": mean,
                "patch_alpha_std": std,
                "patch_alpha_l1_to_prior": l1_dev,
                "patch_alpha_entropy": entropy,
                "patch_lambda": lam,
                "patch_beta": beta,
                "patch_rho": None if rho is None else rho.numpy(),
                "patch_rho_count": None if rho_count is None else rho_count.numpy(),
                "match_mode": self.match_mode,
                "nn_switch_rate": switch_rate,
                "op_timing_ms": op_timing,
                "topk_recall": list(self.epoch_topk_recall),
            }
        )
        return base


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
    parser.add_argument("--threshold_stage1", type=float, default=0.15, help="Pseudo label threshold for early stage (before threshold_epoch_split)")
    parser.add_argument("--threshold_stage2", type=float, default=0.1, help="Pseudo label threshold for late stage (from threshold_epoch_split onward)")
    parser.add_argument("--threshold_epoch_split", type=int, default=1, help="Epoch at which to switch from stage1 to stage2 threshold. 0 means always use stage1.")
    parser.add_argument("-n", "--noise_threshold", type=float, default=0.8)
    parser.add_argument("--use_mad_threshold", action="store_true",
                        help="Use MAD-based per-class dynamic threshold instead of a single global threshold.")
    parser.add_argument("--mad_k", type=float, default=4,
                        help="Default k multiplier for MAD threshold. "
                        "Larger k means stricter normal definition. "
                        "Can be overridden per-class via --mad_k_per_class.")
    parser.add_argument("--mad_k_per_class", type=str, default="screw:9.0",
                        help="Per-class k overrides, e.g. 'bottle:4.0,cable:6.0'. "
                        "Classes not listed fall back to --mad_k.")
    parser.add_argument(
        "--noise",
        type=str,
        default="10%",
        choices=["0%", "1%", "2%", "3%", "5%", "10%", "15%", "20%", "25%"],
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
        "--init_weights_from",
        type=str,
        default="",
        help="Explicit weights-only initialization: load net + fusion_state from a "
        "checkpoint but start a NEW run (fresh optimizer/RNG/epoch/pseudo labels). "
        "Match mode may differ from the checkpoint; architecture must match. "
        "Mutually exclusive with --resume.",
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
        "scorer for all patches. Pseudo-label pipeline (Phase 1/2 normalization, "
        "sample selection, distance normalization, MAD threshold) is also unified globally.",
    )
    parser.add_argument(
        "--normal_sample_selection",
        type=str,
        choices=["threshold", "quantile"],
        default="threshold",
        help="How to select 'normal' images in Phase 2 of pseudo-label pipeline. "
        "'threshold' (default): select images with norm_score < 0.5. "
        "'quantile': select images with norm_score below the top-N quantile (e.g. top 30%%).",
    )
    parser.add_argument(
        "--normal_sample_quantile",
        type=float,
        default=0.3,
        help="Quantile threshold for 'quantile' selection mode. "
        "E.g., 0.3 means top 30%% lowest-scoring images are considered normal candidates. "
        "Only used when --normal_sample_selection=quantile.",
    )
    parser.add_argument(
        "--memory_bank_freeze_start_epoch",
        type=int,
        default=10,
        help="0-based epoch index from which Phase 1/2 memory bank is frozen and only Phase 3 "
        "reruns each epoch. -1 disables. Snapshot is written after Phase 1/2 when "
        "epoch == freeze_start - 1. Must be >= 1 when enabled.",
    )

    # ------------------------------------------------------------------
    # Learned DINOv3 layer fusion (see docs/learnable_dinov3_layer_fusion)
    # ------------------------------------------------------------------
    parser.add_argument(
        "--fusion_mode",
        type=str,
        choices=["fixed_mean", "learned_class", "learned_shared", "class_patch"],
        default="fixed_mean",
        help="fixed_mean: legacy per-class layer table + mean (default, reproducible). "
        "learned_class: class-conditioned softmax layer weights. "
        "learned_shared: one softmax weight vector shared by all classes (E1 ablation). "
        "class_patch: V2 class prior + per-patch gate; reference matching is "
        "controlled by --fusion_match_mode (default prior_top1).",
    )
    parser.add_argument(
        "--fusion_layers",
        type=int,
        nargs="+",
        default=None,
        help="Shared 1-based candidate DINOv3 layers for learned fusion. "
        "Default depends on --feature_model (DINO_FUSION_CANDIDATE_LAYERS).",
    )
    parser.add_argument("--fusion_temperature", type=float, default=1.0,
                        help="Softmax temperature tau for layer weights.")
    parser.add_argument(
        "--fusion_norm",
        type=str,
        choices=["layer_norm", "none"],
        default="none",
        help="Per-layer normalization before weighting (default: none).",
    )
    parser.add_argument("--fusion_norm_affine", type=str2bool, default="False",
                        help="Use affine parameters in the per-layer LayerNorm.")
    parser.add_argument(
        "--reference_refresh_unit",
        type=str,
        choices=["epoch", "step"],
        default="epoch",
        help="Fused reference cache refresh unit. 'epoch' (default) or 'step' "
        "(step=1 is the exact-NN reference mode).",
    )
    parser.add_argument("--reference_refresh_interval", type=int, default=1,
                        help="Refresh every N epochs or N optimizer steps.")
    parser.add_argument(
        "--reference_storage_dtype",
        type=str,
        choices=["float16", "float32"],
        default="float16",
        help="CPU storage dtype for the raw multilayer reference bank.",
    )
    parser.add_argument("--fusion_ref_chunk", type=int, default=1024,
                        help="Rows per chunk when re-fusing selected raw references.")
    parser.add_argument("--fusion_nn_query_chunk", type=int, default=2048,
                        help="Query patch chunk size for cached NN search.")
    parser.add_argument("--fusion_nn_ref_chunk", type=int, default=4096,
                        help="Reference patch chunk size for cached NN search.")
    parser.add_argument(
        "--fusion_stale_probe_interval",
        type=int,
        default=0,
        help="Every N optimizer steps, compare cached NN ids with current-theta "
        "exact NN and log the mismatch rate. 0 disables (expensive).",
    )

    # ------------------------------------------------------------------
    # V2 class-patch adaptive fusion (/class_patch mode)
    # ------------------------------------------------------------------
    parser.add_argument(
        "--fusion_match_mode",
        type=str,
        choices=["prior_top1", "dynamic_topk"],
        default="prior_top1",
        help="How the single reference j* is selected. prior_top1 (default): "
        "class prior only; the patch gate shapes only the residual. "
        "dynamic_topk: historical coarse Top-K + query-conditioned rerank.",
    )
    parser.add_argument(
        "--fusion_topk",
        type=int,
        default=32,
        help="Coarse class-prior candidate count K (dynamic_topk only).",
    )
    parser.add_argument("--fusion_query_chunk", type=int, default=256,
                        help="Query patches per reference-gather/fusion chunk "
                        "(both match modes; memory bound).")
    parser.add_argument("--fusion_candidate_chunk", type=int, default=8,
                        help="Candidates per rerank chunk (memory bound).")
    parser.add_argument("--gate_embedding_dim", type=int, default=32,
                        help="Shared class embedding dimension e_c.")
    parser.add_argument("--gate_hidden_dim", type=int, default=64,
                        help="Patch-gate hidden/projection dimension d.")
    parser.add_argument("--gate_norm", type=str, choices=["layer_norm", "none"],
                        default="layer_norm",
                        help="Internal Gate input norm (independent of --fusion_norm).")
    parser.add_argument("--gate_lambda_init", type=float, default=0.1,
                        help="Initial softplus gate amplitude lambda_c (small positive).")
    parser.add_argument("--gate_use_context", type=str2bool, default="False",
                        help="Add a 3x3 depthwise spatial context branch to the gate.")
    parser.add_argument("--gate_demean_delta", type=str2bool, default="True",
                        help="Remove the per-patch mean of delta (softmax-invariant, "
                        "only improves log interpretability).")
    parser.add_argument(
        "--fusion_no_layer_fusion_classes",
        type=str,
        default="screw",
        help="Comma-separated class names whose class prior is pinned to the "
        "last candidate layer (weight 1, no learned inter-layer fusion). "
        "The patch gate may still deviate per patch. Empty string disables. "
        "class_patch only.",
    )
    parser.add_argument(
        "--fusion_last_layer_prior_logit",
        type=float,
        default=8.0,
        help="Fixed last-layer logit for --fusion_no_layer_fusion_classes. "
        "Initial patch weights are last-layer dominant; a larger value makes "
        "the patch gate work harder to deviate.",
    )
    parser.add_argument(
        "--feature_adapter_mode",
        type=str,
        choices=["legacy", "class_residual"],
        default="legacy",
        help="V2-A: legacy ClassCrossAttentionAdapter. "
        "V2-C: beta/FiLM class-strength residual adapter (class_patch only).",
    )
    parser.add_argument("--adapter_hidden_dim", type=int, default=128,
                        help="Hidden width of the V2-C residual MLP.")
    parser.add_argument("--adapter_beta_init", type=float, default=0.15,
                        help="Initial beta_c value for the V2-C adapter.")
    parser.add_argument("--fusion_lr", type=float, default=None,
                        help="Optional separate optimizer LR for all fusion params "
                        "(default: same as --lr).")
    parser.add_argument(
        "--fusion_topk_recall_probe_interval",
        type=int,
        default=0,
        help="Every N optimizer steps, estimate Top-K recall vs exact dynamic NN "
        "on a small sample. 0 disables (very expensive).",
    )
    parser.add_argument("--fusion_topk_recall_sample_size", type=int, default=32,
                        help="Queries per class sampled for the Top-K recall probe.")
    parser.add_argument(
        "--fusion_profile",
        type=str2bool,
        default="False",
        help="Prior-Top1 only: time coarse projection / NN / gate / ref gather / "
        "residual fusion with CUDA synchronizations and log per-epoch means. "
        "Off by default because synchronizing every batch costs throughput.",
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
    extract_feature_batch_fn=None,
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
        extract_feature_batch_fn=(
            _adapt_extract_fn
            if extract_feature_batch_fn is None
            else extract_feature_batch_fn
        ),
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
    memory_bank_snapshot,
    moe_expert_vis_ctx=None,
    compute_residual_fn=common_utils.compute_residual_feature_batch,
    extract_feature_batch_fn=None,
    fusion_runtime=None,
):
    if fusion_runtime is not None:
        # Refresh before pseudo-label precompute so Phase 1/2/3 and the batch
        # loop use the same current-theta derived cache.
        fusion_runtime.begin_epoch(iteration, epoch)

    total_batch = len(local_loader)
    threshold = args.threshold_stage1 if epoch < args.threshold_epoch_split else args.threshold_stage2
    use_pseudo_label = threshold <= 1 or args.use_mad_threshold

    local_loss = 0
    oto_loss = 0
    bce_loss = 0
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
    mad_threshold_map = None
    moe_vis_enabled = bool(
        moe_expert_vis_ctx is not None
        and moe_expert_vis_ctx.get("enabled", False)
        and args.use_moe_discriminator
    )
    class_expert_count = None
    if moe_vis_enabled:
        num_expert = int(moe_expert_vis_ctx["num_expert"])
        class_expert_count = np.zeros((num_classes, num_expert), dtype=np.float64)

    if use_pseudo_label:
        distance_map, confident_feature_bank, global_dim, mb_time, pl_time, mad_threshold_map = precompute_pseudo_labels_multiclass_residual(
            args=args,
            localnet=localnet,
            feature_extractor=feature_extractor,
            mini_loader=mini_loader,
            num_classes=num_classes,
            reference_memory_by_class=reference_memory_by_class,
            reference_index_by_class=reference_index_by_class,
            device=device,
            extract_feature_batch_fn=(
                _adapt_extract_fn
                if extract_feature_batch_fn is None
                else extract_feature_batch_fn
            ),
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
        if extract_feature_batch_fn is None:
            raw_features, cls_token = extract_dinov3_feature_batch(
                images, feature_extractor, args.class_layer_indices,
                class_indices=class_idx,
                use_cls_token=args.use_cls_token, return_cls_token=True,
            )
        else:
            raw_features, cls_token = extract_feature_batch_fn(
                images, feature_extractor, args,
                class_indices=class_idx, return_cls_token=True,
            )
        x = compute_residual_fn(
            raw_features,
            class_idx,
            reference_memory_by_class,
            reference_index_by_class,
        )
        if fusion_runtime is not None:
            fusion_runtime.maybe_probe_stale(raw_features, class_idx, iteration)
            maybe_topk_probe = getattr(fusion_runtime, "maybe_probe_topk_recall", None)
            if callable(maybe_topk_probe):
                maybe_topk_probe(raw_features, class_idx, iteration)
        # Largest per-step buffer (~B*P*L*C, e.g. ~400 MB for 16x784x8x1024).
        # It is no longer needed once the residual is built; release it before
        # the adapter/MoE forwards to lower peak memory.  When residual mode
        # is off ``x`` may alias this tensor, in which case the reference in
        # ``x`` keeps the storage alive.
        raw_features = None
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
        x_perturbed = None
        if use_pseudo_label and args.gaussian:
            # Keep x on the autograd graph: only the sampled noise is detached.
            # This is required so BCE can supervise fusion / residual (see
            # docs/.../requirements_and_design.md §5.1.1).
            x_perturbed = common_utils.apply_gaussian_feature_noise(
                x,
                distance=distance,
                threshold_map=threshold_map,
                noise_threshold=args.noise_threshold,
                std=args.std,
            )

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
                syn_anomaly = syn_anomaly.to(device, non_blocking=True)
                syn_class = syn_class.to(device, non_blocking=True)
                # The synthetic row is fed to localnet in both branches, so its
                # class index must be appended in both branches as well.
                class_idx_for_oto = torch.cat([class_idx_for_oto, syn_class], dim=0)
                # Both the clean and Gaussian-perturbed branches are fed to
                # localnet, so both must carry the synthetic row to stay
                # aligned with ``local_label`` / ``class_idx_for_oto``.
                x = torch.cat([x, syn_anomaly], dim=0)
                if use_pseudo_label and args.gaussian:
                    x_perturbed = torch.cat([x_perturbed, syn_anomaly], dim=0)
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
            class_idx=class_idx_for_oto,
        )
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
        if use_pseudo_label and args.gaussian:
            x_perturbed = x_perturbed.to(device, non_blocking=True)
            gaussian_cls_token = (
                cls_token if cls_token.shape[0] == x_perturbed.shape[0] else None
            )
            gaussian_patch_class_idx = None
            if args.use_moe_discriminator and args.moe_hard_class_gate:
                token_per_image = int(x_perturbed.shape[1]) if x_perturbed.dim() >= 2 else 1
                gaussian_patch_class_idx = class_idx_for_oto.to(
                    device=x_perturbed.device, dtype=torch.long
                ).reshape(-1, 1, 1).expand(
                    -1, token_per_image, 1
                )
                gaussian_patch_class_idx = gaussian_patch_class_idx.contiguous()
            gaussian_feature, gaussian_pred = localnet(
                x_perturbed,
                cls_token=gaussian_cls_token,
                patch_class_idx=gaussian_patch_class_idx,
                class_idx=class_idx_for_oto,
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
        if args.use_moe_discriminator:
            _local_loss = _local_loss + args.gate_aux_weight * _gate_aux_loss

        # BCE and OTO now share the residual graph (Gaussian BCE is no longer
        # detached).  Keep the two backward passes separated so the BCE and
        # OTO contributions to theta can be logged independently.
        theta_grad_before = (
            fusion_runtime.snapshot_theta_grad() if fusion_runtime is not None else None
        )
        theta_grad_after_bce = None
        if args.alternative:
            _local_loss.backward(retain_graph=bool(_l_loss.requires_grad))
            if fusion_runtime is not None:
                theta_grad_after_bce = fusion_runtime.snapshot_theta_grad()
            localnet_optimizer.step()

            if onetoone_optimizer is not None and _l_loss.requires_grad:
                _l_loss.backward()
                onetoone_optimizer.step()
            if fusion_runtime is not None:
                fusion_runtime.record_theta_grad(
                    before=theta_grad_before, after_bce=theta_grad_after_bce
                )
        else:
            bce_term = _loss
            if args.use_moe_discriminator:
                bce_term = bce_term + args.gate_aux_weight * _gate_aux_loss
            oto_term = args.weight * _l_loss
            bce_term.backward(retain_graph=bool(oto_term.requires_grad))
            if fusion_runtime is not None:
                theta_grad_after_bce = fusion_runtime.snapshot_theta_grad()
            if oto_term.requires_grad:
                oto_term.backward()
            localnet_optimizer.step()
            if fusion_runtime is not None:
                fusion_runtime.record_theta_grad(
                    before=theta_grad_before, after_bce=theta_grad_after_bce
                )

        # Accumulate detached scalars only.  Keeping live loss tensors here
        # would link each step's autograd graph into the next step's
        # accumulator; combined with the retain_graph split below that keeps
        # every step's saved activations alive and grows GPU memory linearly.
        local_loss += _local_loss.detach() / total_batch
        bce_loss += _loss.detach() / total_batch
        oto_loss += _l_loss.detach() / total_batch
        gate_aux_loss += _gate_aux_loss.detach() / total_batch
        iteration += 1

        # The split BCE/OTO backward retains the branch reachable only from
        # the BCE term (the Gaussian branch when --gaussian is set) because
        # the second backward from the OTO root does not traverse it.  Drop
        # every reference to this step's graph so those saved activations are
        # released now instead of accumulating across optimizer steps.
        bce_term = None
        oto_term = None
        pred_for_loss = None
        gaussian_pred = None
        gaussian_feature = None
        local_pred = None
        batch_feature = None
        x_perturbed = None
        _local_loss = None
        _loss = None
        _l_loss = None
        _gate_aux_loss = None

        if fusion_runtime is not None:
            fusion_runtime.after_step(iteration, epoch)

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
    if args.fusion_mode == "class_patch":
        args.fusion_no_layer_fusion_class_names = _parse_no_layer_fusion_classes(
            args.fusion_no_layer_fusion_classes, class_names
        )
        if float(args.fusion_last_layer_prior_logit) < 0.0:
            raise ValueError("--fusion_last_layer_prior_logit must be >= 0.")
    else:
        # The CLI default ("screw") targets class_patch runs only; do not
        # affect learned_class / fixed_mean commands that do not pass it.
        args.fusion_no_layer_fusion_class_names = []

    # Build per-class layer indices dict (class_idx -> list[int])
    # 层表按 backbone 分档，必须用 args.feature_model 去取对应尺寸的那张。
    class_layer_table = get_class_layer_indices(args.feature_model)
    args.class_layer_indices = {
        i: class_layer_table.get(name, [-1])
        for i, name in enumerate(class_names)
    }
    _all_layer_values = sorted(set(
        idx for lst in args.class_layer_indices.values() for idx in lst
    ))

    fusion_enabled = args.fusion_mode != "fixed_mean"
    if args.feature_adapter_mode == "class_residual" and args.fusion_mode != "class_patch":
        raise ValueError(
            "--feature_adapter_mode class_residual requires --fusion_mode class_patch "
            "(V2-C)."
        )
    if args.fusion_mode == "class_patch":
        if str(args.fusion_match_mode) not in ("prior_top1", "dynamic_topk"):
            raise ValueError(
                "--fusion_match_mode must be prior_top1 or dynamic_topk."
            )
        if int(args.fusion_query_chunk) < 1:
            raise ValueError("--fusion_query_chunk must be >= 1.")
        if args.fusion_match_mode == "dynamic_topk":
            if int(args.fusion_topk) < 1:
                raise ValueError("--fusion_topk must be >= 1 (dynamic_topk).")
            if int(args.fusion_candidate_chunk) < 1:
                raise ValueError(
                    "--fusion_candidate_chunk must be >= 1 (dynamic_topk)."
                )
        if int(args.gate_embedding_dim) < 1 or int(args.gate_hidden_dim) < 1:
            raise ValueError("gate embedding/hidden dims must be >= 1.")
        if float(args.gate_lambda_init) < 0:
            raise ValueError("--gate_lambda_init must be >= 0.")
        if int(args.adapter_hidden_dim) < 1:
            raise ValueError("--adapter_hidden_dim must be >= 1.")
        if not 0.0 < float(args.adapter_beta_init) < 1.0:
            raise ValueError("--adapter_beta_init must be in (0, 1).")
    resume_path = (args.resume or "").strip()
    init_path = (args.init_weights_from or "").strip()
    if resume_path and init_path:
        raise ValueError(
            "--resume and --init_weights_from are mutually exclusive: resume "
            "continues the same experiment, init_weights_from starts a new one."
        )
    resume_meta = None
    if resume_path:
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"--resume path not found: {resume_path}")
        resume_meta = common_utils.inspect_train_checkpoint(resume_path)
        if resume_meta.get("has_fusion_state") and not fusion_enabled:
            raise ValueError(
                "This checkpoint was trained with learned layer fusion but "
                "--fusion_mode is fixed_mean. Pass the matching --fusion_mode "
                "(learned_class/learned_shared)."
            )
        if fusion_enabled and not resume_meta.get("has_fusion_state"):
            raise RuntimeError(
                "Current run enables learned fusion but the checkpoint has no "
                "fusion_state. Start a fresh learned-fusion run instead."
            )
    init_meta = None
    if init_path:
        if not os.path.isfile(init_path):
            raise FileNotFoundError(
                f"--init_weights_from path not found: {init_path}"
            )
        init_meta = common_utils.inspect_train_checkpoint(init_path)
        if init_meta.get("has_fusion_state") and not fusion_enabled:
            raise ValueError(
                "init checkpoint contains fusion_state but the current run uses "
                "fixed_mean fusion."
            )
        if fusion_enabled and not init_meta.get("has_fusion_state"):
            raise RuntimeError(
                "Current run enables learned fusion but the init checkpoint has "
                "no fusion_state."
            )

    if fusion_enabled:
        if not args.residual:
            raise ValueError("learned fusion requires --residual True.")
        ckpt_cfg = (resume_meta or {}).get("fusion_config")
        init_cfg = (init_meta or {}).get("fusion_config")
        layer_cfg = ckpt_cfg or init_cfg
        if args.fusion_layers is None:
            if layer_cfg and layer_cfg.get("candidate_layers"):
                candidate_layers = layer_cfg["candidate_layers"]
                print(f"[Fusion] adopting candidate layers from checkpoint: {candidate_layers}")
            else:
                candidate_layers = get_fusion_candidate_layers(args.feature_model)
        else:
            candidate_layers = args.fusion_layers
        candidate_layers = normalize_candidate_layers(candidate_layers, args.feature_model)
        if ckpt_cfg:
            _validate_resume_fusion_config(ckpt_cfg, args, class_names, candidate_layers)
        if init_cfg:
            # Same architecture required, but the reference-match mode may
            # differ (no learned parameters are tied to it).
            _validate_resume_fusion_config(
                init_cfg,
                args,
                class_names,
                candidate_layers,
                ignore_match_mode=True,
            )
        if int(args.reference_refresh_interval) < 1:
            raise ValueError("--reference_refresh_interval must be >= 1.")
        # The extraction hook reads args.fusion_layers; keep it normalized and
        # identical to the candidate layer order used by theta.
        args.fusion_layers = list(candidate_layers)
        print(
            f"[Fusion] mode={args.fusion_mode} | layers={candidate_layers} | "
            f"tau={args.fusion_temperature} | norm={args.fusion_norm} "
            f"(affine={args.fusion_norm_affine}) | "
            f"refresh={args.reference_refresh_unit}/{args.reference_refresh_interval} | "
            f"raw storage={args.reference_storage_dtype}"
        )
        if args.fusion_mode == "class_patch":
            match_info = (
                f"match={args.fusion_match_mode} | "
                + (
                    f"topk={args.fusion_topk} | chunks={args.fusion_query_chunk}x"
                    f"{args.fusion_candidate_chunk} | "
                    if args.fusion_match_mode == "dynamic_topk"
                    else f"query_chunk={args.fusion_query_chunk} | "
                )
            )
            print(
                f"[Fusion V2] adapter={args.feature_adapter_mode} | "
                f"{match_info}"
                f"gate d={args.gate_hidden_dim} "
                f"e={args.gate_embedding_dim} lambda0={args.gate_lambda_init} "
                f"norm={args.gate_norm} context={args.gate_use_context}"
            )
            print(
                "[Fusion V2] last-layer-only class prior: "
                f"{args.fusion_no_layer_fusion_class_names or 'none'} "
                f"(prior_logit={args.fusion_last_layer_prior_logit}); "
                "patch gate may still deviate per patch."
            )
    else:
        if args.fusion_layers is not None:
            warnings.warn("--fusion_layers is ignored when --fusion_mode fixed_mean.")
        args.fusion_layers = None
        candidate_layers = None

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
    inferred_patch_mask_size = infer_dinov3_patch_mask_size(train_dataset, feature_extractor, _all_layer_values, args.use_cls_token, device)
    train_dataset.patch_mask_size = inferred_patch_mask_size
    print(f"inferred patch_mask_size from feature extractor: {inferred_patch_mask_size}")
    if fusion_enabled:
        _selected_blocks = resolve_dino_block_indices(feature_extractor, candidate_layers)
        _display = [b + 1 for b in _selected_blocks]
        print(
            f"[DINOv3] raw fusion candidate layers {_display} "
            f"(0-based blocks {_selected_blocks})"
        )
    else:
        _selected_blocks = resolve_dino_block_indices(feature_extractor, _all_layer_values)
        _display = [b + 1 for b in _selected_blocks]
        print(f"[DINOv3] aggregating layers {_display} (0-based blocks {_selected_blocks})")
        for cls_name, cls_layers in class_layer_table.items():
            if cls_name in class_names:
                _cls_blocks = resolve_dino_block_indices(feature_extractor, cls_layers)
                if set(_cls_blocks) != set(_selected_blocks):
                    print(f"  class {cls_name}: layers {cls_layers} -> blocks {_cls_blocks}")

    fusion = None
    fusion_cache = None
    fusion_runtime = None
    fusion_config = None
    raw_reference_manifest = None
    feature_dim = None

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

        if fusion_enabled:
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
                "num_reference_images_per_class": int(args.num_reference_images_per_class),
                "strict_clean_reference": bool(args.strict_clean_reference),
                "reference_seed": int(args.seed),
                "storage_dtype": str(args.reference_storage_dtype),
            }

            raw_reference_memory_by_class = None
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
                    print(
                        "[Fusion] loaded raw reference bank from checkpoint sidecar: "
                        f"{ckpt_manifest.get('path')} "
                        f"(sha256={str(ckpt_manifest.get('sha256'))[:16]}...)"
                    )

            if raw_reference_memory_by_class is None:
                raw_reference_memory_by_class, raw_info = (
                    common_utils.build_multilayer_reference_memory_bank(
                        train_dataset,
                        reference_indices_by_class,
                        extract_multilayer_fn=lambda images, class_indices=None: (
                            extract_dinov3_multilayer_feature_batch(
                                images, feature_extractor, candidate_layers,
                                use_cls_token=args.use_cls_token,
                            )
                        ),
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        device=device,
                        storage_dtype=args.reference_storage_dtype,
                    )
                )
                raw_metadata.update(raw_info)

            first_raw = next(iter(raw_reference_memory_by_class.values()))
            num_layers = int(first_raw.shape[1])
            feature_dim = int(first_raw.shape[2])
            raw_metadata["num_layers"] = num_layers
            raw_metadata["feature_dim"] = feature_dim

            if loaded_sidecar_path is None:
                raw_reference_manifest = common_utils.save_raw_reference_bank(
                    raw_reference_memory_by_class,
                    raw_metadata,
                    save_dir=saved_dir,
                    prefix=(
                        f"raw_reference_bank_{args.dataset}_{args.noise}_"
                        f"{args.feature_model}"
                    ),
                )
            elif os.path.dirname(loaded_sidecar_path) == os.path.abspath(saved_dir):
                raw_reference_manifest = dict(ckpt_manifest)
            else:
                # Resume into a new save directory: byte-copy the verified
                # content-addressed sidecar so the relative manifest path stays
                # portable and the sha256 identity is preserved.
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
            if num_layers != len(candidate_layers):
                raise RuntimeError(
                    f"Raw bank has {num_layers} layers but {len(candidate_layers)} "
                    "candidate layers were requested."
                )

            if args.fusion_mode == "class_patch":
                fusion = ClassPatchFusionAdapter(
                    num_classes=len(class_names),
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
                    enable_adapter=(args.feature_adapter_mode == "class_residual"),
                    adapter_hidden_dim=args.adapter_hidden_dim,
                    adapter_beta_init=args.adapter_beta_init,
                    no_layer_fusion_classes=[
                        class_names.index(name)
                        for name in args.fusion_no_layer_fusion_class_names
                    ],
                    last_layer_prior_logit=args.fusion_last_layer_prior_logit,
                ).to(device)
            else:
                fusion = ClassConditionedLayerFusion(
                    num_classes=len(class_names),
                    num_layers=num_layers,
                    feature_dim=feature_dim,
                    temperature=args.fusion_temperature,
                    norm_type=args.fusion_norm,
                    norm_affine=args.fusion_norm_affine,
                    shared_weights=bool(args.fusion_mode == "learned_shared"),
                ).to(device)

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
            fusion_config = _build_fusion_config(
                args, class_names, candidate_layers, feature_dim
            )

            # Raw references live in the sidecar + cache; the legacy inline
            # per-class fused bank is intentionally not built in learned mode.
            reference_memory_by_class_cpu = {}
            reference_memory_by_class = {}
            reference_index_by_class = {}
            compute_residual_fn = fusion_runtime.residual
        else:
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
            reference_memory_by_class = common_utils.move_reference_memory_to_gpu(reference_memory_by_class_cpu, device)
            compute_residual_fn = common_utils.compute_residual_feature_batch

        common_utils.remove_reference_samples_from_dataset(train_dataset, reference_indices_by_class)
        print(f"train samples after removing references: {len(train_dataset)}")
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
    if feature_dim is None:
        feature_dim = infer_dinov3_feature_dim(feature_extractor, mini_loader, _all_layer_values, args.use_cls_token, device)

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
        feature_adapter_mode=args.feature_adapter_mode,
    ).to(device)
    if args.feature_adapter_mode == "class_residual":
        # Non-owning pointer: fusion params stay registered once and are
        # optimized through the fusion parameter list below.
        localnet.set_feature_transform_adapter(fusion)
    if args.use_moe_discriminator:
        discriminator = getattr(localnet, "discriminator", None)
        if discriminator is not None and hasattr(discriminator, "enable_gate_stats"):
            discriminator.enable_gate_stats(bool(args.moe_expert_vis_enable))
    # Fusion parameters are appended after the localnet parameters so the
    # optimizer parameter order (and therefore Adam/RMSprop state loading on
    # resume) stays deterministic and identical to the legacy layout when
    # fusion is disabled.
    localnet_params = list(localnet.parameters())
    fusion_params = list(fusion.parameters()) if fusion is not None else []
    trainable_params = localnet_params + fusion_params
    def build_param_groups():
        # Fresh dicts per optimizer: add_param_group stores the dict object
        # itself, so sharing one list would make RMSprop and Adam alias the
        # same groups.
        if args.fusion_lr is not None and fusion_params:
            return [
                {"params": localnet_params, "lr": float(args.lr)},
                {"params": fusion_params, "lr": float(args.fusion_lr)},
            ]
        return [{"params": trainable_params, "lr": float(args.lr)}]

    if args.fusion_lr is not None and float(args.fusion_lr) <= 0:
        raise ValueError("--fusion_lr must be > 0.")
    localnet_optimizer = optim.RMSprop(
        build_param_groups(), lr=args.lr, momentum=0.2
    )
    onetoone_optimizer = (
        optim.Adam(build_param_groups(), lr=args.lr) if args.alternative else None
    )
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
    if fusion_enabled:
        run_name += (
            "_fusion_"
            + str(args.fusion_mode)
            + "_layers_"
            + "-".join(str(v) for v in candidate_layers)
        )
        if args.fusion_mode == "class_patch":
            run_name += "_adapter_" + str(args.feature_adapter_mode)
            if args.fusion_match_mode == "dynamic_topk":
                # Keep the historical run name for reproducible old runs.
                run_name += "_topk_" + str(args.fusion_topk)
            else:
                run_name += "_match_" + str(args.fusion_match_mode)
    moe_expert_vis_ctx = {
        "enabled": bool(args.moe_expert_vis_enable),
        "save_interval": max(1, int(args.moe_expert_vis_interval)),
        "save_dir": os.path.join(saved_dir, args.moe_expert_vis_dirname, run_name),
        "class_names": class_names,
        "num_expert": int(moe_num_expert_runtime),
    }
    if moe_expert_vis_ctx["enabled"] and args.use_moe_discriminator:
        os.makedirs(moe_expert_vis_ctx["save_dir"], exist_ok=True)

    start_epoch = 0
    iteration = 0
    best_mean = -1.0
    best_result_by_class = {}
    memory_bank_snapshot: dict = {}

    if init_path:
        print(
            f"[init] weights-only initialization from: {init_path}\n"
            "[init] net + fusion_state only; optimizer, RNG, epoch, memory-bank "
            "and pseudo-label state start fresh (new experiment, not a resume)."
        )
        init_info = common_utils.load_weights_only_checkpoint(
            init_path, localnet, fusion=fusion
        )
        if fusion_enabled:
            # The cache was refreshed at the fresh-init theta above; rebuild it
            # for the loaded prior before any matching happens.
            fusion_cache.refresh(step=0, epoch=0)
        print(
            "[init] loaded checkpoint epoch="
            f"{init_info['epoch']} iteration={init_info['iteration']}"
        )

    if resume_path:
        print(f"[resume] loading training checkpoint: {resume_path}")
        (
            last_done,
            iteration,
            best_mean,
            best_result_by_class,
            ckpt_mb_freeze,
            ckpt_mb_snapshot,
            resume_checkpoint_meta,
        ) = common_utils.load_train_checkpoint(
            resume_path,
            localnet,
            localnet_optimizer,
            onetoone_optimizer,
            fusion=fusion,
            return_metadata=True,
        )
        if fusion_enabled:
            _validate_resume_fusion_config(
                resume_checkpoint_meta.get("fusion_config"),
                args,
                class_names,
                candidate_layers,
            )
            ckpt_raw_manifest = resume_checkpoint_meta.get("raw_reference_manifest")
            if ckpt_raw_manifest and raw_reference_manifest:
                # Paths may differ when resuming into a new save directory;
                # the content hash is the identity of the raw bank.
                if (
                    ckpt_raw_manifest.get("sha256")
                    != raw_reference_manifest.get("sha256")
                ):
                    raise RuntimeError(
                        "Raw reference bank sidecar used for resume does not match "
                        "the manifest stored in the checkpoint (sha256 mismatch)."
                    )
            print(
                "[resume] restored fusion_state | "
                f"cache was v{fusion_cache.version}, refreshing at current theta"
            )
            fusion_cache.refresh(step=iteration, epoch=last_done + 1)
        ckpt_mb_freeze = int(ckpt_mb_freeze)
        arg_mb_freeze = int(getattr(args, "memory_bank_freeze_start_epoch", -1))
        if arg_mb_freeze != ckpt_mb_freeze:
            print(
                "[resume] memory_bank_freeze_start_epoch mismatch "
                f"(checkpoint {ckpt_mb_freeze} vs args {arg_mb_freeze}); "
                "not loading memory bank snapshot from checkpoint"
            )
        elif ckpt_mb_snapshot and ckpt_mb_snapshot.get("reduced_features_by_class"):
            ds_n = len(mini_loader.dataset)
            cs = ckpt_mb_snapshot.get("class_stack")
            if cs is not None and len(cs) == ds_n:
                snap = common_utils.clone_memory_bank_snapshot_for_checkpoint(ckpt_mb_snapshot)
                memory_bank_snapshot.clear()
                memory_bank_snapshot.update(snap)
                print("[resume] restored memory_bank_snapshot from checkpoint")
            else:
                print(
                    "[resume] memory_bank_snapshot in checkpoint incompatible with current "
                    f"dataset (class_stack len {0 if cs is None else len(cs)} vs {ds_n}); not loading"
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

    def fusion_checkpoint_extras():
        """Extra checkpoint fields; fixed_mean keeps the legacy payload."""
        if not fusion_enabled:
            return {}
        return {
            "fusion_state": fusion.state_dict(),
            "fusion_config": fusion_config,
            "raw_reference_manifest": raw_reference_manifest,
            "fusion_cache_metadata": fusion_cache.metadata(),
            "reference_memory_format": "multilayer_raw_v1",
        }

    for epoch in range(start_epoch, args.epoch):
        (
            local_loss_value,
            bce_loss_value,
            oto_loss_value,
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
            memory_bank_snapshot=memory_bank_snapshot,
            moe_expert_vis_ctx=moe_expert_vis_ctx,
            compute_residual_fn=compute_residual_fn,
            extract_feature_batch_fn=(
                fusion_runtime.extract if fusion_runtime is not None else None
            ),
            fusion_runtime=fusion_runtime,
        )

        print_epoch_losses(
            epoch,
            local_loss_value,
            bce_loss_value,
            oto_loss_value,
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
        if fusion_runtime is not None:
            fusion_runtime.log_epoch(epoch, class_names, save_dir=saved_dir)

        if (epoch + 1) % args.eval_interval == 0:
            if fusion_runtime is not None:
                # Evaluation must use the current checkpoint theta exactly.
                fusion_runtime.before_eval(iteration, epoch)
            eval_rows = []
            mean_img_list = []
            mean_pixel_list = []
            for class_idx_eval, class_name in enumerate(class_names):
                test_loader = build_test_loader(args, class_name)
                (
                    auroc,
                    pixel_auroc,
                ) = evaluate_epoch(
                    localnet,
                    feature_extractor,
                    test_loader,
                    args,
                    class_idx_eval=class_idx_eval,
                    reference_memory_by_class=reference_memory_by_class,
                    reference_index_by_class=reference_index_by_class,
                    compute_residual_fn=compute_residual_fn,
                    extract_feature_batch_fn=(
                        fusion_runtime.extract if fusion_runtime is not None else None
                    ),
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
                best_payload = {
                    "net": localnet.state_dict(),
                    "reference_memory_by_class": reference_memory_by_class_cpu,
                    "reference_indices_by_class": reference_indices_by_class,
                }
                best_payload.update(fusion_checkpoint_extras())
                torch.save(
                    best_payload,
                    os.path.join(saved_dir, run_name + "_localnet.pt"),
                )

            if args.save_log:
                with open(os.path.join(saved_dir, "log.txt"), "a") as file:
                    file.write(
                        f"epoch {epoch + 1} | total loss: {local_loss_value:.6f} | bce loss: {bce_loss_value:.6f} | one-to-one loss: {oto_loss_value:.6f} | gate aux loss: {gate_aux_loss_value:.6f} | gate aux weight: {args.gate_aux_weight:.6f} | multiclass I-AUROC mean: {epoch_mean_img:.5f} | multiclass P-AUROC mean: {epoch_mean_pixel:.5f}\n"
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
            memory_bank_freeze_start_epoch=int(
                getattr(args, "memory_bank_freeze_start_epoch", -1)
            ),
            memory_bank_snapshot=memory_bank_snapshot,
            extra=fusion_checkpoint_extras(),
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
