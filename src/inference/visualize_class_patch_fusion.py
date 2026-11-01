"""Checkpoint-only visualisation of class-and-patch DINOv3 layer fusion.

This entry point mirrors ``infer_multiclass_residual.py`` exactly for model
construction, reference matching and score post-processing, but additionally
exports the fusion weights and six diagnostic figures:

    fig1  class x DINOv3-block global prior
    fig2  per-class patch weight mean +/- std
    fig3  dominant block per patch for selected cases
    fig4  per-patch weight entropy for selected cases
    fig5  per-class patch-to-prior L1 distance
    fig6  detection / fusion case panels

Hard constraints (see ``docs/visualization/class_patch_fusion_inference_visualization_plan.md``):

* inference only: no optimizer, no ``train()``, no ``backward()``, no checkpoint
  writes;
* the checkpoint schema (backbone, layer order, class order, preprocessing,
  match mode) is validated against the CLI before any forward pass;
* the fused reference cache is rebuilt from the raw sidecar, never from stale
  checkpoint tensors;
* score maps use the original resize + Gaussian post-processing; figure colour
  ranges are fixed and recorded, never min-max stretched per image.

Example::

    python src/inference/visualize_class_patch_fusion.py \
      --checkpoint_path /path/to/*_fusion_class_patch_*_localnet.pt \
      --data_path /path/to/MVTec \
      --output_dir ./outputs/fusion_viz \
      --stat_class_names bottle cable capsule \
      --case_class_names leather capsule transistor \
      --cases_per_class 2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

# Ensure project root is on sys.path (script lives at src/inference/...).
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import cv2
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader

from dataset import dataset_extract
from dataset.feature_extract import (
    _FEATURE_MODEL_CHOICES,
    extract_dinov3_multilayer_feature_batch,
)
from dataset.multiclass_feature_dataset import get_all_class_names
from src.inference.infer_multiclass_residual import (
    FUSION_REFERENCE_FORMAT,
    build_feature_extractor,
    denormalize_imagenet,
    detect_learned_fusion,
    find_fusion_num_patches,
    infer_model_params_from_checkpoint,
    resolve_fusion_match_mode,
    str2bool,
    validate_fusion_inference_config,
    validate_raw_bank_metadata,
)
from src.model import model
from src.model.class_patch_fusion_adapter import ClassPatchFusionAdapter
import utils.train_utils as common_utils
from utils.evaluate import _cv2_resize_dsize_from_mask
from utils.fusion_viz_plots import (
    configure_matplotlib,
    plot_dominant_layer,
    plot_fusion_detection,
    plot_global_prior,
    plot_l1_by_class,
    plot_patch_entropy,
    plot_patch_mean_std,
)
from utils.fusion_viz_stats import (
    FusionStatsCollector,
    patch_entropy,
    patch_l1_to_prior,
    stats_to_row,
    validate_global_prior,
)

DEFAULT_FIG2_CLASSES = ("leather", "capsule", "transistor")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        "visualize_class_patch_fusion",
        description="Inference-only visualisation of a trained class_patch fusion checkpoint.",
    )
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Trusted training checkpoint (with raw-bank sidecar beside it).")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Dataset root (same layout expected by MyDataset).")
    parser.add_argument("--dataset", type=str, default="mvtec", choices=["mvtec", "visa"])
    parser.add_argument("--output_dir", type=str, default="./outputs/fusion_viz")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)

    # Statistics scope.
    parser.add_argument("--stat_class_names", type=str, nargs="+", default=None,
                        help="Classes for Fig.2/Fig.5 and the statistics CSV "
                             "(default: all; 'all' also accepted).")
    parser.add_argument("--max_images_per_class", type=int, default=-1,
                        help="Cap images per class for statistics; -1 = full test split.")
    parser.add_argument("--fig2_class_names", type=str, nargs="+", default=None,
                        help="Classes shown in Fig.2 small multiples "
                             f"(default: {' '.join(DEFAULT_FIG2_CLASSES)}; 'all' = every "
                             "statistics class).")

    # Case gallery (Fig.3/4/6).
    parser.add_argument("--case_class_names", type=str, nargs="+", default=None,
                        help="Classes for case figures (default: Fig.2 classes; "
                             "'all' = every class).")
    parser.add_argument("--case_types", type=str, nargs="+", default=["good", "anomaly"],
                        choices=["good", "anomaly"])
    parser.add_argument("--cases_per_class", type=int, default=1,
                        help="Cases per requested type and class.")
    parser.add_argument("--case_selection", type=str, default="first",
                        choices=["first", "seeded_random", "explicit"])
    parser.add_argument("--case_paths", type=str, nargs="+", default=None,
                        help="Explicit test image paths (used with --case_selection explicit).")
    parser.add_argument("--focus_layer", type=int, default=None,
                        help="Real 1-based DINOv3 block for the Fig.6 alpha panel. "
                             "Default: per-class patch-mean argmax (recorded in metadata).")

    # Backbone / preprocessing (must match the checkpoint).
    parser.add_argument(
        "--feature_model", type=str, choices=_FEATURE_MODEL_CHOICES,
        default="dinov3_vitl16",
    )
    parser.add_argument("--dinov3_hub", type=str, default=None)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--crop_size", type=int, default=448)
    parser.add_argument("--use_cls_token", type=str2bool, default=False)
    parser.add_argument(
        "--moe_use_cls_token", type=str2bool, default=False,
        help="MoE gate routing input; must match training (often True).",
    )

    # Optional model-construction overrides (auto-detected when omitted).
    parser.add_argument("--use_moe_discriminator", type=str2bool, default=None)
    parser.add_argument("--moe_num_expert", type=int, default=None)
    parser.add_argument("--moe_top_k", type=int, default=None)
    parser.add_argument("--moe_hard_class_gate", type=str2bool, default=None)

    # Reference index.
    parser.add_argument("--faiss_cpu_index", action="store_true")
    parser.add_argument("--faiss_gpu_temp_mem_mb", type=int, default=256)

    # Score post-processing / figures.
    parser.add_argument("--gaussian_sigma", type=float, default=4.0)
    parser.add_argument("--heatmap_score_min", type=float, default=0.0)
    parser.add_argument("--heatmap_score_max", type=float, default=1.0)
    parser.add_argument("--img_score_topk_ratio", type=float, default=0.01)
    parser.add_argument("--export_formats", type=str, nargs="+",
                        default=["png", "pdf"], choices=["png", "pdf", "svg"])
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--hash_checkpoint", type=str2bool, default=True,
                        help="Record the checkpoint SHA-256 in metadata.json.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def _ensure_dirs(output_dir: str) -> Dict[str, str]:
    dirs = {
        "root": output_dir,
        "figures": os.path.join(output_dir, "figures"),
        "tables": os.path.join(output_dir, "tables"),
        "cases": os.path.join(output_dir, "cases"),
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs


def _sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_project_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def _sanitize_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return value or "sample"


def _class_output_name(class_name: str) -> str:
    return _sanitize_name(class_name)


def _expand_class_scope(values, all_class_names: Sequence[str]):
    """Expand the ``all`` shorthand; ``None`` stays ``None``."""
    if values is None:
        return None
    values = list(values)
    if len(values) == 1 and str(values[0]).lower() == "all":
        return list(all_class_names)
    return values


# ---------------------------------------------------------------------------
# Loading (strictly mirrors infer_multiclass_residual.main)
# ---------------------------------------------------------------------------
def load_checkpoint(path: str, device: torch.device) -> dict:
    print(f"Loading checkpoint: {path}", flush=True)
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if "net" not in checkpoint:
        raise KeyError("checkpoint 缺少 'net' 键。")
    return checkpoint


def build_models_and_cache(args, checkpoint: dict, all_class_names: List[str],
                           feature_extractor, device: torch.device,
                           num_patches: int):
    """Build localnet + ClassPatchFusionAdapter + fused raw-bank cache."""
    fusion_cfg = checkpoint.get("fusion_config") or {}
    feature_dim = int(fusion_cfg["feature_dim"])
    match_mode = resolve_fusion_match_mode(fusion_cfg)

    state_dict_keys = set(checkpoint["net"].keys())
    model_params = infer_model_params_from_checkpoint(state_dict_keys, args)
    has_class_adapter = model_params["class_conditioned_adapter"]
    use_moe = model_params["use_moe_discriminator"]
    moe_num_expert = model_params["moe_num_expert"]
    moe_top_k = model_params["moe_top_k"]
    moe_hard_class_gate = model_params["moe_hard_class_gate"]
    feature_adapter_mode = str(fusion_cfg.get("feature_adapter_mode", "legacy"))

    print(
        f"[model] class_conditioned_adapter={has_class_adapter} | "
        f"use_moe={use_moe} (experts={moe_num_expert}, top_k={moe_top_k}, "
        f"hard_class_gate={moe_hard_class_gate}) | "
        f"feature_adapter_mode={feature_adapter_mode}",
        flush=True,
    )

    localnet = model.localnet(
        len_feature=feature_dim,
        use_moe_discriminator=use_moe,
        moe_num_expert=moe_num_expert,
        moe_top_k=moe_top_k,
        moe_use_cls_token=args.moe_use_cls_token,
        moe_hard_class_gate=moe_hard_class_gate,
        num_classes=(
            len(all_class_names)
            if (has_class_adapter or feature_adapter_mode == "class_residual")
            else None
        ),
        class_conditioned_adapter=has_class_adapter,
        feature_adapter_mode=feature_adapter_mode,
    ).to(device)
    localnet.load_state_dict(checkpoint["net"], strict=True)
    localnet.eval()

    no_layer_names = [
        str(name) for name in (fusion_cfg.get("no_layer_fusion_classes") or [])
    ]
    fusion = ClassPatchFusionAdapter(
        num_classes=int(fusion_cfg["num_classes"]),
        num_layers=int(fusion_cfg["num_layers"]),
        feature_dim=feature_dim,
        temperature=float(fusion_cfg.get("temperature", 1.0)),
        norm_type=str(fusion_cfg.get("norm_type", "none")),
        norm_affine=bool(fusion_cfg.get("norm_affine", False)),
        class_embedding_dim=int(fusion_cfg.get("gate_embedding_dim", 32)),
        gate_hidden_dim=int(fusion_cfg.get("gate_hidden_dim", 64)),
        gate_norm=str(fusion_cfg.get("gate_norm", "layer_norm")),
        gate_lambda_init=float(fusion_cfg.get("gate_lambda_init", 0.1)),
        gate_use_context=bool(fusion_cfg.get("gate_use_context", False)),
        gate_demean_delta=bool(fusion_cfg.get("gate_demean_delta", True)),
        enable_adapter=(feature_adapter_mode == "class_residual"),
        adapter_hidden_dim=int(fusion_cfg.get("adapter_hidden_dim", 128)),
        adapter_beta_init=float(fusion_cfg.get("adapter_beta_init", 0.15)),
        no_layer_fusion_classes=[
            all_class_names.index(name) for name in no_layer_names
        ],
        last_layer_prior_logit=float(fusion_cfg.get("last_layer_prior_logit", 8.0)),
    ).to(device)
    fusion.load_state_dict(checkpoint["fusion_state"], strict=True)
    fusion.eval()
    if feature_adapter_mode == "class_residual":
        localnet.set_feature_transform_adapter(fusion)

    base_dir = os.path.dirname(os.path.abspath(args.checkpoint_path))
    raw_bank, raw_meta = common_utils.load_raw_reference_bank(
        checkpoint["raw_reference_manifest"], base_dir=base_dir
    )
    validate_raw_bank_metadata(raw_meta, fusion_cfg, raw_bank)

    fusion_cache = common_utils.FusedReferenceCache(
        raw_bank,
        fusion,
        device,
        expected_num_layers=int(fusion_cfg["num_layers"]),
        expected_feature_dim=feature_dim,
        storage_dtype=str(fusion_cfg.get("reference_storage_dtype", "float16")),
        use_faiss=True,
        faiss_use_cuda=torch.cuda.is_available(),
        faiss_cpu_index=bool(args.faiss_cpu_index),
        faiss_gpu_temp_mem_mb=int(args.faiss_gpu_temp_mem_mb),
    )
    fusion_cache.refresh(step=0, epoch=0)
    print(
        f"[fusion] rebuilt fused reference cache v{fusion_cache.version} from raw "
        f"sidecar ({sum(int(v.shape[0]) for v in raw_bank.values())} rows)",
        flush=True,
    )

    context = {
        "fusion_cfg": fusion_cfg,
        "feature_dim": feature_dim,
        "match_mode": match_mode,
        "fusion": fusion,
        "fusion_cache": fusion_cache,
        "localnet": localnet,
        "feature_adapter_mode": feature_adapter_mode,
        "use_moe": use_moe,
        "moe_hard_class_gate": moe_hard_class_gate,
        "raw_meta": raw_meta,
        "model_params": model_params,
        "num_patches": int(num_patches),
    }
    return context


# ---------------------------------------------------------------------------
# Case selection
# ---------------------------------------------------------------------------
def select_cases(test_set, class_name: str, args) -> Dict[int, Dict[str, object]]:
    """Return ``{dataset_index: meta}`` for the requested case types."""
    paths = list(test_set.x)
    labels = np.asarray(test_set.y, dtype=np.int64)
    requested_types = set(args.case_types)
    selected: Dict[int, Dict[str, object]] = {}

    def _tag(label: int) -> str:
        return "anomaly" if int(label) == 1 else "good"

    if args.case_selection == "explicit":
        if not args.case_paths:
            raise ValueError("--case_selection explicit 需要 --case_paths。")
        for raw_path in args.case_paths:
            target = os.path.normpath(str(raw_path))
            if os.path.isabs(target):
                # Full paths must match exactly; never fall back to basename,
                # which would also hit the same filename under another defect
                # type and silently swap a good image for an anomaly.
                matches = [
                    idx for idx, path in enumerate(paths)
                    if os.path.normpath(path) == target
                ]
            elif os.sep in target:
                # Relative suffix such as "bottle/test/good/003.png".
                matches = [
                    idx for idx, path in enumerate(paths)
                    if os.path.normpath(path) == target
                    or os.path.normpath(path).endswith(os.sep + target)
                ]
            else:
                # Bare filename; unique within the class in practice.
                matches = [
                    idx for idx, path in enumerate(paths)
                    if os.path.basename(path) == target
                ]
            if not matches:
                warnings.warn(
                    f"[{class_name}] explicit case path not found in test split: {raw_path}",
                    RuntimeWarning,
                )
                continue
            if len(matches) > 1:
                warnings.warn(
                    f"[{class_name}] explicit path {raw_path} matched "
                    f"{len(matches)} test images; all are used.",
                    RuntimeWarning,
                )
            for idx in matches:
                tag = _tag(labels[idx])
                if tag not in requested_types:
                    warnings.warn(
                        f"[{class_name}] skip {raw_path} (label={tag}, "
                        f"requested={sorted(requested_types)})",
                        RuntimeWarning,
                    )
                    continue
                selected[int(idx)] = {
                    "tag": tag,
                    "label": int(labels[idx]),
                    "src_path": paths[idx],
                }
        return dict(sorted(selected.items()))

    for tag in ("good", "anomaly"):
        if tag not in requested_types:
            continue
        label = 0 if tag == "good" else 1
        candidates = np.where(labels == label)[0].tolist()
        if not candidates:
            warnings.warn(
                f"[{class_name}] no '{tag}' test image found; skipping that case type.",
                RuntimeWarning,
            )
            continue
        count = min(int(args.cases_per_class), len(candidates))
        if args.case_selection == "first":
            chosen = candidates[:count]
        elif args.case_selection == "seeded_random":
            rng = random.Random(int(args.seed) + label)
            chosen = sorted(rng.sample(candidates, count))
        else:
            raise ValueError(f"unsupported case_selection={args.case_selection!r}")
        for idx in chosen:
            selected[int(idx)] = {
                "tag": tag,
                "label": int(labels[idx]),
                "src_path": paths[idx],
            }
    return dict(sorted(selected.items()))


# ---------------------------------------------------------------------------
# Per-batch fusion + scoring
# ---------------------------------------------------------------------------
def compute_batch(context, raw_multilayer, class_idx_batch):
    """Return ``(residual, alpha, prior, diagnostics)`` using the checkpoint mode."""
    fusion = context["fusion"]
    cache = context["fusion_cache"]
    fusion_cfg = context["fusion_cfg"]
    match_mode = context["match_mode"]

    if match_mode == "prior_top1":
        residual, diagnostics = common_utils.compute_class_patch_prior_top1_residual_batch(
            raw_multilayer,
            class_idx_batch,
            cache,
            fusion,
            query_chunk_size=int(fusion_cfg.get("fusion_query_chunk", 512)),
            return_diagnostics=True,
        )
    elif match_mode == "dynamic_topk":
        residual, diagnostics = common_utils.compute_class_patch_fused_residual_batch(
            raw_multilayer,
            class_idx_batch,
            cache,
            fusion,
            topk=int(fusion_cfg.get("fusion_topk", 32)),
            query_chunk_size=int(fusion_cfg.get("fusion_query_chunk", 256)),
            candidate_chunk_size=int(fusion_cfg.get("fusion_candidate_chunk", 8)),
            return_diagnostics=True,
        )
    else:
        raise ValueError(f"unsupported fusion match_mode: {match_mode!r}")

    alpha = diagnostics["alpha"].detach()
    prior = fusion.global_weights(class_idx_batch).detach()
    return residual, alpha, prior, diagnostics


def save_case_npz(
    cases_dir: str,
    class_name: str,
    case_id: str,
    *,
    rgb: np.ndarray,
    gt_mask: np.ndarray,
    alpha: np.ndarray,
    prior: np.ndarray,
    score_patch: np.ndarray,
    score_map: np.ndarray,
    entropy: np.ndarray,
    l1: np.ndarray,
    dominant: np.ndarray,
    candidate_layers: Sequence[int],
    class_idx: int,
    sample_index: int,
    label: int,
    src_path: str,
) -> str:
    class_dir = os.path.join(cases_dir, _class_output_name(class_name))
    os.makedirs(class_dir, exist_ok=True)
    path = os.path.join(class_dir, f"{case_id}.npz")
    np.savez_compressed(
        path,
        image_rgb=np.asarray(rgb, dtype=np.uint8),
        gt_mask=np.asarray(gt_mask, dtype=np.uint8),
        alpha_patch=np.asarray(alpha, dtype=np.float32),
        alpha_global=np.asarray(prior, dtype=np.float32),
        score_patch=np.asarray(score_patch, dtype=np.float32),
        score_map=np.asarray(score_map, dtype=np.float32),
        entropy=np.asarray(entropy, dtype=np.float32),
        l1=np.asarray(l1, dtype=np.float32),
        dominant_layer=np.asarray(dominant, dtype=np.int16),
        candidate_layers=np.asarray(candidate_layers, dtype=np.int32),
        class_name=np.asarray(class_name),
        class_index=np.asarray(int(class_idx), dtype=np.int32),
        sample_index=np.asarray(int(sample_index), dtype=np.int32),
        label=np.asarray(int(label), dtype=np.uint8),
        src_path=np.asarray(str(src_path)),
    )
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    common_utils.fix_seed(args.seed)
    if args.max_images_per_class == 0:
        raise ValueError("--max_images_per_class must be -1 (all) or >= 1.")
    if args.cases_per_class < 1:
        raise ValueError("--cases_per_class must be >= 1.")
    if args.case_selection == "explicit" and not args.case_paths:
        raise ValueError("--case_selection explicit requires --case_paths.")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    dirs = _ensure_dirs(args.output_dir)
    configure_matplotlib()
    started_at = datetime.now(timezone.utc).isoformat()

    all_class_names = get_all_class_names(args.dataset)
    class_to_idx = {name: idx for idx, name in enumerate(all_class_names)}

    checkpoint = load_checkpoint(args.checkpoint_path, device)
    if not detect_learned_fusion(checkpoint):
        raise RuntimeError(
            "checkpoint 不是 learned-fusion 运行（缺少 fusion_state / raw bank 标记）。"
        )
    if checkpoint.get("reference_memory_format") != FUSION_REFERENCE_FORMAT:
        raise RuntimeError(
            "learned-fusion checkpoint 的 reference_memory_format 不正确: "
            f"{checkpoint.get('reference_memory_format')!r}"
        )
    if checkpoint.get("fusion_state") is None:
        raise KeyError("learned-fusion checkpoint 缺少 'fusion_state'。")
    if checkpoint.get("raw_reference_manifest") is None:
        raise KeyError("learned-fusion checkpoint 缺少 'raw_reference_manifest'。")

    fusion_cfg = checkpoint.get("fusion_config") or {}
    if fusion_cfg.get("fusion_mode") != "class_patch":
        raise RuntimeError(
            "本脚本只支持 class_patch fusion checkpoint；当前 fusion_mode="
            f"{fusion_cfg.get('fusion_mode')!r}。V1 运行请使用 infer_multiclass_residual.py。"
        )
    candidate_layers = validate_fusion_inference_config(
        fusion_cfg, args, all_class_names
    )
    match_mode = resolve_fusion_match_mode(fusion_cfg)
    no_layer_names = [
        str(name) for name in (fusion_cfg.get("no_layer_fusion_classes") or [])
    ]
    print(
        f"[fusion] schema=v{fusion_cfg.get('schema_version')} | mode=class_patch | "
        f"match={match_mode} | blocks={candidate_layers} | "
        f"tau={fusion_cfg.get('temperature')} | feature_dim={fusion_cfg.get('feature_dim')} | "
        f"pinned={no_layer_names or 'none'}",
        flush=True,
    )

    feature_extractor = build_feature_extractor(args, device)
    feature_extractor.eval()

    num_patches = find_fusion_num_patches(
        args, feature_extractor, device, all_class_names, candidate_layers
    )
    patch_side = int(np.sqrt(num_patches))
    if patch_side * patch_side != num_patches:
        raise RuntimeError(
            f"patch 数 {num_patches} 不是完全平方数，无法形成方形热力图网格。"
        )
    if len(candidate_layers) != int(fusion_cfg["num_layers"]):
        raise RuntimeError("candidate_layers 长度与 fusion_config.num_layers 不一致。")
    print(
        f"[data] patches={num_patches} ({patch_side}x{patch_side}) | "
        f"layers={len(candidate_layers)} | feature_dim={fusion_cfg['feature_dim']}",
        flush=True,
    )

    context = build_models_and_cache(
        args, checkpoint, all_class_names, feature_extractor, device, num_patches
    )
    fusion = context["fusion"]
    localnet = context["localnet"]
    feature_dim = context["feature_dim"]

    # ------------------------------------------------------------------
    # Global prior export (Fig.1) - independent of the test images.
    # ------------------------------------------------------------------
    alpha_global = (
        fusion.get_layer_weights().detach().float().cpu().numpy()
    )
    alpha_global = validate_global_prior(alpha_global)
    patch_lambda = fusion.get_patch_lambda().detach().float().cpu().numpy()
    beta = fusion.get_beta().detach().float().cpu().numpy()
    pinned_flags = [name in no_layer_names for name in all_class_names]

    prior_rows = []
    for idx, name in enumerate(all_class_names):
        row = {"class_index": idx, "class": name}
        for pos, block in enumerate(candidate_layers):
            row[f"block_{int(block)}"] = float(alpha_global[idx, pos])
        row["pinned_prior"] = bool(pinned_flags[idx])
        row["lambda"] = float(patch_lambda[idx])
        row["beta"] = float(beta[idx])
        row["temperature"] = float(fusion_cfg.get("temperature", 1.0))
        prior_rows.append(row)
    prior_csv = os.path.join(dirs["tables"], "global_prior.csv")
    pd.DataFrame(prior_rows).to_csv(prior_csv, index=False)
    fig1_paths = plot_global_prior(
        alpha_global,
        all_class_names,
        candidate_layers,
        pinned_flags,
        dirs["figures"],
        formats=args.export_formats,
        dpi=args.dpi,
        subtitle=(
            f"{fusion_cfg.get('feature_model')} · schema "
            f"v{fusion_cfg.get('schema_version')} · match={match_mode}"
        ),
    )
    print(f"[fig1] wrote {prior_csv} and {len(fig1_paths)} figure file(s)", flush=True)

    # ------------------------------------------------------------------
    # Resolve scopes.
    # ------------------------------------------------------------------
    stat_scope = _expand_class_scope(args.stat_class_names, all_class_names)
    stat_class_names = stat_scope if stat_scope else list(all_class_names)
    unknown = sorted(set(stat_class_names) - set(all_class_names))
    if unknown:
        raise ValueError(f"unknown --stat_class_names: {unknown}")

    fig2_scope = _expand_class_scope(args.fig2_class_names, all_class_names)
    requested_fig2 = fig2_scope if fig2_scope else list(DEFAULT_FIG2_CLASSES)
    fig2_class_names = [name for name in requested_fig2 if name in stat_class_names]
    if not fig2_class_names:
        fig2_class_names = stat_class_names[: min(3, len(stat_class_names))]
        warnings.warn(
            "none of the requested Fig.2 classes are in the statistics scope; "
            f"falling back to {fig2_class_names}.",
            RuntimeWarning,
        )

    case_scope = _expand_class_scope(args.case_class_names, all_class_names)
    if case_scope:
        case_class_names = case_scope
        unknown_cases = sorted(set(case_class_names) - set(all_class_names))
        if unknown_cases:
            raise ValueError(f"unknown --case_class_names: {unknown_cases}")
    else:
        case_class_names = list(fig2_class_names)

    focus_layer_arg = None
    if args.focus_layer is not None:
        if int(args.focus_layer) not in candidate_layers:
            raise ValueError(
                f"--focus_layer {args.focus_layer} is not a candidate block "
                f"({candidate_layers})."
            )
        focus_layer_arg = int(args.focus_layer)

    stat_set = set(stat_class_names)
    case_set = set(case_class_names)
    process_classes = [
        name for name in all_class_names if name in stat_set or name in case_set
    ]
    if not process_classes:
        raise RuntimeError("no class selected for statistics or cases.")

    print(
        f"[scope] statistics={stat_class_names} | fig2={fig2_class_names} | "
        f"cases={case_class_names} ({args.case_types}, "
        f"{args.cases_per_class}/type, {args.case_selection})",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Main pass: stream statistics and selected cases.
    # ------------------------------------------------------------------
    collector = FusionStatsCollector(context["fusion_cfg"]["num_layers"])
    case_records: List[Dict[str, object]] = []
    missing_case_labels: List[str] = []

    for class_name in process_classes:
        class_idx = class_to_idx[class_name]
        need_stats = class_name in stat_set
        need_cases = class_name in case_set
        test_set = dataset_extract.MyDataset(
            dataset_path=args.data_path,
            dataset=args.dataset,
            class_name=class_name,
            is_train=False,
            resize=args.image_size,
            cropsize=args.crop_size,
        )
        if len(test_set) == 0:
            warnings.warn(f"[{class_name}] empty test split; skipped.", RuntimeWarning)
            continue

        case_map = select_cases(test_set, class_name, args) if need_cases else {}
        if need_cases and not case_map:
            missing_case_labels.append(class_name)
            warnings.warn(
                f"[{class_name}] no case matched the requested types; "
                "case figures will be skipped for this class.",
                RuntimeWarning,
            )

        stat_limit = int(args.max_images_per_class)
        stat_stop = (
            len(test_set) if stat_limit <= 0 else min(stat_limit, len(test_set))
        )
        case_stop = (max(case_map) + 1) if case_map else 0
        stop_at = max(stat_stop if need_stats else 0, case_stop if need_cases else 0)
        if stop_at <= 0:
            continue

        loader = DataLoader(
            test_set,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            pin_memory=False,
            drop_last=False,
        )
        print(
            f"[{class_name}] test images={len(test_set)} | processed<={min(stop_at, len(test_set))} | "
            f"stat={need_stats} | cases={len(case_map)}",
            flush=True,
        )

        used_case_ids = set()
        sample_offset = 0
        with torch.inference_mode():
            for images, y, mask in loader:
                if sample_offset >= stop_at:
                    break
                images = images.to(device, non_blocking=False)
                batch = int(images.shape[0])
                class_idx_batch = torch.full(
                    (batch,), int(class_idx), dtype=torch.long, device=device
                )

                raw_multilayer, cls_token = extract_dinov3_multilayer_feature_batch(
                    images,
                    feature_extractor,
                    candidate_layers,
                    use_cls_token=args.use_cls_token,
                    return_cls_token=True,
                )
                if (
                    raw_multilayer.shape[1] != num_patches
                    or raw_multilayer.shape[2] != int(context["fusion_cfg"]["num_layers"])
                    or raw_multilayer.shape[3] != feature_dim
                ):
                    raise RuntimeError(
                        "extracted raw features do not match the checkpoint schema: "
                        f"got {tuple(raw_multilayer.shape)}, expected "
                        f"[B, {num_patches}, {context['fusion_cfg']['num_layers']}, "
                        f"{feature_dim}]."
                    )

                model_features, alpha, prior, _diag = compute_batch(
                    context, raw_multilayer, class_idx_batch
                )

                patch_class_idx = None
                if context["use_moe"] and context["moe_hard_class_gate"]:
                    patch_class_idx = (
                        class_idx_batch.to(device=model_features.device, dtype=torch.long)
                        .reshape(-1, 1, 1)
                        .expand(-1, int(model_features.shape[1]), 1)
                        .contiguous()
                    )
                _, score = localnet(
                    model_features,
                    cls_token=cls_token,
                    patch_class_idx=patch_class_idx,
                    class_idx=class_idx_batch,
                )
                score_np = score.detach().float().cpu().numpy().reshape(batch, -1)
                if score_np.shape[1] != num_patches:
                    raise RuntimeError(
                        f"localnet returned {score_np.shape[1]} patch scores, "
                        f"expected {num_patches}."
                    )

                n_stat = batch
                if need_stats and stat_limit > 0:
                    n_stat = max(0, min(batch, stat_limit - sample_offset))
                if n_stat > 0:
                    collector.update(class_idx, alpha[:n_stat], prior[:n_stat])

                if case_map:
                    image_np = images.detach().float().cpu().numpy()
                    y_np = y.detach().cpu().numpy().astype(np.int64)
                    mask_np = mask.detach().cpu().numpy().astype(np.uint8)
                    alpha_np = alpha.detach().float().cpu().numpy()
                    prior_np = prior.detach().float().cpu().numpy()

                    for local_i in range(batch):
                        global_idx = sample_offset + local_i
                        meta = case_map.get(global_idx)
                        if meta is None:
                            continue
                        tag = str(meta["tag"])
                        label = int(meta["label"])
                        src_path = str(meta["src_path"])
                        stem = os.path.splitext(os.path.basename(src_path))[0]
                        case_id = _sanitize_name(f"{tag}_{stem}")
                        base_id = case_id
                        suffix = 1
                        while case_id in used_case_ids:
                            suffix += 1
                            case_id = f"{base_id}_{suffix}"
                        used_case_ids.add(case_id)

                        score_patch = score_np[local_i]
                        score_map = cv2.resize(
                            score_patch.reshape(patch_side, patch_side),
                            _cv2_resize_dsize_from_mask(mask_np[local_i]),
                            interpolation=cv2.INTER_LINEAR,
                        )
                        score_map = gaussian_filter(
                            score_map, sigma=float(args.gaussian_sigma)
                        )
                        image_score = float(
                            common_utils.aggregate_image_scores(
                                score_patch.reshape(1, -1),
                                topk_ratio=float(args.img_score_topk_ratio),
                            )[0]
                        )

                        alpha_i = alpha_np[local_i].astype(np.float64)
                        prior_i = prior_np[local_i].astype(np.float64)
                        entropy_i = patch_entropy(alpha_i)
                        l1_i = patch_l1_to_prior(
                            alpha_i[None, ...], prior_i[None, :]
                        )[0]
                        dominant_i = alpha_i.argmax(axis=-1).astype(np.int16)
                        rgb_u8 = denormalize_imagenet(image_np[local_i])
                        gt_u8 = mask_np[local_i].squeeze(0).astype(np.uint8)

                        npz_path = save_case_npz(
                            dirs["cases"],
                            class_name,
                            case_id,
                            rgb=rgb_u8,
                            gt_mask=gt_u8,
                            alpha=alpha_i,
                            prior=prior_i,
                            score_patch=score_patch,
                            score_map=score_map,
                            entropy=entropy_i,
                            l1=l1_i,
                            dominant=dominant_i,
                            candidate_layers=candidate_layers,
                            class_idx=class_idx,
                            sample_index=global_idx,
                            label=label,
                            src_path=src_path,
                        )
                        case_records.append(
                            {
                                "class_name": class_name,
                                "class_idx": int(class_idx),
                                "sample_index": int(global_idx),
                                "case_id": case_id,
                                "tag": tag,
                                "label": label,
                                "src_path": src_path,
                                "rgb": rgb_u8,
                                "gt": gt_u8,
                                "alpha": alpha_i.astype(np.float32),
                                "prior": prior_i.astype(np.float32),
                                "score_patch": score_patch.astype(np.float32),
                                "score_map": score_map.astype(np.float32),
                                "entropy": entropy_i.astype(np.float32),
                                "l1": l1_i.astype(np.float32),
                                "dominant": dominant_i,
                                "image_score": image_score,
                                "npz_path": npz_path,
                            }
                        )

                sample_offset += batch
                if sample_offset >= stop_at:
                    break

    if not collector.class_indices:
        raise RuntimeError("no statistics were collected; check the dataset and scope.")

    # ------------------------------------------------------------------
    # Fig.2 / Fig.5 + statistics CSV.
    # ------------------------------------------------------------------
    stat_results: Dict[str, Dict[str, object]] = {}
    stat_rows = []
    for name in stat_class_names:
        idx = class_to_idx[name]
        if not collector.has(idx):
            warnings.warn(
                f"[{name}] requested for statistics but no image was processed; "
                "excluded from tables and figures.",
                RuntimeWarning,
            )
            continue
        result = collector.result(idx)
        stat_results[name] = result
        extra = {
            "lambda": float(patch_lambda[idx]),
            "beta": float(beta[idx]),
        }
        stat_rows.append(
            stats_to_row(idx, name, result, candidate_layers, extra=extra)
        )

    if not stat_rows:
        raise RuntimeError("statistics CSV would be empty.")
    stats_csv = os.path.join(dirs["tables"], "patch_statistics.csv")
    stats_df = pd.DataFrame(stat_rows)
    stats_df.to_csv(stats_csv, index=False)
    print(
        f"[stats] wrote {stats_csv} ({len(stats_df)} classes, "
        f"{int(stats_df['n_patches'].sum()):,} patches)",
        flush=True,
    )

    coverage_note = None
    if set(stat_class_names) != set(all_class_names):
        coverage_note = (
            f"statistics cover {len(stat_results)}/{len(all_class_names)} classes"
        )

    fig2_paths = []
    fig2_available = [name for name in fig2_class_names if name in stat_results]
    if fig2_available:
        means = np.stack([stat_results[n]["mean"] for n in fig2_available])
        stds = np.stack([stat_results[n]["std"] for n in fig2_available])
        priors = np.stack([alpha_global[class_to_idx[n]] for n in fig2_available])
        counts = [
            (int(stat_results[n]["n_images"]), int(stat_results[n]["n_patches"]))
            for n in fig2_available
        ]
        fig2_paths = plot_patch_mean_std(
            fig2_available,
            candidate_layers,
            means,
            stds,
            priors,
            counts,
            dirs["figures"],
            formats=args.export_formats,
            dpi=args.dpi,
            subtitle=coverage_note,
        )
        print(f"[fig2] wrote {len(fig2_paths)} figure file(s)", flush=True)
    else:
        warnings.warn("Fig.2 skipped: none of the Fig.2 classes have statistics.",
                      RuntimeWarning)

    fig5_names = [name for name in stat_class_names if name in stat_results]
    fig5_paths = plot_l1_by_class(
        fig5_names,
        [float(stat_results[n]["l1_mean"]) for n in fig5_names],
        [float(stat_results[n]["l1_std"]) for n in fig5_names],
        dirs["figures"],
        formats=args.export_formats,
        dpi=args.dpi,
        subtitle=coverage_note,
    )
    print(f"[fig5] wrote {len(fig5_paths)} figure file(s)", flush=True)

    # ------------------------------------------------------------------
    # Case figures (Fig.3 / Fig.4 / Fig.6).
    # ------------------------------------------------------------------
    focus_by_class: Dict[str, int] = {}
    for class_name in case_class_names:
        idx = class_to_idx[class_name]
        if focus_layer_arg is not None:
            focus_block = focus_layer_arg
        elif collector.has(idx):
            mean = np.asarray(collector.result(idx)["mean"], dtype=np.float64)
            focus_block = int(candidate_layers[int(np.argmax(mean))])
        else:
            focus_block = int(candidate_layers[int(np.argmax(alpha_global[idx]))])
        focus_by_class[class_name] = focus_block

    figure_paths: List[str] = list(fig1_paths) + list(fig2_paths) + list(fig5_paths)
    rendered_cases = 0
    for record in case_records:
        class_name = str(record["class_name"])
        focus_block = focus_by_class[class_name]
        focus_pos = candidate_layers.index(focus_block)
        class_tag = _class_output_name(class_name)
        stem = _sanitize_name(f"{class_name}_{record['case_id']}")
        sample_rel = os.path.relpath(str(record["src_path"]), args.data_path)
        subtitle = (
            f"class={class_name} · sample={sample_rel} · "
            f"source alpha_patch: cases/{class_tag}/{record['case_id']}.npz"
        )
        focus_rule = (
            "explicit --focus_layer"
            if focus_layer_arg is not None
            else "per-class patch-mean argmax"
        )

        figure_paths.extend(
            plot_dominant_layer(
                record["rgb"],
                np.asarray(record["dominant"]).reshape(patch_side, patch_side),
                candidate_layers,
                dirs["figures"],
                stem=f"fig3_dominant_layer_{stem}",
                formats=args.export_formats,
                dpi=args.dpi,
                subtitle=subtitle,
            )
        )
        figure_paths.extend(
            plot_patch_entropy(
                record["rgb"],
                np.asarray(record["entropy"]).reshape(patch_side, patch_side),
                len(candidate_layers),
                dirs["figures"],
                stem=f"fig4_patch_entropy_{stem}",
                formats=args.export_formats,
                dpi=args.dpi,
                subtitle=subtitle,
            )
        )
        alpha_map = np.asarray(record["alpha"])[:, focus_pos].reshape(
            patch_side, patch_side
        )
        figure_paths.extend(
            plot_fusion_detection(
                record["rgb"],
                record["gt"],
                record["score_map"],
                alpha_map,
                np.asarray(record["l1"]).reshape(patch_side, patch_side),
                dirs["figures"],
                stem=f"fig6_fusion_detection_{stem}",
                class_name=class_name,
                case_id=str(record["case_id"]),
                label=int(record["label"]),
                image_score=float(record["image_score"]),
                focus_block=focus_block,
                score_min=float(args.heatmap_score_min),
                score_max=float(args.heatmap_score_max),
                formats=args.export_formats,
                dpi=args.dpi,
                subtitle=f"{subtitle} · focus rule: {focus_rule}",
            )
        )
        rendered_cases += 1

    if rendered_cases == 0:
        warnings.warn(
            "no case figures were generated (no matching test samples).", RuntimeWarning
        )
    else:
        print(f"[cases] rendered {rendered_cases} case(s)", flush=True)

    # ------------------------------------------------------------------
    # Metadata.
    # ------------------------------------------------------------------
    checkpoint_stat = os.stat(args.checkpoint_path)
    metadata = {
        "generated_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "checkpoint": {
            "path": os.path.abspath(args.checkpoint_path),
            "size_bytes": int(checkpoint_stat.st_size),
            "sha256": (
                _sha256_file(args.checkpoint_path) if args.hash_checkpoint else None
            ),
        },
        "data": {
            "path": os.path.abspath(args.data_path),
            "dataset": args.dataset,
        },
        "schema": {
            "schema_version": fusion_cfg.get("schema_version"),
            "fusion_mode": fusion_cfg.get("fusion_mode"),
            "match_mode": match_mode,
            "candidate_layers": list(candidate_layers),
            "class_names": list(all_class_names),
            "feature_model": fusion_cfg.get("feature_model"),
            "feature_dim": int(feature_dim),
            "num_layers": int(fusion_cfg["num_layers"]),
            "temperature": float(fusion_cfg.get("temperature", 1.0)),
            "norm_type": fusion_cfg.get("norm_type"),
            "norm_affine": bool(fusion_cfg.get("norm_affine", False)),
            "feature_adapter_mode": context["feature_adapter_mode"],
            "no_layer_fusion_classes": no_layer_names,
            "last_layer_prior_logit": float(
                fusion_cfg.get("last_layer_prior_logit", 8.0)
            ),
            "preprocess": fusion_cfg.get("preprocess"),
            "raw_bank_metadata": context["raw_meta"],
        },
        "inference": {
            "image_size": int(args.image_size),
            "crop_size": int(args.crop_size),
            "use_cls_token": bool(args.use_cls_token),
            "moe_use_cls_token": bool(args.moe_use_cls_token),
            "use_moe_discriminator": bool(context["use_moe"]),
            "moe_hard_class_gate": bool(context["moe_hard_class_gate"]),
            "batch_size": int(args.batch_size),
            "seed": int(args.seed),
            "gaussian_sigma": float(args.gaussian_sigma),
            "heatmap_score_range": [
                float(args.heatmap_score_min), float(args.heatmap_score_max)
            ],
            "img_score_topk_ratio": float(args.img_score_topk_ratio),
            "score_postprocess": "cv2.resize(INTER_LINEAR) + gaussian_filter",
            "device": str(device),
        },
        "statistics": {
            "class_names": fig5_names,
            "max_images_per_class": int(args.max_images_per_class),
            "n_images": int(sum(int(stat_results[n]["n_images"]) for n in fig5_names)),
            "n_patches": int(sum(int(stat_results[n]["n_patches"]) for n in fig5_names)),
            "coverage_note": coverage_note,
            "lambda": {name: float(patch_lambda[class_to_idx[name]]) for name in fig5_names},
            "beta": {name: float(beta[class_to_idx[name]]) for name in fig5_names},
        },
        "cases": {
            "class_names": case_class_names,
            "types": list(args.case_types),
            "cases_per_class": int(args.cases_per_class),
            "selection": args.case_selection,
            "seed": int(args.seed),
            "explicit_paths": list(args.case_paths or []),
            "focus_layer_rule": (
                f"explicit block {focus_layer_arg}"
                if focus_layer_arg is not None
                else "per-class patch-mean argmax"
            ),
            "focus_block_by_class": focus_by_class,
            "missing_case_classes": missing_case_labels,
            "rendered": rendered_cases,
            "samples": [
                {
                    "class": str(record["class_name"]),
                    "case_id": str(record["case_id"]),
                    "sample_index": int(record["sample_index"]),
                    "label": int(record["label"]),
                    "src_path": str(record["src_path"]),
                    "image_score": float(record["image_score"]),
                    "npz": os.path.relpath(record["npz_path"], args.output_dir),
                }
                for record in case_records
            ],
        },
        "outputs": {
            "figures": [os.path.relpath(p, args.output_dir) for p in figure_paths],
            "tables": [
                os.path.relpath(prior_csv, args.output_dir),
                os.path.relpath(stats_csv, args.output_dir),
            ],
        },
    }

    metadata_path = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(_jsonable(metadata), handle, indent=2, ensure_ascii=False)

    print(f"[done] metadata -> {metadata_path}", flush=True)
    print(f"[done] figures  -> {dirs['figures']}", flush=True)
    print(f"[done] cases    -> {dirs['cases']}", flush=True)


if __name__ == "__main__":
    main()
