import argparse
import hashlib
import os
import random
import time
import warnings
from typing import Any, Dict, Optional, Tuple

import faiss
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from PIL import Image

_FAISS_GPU_RESOURCES = None


def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y", "t"}:
        return True
    if lowered in {"false", "0", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected, e.g. true/false")


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


def get_oto_loss(oto_loss_name, device):
    if oto_loss_name == "mae":
        return nn.L1Loss().to(device)
    if oto_loss_name == "mse":
        return nn.MSELoss().to(device)
    return nn.KLDivLoss(reduction="batchmean").to(device)


def aggregate_image_scores(score_2d, topk_ratio):
    if score_2d.ndim != 2:
        raise ValueError(f"Expected 2D score array, got shape={score_2d.shape}")
    patch_count = int(score_2d.shape[1])
    safe_ratio = float(np.clip(topk_ratio, 0.0, 1.0))
    k = max(1, int(np.ceil(patch_count * safe_ratio)))
    k = min(k, patch_count)
    topk = np.partition(score_2d, patch_count - k, axis=1)[:, -k:]
    return topk.mean(axis=1)


def _get_faiss_gpu_resources(temp_mem_mb=None):
    global _FAISS_GPU_RESOURCES
    if _FAISS_GPU_RESOURCES is None:
        _FAISS_GPU_RESOURCES = faiss.StandardGpuResources()
        if temp_mem_mb is not None and temp_mem_mb > 0:
            _FAISS_GPU_RESOURCES.setTempMemory(int(temp_mem_mb) * 1024 * 1024)
    return _FAISS_GPU_RESOURCES


def build_faiss_index(feature_np, use_cuda, use_cpu_index=False, gpu_temp_mem_mb=256):
    faiss.omp_set_num_threads(4)
    dim = int(feature_np.shape[-1])
    feat = np.ascontiguousarray(feature_np.astype(np.float32))
    if use_cuda and (not use_cpu_index):
        try:
            resources = _get_faiss_gpu_resources(gpu_temp_mem_mb)
            index = faiss.GpuIndexFlatL2(resources, dim, faiss.GpuIndexFlatConfig())
        except RuntimeError:
            index = faiss.IndexFlatL2(dim)
    else:
        index = faiss.IndexFlatL2(dim)
    index.add(feat)
    return index


def compute_distance(feature, use_cuda, use_cpu_index=False, gpu_temp_mem_mb=256):
    embedding = np.ascontiguousarray(feature.astype(np.float32))
    index = build_faiss_index(
        embedding,
        use_cuda=use_cuda,
        use_cpu_index=use_cpu_index,
        gpu_temp_mem_mb=gpu_temp_mem_mb,
    )
    distance, id_array = index.search(embedding, k=2)
    distance = distance.T[-1]
    id_array = id_array.T[-1]
    return np.expand_dims(distance, axis=-1), id_array


def update_topk_features(top_feat, top_dist, cand_feat, cand_dist, k):
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


# ---------------------------------------------------------------------------
# RNG state management
# ---------------------------------------------------------------------------

def collect_rng_state():
    state = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "random": random.getstate(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    if state is None:
        return
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["random"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------

def default_train_checkpoint_path(saved_dir: str, run_name: str) -> str:
    return os.path.join(saved_dir, run_name + "_train_checkpoint.pt")


def _torch_load_train_checkpoint(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def clone_memory_bank_snapshot_for_checkpoint(snapshot: dict) -> dict:
    """Deep-copy numpy arrays for checkpoint I/O (CPU)."""
    rf_in = snapshot.get("reduced_features_by_class") or {}
    return {
        "reduced_features_by_class": {
            int(k): np.asarray(v, dtype=np.float32).copy() for k, v in rf_in.items()
        },
        "class_stack": np.asarray(snapshot["class_stack"], dtype=np.int64).copy(),
        "gt_patch_masks": np.asarray(snapshot["gt_patch_masks"], dtype=np.uint8).copy(),
        "global_dim": int(snapshot["global_dim"]),
    }


def save_train_checkpoint(
    path: str,
    epoch_completed: int,
    iteration: int,
    localnet: nn.Module,
    localnet_optimizer: optim.Optimizer,
    onetoone_optimizer,
    best_mean: float,
    best_result_by_class: dict,
    memory_bank_freeze_start_epoch: int = -1,
    memory_bank_snapshot: Optional[dict] = None,
    extra: Optional[dict] = None,
):
    """epoch_completed: last finished epoch index (0-based).

    ``extra`` carries learned-fusion state/metadata (fusion_state,
    fusion_config, raw_reference_manifest, ...).  Core keys cannot be
    overridden silently.
    """
    if memory_bank_snapshot is None:
        memory_bank_snapshot = {}
    payload = {
        "epoch": int(epoch_completed),
        "iteration": int(iteration),
        "net": localnet.state_dict(),
        "localnet_optimizer": localnet_optimizer.state_dict(),
        "onetoone_optimizer": (
            onetoone_optimizer.state_dict() if onetoone_optimizer is not None else None
        ),
        "best_mean": float(best_mean),
        "best_result_by_class": best_result_by_class,
        "rng_state": collect_rng_state(),
        "memory_bank_freeze_start_epoch": int(memory_bank_freeze_start_epoch),
        "memory_bank_snapshot": (
            clone_memory_bank_snapshot_for_checkpoint(memory_bank_snapshot)
            if memory_bank_snapshot.get("reduced_features_by_class")
            else None
        ),
    }
    if extra:
        overlap = sorted(set(extra).intersection(payload))
        if overlap:
            raise KeyError(f"extra checkpoint keys would override core keys: {overlap}")
        payload.update(extra)
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_train_checkpoint(
    path: str,
    localnet: nn.Module,
    localnet_optimizer: optim.Optimizer,
    onetoone_optimizer,
    fusion: Optional[nn.Module] = None,
    return_metadata: bool = False,
):
    ckpt = _torch_load_train_checkpoint(path)
    localnet.load_state_dict(ckpt["net"])
    localnet_optimizer.load_state_dict(ckpt["localnet_optimizer"])
    oto_sd = ckpt.get("onetoone_optimizer")
    if oto_sd is not None:
        if onetoone_optimizer is None:
            raise RuntimeError(
                "Checkpoint has onetoone_optimizer state but current run has --alternative disabled."
            )
        onetoone_optimizer.load_state_dict(oto_sd)
    elif onetoone_optimizer is not None:
        raise RuntimeError(
            "Checkpoint has no onetoone_optimizer state but current run uses --alternative."
        )

    fusion_sd = ckpt.get("fusion_state")
    if fusion_sd is not None and fusion is None:
        raise RuntimeError(
            "Checkpoint contains learned-fusion state but the current run uses "
            "--fusion_mode fixed_mean. Pass the matching --fusion_mode or load it "
            "for inference."
        )
    if fusion is not None and fusion_sd is None:
        raise RuntimeError(
            "Current run enables learned fusion but the checkpoint has no "
            "'fusion_state'. Start a new run or resume a learned-fusion checkpoint."
        )
    if fusion is not None and fusion_sd is not None:
        fusion.load_state_dict(fusion_sd, strict=True)

    restore_rng_state(ckpt.get("rng_state"))
    base = (
        int(ckpt["epoch"]),
        int(ckpt["iteration"]),
        float(ckpt.get("best_mean", -1.0)),
        ckpt.get("best_result_by_class") or {},
        int(ckpt.get("memory_bank_freeze_start_epoch", -1)),
        ckpt.get("memory_bank_snapshot"),
    )
    if not return_metadata:
        return base
    metadata = {
        "fusion_config": ckpt.get("fusion_config"),
        "raw_reference_manifest": ckpt.get("raw_reference_manifest"),
        "fusion_cache_metadata": ckpt.get("fusion_cache_metadata"),
        "has_fusion_state": fusion_sd is not None,
        "reference_memory_format": ckpt.get("reference_memory_format"),
    }
    return base + (metadata,)


def inspect_train_checkpoint(path: str) -> dict:
    """Load only learned-fusion-related metadata from a training checkpoint."""
    ckpt = _torch_load_train_checkpoint(path)
    return {
        "fusion_config": ckpt.get("fusion_config"),
        "raw_reference_manifest": ckpt.get("raw_reference_manifest"),
        "fusion_cache_metadata": ckpt.get("fusion_cache_metadata"),
        "has_fusion_state": "fusion_state" in ckpt,
        "reference_memory_format": ckpt.get("reference_memory_format"),
    }


# ---------------------------------------------------------------------------
# DataLoader helpers
# ---------------------------------------------------------------------------

def build_loader_kwargs(
    num_workers, pin_memory=True, prefetch_factor=1, persistent_workers=True
):
    kwargs = {
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
    }
    if int(num_workers) > 0:
        kwargs["persistent_workers"] = bool(persistent_workers)
        kwargs["prefetch_factor"] = int(prefetch_factor)
    return kwargs


# ---------------------------------------------------------------------------
# Reference memory & residual feature computation
# ---------------------------------------------------------------------------

def build_reference_memory_bank(
    train_dataset, reference_indices_by_class, extract_feature_fn, batch_size, num_workers, device,
):
    """Extract features from reference images and build per-class memory + FAISS index."""
    all_indices = []
    for class_idx in sorted(reference_indices_by_class.keys()):
        all_indices.extend(reference_indices_by_class[class_idx])

    if len(all_indices) == 0:
        raise RuntimeError("没有参考图可用于构建残差记忆库。")

    reference_subset = Subset(train_dataset, all_indices)
    loader_kwargs = build_loader_kwargs(num_workers, pin_memory=True, prefetch_factor=2)
    reference_loader = DataLoader(
        reference_subset,
        batch_size=max(1, min(batch_size, 16)),
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    memory_by_class = {class_idx: [] for class_idx in reference_indices_by_class.keys()}
    with torch.no_grad():
        for batch in reference_loader:
            images, class_idx = batch[0], batch[1]
            images = images.to(device, non_blocking=True)
            base_features = extract_feature_fn(images, class_indices=class_idx)
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
        index_by_class[int(class_idx)] = build_faiss_index(cls_memory,
            use_cuda=torch.cuda.is_available())

    if len(index_by_class) == 0:
        raise RuntimeError("参考图特征提取失败，无法构建残差记忆库。")

    return memory_np_by_class, index_by_class


def move_reference_memory_to_gpu(reference_memory_by_class, device):
    reference_memory_gpu_by_class = {}
    for class_idx, memory_np in reference_memory_by_class.items():
        if memory_np is None or memory_np.shape[0] == 0:
            continue
        reference_memory_gpu_by_class[int(class_idx)] = torch.as_tensor(
            memory_np, dtype=torch.float32, device=device
        )
    return reference_memory_gpu_by_class


def compute_residual_feature_batch(
    features, class_idx_batch, reference_memory_by_class, reference_index_by_class
):
    """Compute residual features: feat - nearest reference feature per class."""
    del reference_index_by_class  # Kept for API compatibility.
    dim = int(features.shape[-1])
    num_patches = int(features.shape[1])
    residual_features = features
    unique_class_idx = torch.unique(class_idx_batch)

    for cls_tensor in unique_class_idx:
        cls = int(cls_tensor.item())
        reference_memory = reference_memory_by_class.get(cls, None)
        if reference_memory is None:
            continue

        cls_mask = class_idx_batch == cls_tensor
        if not bool(cls_mask.any().item()):
            continue

        cls_feat = features[cls_mask].reshape(-1, dim)
        distance = torch.cdist(
            cls_feat.to(dtype=torch.float32),
            reference_memory,
            p=2.0,
        )
        nearest_id = torch.argmin(distance, dim=1)
        nearest_feat = reference_memory.index_select(0, nearest_id).to(dtype=features.dtype)
        cls_residual = (cls_feat - nearest_feat).reshape(-1, num_patches, dim)
        residual_features[cls_mask] = cls_residual

    return residual_features


def identity_residual(features, class_idx, reference_memory_by_class, reference_index_by_class):
    """Pass features through unchanged (used when residual is disabled)."""
    return features


# ---------------------------------------------------------------------------
# Learned multi-layer fusion: raw reference bank, fused cache, NN residual
# ---------------------------------------------------------------------------

_RAW_BANK_FORMAT = "multilayer_raw_v1"


def _normalize_storage_dtype(storage_dtype):
    if storage_dtype is None:
        return None
    if isinstance(storage_dtype, torch.dtype):
        return storage_dtype
    text = str(storage_dtype).lower()
    if text in ("float16", "fp16", "half"):
        return torch.float16
    if text in ("float32", "fp32", "float"):
        return torch.float32
    raise ValueError(
        f"Unsupported reference_storage_dtype={storage_dtype!r}; use float16 or float32."
    )


def _storage_dtype_name(dtype):
    return "float16" if dtype == torch.float16 else "float32"


def apply_gaussian_feature_noise(
    x,
    distance,
    threshold_map,
    noise_threshold,
    std=None,
):
    """Perturb uncertain patches while keeping ``x`` on the autograd graph.

    The previous implementation used ``x.detach().clone()`` which cut the
    Gaussian BCE branch off from residual/fusion parameters.  Noise statistics
    and sampling stay under ``torch.no_grad()``; only the added noise is
    detached.

    Args:
        x: ``[B, P, C]`` residual features (may require grad).
        distance: ``[B, P]`` or ``[B*P]`` patch distances.
        threshold_map: same shape as ``distance``.
        noise_threshold: upper bound of the uncertain interval.
        std: optional scalar noise std; when None use per-channel max patch std
            of ``x.detach()`` (original FUNAD behaviour).

    Returns:
        ``x_perturbed`` with the same shape/dtype/device as ``x``.
    """
    if x.dim() != 3:
        raise ValueError(f"Expected x [B,P,C], got {tuple(x.shape)}")
    batch, patches, channels = x.shape

    def _entry_count(value):
        # ``len(np.ndarray)`` only returns the first dimension, so use size()
        # for numpy inputs (distance/threshold maps are [B, P] in training).
        if torch.is_tensor(value):
            return int(value.numel())
        return int(np.asarray(value).size)

    distance_entries = _entry_count(distance)
    threshold_entries = _entry_count(threshold_map)
    if distance_entries != batch * patches:
        raise ValueError(
            "distance must contain B*P entries; "
            f"got {distance_entries} for B={batch}, P={patches}."
        )
    if threshold_entries != batch * patches:
        raise ValueError(
            "threshold_map must contain B*P entries; "
            f"got {threshold_entries} for B={batch}, P={patches}."
        )

    with torch.no_grad():
        dist = torch.as_tensor(distance, dtype=torch.float32, device=x.device).reshape(-1)
        thr = torch.as_tensor(threshold_map, dtype=torch.float32, device=x.device).reshape(-1)
        uncertain_mask = (dist > thr) & (dist < float(noise_threshold))

        x_flat = x.detach().reshape(-1, channels)
        noise = torch.zeros_like(x_flat)
        n_uncertain = int(uncertain_mask.sum().item())
        if n_uncertain > 0:
            if std is None:
                # Match the original FUNAD scale: per-channel std over the
                # batch, maxed over patch positions -> [C].
                x_std = x.detach().std(dim=0).max(dim=0)[0]
                scale = x_std.repeat_interleave(n_uncertain).reshape(-1, n_uncertain).T
                noise[uncertain_mask] = torch.normal(mean=0.0, std=scale)
            else:
                noise[uncertain_mask] = torch.normal(
                    mean=0.0,
                    std=float(std),
                    size=(n_uncertain, channels),
                    device=x.device,
                )

    # x + detached noise: gradients flow to x, not to the noise sample.
    return (x.reshape(-1, channels) + noise).reshape(batch, patches, channels)


def build_multilayer_reference_memory_bank(
    train_dataset,
    reference_indices_by_class,
    extract_multilayer_fn,
    batch_size,
    num_workers,
    device,
    storage_dtype="float16",
    require_all_classes=True,
):
    """Extract and store raw per-layer reference patches ``[N_c, L, C]``.

    The L axis is preserved (no mean).  All layers of a row come from the same
    reference patch because DINOv3 layers are stacked before flattening.

    Returns:
        memory_np_by_class: dict[int, np.ndarray] with dtype float16/float32.
        info: dict with num_layers, feature_dim, shapes, storage_dtype.
    """
    all_indices = []
    for class_idx in sorted(reference_indices_by_class.keys()):
        all_indices.extend(reference_indices_by_class[class_idx])
    if len(all_indices) == 0:
        raise RuntimeError("没有参考图可用于构建多层参考记忆库。")

    reference_subset = Subset(train_dataset, all_indices)
    loader_kwargs = build_loader_kwargs(num_workers, pin_memory=True, prefetch_factor=2)
    reference_loader = DataLoader(
        reference_subset,
        batch_size=max(1, min(batch_size, 16)),
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    storage_dtype = _normalize_storage_dtype(storage_dtype)
    np_storage_dtype = np.float16 if storage_dtype == torch.float16 else np.float32
    buffers = {int(class_idx): [] for class_idx in reference_indices_by_class.keys()}
    num_layers = None
    feature_dim = None

    with torch.no_grad():
        for batch in reference_loader:
            images, class_idx = batch[0], batch[1]
            images = images.to(device, non_blocking=True)
            features = extract_multilayer_fn(images, class_indices=class_idx)
            if features.dim() != 4:
                raise RuntimeError(
                    "extract_multilayer_fn must return [B, P, L, C]; "
                    f"got {tuple(features.shape)}."
                )
            batch_size_actual, num_patches, layers, dim = features.shape
            if num_layers is None:
                num_layers = int(layers)
                feature_dim = int(dim)
            elif int(layers) != num_layers or int(dim) != feature_dim:
                raise RuntimeError(
                    "Inconsistent multilayer feature shape across batches: "
                    f"{(layers, dim)} vs {(num_layers, feature_dim)}."
                )

            feat_np = features.detach().cpu().numpy()
            class_np = class_idx.detach().cpu().numpy().astype(np.int64)
            for cls in np.unique(class_np).tolist():
                cls = int(cls)
                cls_mask = class_np == cls
                # [N_images, P, L, C] -> [N_images*P, L, C]
                cls_feat = feat_np[cls_mask].reshape(-1, num_layers, feature_dim)
                buffers.setdefault(cls, []).append(cls_feat)

    memory_np_by_class = {}
    for class_idx, feat_list in buffers.items():
        if len(feat_list) == 0:
            if require_all_classes:
                raise RuntimeError(
                    f"class {class_idx} 没有可用的多层参考特征；learned fusion 要求所有类别都有参考图。"
                )
            continue
        cls_memory = np.concatenate(feat_list, axis=0).astype(np_storage_dtype, copy=False)
        memory_np_by_class[int(class_idx)] = cls_memory

    if len(memory_np_by_class) == 0:
        raise RuntimeError("参考图多层特征提取失败，无法构建原始参考库。")

    info = {
        "format": _RAW_BANK_FORMAT,
        "num_layers": int(num_layers),
        "feature_dim": int(feature_dim),
        "storage_dtype": _storage_dtype_name(storage_dtype),
        "shapes": {
            int(c): [int(v) for v in arr.shape]
            for c, arr in memory_np_by_class.items()
        },
    }
    return memory_np_by_class, info


def _sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def save_raw_reference_bank(
    raw_memory_by_class: Dict[int, Any],
    metadata: dict,
    save_dir: str,
    prefix: str,
):
    """Persist the raw multilayer bank as a content-addressed sidecar.

    The returned manifest stores a relative path + sha256 so checkpoints stay
    portable across machines (no absolute paths).
    """
    os.makedirs(save_dir, exist_ok=True)
    arrays = {}
    shapes = {}
    for class_idx, arr in raw_memory_by_class.items():
        arr_np = np.asarray(arr)
        if arr_np.ndim != 3:
            raise ValueError(
                f"raw bank for class {class_idx} must be [N,L,C]; got {arr_np.shape}"
            )
        arrays[int(class_idx)] = arr_np
        shapes[int(class_idx)] = [int(v) for v in arr_np.shape]

    payload_metadata = dict(metadata)
    payload_metadata["format"] = _RAW_BANK_FORMAT
    payload_metadata["shapes"] = shapes
    payload = {
        "format": _RAW_BANK_FORMAT,
        "metadata": payload_metadata,
        "raw_reference_memory_by_class": arrays,
    }

    tmp_path = os.path.join(save_dir, prefix + ".tmp")
    torch.save(payload, tmp_path)
    sha = _sha256_file(tmp_path)
    final_name = f"{prefix}_{sha[:16]}.pt"
    final_path = os.path.join(save_dir, final_name)

    if os.path.isfile(final_path):
        if _sha256_file(final_path) != sha:
            os.remove(tmp_path)
            raise RuntimeError(
                f"Content-addressed sidecar collision for {final_path}; "
                "remove the stale file and retry."
            )
        os.remove(tmp_path)
    else:
        os.replace(tmp_path, final_path)

    manifest = dict(payload_metadata)
    manifest["path"] = final_name
    manifest["sha256"] = sha
    return manifest


def load_raw_reference_bank(manifest: dict, base_dir: str):
    """Load and verify a raw bank sidecar described by ``manifest``.

    Returns:
        (raw_memory_by_class, metadata)
    """
    if manifest is None:
        raise KeyError("checkpoint 缺少 raw_reference_manifest。")
    if manifest.get("format") != _RAW_BANK_FORMAT:
        raise ValueError(
            f"不支持的 raw reference bank format: {manifest.get('format')!r}."
        )
    rel_path = manifest.get("path")
    if not rel_path or os.path.isabs(rel_path):
        raise ValueError(
            "raw_reference_manifest['path'] 必须是 checkpoint 同目录下的相对路径，"
            f"当前为 {rel_path!r}。"
        )
    base_dir = os.path.abspath(base_dir)
    resolved = os.path.abspath(os.path.join(base_dir, rel_path))
    if os.path.commonpath([base_dir, resolved]) != base_dir:
        raise ValueError(f"raw reference bank 路径越界: {resolved}")
    if not os.path.isfile(resolved):
        raise FileNotFoundError(
            f"找不到 raw reference bank sidecar: {resolved}. "
            "请将 checkpoint 与 sidecar 一起迁移，或用训练数据重新提取。"
        )

    expected_sha = manifest.get("sha256")
    actual_sha = _sha256_file(resolved)
    if expected_sha and actual_sha != expected_sha:
        raise ValueError(
            "raw reference bank 哈希不匹配："
            f"manifest={expected_sha[:16]}..., file={actual_sha[:16]}..."
        )

    payload = _torch_load_train_checkpoint(resolved)
    payload_metadata = payload.get("metadata") or {}
    if payload.get("format") != _RAW_BANK_FORMAT:
        raise ValueError("raw reference bank sidecar 内部 format 不正确。")
    manifest_metadata = {
        k: v for k, v in manifest.items() if k not in ("path", "sha256")
    }
    mismatched = [
        key for key, value in manifest_metadata.items()
        if key in payload_metadata and payload_metadata[key] != value
    ]
    if mismatched:
        raise ValueError(
            f"raw reference bank metadata 与 manifest 不一致: {mismatched}"
        )

    raw = payload.get("raw_reference_memory_by_class")
    if not isinstance(raw, dict) or len(raw) == 0:
        raise RuntimeError("raw reference bank sidecar 中没有可用特征。")
    return {int(k): np.asarray(v) for k, v in raw.items()}, payload_metadata


def should_refresh_reference_cache(unit: str, interval: int, step=None, epoch=None) -> bool:
    """Return whether the fused reference cache should refresh now."""
    interval = max(1, int(interval))
    if unit == "step":
        return step is not None and int(step) % interval == 0
    if unit == "epoch":
        return epoch is not None and int(epoch) % interval == 0
    raise ValueError(f"Unknown refresh unit {unit!r}; expected 'epoch' or 'step'.")


class FusedReferenceCache:
    """Derived fused reference cache + exact NN search over it.

    Lifecycle:
      * raw bank is fixed (CPU, usually float16);
      * ``refresh()`` re-fuses the whole raw bank with the current theta and
        atomically swaps the fused tensors / indices;
      * queries are fused with the current theta every step, so between
        refreshes NN search is a controlled approximation.

    Search uses chunked exact L2 by default.  ``use_faiss=True`` builds a FAISS
    index per class (used by independent inference).
    """

    def __init__(
        self,
        raw_memory_by_class: Dict[int, Any],
        fusion,
        device,
        expected_num_layers: Optional[int] = None,
        expected_feature_dim: Optional[int] = None,
        storage_dtype=None,
        use_faiss: bool = False,
        faiss_use_cuda: bool = False,
        faiss_cpu_index: bool = False,
        faiss_gpu_temp_mem_mb: int = 256,
        query_chunk_size: int = 2048,
        ref_chunk_size: int = 4096,
        fuse_chunk_size: int = 1024,
        drift_sample_size: int = 512,
    ):
        self.fusion = fusion
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.raw_cpu = {}
        for class_idx, arr in raw_memory_by_class.items():
            tensor = torch.as_tensor(np.asarray(arr))
            if tensor.dim() != 3:
                raise ValueError(
                    f"raw bank for class {class_idx} must be [N,L,C]; got {tuple(tensor.shape)}"
                )
            self.raw_cpu[int(class_idx)] = tensor.cpu()

        if len(self.raw_cpu) == 0:
            raise RuntimeError("Raw reference bank is empty.")

        first = next(iter(self.raw_cpu.values()))
        self.num_layers = int(first.shape[1])
        self.feature_dim = int(first.shape[2])
        if expected_num_layers is not None and self.num_layers != int(expected_num_layers):
            raise ValueError(
                f"raw bank has {self.num_layers} layers, expected {expected_num_layers}."
            )
        if expected_feature_dim is not None and self.feature_dim != int(expected_feature_dim):
            raise ValueError(
                f"raw bank has feature_dim={self.feature_dim}, expected {expected_feature_dim}."
            )
        for cls, tensor in self.raw_cpu.items():
            if int(tensor.shape[1]) != self.num_layers or int(tensor.shape[2]) != self.feature_dim:
                raise ValueError(f"raw bank for class {cls} has inconsistent L/C shape.")

        self.storage_dtype = _normalize_storage_dtype(storage_dtype) or first.dtype
        self.use_faiss = bool(use_faiss)
        self.faiss_use_cuda = bool(faiss_use_cuda)
        self.faiss_cpu_index = bool(faiss_cpu_index)
        self.faiss_gpu_temp_mem_mb = int(faiss_gpu_temp_mem_mb)
        self.query_chunk_size = int(query_chunk_size)
        self.ref_chunk_size = int(ref_chunk_size)
        self.fuse_chunk_size = int(fuse_chunk_size)
        self.drift_sample_size = int(drift_sample_size)

        self.fused_by_class: Dict[int, torch.Tensor] = {}
        self.index_by_class: Dict[int, object] = {}
        self.version = 0
        self.refresh_count = 0
        self.refresh_step = -1
        self.refresh_epoch = -1
        self.last_refresh_time = 0.0
        self.total_refresh_time = 0.0
        self.last_drift_rate_by_class: Dict[int, float] = {}

    # -- lifecycle ---------------------------------------------------------
    def _fuse_raw(self, raw_cpu: torch.Tensor, class_idx: int) -> torch.Tensor:
        chunks = []
        for start in range(0, raw_cpu.shape[0], self.fuse_chunk_size):
            chunk = raw_cpu[start:start + self.fuse_chunk_size].to(
                self.device, dtype=torch.float32
            )
            chunks.append(self.fusion(chunk, int(class_idx)))
        if not chunks:
            return torch.empty(
                (0, self.feature_dim), dtype=torch.float32, device=self.device
            )
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)

    def _estimate_drift(self, old_fused: torch.Tensor, new_fused: torch.Tensor) -> float:
        total = old_fused.shape[0]
        sample = min(max(1, self.drift_sample_size), total)
        sample_idx = torch.linspace(
            0, total - 1, steps=sample, device=old_fused.device
        ).round().long().unique()
        anchors = old_fused.index_select(0, sample_idx)
        nn_old = self._chunked_nearest_ids(anchors, old_fused)
        nn_new = self._chunked_nearest_ids(anchors, new_fused)
        return float((nn_old != nn_new).float().mean().item())

    def refresh(self, step=None, epoch=None) -> dict:
        """Re-fuse the whole raw bank with the current theta and swap atomically."""
        start = time.perf_counter()
        new_fused: Dict[int, torch.Tensor] = {}
        new_index: Dict[int, object] = {}
        drift: Dict[int, float] = {}

        with torch.no_grad():
            for class_idx, raw in self.raw_cpu.items():
                fused = self._fuse_raw(raw, class_idx)
                old = self.fused_by_class.get(class_idx)
                if old is not None and old.shape == fused.shape:
                    drift[class_idx] = self._estimate_drift(old, fused)
                new_fused[class_idx] = fused
                if self.use_faiss:
                    new_index[class_idx] = build_faiss_index(
                        fused.detach().cpu().numpy(),
                        use_cuda=self.faiss_use_cuda,
                        use_cpu_index=self.faiss_cpu_index,
                        gpu_temp_mem_mb=self.faiss_gpu_temp_mem_mb,
                    )

        # Atomic switch: both the fused tensors and indices change together.
        self.fused_by_class = new_fused
        self.index_by_class = new_index
        self.version += 1
        self.refresh_count += 1
        self.refresh_step = -1 if step is None else int(step)
        self.refresh_epoch = -1 if epoch is None else int(epoch)
        self.last_drift_rate_by_class = dict(drift)
        self.last_refresh_time = time.perf_counter() - start
        self.total_refresh_time += self.last_refresh_time

        mean_drift = float(np.mean(list(drift.values()))) if drift else 0.0
        return {
            "version": self.version,
            "refresh_step": self.refresh_step,
            "refresh_epoch": self.refresh_epoch,
            "elapsed": self.last_refresh_time,
            "drift_rate_by_class": dict(drift),
            "mean_drift": mean_drift,
        }

    def metadata(self) -> dict:
        return {
            "version": self.version,
            "refresh_count": self.refresh_count,
            "refresh_step": self.refresh_step,
            "refresh_epoch": self.refresh_epoch,
            "num_classes": len(self.raw_cpu),
            "num_layers": self.num_layers,
            "feature_dim": self.feature_dim,
            "storage_dtype": _storage_dtype_name(self.storage_dtype),
            "last_drift_rate_by_class": dict(self.last_drift_rate_by_class),
            "last_refresh_time": self.last_refresh_time,
            "total_refresh_time": self.total_refresh_time,
        }

    # -- search ------------------------------------------------------------
    def _chunked_nearest_ids(self, query: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if ref.shape[0] == 0:
            raise RuntimeError("Cannot search an empty reference bank.")
        query = query.detach().to(dtype=torch.float32)
        ref = ref.detach().to(dtype=torch.float32)
        total = query.shape[0]
        out = torch.empty(total, dtype=torch.long, device=query.device)
        ref_chunk = max(1, self.ref_chunk_size)
        query_chunk = max(1, self.query_chunk_size)
        for q_start in range(0, total, query_chunk):
            q = query[q_start:q_start + query_chunk]
            best_dist = torch.full((q.shape[0],), float("inf"), device=q.device)
            best_idx = torch.zeros(q.shape[0], dtype=torch.long, device=q.device)
            for r_start in range(0, ref.shape[0], ref_chunk):
                r = ref[r_start:r_start + ref_chunk]
                dist = torch.cdist(q, r, p=2.0)
                min_dist, min_idx = dist.min(dim=1)
                better = min_dist < best_dist
                best_dist = torch.where(better, min_dist, best_dist)
                best_idx = torch.where(better, min_idx + r_start, best_idx)
            out[q_start:q_start + q.shape[0]] = best_idx
        return out

    def nearest_ids(self, fused_query: torch.Tensor, class_idx_batch: torch.Tensor) -> torch.Tensor:
        """Exact NN against the **cached** fused reference version.

        Args:
            fused_query: ``[B, P, C]`` or ``[M, C]``.
            class_idx_batch: ``[B]`` (or ``[1]``/empty for 2D input).
        Returns:
            ``[B, P]`` long tensor on the query device.
        """
        squeeze = fused_query.dim() == 2
        if squeeze:
            fused_query = fused_query.unsqueeze(0)
        if fused_query.dim() != 3:
            raise ValueError(f"Expected fused_query [B,P,C], got {tuple(fused_query.shape)}")

        batch, patches, dim = fused_query.shape
        if dim != self.feature_dim:
            raise ValueError(
                f"fused_query feature_dim={dim}, cache expects {self.feature_dim}."
            )
        class_idx_batch = torch.as_tensor(
            class_idx_batch, dtype=torch.long, device=fused_query.device
        ).reshape(-1)
        if class_idx_batch.numel() == 1 and batch > 1:
            class_idx_batch = class_idx_batch.expand(batch)
        if class_idx_batch.numel() != batch:
            raise ValueError(
                f"class_idx has {class_idx_batch.numel()} entries for batch {batch}."
            )

        out = torch.empty(
            (batch, patches), dtype=torch.long, device=fused_query.device
        )
        for cls_tensor in torch.unique(class_idx_batch):
            cls = int(cls_tensor.item())
            if cls not in self.fused_by_class:
                raise KeyError(f"class {cls} has no fused reference cache.")
            mask = class_idx_batch == cls_tensor
            queries = fused_query[mask].reshape(-1, dim).detach().to(dtype=torch.float32)

            if self.use_faiss and cls in self.index_by_class:
                q_np = np.ascontiguousarray(queries.cpu().numpy())
                _, ids_np = self.index_by_class[cls].search(q_np, k=1)
                ids = torch.as_tensor(
                    ids_np.reshape(-1), dtype=torch.long, device=fused_query.device
                )
            else:
                ids = self._chunked_nearest_ids(queries, self.fused_by_class[cls])

            out[mask] = ids.reshape(-1, patches)

        return out.squeeze(0) if squeeze else out

    def raw_rows(self, class_idx: int, ids: torch.Tensor, dtype=torch.float32) -> torch.Tensor:
        """Gather raw ``[K,L,C]`` rows from the fixed CPU bank onto the device."""
        raw = self.raw_cpu[int(class_idx)]
        selected = raw[ids.detach().cpu().to(dtype=torch.long)]
        return selected.to(device=self.device, dtype=dtype, non_blocking=True)

    def nn_mismatch_rate(self, fused_query: torch.Tensor, class_idx_batch: torch.Tensor) -> float:
        """Fraction of queries whose cached NN differs from current-theta NN.

        This re-fuses every reference with the **current** theta and is meant
        for periodic diagnostics only (expensive).
        """
        with torch.no_grad():
            cached_ids = self.nearest_ids(fused_query, class_idx_batch)
            squeeze = fused_query.dim() == 2
            q = fused_query.unsqueeze(0) if squeeze else fused_query
            c = torch.as_tensor(class_idx_batch, dtype=torch.long, device=q.device).reshape(-1)
            if c.numel() == 1 and q.shape[0] > 1:
                c = c.expand(q.shape[0])
            total = 0
            mismatch = 0
            for cls_tensor in torch.unique(c):
                cls = int(cls_tensor.item())
                mask = c == cls_tensor
                queries = q[mask].reshape(-1, self.feature_dim)
                current_ref = self._fuse_raw(self.raw_cpu[cls], cls)
                exact = self._chunked_nearest_ids(queries, current_ref)
                cached = cached_ids[mask].reshape(-1)
                total += int(exact.numel())
                mismatch += int((exact != cached).sum().item())
        return float(mismatch) / float(total) if total else 0.0


def compute_fused_residual_batch(
    raw_query,
    class_idx_batch,
    fusion_cache: FusedReferenceCache,
    fusion,
    fuse_chunk_size: Optional[int] = None,
    return_neighbor_ids: bool = False,
):
    """Fusion → cached-NN → re-fuse selected raw refs → fused residual.

    The NN ids come from the (possibly stale) fused cache and are discrete.
    The selected reference rows are fetched from the **fixed raw bank** and
    fused again with the **current** theta so gradients reach both query and
    reference sides.

    Args:
        raw_query: ``[B, P, L, C]`` from the frozen backbone (no grad).
        class_idx_batch: ``[B]``.
        fusion_cache: ``FusedReferenceCache``.
        fusion: ``ClassConditionedLayerFusion`` (same instance as cache.fusion).
        return_neighbor_ids: also return ``[B, P]`` nearest ids.
    Returns:
        ``[B, P, C]`` residual (or ``(residual, ids)``).
    """
    if raw_query.dim() != 4:
        raise ValueError(f"Expected raw_query [B,P,L,C], got {tuple(raw_query.shape)}")
    if raw_query.shape[-2] != fusion_cache.num_layers:
        raise ValueError(
            f"raw_query has {raw_query.shape[-2]} layers, cache expects "
            f"{fusion_cache.num_layers}."
        )
    class_idx_batch = torch.as_tensor(
        class_idx_batch, dtype=torch.long, device=raw_query.device
    ).reshape(-1)
    batch, patches, _, dim = raw_query.shape
    if class_idx_batch.numel() == 1 and batch > 1:
        class_idx_batch = class_idx_batch.expand(batch)
    if class_idx_batch.numel() != batch:
        raise ValueError(
            f"class_idx has {class_idx_batch.numel()} entries for batch {batch}."
        )

    fused_query = fusion(raw_query, class_idx_batch)  # [B, P, C], keeps grad
    neighbor_ids = fusion_cache.nearest_ids(fused_query.detach(), class_idx_batch)
    residual = torch.zeros_like(fused_query)
    chunk = int(fuse_chunk_size or fusion_cache.fuse_chunk_size)

    for cls_tensor in torch.unique(class_idx_batch):
        cls = int(cls_tensor.item())
        mask = class_idx_batch == cls_tensor
        query = fused_query[mask].reshape(-1, dim)
        cls_ids = neighbor_ids[mask].reshape(-1)

        # Fuse each unique selected reference once, then scatter back.
        unique_ids, inverse = torch.unique(cls_ids, return_inverse=True)
        raw_selected = fusion_cache.raw_rows(cls, unique_ids)
        fused_ref_chunks = [
            fusion(part, cls)
            for part in torch.split(raw_selected, max(1, chunk), dim=0)
        ]
        fused_ref = (
            fused_ref_chunks[0]
            if len(fused_ref_chunks) == 1
            else torch.cat(fused_ref_chunks, dim=0)
        )
        selected_ref = fused_ref.index_select(0, inverse)
        residual[mask] = (query - selected_ref).reshape(-1, patches, dim)

    if return_neighbor_ids:
        return residual, neighbor_ids
    return residual


# ---------------------------------------------------------------------------
# Reference selection & dataset manipulation
# ---------------------------------------------------------------------------

def select_reference_indices_by_class(train_dataset, num_classes, num_reference_images_per_class, seed, strict_clean_reference=False):
    rng = np.random.default_rng(seed)
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
            if strict_clean_reference:
                raise RuntimeError(msg)
            warnings.warn(msg)
            continue

        sample_n = min(max(1, num_reference_images_per_class), len(cls_candidates))
        selected = rng.choice(np.array(cls_candidates), size=sample_n, replace=False)
        reference_indices_by_class[class_idx] = sorted(selected.tolist())

        # print selected reference image paths
        dataset_samples = train_dataset.samples
        print(f"[Reference] class {class_idx} ({sample_n} images):")
        for sel_idx in reference_indices_by_class[class_idx]:
            sel_path = dataset_samples[sel_idx][0]
            print(f"  {sel_path}")

    return reference_indices_by_class


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
