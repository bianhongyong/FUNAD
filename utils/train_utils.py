import argparse
import os
import random
import warnings
from typing import Optional

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
):
    """epoch_completed: last finished epoch index (0-based)."""
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
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_train_checkpoint(
    path: str,
    localnet: nn.Module,
    localnet_optimizer: optim.Optimizer,
    onetoone_optimizer,
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
    restore_rng_state(ckpt.get("rng_state"))
    return (
        int(ckpt["epoch"]),
        int(ckpt["iteration"]),
        float(ckpt.get("best_mean", -1.0)),
        ckpt.get("best_result_by_class") or {},
        int(ckpt.get("memory_bank_freeze_start_epoch", -1)),
        ckpt.get("memory_bank_snapshot"),
    )


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
