import argparse
import random

import faiss
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

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
