import os
import time

import faiss
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader, Subset
from utils.memory_bank_stats import print_greedy_memory_bank_anomaly_stats
from utils.sampler import ApproximateGreedyCoresetSampler


def _as_1d_float32(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.shape == ():
        x = np.array([x], dtype=np.float32)
    return x.astype(np.float32).reshape(-1)


def _minmax_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    vmin = float(values.min())
    vmax = float(values.max())
    if vmax == vmin:
        return np.zeros_like(values, dtype=np.float32)
    return (values - vmin) / (vmax - vmin)


def _postprocess_faiss_distances(distance: np.ndarray, k_number: int) -> np.ndarray:
    """
    distance: shape (N, k). Distances below 1e-4 are treated as 0 (self / duplicate).
    Returns shape (N,): per row, mean of neighbors whose distance is > 0.
    If all k neighbors are 0, returns 0 for that row.
    k_number is unused but kept for call-site compatibility with FAISS k search.
    """
    _ = k_number
    distance = np.asarray(distance, dtype=np.float32)
    distance[distance < 1e-4] = 0
    mask = distance > 0
    sums = (distance * mask).sum(axis=1)
    counts = mask.sum(axis=1)
    return np.where(counts > 0, sums / counts.astype(np.float32), 0.0).astype(np.float32)

def _plot_distance_distribution_normal_vs_anomaly(
    args, distance_values, gt_patch_masks, epoch=None
):
    values = np.asarray(distance_values, dtype=np.float32).reshape(-1)
    labels = np.asarray(gt_patch_masks, dtype=np.uint8).reshape(-1) > 0
    if values.size == 0 or labels.size != values.size:
        return

    normal_values = values[~labels]
    anomaly_values = values[labels]
    if normal_values.size == 0 and anomaly_values.size == 0:
        return

    # Keep plotting lightweight even for large datasets.
    max_points = 200000
    if normal_values.size > max_points:
        normal_values = np.random.choice(normal_values, size=max_points, replace=False)
    if anomaly_values.size > max_points:
        anomaly_values = np.random.choice(anomaly_values, size=max_points, replace=False)

    save_dir = os.path.join(args.save_path, args.dataset, args.noise)
    os.makedirs(save_dir, exist_ok=True)
    if epoch is None:
        filename = "distance_distribution_normal_vs_anomaly.png"
    else:
        filename = f"distance_distribution_normal_vs_anomaly_epoch_{int(epoch) + 1:03d}.png"
    save_path = os.path.join(save_dir, filename)

    plt.figure(figsize=(10, 6))
    bins = np.linspace(0.0, 1.0, 101)
    if normal_values.size > 0:
        plt.hist(
            normal_values,
            bins=bins,
            density=True,
            alpha=0.55,
            color="tab:blue",
            label=f"normal ({normal_values.size})",
        )
    if anomaly_values.size > 0:
        plt.hist(
            anomaly_values,
            bins=bins,
            density=True,
            alpha=0.55,
            color="tab:red",
            label=f"anomaly ({anomaly_values.size})",
        )

    plt.xlabel("Normalized distance")
    plt.ylabel("Density")
    plt.title("Distance Distribution: Normal vs Anomaly Patches")
    plt.xlim(0.0, 1.0)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[PseudoLabel-Plot] saved: {save_path}")


def _plot_image_norm_score_distribution_normal_vs_anomaly(
    args, image_norm_scores, gt_patch_masks, epoch=None
):
    scores = np.asarray(image_norm_scores, dtype=np.float32).reshape(-1)
    patch_masks = np.asarray(gt_patch_masks, dtype=np.uint8)
    if scores.size == 0 or patch_masks.ndim != 2 or patch_masks.shape[0] != scores.size:
        return

    # Image is anomaly if any patch is anomaly.
    image_is_anomaly = patch_masks.reshape(patch_masks.shape[0], -1).sum(axis=1) > 0
    normal_scores = scores[~image_is_anomaly]
    anomaly_scores = scores[image_is_anomaly]
    if normal_scores.size == 0 and anomaly_scores.size == 0:
        return

    save_dir = os.path.join(args.save_path, args.dataset, args.noise)
    os.makedirs(save_dir, exist_ok=True)
    if epoch is None:
        filename = "image_norm_scores_distribution_normal_vs_anomaly.png"
    else:
        filename = (
            f"image_norm_scores_distribution_normal_vs_anomaly_epoch_{int(epoch) + 1:03d}.png"
        )
    save_path = os.path.join(save_dir, filename)

    plt.figure(figsize=(10, 6))
    bins = np.linspace(0.0, 1.0, 51)
    if normal_scores.size > 0:
        plt.hist(
            normal_scores,
            bins=bins,
            density=True,
            alpha=0.55,
            color="tab:blue",
            label=f"normal ({normal_scores.size})",
        )
    if anomaly_scores.size > 0:
        plt.hist(
            anomaly_scores,
            bins=bins,
            density=True,
            alpha=0.55,
            color="tab:red",
            label=f"anomaly ({anomaly_scores.size})",
        )

    plt.xlabel("Image normalized score")
    plt.ylabel("Density")
    plt.title("Image Norm Score Distribution: Normal vs Anomaly Images")
    plt.xlim(0.0, 1.0)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[PseudoLabel-Plot] saved: {save_path}")

def precompute_pseudo_labels_feature(
    args,
    localnet,
    mini_loader,
    device,
    update_topk_features_fn,
):
    memory_bank_start = time.perf_counter()

    dataset_size = len(mini_loader.dataset)
    score_stack = np.zeros(dataset_size, dtype=np.float32)
    dim = None

    with torch.no_grad():
        localnet.eval()
        for mini_x, sample_idx in mini_loader:
            features, score = localnet(mini_x.to(device))
            if features.shape[0] > args.batch_size:
                features = features.unsqueeze(0)

            if dim is None:
                dim = int(features.shape[-1])

            score = score.detach().cpu().numpy()
            score = _as_1d_float32(score.max(axis=-1))

            sample_idx_np = sample_idx.detach().cpu().numpy().astype(np.int64)
            score_stack[sample_idx_np] = score.astype(np.float32)

    if dim is None:
        raise RuntimeError("未能从 mini_loader 中获取特征维度。")

    normalized_score = _minmax_normalize(score_stack)

    normal_indices = np.where(normalized_score < 0.5)[0]
    if args.random < 1:
        sample_num = int(normal_indices.shape[0] * args.random)
        selected_normal_indices = np.random.choice(
            normal_indices, size=sample_num, replace=False
        )
    else:
        selected_normal_indices = normal_indices

    selected_normal_indices = np.asarray(selected_normal_indices, dtype=np.int64)

    normal_features = []
    with torch.no_grad():
        localnet.eval()
        for mini_x, sample_idx in mini_loader:
            features, _ = localnet(mini_x.to(device))
            features = features.detach().cpu().numpy()
            sample_idx_np = sample_idx.detach().cpu().numpy().astype(np.int64)

            keep_mask = np.isin(sample_idx_np, selected_normal_indices)
            if np.any(keep_mask):
                normal_features.append(features[keep_mask].reshape(-1, dim))

    if len(normal_features) == 0:
        raise RuntimeError("构建 memory bank 时未选中任何 normal 特征，请检查 threshold/random。")

    normal_features = np.concatenate(normal_features, axis=0)

    faiss.omp_set_num_threads(4)
    index = faiss.GpuIndexFlatL2(
        faiss.StandardGpuResources(),
        dim,
        faiss.GpuIndexFlatConfig(),
    )
    index.add(normal_features)
    memory_bank_time = time.perf_counter() - memory_bank_start

    pseudo_start = time.perf_counter()
    distance_map = np.zeros((dataset_size, 784), dtype=np.float32)
    global_min = np.inf
    global_max = -np.inf

    top_feat = None
    top_dist = None
    with torch.no_grad():
        localnet.eval()
        for mini_x, sample_idx in mini_loader:
            features, _ = localnet(mini_x.to(device))
            features = features.detach().cpu().numpy()
            sample_idx_np = sample_idx.detach().cpu().numpy().astype(np.int64)

            batch_features = features.reshape(-1, dim)
            distance, _ = index.search(np.ascontiguousarray(batch_features), k=args.k_number)
            distance = _postprocess_faiss_distances(distance, args.k_number)
            if distance.size > 0:
                global_min = min(global_min, float(distance.min()))
                global_max = max(global_max, float(distance.max()))
            distance_map[sample_idx_np] = distance.reshape(-1, 784)

            if args.beta:
                high_sample_mask = normalized_score[sample_idx_np] > 0.5
                if np.any(high_sample_mask):
                    high_patch_mask = np.repeat(high_sample_mask, 784)
                    cand_feat = batch_features[high_patch_mask]
                    cand_dist = distance[high_patch_mask]
                    top_feat, top_dist = update_topk_features_fn(
                        top_feat, top_dist, cand_feat, cand_dist, args.beta_number
                    )

    if not np.isfinite(global_min) or not np.isfinite(global_max) or global_max == global_min:
        distance_map = np.zeros_like(distance_map)
    else:
        distance_map = (distance_map - global_min) / (global_max - global_min)

    pseudo_label_time = time.perf_counter() - pseudo_start

    confident_features = None
    if args.beta:
        if top_feat is None or top_feat.shape[0] == 0:
            confident_features = torch.zeros((0, dim), dtype=torch.float32)
        else:
            confident_features = torch.as_tensor(top_feat, dtype=torch.float32)

    return distance_map, confident_features, dim, memory_bank_time, pseudo_label_time


def precompute_pseudo_labels_multiclass(
    args,
    localnet,
    feature_extractor,
    mini_loader,
    num_classes,
    device,
    extract_feature_batch_fn,
    aggregate_image_scores_fn,
    build_faiss_index_fn,
    update_topk_features_fn,
):
    memory_bank_start = time.perf_counter()

    dataset_size = len(mini_loader.dataset)
    image_scores = np.zeros(dataset_size, dtype=np.float32)
    class_stack = np.zeros(dataset_size, dtype=np.int64)
    global_dim = None

    with torch.no_grad():
        localnet.eval()
        feature_extractor.eval()
        for batch in mini_loader:
            images, mini_class_idx, mini_sample_idx = batch[0], batch[1], batch[2]
            images = images.to(device)
            image_features = extract_feature_batch_fn(images, feature_extractor, args)

            features, score = localnet(image_features)
            if features.shape[0] > args.batch_size:
                features = features.unsqueeze(0)

            if global_dim is None:
                global_dim = int(features.shape[-1])

            score = score.detach().cpu().numpy()
            score = _as_1d_float32(
                aggregate_image_scores_fn(score, topk_ratio=args.img_score_topk_ratio)
            )

            sample_idx_np = mini_sample_idx.detach().cpu().numpy().astype(np.int64)
            class_np = mini_class_idx.detach().cpu().numpy().astype(np.int64)
            image_scores[sample_idx_np] = score.astype(np.float32)
            class_stack[sample_idx_np] = class_np

    if global_dim is None:
        raise RuntimeError("未能从 mini_loader 获取到特征维度。")

    image_norm_scores = np.zeros(dataset_size, dtype=np.float32)
    selected_images_by_class = {}

    for class_idx in range(num_classes):
        cls_idx = np.where(class_stack == class_idx)[0]
        if cls_idx.shape[0] == 0:
            selected_images_by_class[class_idx] = np.array([], dtype=np.int64)
            continue

        sampled_num = max(1, int(np.ceil(cls_idx.shape[0] * args.bank_sample_ratio)))
        sampled_num = min(sampled_num, cls_idx.shape[0])
        sampled_cls_idx = np.random.choice(cls_idx, size=sampled_num, replace=False)

        sampled_scores = image_scores[sampled_cls_idx]
        normalized_score = _minmax_normalize(sampled_scores)
        image_norm_scores[sampled_cls_idx] = normalized_score.astype(np.float32)

        normal_indices = np.where(normalized_score < 0.5)[0]
        if normal_indices.shape[0] == 0:
            selected_local = np.arange(sampled_cls_idx.shape[0])
        elif args.random < 1:
            sample_num = max(1, int(normal_indices.shape[0] * args.random))
            selected_local = np.random.choice(
                normal_indices, size=sample_num, replace=False
            )
        else:
            selected_local = normal_indices

        selected_global = sampled_cls_idx[selected_local]
        if args.max_bank_images > 0 and selected_global.shape[0] > args.max_bank_images:
            pick = np.random.choice(
                selected_global.shape[0], size=args.max_bank_images, replace=False
            )
            selected_global = selected_global[pick]
        selected_images_by_class[class_idx] = selected_global.astype(np.int64)

    class_feature_buffers = {class_idx: [] for class_idx in range(num_classes)}
    selected_indices = [
        selected_images_by_class[class_idx]
        for class_idx in range(num_classes)
        if selected_images_by_class[class_idx].shape[0] > 0
    ]

    if len(selected_indices) > 0:
        selected_indices = np.concatenate(selected_indices).astype(np.int64)
        selected_dataset = Subset(mini_loader.dataset, selected_indices.tolist())
        selected_loader = DataLoader(
            selected_dataset,
            batch_size=mini_loader.batch_size,
            pin_memory=mini_loader.pin_memory,
            shuffle=False,
            num_workers=mini_loader.num_workers,
            drop_last=False,
        )

        with torch.no_grad():
            localnet.eval()
            feature_extractor.eval()
            for batch in selected_loader:
                images, mini_class_idx = batch[0], batch[1]
                images = images.to(device)
                image_features = extract_feature_batch_fn(images, feature_extractor, args)
                features, _ = localnet(image_features)

                features_np = features.detach().cpu().numpy()
                class_np = mini_class_idx.detach().cpu().numpy().astype(np.int64)
                for class_idx in np.unique(class_np).tolist():
                    class_idx = int(class_idx)
                    class_mask = class_np == class_idx
                    if not np.any(class_mask):
                        continue
                    cls_features = features_np[class_mask].reshape(-1, global_dim)
                    class_feature_buffers[class_idx].append(cls_features)

    distance_map = np.zeros((dataset_size, 784), dtype=np.float16)
    memory_bank = {}
    confident_feature_bank = {}

    for class_idx in range(num_classes):
        if len(class_feature_buffers[class_idx]) == 0:
            continue
        normal_features = np.concatenate(class_feature_buffers[class_idx], axis=0)
        memory_bank[class_idx] = build_faiss_index_fn(normal_features)

    if args.beta:
        top_feat_by_class = {class_idx: None for class_idx in range(num_classes)}
        top_dist_by_class = {class_idx: None for class_idx in range(num_classes)}

    class_min_distance = {class_idx: np.inf for class_idx in range(num_classes)}
    class_max_distance = {class_idx: -np.inf for class_idx in range(num_classes)}
    memory_bank_time = time.perf_counter() - memory_bank_start

    pseudo_start = time.perf_counter()
    with torch.no_grad():
        localnet.eval()
        feature_extractor.eval()
        for batch in mini_loader:
            images, mini_class_idx, mini_sample_idx = batch[0], batch[1], batch[2]
            images = images.to(device)
            image_features = extract_feature_batch_fn(images, feature_extractor, args)
            features, _ = localnet(image_features)
            features_np = features.detach().cpu().numpy()
            sample_idx_np = mini_sample_idx.detach().cpu().numpy().astype(np.int64)
            class_np = mini_class_idx.detach().cpu().numpy().astype(np.int64)

            for class_idx in np.unique(class_np).tolist():
                class_idx = int(class_idx)
                if class_idx not in memory_bank:
                    continue
                class_mask = class_np == class_idx
                if not np.any(class_mask):
                    continue

                cls_features = features_np[class_mask].reshape(-1, global_dim)
                cls_distance, _ = memory_bank[class_idx].search(
                    np.ascontiguousarray(cls_features), k=args.k_number
                )
                cls_distance = _postprocess_faiss_distances(cls_distance, args.k_number)

                if cls_distance.size > 0:
                    class_min_distance[class_idx] = min(
                        class_min_distance[class_idx], float(cls_distance.min())
                    )
                    class_max_distance[class_idx] = max(
                        class_max_distance[class_idx], float(cls_distance.max())
                    )

                cls_distance_map = cls_distance.reshape(-1, 784)
                cls_sample_ids = sample_idx_np[class_mask]
                distance_map[cls_sample_ids] = cls_distance_map.astype(np.float16)

                if args.beta:
                    high_sample_mask = image_norm_scores[cls_sample_ids] > 0.5
                    if np.any(high_sample_mask):
                        high_patch_mask = np.repeat(high_sample_mask, 784)
                        cand_feat = cls_features[high_patch_mask]
                        cand_dist = cls_distance[high_patch_mask]
                        top_feat_by_class[class_idx], top_dist_by_class[class_idx] = (
                            update_topk_features_fn(
                                top_feat_by_class[class_idx],
                                top_dist_by_class[class_idx],
                                cand_feat,
                                cand_dist,
                                args.beta_number,
                            )
                        )

    for class_idx in range(num_classes):
        cls_idx = np.where(class_stack == class_idx)[0]
        if cls_idx.shape[0] == 0:
            continue

        min_val = class_min_distance[class_idx]
        max_val = class_max_distance[class_idx]
        if not np.isfinite(min_val) or not np.isfinite(max_val) or max_val == min_val:
            distance_map[cls_idx] = 0
        else:
            cls_values = distance_map[cls_idx].astype(np.float32)
            cls_values = (cls_values - min_val) / (max_val - min_val)
            distance_map[cls_idx] = cls_values.astype(np.float16)

        if args.beta:
            top_feat = top_feat_by_class.get(class_idx, None)
            if top_feat is None or top_feat.shape[0] == 0:
                confident_feature_bank[class_idx] = torch.zeros(
                    (0, global_dim), dtype=torch.float32
                )
            else:
                confident_feature_bank[class_idx] = torch.as_tensor(
                    top_feat, dtype=torch.float32
                )

    pseudo_label_time = time.perf_counter() - pseudo_start
    return distance_map, confident_feature_bank, global_dim, memory_bank_time, pseudo_label_time


def precompute_pseudo_labels_multiclass_residual(
    args,
    localnet,
    feature_extractor,
    mini_loader,
    num_classes,
    reference_memory_by_class,
    reference_index_by_class,
    device,
    extract_feature_batch_fn,
    compute_residual_feature_batch_fn,
    aggregate_image_scores_fn,
    build_faiss_index_fn,
    update_topk_features_fn,
    print_selected_score_distribution_fn,
    print_selected_clean_ratio_fn,
    print_confusion_matrix_fn,
    epoch=None,
):
    memory_bank_start = time.perf_counter()

    dataset_size = len(mini_loader.dataset)
    image_scores = np.zeros(dataset_size, dtype=np.float32)
    class_stack = np.zeros(dataset_size, dtype=np.int64)
    gt_patch_masks = np.zeros((dataset_size, 784), dtype=np.uint8)
    global_dim = None

    print("[Phase 1/3] Computing image scores...")
    with torch.no_grad():
        localnet.eval()
        feature_extractor.eval()
        for batch in tqdm.tqdm(mini_loader, desc="Phase 1"):
            if len(batch) == 4:
                images, mini_class_idx, mini_sample_idx, mini_patch_mask = batch
            else:
                images, mini_class_idx, mini_sample_idx = batch
                mini_patch_mask = None
            images = images.to(device)
            image_features = extract_feature_batch_fn(images, feature_extractor, args)
            residual_features = compute_residual_feature_batch_fn(
                image_features,
                mini_class_idx.to(device),
                reference_memory_by_class,
                reference_index_by_class,
            )

            features, score = localnet(residual_features)
            if features.shape[0] > args.batch_size:
                features = features.unsqueeze(0)

            if global_dim is None:
                global_dim = int(features.shape[-1])

            score_np = score.detach().cpu().numpy()
            score_np = _as_1d_float32(
                aggregate_image_scores_fn(
                    score_np,
                    topk_ratio=args.img_score_topk_ratio,
                )
            )

            sample_idx_np = mini_sample_idx.detach().cpu().numpy().astype(np.int64)
            class_np = mini_class_idx.detach().cpu().numpy().astype(np.int64)
            image_scores[sample_idx_np] = score_np.astype(np.float32)
            class_stack[sample_idx_np] = class_np
            if mini_patch_mask is not None:
                mask_np = mini_patch_mask.detach().cpu().numpy()
                gt_patch_masks[sample_idx_np] = (mask_np > 0).astype(np.uint8)
    image_norm_scores = np.zeros(dataset_size, dtype=np.float32)
    normalized_score = np.zeros_like(image_scores, dtype=np.float32)
    #分类归一化
    for cls in range(num_classes):
        cls_mask = class_stack == cls
        if not np.any(cls_mask):
            continue
        cls_scores = image_scores[cls_mask]
        cls_norm = _minmax_normalize(cls_scores)
        normalized_score[cls_mask] = cls_norm.astype(np.float32)

    image_norm_scores[:] = normalized_score.astype(np.float32)
    _plot_image_norm_score_distribution_normal_vs_anomaly(
        args=args,
        image_norm_scores=image_norm_scores,
        gt_patch_masks=gt_patch_masks,
        epoch=epoch,
    )
    
    selected_local_list = []
    print("[Phase 2/3] Building memory bank from normal samples...")
    #下采样选取
    for cls in range(num_classes):
        cls_mask = class_stack == cls
        if not np.any(cls_mask):
            continue
        cls_indices = np.where(cls_mask)[0].astype(np.int64)
        cls_norm_scores = normalized_score[cls_indices]

        cls_normal_local = np.where(cls_norm_scores < 0.5)[0]
        if cls_normal_local.shape[0] == 0:
            cls_selected = cls_indices
        elif args.random < 1:
            cls_sample_num = max(1, int(cls_normal_local.shape[0] * args.random))
            pick_local = np.random.choice(
                cls_normal_local, size=cls_sample_num, replace=False
            )
            cls_selected = cls_indices[pick_local]
        else:
            cls_selected = cls_indices[cls_normal_local]
        selected_local_list.append(cls_selected)

    if len(selected_local_list) == 0:
        selected_indices = np.arange(dataset_size, dtype=np.int64)
    else:
        selected_indices = np.concatenate(selected_local_list, axis=0).astype(np.int64)
    print_selected_score_distribution_fn(
        selected_indices=selected_indices,
        class_stack=class_stack,
        image_scores=image_scores,
    )
    print_selected_clean_ratio_fn(
        selected_indices=selected_indices,
        dataset=mini_loader.dataset,
    )

    distance_map = np.zeros((dataset_size, 784), dtype=np.float16)
    confident_feature_bank = torch.zeros((0, global_dim), dtype=torch.float32)
    if selected_indices.shape[0] == 0:
        pseudo_label_time = time.perf_counter() - memory_bank_start
        return distance_map, confident_feature_bank, global_dim, 0.0, pseudo_label_time

    # 按类别记录被选中的图像索引
    selected_images_by_class = {}
    for cls in range(num_classes):
        cls_mask = class_stack[selected_indices] == cls
        selected_images_by_class[cls] = selected_indices[cls_mask]

    # Phase 2: 提取选中样本的 residual 特征，并按类别缓存
    class_feature_buffers = {cls: [] for cls in range(num_classes)}
    with torch.no_grad():
        localnet.eval()
        feature_extractor.eval()
        selected_subset = Subset(mini_loader.dataset, selected_indices.tolist())
        selected_loader = DataLoader(
            selected_subset,
            batch_size=mini_loader.batch_size,
            pin_memory=mini_loader.pin_memory,
            shuffle=False,
            num_workers=mini_loader.num_workers,
            drop_last=False,
        )
        for batch in tqdm.tqdm(selected_loader, desc="Phase 2"):
            if len(batch) == 4:
                images, mini_class_idx, _mini_sample_idx, _mini_patch_mask = batch
            else:
                images, mini_class_idx, _mini_sample_idx = batch
            images = images.to(device)
            image_features = extract_feature_batch_fn(images, feature_extractor, args)
            residual_features = compute_residual_feature_batch_fn(
                image_features,
                mini_class_idx.to(device),
                reference_memory_by_class,
                reference_index_by_class,
            )
            features, _ = localnet(residual_features)
            features_np = features.detach().cpu().numpy()
            batch_class_np = mini_class_idx.detach().cpu().numpy().astype(np.int64)

            for cls in np.unique(batch_class_np).tolist():
                cls = int(cls)
                cls_mask = batch_class_np == cls
                if not np.any(cls_mask):
                    continue
                cls_features = features_np[cls_mask].reshape(-1, global_dim)
                class_feature_buffers[cls].append(cls_features)

    # 按类别做 GreedyCoreset，下采样到「约等于 2 张图片的 patch 数」
    reduced_features_by_class = {}
    coreset_indices_by_class = {}
    for cls in range(num_classes):
        if len(class_feature_buffers[cls]) == 0:
            continue
        normal_features_cls = np.concatenate(class_feature_buffers[cls], axis=0)

        num_selected_images_cls = selected_images_by_class.get(cls, np.array([], dtype=np.int64)).shape[0]
        if num_selected_images_cls > 0 and normal_features_cls.shape[0] > 0:
            patches_per_image = normal_features_cls.shape[0] // num_selected_images_cls
        else:
            patches_per_image = normal_features_cls.shape[0]

        target_images = min(2, num_selected_images_cls) if num_selected_images_cls > 0 else 1
        target_features = patches_per_image * target_images

        if 0 < target_features < normal_features_cls.shape[0]:
            percentage = float(target_features) / float(normal_features_cls.shape[0])
            sampler = ApproximateGreedyCoresetSampler(
                percentage=percentage,
                device=device,
            )
            normal_features_cls, core_idx = sampler.run(
                normal_features_cls, return_indices=True
            )
        else:
            core_idx = np.arange(normal_features_cls.shape[0], dtype=np.int64)

        reduced_features_by_class[cls] = normal_features_cls
        coreset_indices_by_class[cls] = core_idx

    if len(reduced_features_by_class) > 0:
        print_greedy_memory_bank_anomaly_stats(
            num_classes=num_classes,
            selected_images_by_class=selected_images_by_class,
            gt_patch_masks=gt_patch_masks,
            reduced_features_by_class=reduced_features_by_class,
            coreset_indices_by_class=coreset_indices_by_class,
            class_names=getattr(args, "class_names", None),
            epoch=epoch,
        )

    if len(reduced_features_by_class) == 0:
        pseudo_label_time = time.perf_counter() - memory_bank_start
        return distance_map, confident_feature_bank, global_dim, 0.0, pseudo_label_time
    
    # 按类别构建多个索引：memory_bank_by_class[cls]
    memory_bank_by_class = {}
    for cls, feats in reduced_features_by_class.items():
        if feats is None or feats.shape[0] == 0:
            continue
        memory_bank_by_class[int(cls)] = build_faiss_index_fn(feats)

    if args.beta:
        top_feat_global = None
        top_dist_global = None

    class_min_distance = np.full(num_classes, np.inf, dtype=np.float32)
    class_max_distance = np.full(num_classes, -np.inf, dtype=np.float32)
    memory_bank_time = time.perf_counter() - memory_bank_start

    print("[Phase 3/3] Computing distance map for all patches...")
    pseudo_start = time.perf_counter()
    with torch.no_grad():
        localnet.eval()
        feature_extractor.eval()
        for batch in tqdm.tqdm(mini_loader, desc="Phase 3"):
            if len(batch) == 4:
                images, mini_class_idx, mini_sample_idx, _mini_patch_mask = batch
            else:
                images, mini_class_idx, mini_sample_idx = batch
            images = images.to(device)
            sample_idx_np = mini_sample_idx.detach().cpu().numpy().astype(np.int64)

            image_features = extract_feature_batch_fn(images, feature_extractor, args)
            residual_features = compute_residual_feature_batch_fn(
                image_features,
                mini_class_idx.to(device),
                reference_memory_by_class,
                reference_index_by_class,
            )
            features, _ = localnet(residual_features)
            features_np = features.detach().cpu().numpy()
            batch_class_np = mini_class_idx.detach().cpu().numpy().astype(np.int64)
            batch_size = int(batch_class_np.shape[0])

            # 将距离按 batch 原顺序拼回：shape = (B, 784)
            cls_distance_map = np.zeros((batch_size, 784), dtype=np.float32)

            for cls in np.unique(batch_class_np).tolist():
                cls = int(cls)
                if cls not in memory_bank_by_class:
                    continue
                cls_mask = batch_class_np == cls
                if not np.any(cls_mask):
                    continue

                cls_features_2d = features_np[cls_mask].reshape(-1, global_dim)
                cls_distance, _ = memory_bank_by_class[cls].search(
                    np.ascontiguousarray(cls_features_2d), k=args.k_number
                )
                cls_distance = _postprocess_faiss_distances(cls_distance, args.k_number)

                cls_distance_map[cls_mask] = cls_distance.reshape(-1, 784).astype(
                    np.float32
                )

            row_min = cls_distance_map.min(axis=1)
            row_max = cls_distance_map.max(axis=1)
            for cls in np.unique(batch_class_np):
                cls_mask = batch_class_np == cls
                if not np.any(cls_mask):
                    continue
                cls_row_min = float(row_min[cls_mask].min())
                cls_row_max = float(row_max[cls_mask].max())
                class_min_distance[cls] = min(class_min_distance[cls], cls_row_min)
                class_max_distance[cls] = max(class_max_distance[cls], cls_row_max)

            distance_map[sample_idx_np] = cls_distance_map.astype(np.float16)

            if args.beta:
                high_sample_mask = image_norm_scores[sample_idx_np] > 0.5
                if np.any(high_sample_mask):
                    high_patch_mask = np.repeat(high_sample_mask, 784)
                    features_2d = features_np.reshape(-1, global_dim)
                    cand_feat = features_2d[high_patch_mask]
                    cand_dist = cls_distance_map.reshape(-1)[high_patch_mask]
                    top_feat_global, top_dist_global = update_topk_features_fn(
                        top_feat_global,
                        top_dist_global,
                        cand_feat,
                        cand_dist,
                        args.beta_number,
                    )

    finite_min_mask = np.isfinite(class_min_distance)
    finite_max_mask = np.isfinite(class_max_distance)
    valid_cls_mask = finite_min_mask & finite_max_mask & (
        class_max_distance > class_min_distance
    )

    if not np.any(valid_cls_mask):
        distance_map[:] = 0
    else:
        values = distance_map.astype(np.float32)
        for cls in range(num_classes):
            cls_indices = np.where(class_stack == cls)[0]
            if cls_indices.size == 0:
                continue
            if not (
                np.isfinite(class_min_distance[cls])
                and np.isfinite(class_max_distance[cls])
                and class_max_distance[cls] > class_min_distance[cls]
            ):
                values[cls_indices] = 0.0
                continue

            cls_min = class_min_distance[cls]
            cls_max = class_max_distance[cls]
            cls_values = values[cls_indices]
            cls_values = (cls_values - cls_min) / (cls_max - cls_min)
            values[cls_indices] = cls_values
        distance_map = values.astype(np.float16)

    if args.beta:
        if top_feat_global is None or top_feat_global.shape[0] == 0:
            confident_feature_bank = torch.zeros((0, global_dim), dtype=torch.float32)
        else:
            confident_feature_bank = torch.as_tensor(top_feat_global, dtype=torch.float32)

    # 统计伪标签分配准确率（基于真实 patch mask）
    values = distance_map.astype(np.float32)
    gt_mask_bool = gt_patch_masks.astype(bool)

    pred_normal = values < float(args.threshold)
    pred_anomaly = values > float(args.noise_threshold)

    pred_normal_count = int(pred_normal.sum())
    pred_anomaly_count = int(pred_anomaly.sum())
    pred_normal_correct = int(np.logical_and(pred_normal, ~gt_mask_bool).sum())
    pred_anomaly_correct = int(np.logical_and(pred_anomaly, gt_mask_bool).sum())

    normal_precision = (
        float(pred_normal_correct) / float(pred_normal_count)
        if pred_normal_count > 0
        else 0.0
    )
    anomaly_precision = (
        float(pred_anomaly_correct) / float(pred_anomaly_count)
        if pred_anomaly_count > 0
        else 0.0
    )
    print(
        "[PseudoLabel-Stats] "
        f"distance<threshold normal-precision: {normal_precision:.4f} "
        f"({pred_normal_correct}/{pred_normal_count}) | "
        f"distance>noise_threshold anomaly-precision: {anomaly_precision:.4f} "
        f"({pred_anomaly_correct}/{pred_anomaly_count})"
    )
    _plot_distance_distribution_normal_vs_anomaly(
        args=args,
        distance_values=values,
        gt_patch_masks=gt_patch_masks,
        epoch=epoch,
    )

    pseudo_label_time = time.perf_counter() - pseudo_start
    return distance_map, confident_feature_bank, global_dim, memory_bank_time, pseudo_label_time
