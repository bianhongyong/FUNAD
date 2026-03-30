import argparse
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from matplotlib.colors import to_rgb
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Subset

import dataset_extract
import model


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y", "t"}:
        return True
    if value in {"false", "0", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected, e.g. true/false")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("t-SNE visualization for residual features")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/media/honeywell/D/bhy/dataset/MVTec",
        help="Dataset root path (e.g., /path/to/mvtec)",
    )
    parser.add_argument(
        "--output", type=str, default="tsne_residual.png", help="Output image path"
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=[
            "bottle",
            "cable",
            "capsule",
            "carpet",
            "grid",
            "hazelnut",
            "leather",
            "metal_nut",
            "pill",
            "screw",
            "tile",
            "toothbrush",
            "transistor",
            "wood",
            "zipper",
        ],
        help="Class names to visualize",
    )
    parser.add_argument(
        "--max_per_group", type=int, default=50, help="Max patch samples per group"
    )
    parser.add_argument(
        "--perplexity", type=float, default=30.0, help="t-SNE perplexity"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for feature inference"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of dataloader workers"
    )
    parser.add_argument(
        "--use_cls_token",
        type=str2bool,
        default="false",
        help="Whether to concatenate cls token with patch tokens",
    )
    parser.add_argument(
        "--num_reference_images",
        type=int,
        default=16,
        help="Number of normal images to use as reference from training set",
    )
    parser.add_argument(
        "--avg_batch_feature_before_knn",
        type=str2bool,
        default=True,
        help=(
            "Average reference encoder patch features over batch dimension while "
            "building memory before 1-NN matching (reduces search size)."
        ),
    )
    parser.add_argument(
        "--save_npz",
        type=str,
        default=None,
        help="Optional path to save 2D embeddings and group labels as npz",
    )
    parser.add_argument(
        "--save_tsne_features",
        type=str,
        default="tsne_residual_features_2d.npz",
        help="Path to save t-SNE 2D features with normal/anomaly labels",
    )
    parser.add_argument(
        "--distance_output",
        type=str,
        default="tsne_distance_distribution.png",
        help="Output path for distance-to-origin distribution figure",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="tsne_outputs",
        help="Directory to save all generated files",
    )
    return parser.parse_args()


def _extract_dino_features(
    images: torch.Tensor, dino: torch.nn.Module, use_cls_token: bool
) -> torch.Tensor:
    with torch.no_grad():
        dino.eval()
        feature = dino.get_intermediate_layers(images)[0]
        patch_tokens = feature[:, 1:, :]
        if use_cls_token:
            cls_tokens = feature[:, 0, :]
            cls_tokens = torch.repeat_interleave(
                cls_tokens.unsqueeze(1), patch_tokens.shape[1], dim=1
            )
            patch_tokens = torch.cat([cls_tokens, patch_tokens], dim=-1)
    return patch_tokens


def _subsample(
    features: np.ndarray, max_count: int, rng: np.random.Generator
) -> np.ndarray:
    if max_count <= 0 or features.shape[0] <= max_count:
        return features
    indices = rng.choice(features.shape[0], size=max_count, replace=False)
    return features[indices]


def build_reference_memory(
    data_path: str,
    class_name: str,
    num_reference_images: int,
    reference_batch_size: int,
    avg_batch_feature_before_knn: bool,
    dino: torch.nn.Module,
    use_cls_token: bool,
    device: torch.device,
) -> np.ndarray:
    train_set = dataset_extract.MyDataset(
        dataset_path=data_path,
        dataset="mvtec",
        class_name=class_name,
        is_train=True,
    )

    rng = np.random.default_rng(42)
    num_samples = min(num_reference_images, len(train_set))
    selected_indices = rng.choice(len(train_set), size=num_samples, replace=False)
    selected_subset = Subset(train_set, selected_indices.tolist())

    reference_patches = []
    train_loader = DataLoader(
        selected_subset,
        batch_size=max(1, reference_batch_size),
        shuffle=False,
    )
    for image in tqdm.tqdm(train_loader, desc=f"构建{class_name}参考记忆"):
        image = image.to(device)
        features = _extract_dino_features(image, dino, use_cls_token=use_cls_token)
        if avg_batch_feature_before_knn and features.shape[0] > 1:
            # Build a compact reference memory by averaging features within each batch.
            features = features.mean(dim=0, keepdim=True)
        features_np = features.detach().cpu().numpy().reshape(-1, features.shape[-1])
        reference_patches.append(features_np)

    reference_memory = np.concatenate(reference_patches, axis=0)
    return reference_memory


def compute_residual_features(
    test_features: np.ndarray,
    reference_memory: np.ndarray,
) -> np.ndarray:
    residual_features = []
    test_features_tensor = torch.from_numpy(test_features).float()
    reference_memory_tensor = torch.from_numpy(reference_memory).float()

    for i in tqdm.tqdm(range(test_features.shape[0]), desc="计算残差特征"):
        test_patch = test_features_tensor[i : i + 1]
        distances = torch.cdist(test_patch, reference_memory_tensor).squeeze()
        nearest_idx = torch.argmin(distances).item()
        nearest_neighbor = reference_memory[nearest_idx]
        residual = test_features[i] - nearest_neighbor
        residual_features.append(residual)

    return np.array(residual_features)


def build_groups(
    data_path: str,
    class_names: List[str],
    max_per_group: int,
    seed: int,
    batch_size: int,
    num_workers: int,
    use_cls_token: bool,
    num_reference_images: int,
    avg_batch_feature_before_knn: bool,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    groups: Dict[str, np.ndarray] = {}
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    dino = torch.hub.load("facebookresearch/dino:main", "dino_vitb8").to(device)
    dino.eval()

    for class_name in tqdm.tqdm(class_names, desc="处理类别"):
        reference_memory = build_reference_memory(
            data_path=data_path,
            class_name=class_name,
            num_reference_images=num_reference_images,
            reference_batch_size=batch_size,
            avg_batch_feature_before_knn=avg_batch_feature_before_knn,
            dino=dino,
            use_cls_token=use_cls_token,
            device=device,
        )

        test_set = dataset_extract.MyDataset(
            dataset_path=data_path,
            dataset="mvtec",
            class_name=class_name,
            is_train=False,
        )
        test_loader = DataLoader(
            test_set,
            batch_size=batch_size,
            pin_memory=use_cuda,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )

        patch_features_all = []
        patch_labels_all = []

        for images, _, masks in tqdm.tqdm(test_loader, desc=f"提取{class_name}特征"):
            images = images.to(device)
            enc_features = _extract_dino_features(
                images, dino, use_cls_token=use_cls_token
            )

            feat_np = (
                enc_features.detach().cpu().numpy().reshape(-1, enc_features.shape[-1])
            )
            patch_features_all.append(feat_np)

            mask_np = masks.detach().cpu().numpy()
            num_patches = enc_features.shape[1]
            patch_h = mask_np.shape[2] // int(np.sqrt(num_patches))
            patch_w = mask_np.shape[3] // int(np.sqrt(num_patches))
            grid_size = int(np.sqrt(num_patches))
            patch_mask = mask_np.reshape(
                mask_np.shape[0], 1, grid_size, patch_h, grid_size, patch_w
            ).max(axis=(3, 5))
            patch_labels = patch_mask.reshape(-1).astype(np.int32)
            patch_labels_all.append(patch_labels)

        if len(patch_features_all) == 0:
            raise ValueError(f"No test samples found for class: {class_name}")

        feat = np.concatenate(patch_features_all, axis=0)
        gt = np.concatenate(patch_labels_all, axis=0)

        normal_mask = gt == 0
        anomaly_mask = gt != 0

        if normal_mask.sum() == 0:
            raise ValueError(f"No normal samples found for class: {class_name}")
        if anomaly_mask.sum() == 0:
            raise ValueError(f"No anomaly samples found for class: {class_name}")

        normal_features = feat[normal_mask]
        anomaly_features = feat[anomaly_mask]

        normal_features_sub = _subsample(normal_features, max_per_group, rng)
        anomaly_features_sub = _subsample(anomaly_features, max_per_group, rng)

        normal_residual = compute_residual_features(
            normal_features_sub, reference_memory
        )
        anomaly_residual = compute_residual_features(
            anomaly_features_sub, reference_memory
        )

        groups[f"{class_name}-normal"] = normal_residual
        groups[f"{class_name}-anomaly"] = anomaly_residual

    return groups


def run_tsne(features: np.ndarray, perplexity: float, seed: int) -> np.ndarray:
    n_samples = features.shape[0]
    if n_samples < 3:
        raise ValueError("Need at least 3 samples for t-SNE")

    valid_perplexity = min(perplexity, float(n_samples - 1))
    if valid_perplexity < 2.0:
        valid_perplexity = 2.0

    tsne = TSNE(
        n_components=2,
        perplexity=valid_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    return tsne.fit_transform(features)


def plot_embeddings(
    embedding: np.ndarray,
    group_ids: np.ndarray,
    group_names: np.ndarray,
    output_path: str,
) -> None:
    class_names = []
    for group_name in group_names:
        class_name = str(group_name).rsplit("-", 1)[0]
        if class_name not in class_names:
            class_names.append(class_name)

    cmap = plt.get_cmap("tab10")
    class_color = {name: to_rgb(cmap(i % 10)) for i, name in enumerate(class_names)}

    def _shift_lightness(
        color: Tuple[float, float, float], factor: float
    ) -> Tuple[float, float, float]:
        rgb = np.array(color)
        if factor >= 1.0:
            shifted = rgb + (1.0 - rgb) * (factor - 1.0)
        else:
            shifted = rgb * factor
        shifted = np.clip(shifted, 0.0, 1.0)
        return float(shifted[0]), float(shifted[1]), float(shifted[2])

    plt.figure(figsize=(8, 6))
    for idx, group_name in enumerate(group_names):
        mask = group_ids == idx
        group_name = str(group_name)
        class_name, status = group_name.rsplit("-", 1)
        base_color = class_color.get(class_name, (0.5, 0.5, 0.5))
        color = _shift_lightness(base_color, 1.25 if status == "normal" else 0.70)
        marker = "^" if status == "anomaly" else "o"
        plt.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=18,
            alpha=0.75,
            c=[color],
            marker=marker,
            label=group_name,
        )

    plt.title("t-SNE of Residual Features (Reference Memory)")
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    #plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def build_binary_labels(
    group_ids: np.ndarray, group_names: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    point_group_names = group_names[group_ids]
    point_status = np.array(
        [str(name).rsplit("-", 1)[-1] for name in point_group_names], dtype=object
    )
    binary_labels = np.array([1 if status == "anomaly" else 0 for status in point_status])
    return binary_labels, point_status


def plot_distance_distribution(
    embedding: np.ndarray, binary_labels: np.ndarray, output_path: str
) -> np.ndarray:
    distances = np.linalg.norm(embedding, axis=1)
    normal_distances = distances[binary_labels == 0]
    anomaly_distances = distances[binary_labels == 1]

    plt.figure(figsize=(8, 6))
    bins = 40
    if normal_distances.size > 0:
        plt.hist(
            normal_distances,
            bins=bins,
            alpha=0.55,
            color="#1f77b4",
            density=True,
            label="normal",
        )
    if anomaly_distances.size > 0:
        plt.hist(
            anomaly_distances,
            bins=bins,
            alpha=0.55,
            color="#d62728",
            density=True,
            label="anomaly",
        )

    plt.title("Distance Distribution to Origin in t-SNE Space")
    plt.xlabel("Distance to origin (0, 0)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return distances


def summarize_distance_means(
    distances: np.ndarray, binary_labels: np.ndarray
) -> Tuple[float, float]:
    normal_distances = distances[binary_labels == 0]
    anomaly_distances = distances[binary_labels == 1]

    normal_mean = float(normal_distances.mean()) if normal_distances.size > 0 else float("nan")
    anomaly_mean = (
        float(anomaly_distances.mean()) if anomaly_distances.size > 0 else float("nan")
    )
    return normal_mean, anomaly_mean


def summarize_distance_means_per_base_class(
    distances: np.ndarray,
    all_group_ids: np.ndarray,
    group_names: np.ndarray,
    base_class_names: List[str],
) -> List[Tuple[str, float, float]]:
    """Mean distance to origin per dataset class (normal vs anomaly subgroups)."""
    names_arr = np.asarray(group_names)
    name_to_idx = {str(names_arr[i]): i for i in range(len(names_arr))}
    out: List[Tuple[str, float, float]] = []
    for cn in base_class_names:
        idx_n = name_to_idx.get(f"{cn}-normal")
        idx_a = name_to_idx.get(f"{cn}-anomaly")
        d_n = (
            distances[all_group_ids == idx_n]
            if idx_n is not None
            else np.array([], dtype=np.float64)
        )
        d_a = (
            distances[all_group_ids == idx_a]
            if idx_a is not None
            else np.array([], dtype=np.float64)
        )
        mn = float(d_n.mean()) if d_n.size > 0 else float("nan")
        ma = float(d_a.mean()) if d_a.size > 0 else float("nan")
        out.append((cn, mn, ma))
    return out


def _append_suffix_to_stem(file_path: str, suffix: str) -> str:
    stem, ext = os.path.splitext(os.path.basename(file_path))
    return f"{stem}{suffix}{ext}"


def main() -> None:
    args = parse_args()
    class_names = list(dict.fromkeys(args.classes))
    if len(class_names) == 0:
        raise ValueError("--classes must contain at least one class name")

    groups = build_groups(
        data_path=args.data_path,
        class_names=class_names,
        max_per_group=args.max_per_group,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_cls_token=args.use_cls_token,
        num_reference_images=args.num_reference_images,
        avg_batch_feature_before_knn=args.avg_batch_feature_before_knn,
    )

    group_names = np.array(list(groups.keys()))
    features_list = []
    group_ids_list = []

    for idx, name in enumerate(group_names):
        feat = groups[name]
        features_list.append(feat)
        group_ids_list.append(np.full(feat.shape[0], idx, dtype=np.int32))

    all_features = np.concatenate(features_list, axis=0)
    all_group_ids = np.concatenate(group_ids_list, axis=0)

    os.makedirs(args.output_dir, exist_ok=True)
    ref_suffix = f"_ref{args.num_reference_images}"
    tsne_output_path = os.path.join(
        args.output_dir, _append_suffix_to_stem(args.output, ref_suffix)
    )
    distance_output_path = os.path.join(
        args.output_dir, _append_suffix_to_stem(args.distance_output, ref_suffix)
    )
    tsne_features_path = os.path.join(
        args.output_dir, os.path.basename(args.save_tsne_features)
    )
    save_npz_path = (
        os.path.join(args.output_dir, os.path.basename(args.save_npz))
        if args.save_npz is not None
        else None
    )

    embedding = run_tsne(all_features, perplexity=args.perplexity, seed=args.seed)
    plot_embeddings(embedding, all_group_ids, group_names, tsne_output_path)
    binary_labels, status_labels = build_binary_labels(all_group_ids, group_names)
    distances = plot_distance_distribution(
        embedding, binary_labels, distance_output_path
    )
    normal_mean_distance, anomaly_mean_distance = summarize_distance_means(
        distances, binary_labels
    )
    per_class_means = summarize_distance_means_per_base_class(
        distances, all_group_ids, group_names, class_names
    )

    np.savez(
        tsne_features_path,
        embedding=embedding,
        group_ids=all_group_ids,
        group_names=group_names,
        binary_labels=binary_labels,
        status_labels=status_labels,
        distances_to_origin=distances,
        mean_normal_distance_to_origin=normal_mean_distance,
        mean_anomaly_distance_to_origin=anomaly_mean_distance,
    )

    if save_npz_path is not None:
        np.savez(
            save_npz_path,
            embedding=embedding,
            group_ids=all_group_ids,
            group_names=group_names,
        )

    print(f"Saved t-SNE figure to: {tsne_output_path}")
    print(f"Saved distance distribution figure to: {distance_output_path}")
    print(f"Mean distance to origin | normal (all classes): {normal_mean_distance:.6f}")
    print(f"Mean distance to origin | anomaly (all classes): {anomaly_mean_distance:.6f}")
    print("Mean distance to origin | per class:")
    for cn, mn, ma in per_class_means:
        print(f"  {cn} | normal: {mn:.6f} | anomaly: {ma:.6f}")
    print(f"Saved t-SNE 2D features to: {tsne_features_path}")
    if save_npz_path is not None:
        print(f"Saved embedding data to: {save_npz_path}")


if __name__ == "__main__":
    main()
