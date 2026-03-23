import argparse
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from sklearn.manifold import TSNE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("t-SNE visualization for MVTec bottle/cable normal vs anomaly")
    parser.add_argument(
        "--feature_path",
        type=str,
        required=True,
        help="Directory containing files like bottle_test.npy, bottle_gt.npy, cable_test.npy, cable_gt.npy",
    )
    parser.add_argument("--output", type=str, default="tsne_bottle_cable.png", help="Output image path")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["bottle", "cable"],
        help="Class names to visualize from test features, e.g. --classes bottle cable",
    )
    parser.add_argument("--max_per_group", type=int, default=500, help="Max patch samples per group")
    parser.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--save_npz",
        type=str,
        default=None,
        help="Optional path to save 2D embeddings and group labels as npz",
    )
    return parser.parse_args()


def _mask_to_patch_labels(mask: np.ndarray, num_patches: int) -> np.ndarray:
    if mask.ndim == 4 and mask.shape[1] == 1:
        mask = mask[:, 0]
    elif mask.ndim == 4 and mask.shape[-1] == 1:
        mask = mask[..., 0]

    if mask.ndim != 3:
        raise ValueError(f"Unexpected mask shape: {mask.shape}")

    num_images, height, width = mask.shape
    grid_size = int(np.sqrt(num_patches))
    if grid_size * grid_size != num_patches:
        raise ValueError(f"num_patches={num_patches} is not a square number")

    if height % grid_size != 0 or width % grid_size != 0:
        raise ValueError(
            f"Mask size {(height, width)} is not divisible by patch grid {(grid_size, grid_size)}"
        )

    patch_h = height // grid_size
    patch_w = width // grid_size
    patch_mask = mask.reshape(num_images, grid_size, patch_h, grid_size, patch_w).max(axis=(2, 4))
    return (patch_mask.reshape(-1) > 0).astype(np.int32)


def _to_patch_level(features: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if features.ndim < 2:
        raise ValueError(f"Unexpected feature shape: {features.shape}")

    labels = labels.reshape(-1).astype(int)
    num_images = features.shape[0]
    if num_images != labels.shape[0]:
        raise ValueError(f"Mismatched sample count: features={num_images}, labels={labels.shape[0]}")

    if features.ndim == 2:
        return features, (labels != 0).astype(np.int32)

    if mask.shape[0] != num_images:
        raise ValueError(f"Mismatched sample count: features={num_images}, mask={mask.shape[0]}")

    if features.ndim == 3:
        num_patches = features.shape[1]
        patch_features = features.reshape(num_images * num_patches, features.shape[2])
        patch_labels = _mask_to_patch_labels(mask, num_patches)
        if patch_features.shape[0] != patch_labels.shape[0]:
            raise ValueError(
                f"Mismatched patch count: features={patch_features.shape[0]}, patch_labels={patch_labels.shape[0]}"
            )
        return patch_features, patch_labels

    feature_dim = int(np.prod(features.shape[2:]))
    num_patches = features.shape[1]
    patch_features = features.reshape(num_images * num_patches, feature_dim)
    patch_labels = _mask_to_patch_labels(mask, num_patches)
    if patch_features.shape[0] != patch_labels.shape[0]:
        raise ValueError(
            f"Mismatched patch count: features={patch_features.shape[0]}, patch_labels={patch_labels.shape[0]}"
        )
    return patch_features, patch_labels


def _load_class_test_data(feature_path: str, class_name: str) -> Tuple[np.ndarray, np.ndarray]:
    test_path = os.path.join(feature_path, f"{class_name}_test.npy")
    gt_path = os.path.join(feature_path, f"{class_name}_gt.npy")
    mask_path = os.path.join(feature_path, f"{class_name}_mask.npy")

    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing feature file: {test_path}")
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Missing gt file: {gt_path}")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Missing mask file: {mask_path}")

    features = np.load(test_path)
    labels = np.load(gt_path)
    mask = np.load(mask_path)
    patch_features, patch_labels = _to_patch_level(features, labels, mask)
    return patch_features, patch_labels


def _subsample(features: np.ndarray, max_count: int, rng: np.random.Generator) -> np.ndarray:
    if max_count <= 0 or features.shape[0] <= max_count:
        return features
    indices = rng.choice(features.shape[0], size=max_count, replace=False)
    return features[indices]


def build_groups(feature_path: str, class_names: List[str], max_per_group: int, seed: int) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    groups: Dict[str, np.ndarray] = {}

    for class_name in class_names:
        feat, gt = _load_class_test_data(feature_path, class_name)
        normal = feat[gt == 0]
        anomaly = feat[gt != 0]

        if normal.shape[0] == 0:
            raise ValueError(f"No normal samples found for class: {class_name}")
        if anomaly.shape[0] == 0:
            raise ValueError(f"No anomaly samples found for class: {class_name}")

        groups[f"{class_name}-normal"] = _subsample(normal, max_per_group, rng)
        groups[f"{class_name}-anomaly"] = _subsample(anomaly, max_per_group, rng)

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


def plot_embeddings(embedding: np.ndarray, group_ids: np.ndarray, group_names: np.ndarray, output_path: str) -> None:
    class_names = []
    for group_name in group_names:
        class_name = str(group_name).rsplit("-", 1)[0]
        if class_name not in class_names:
            class_names.append(class_name)

    cmap = plt.get_cmap("tab10")
    class_color = {name: to_rgb(cmap(i % 10)) for i, name in enumerate(class_names)}

    def _shift_lightness(color: Tuple[float, float, float], factor: float) -> Tuple[float, float, float]:
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
        plt.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=18,
            alpha=0.75,
            c=[color],
            marker="o",
            label=group_name,
        )

    plt.title("t-SNE of DINO ViT-B/8 Features (MVTec)")
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    args = parse_args()
    class_names = list(dict.fromkeys(args.classes))
    if len(class_names) == 0:
        raise ValueError("--classes must contain at least one class name")

    groups = build_groups(
        feature_path=args.feature_path,
        class_names=class_names,
        max_per_group=args.max_per_group,
        seed=args.seed,
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

    embedding = run_tsne(all_features, perplexity=args.perplexity, seed=args.seed)
    plot_embeddings(embedding, all_group_ids, group_names, args.output)

    if args.save_npz is not None:
        np.savez(args.save_npz, embedding=embedding, group_ids=all_group_ids, group_names=group_names)

    print(f"Saved t-SNE figure to: {args.output}")
    if args.save_npz is not None:
        print(f"Saved embedding data to: {args.save_npz}")


if __name__ == "__main__":
    main()
