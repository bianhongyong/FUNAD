import argparse
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import to_rgb
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

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
    parser = argparse.ArgumentParser("t-SNE visualization for MVTec using online feature inference")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Dataset root path (e.g., /path/to/mvtec)",
    )
    parser.add_argument("--output", type=str, default="tsne.png", help="Output image path")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["cable","capsule","bottle"],
        help="Class names to visualize from test features, e.g. --classes bottle cable",
    )
    parser.add_argument("--max_per_group", type=int, default=100, help="Max patch samples per group")
    parser.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for feature inference")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument(
        "--use_cls_token",
        type=str2bool,
        default=True,
        help="Whether to concatenate cls token with patch tokens",
    )
    parser.add_argument(
        "--feature_source",
        type=str,
        choices=["encoder", "adaptor"],
        default="encoder",
        help="Choose t-SNE feature source: dino encoder output or feature adaptor output",
    )
    parser.add_argument(
        "--adaptor_ckpt",
        type=str,
        default="/media/honeywell/E/bhy/FUNAD/save_results/results/bottle/gaussian_True_noise_10%_balancing_True_oto_True_weight_2.5_synthetic_False_localnet.pt",
        help="Path to trained localnet checkpoint (.pt) containing key 'net'",
    )
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


def _extract_dino_features(images: torch.Tensor, dino: torch.nn.Module, use_cls_token: bool) -> torch.Tensor:
    with torch.no_grad():
        dino.eval()
        feature = dino.get_intermediate_layers(images)[0]
        patch_tokens = feature[:, 1:, :]
        if use_cls_token:
            cls_tokens = feature[:, 0, :]
            cls_tokens = torch.repeat_interleave(cls_tokens.unsqueeze(1), patch_tokens.shape[1], dim=1)
            patch_tokens = torch.cat([cls_tokens, patch_tokens], dim=-1)
    return patch_tokens


def _build_adaptor(feature_dim: int, ckpt_path: str, device: torch.device) -> torch.nn.Module:
    localnet = model.localnet(len_feature=feature_dim)
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["net"] if isinstance(ckpt, dict) and "net" in ckpt else ckpt
    localnet.load_state_dict(state_dict)
    localnet = localnet.to(device)
    localnet.eval()
    return localnet.adaptor


def _subsample(features: np.ndarray, max_count: int, rng: np.random.Generator) -> np.ndarray:
    if max_count <= 0 or features.shape[0] <= max_count:
        return features
    indices = rng.choice(features.shape[0], size=max_count, replace=False)
    return features[indices]


def build_groups(
    data_path: str,
    class_names: List[str],
    max_per_group: int,
    seed: int,
    batch_size: int,
    num_workers: int,
    use_cls_token: bool,
    feature_source: str,
    adaptor_ckpt: str,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    groups: Dict[str, np.ndarray] = {}
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    dino = torch.hub.load("facebookresearch/dino:main", "dino_vitb8").to(device)
    dino.eval()

    adaptor = None

    for class_name in class_names:
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

        for images, _, masks in test_loader:
            images = images.to(device)
            enc_features = _extract_dino_features(images, dino, use_cls_token=use_cls_token)

            if feature_source == "adaptor":
                if adaptor is None:
                    if adaptor_ckpt is None:
                        raise ValueError("feature_source=adaptor 时必须提供 --adaptor_ckpt")
                    adaptor = _build_adaptor(
                        feature_dim=int(enc_features.shape[-1]),
                        ckpt_path=adaptor_ckpt,
                        device=device,
                    )
                with torch.no_grad():
                    feat = adaptor(enc_features)
            else:
                feat = enc_features

            feat_np = feat.detach().cpu().numpy().reshape(-1, feat.shape[-1])
            mask_np = masks.detach().cpu().numpy()
            patch_labels = _mask_to_patch_labels(mask_np, num_patches=feat.shape[1])

            patch_features_all.append(feat_np)
            patch_labels_all.append(patch_labels)

        if len(patch_features_all) == 0:
            raise ValueError(f"No test samples found for class: {class_name}")

        feat = np.concatenate(patch_features_all, axis=0)
        gt = np.concatenate(patch_labels_all, axis=0)

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

    plt.title("t-SNE of MVTec Patch Features")
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
        data_path=args.data_path,
        class_names=class_names,
        max_per_group=args.max_per_group,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_cls_token=args.use_cls_token,
        feature_source=args.feature_source,
        adaptor_ckpt=args.adaptor_ckpt,
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

    print(f"Feature source: {args.feature_source}")
    print(f"Saved t-SNE figure to: {args.output}")
    if args.save_npz is not None:
        print(f"Saved embedding data to: {args.save_npz}")


if __name__ == "__main__":
    main()
