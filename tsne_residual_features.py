import argparse
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from matplotlib.colors import to_rgb
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Subset

import dataset_extract
import model
from multiclass_feature_dataset import get_all_class_names

VALID_STAGES = {"before", "after"}


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
        default=4,
        help="Number of normal images to use as reference from training set",
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
        "--difference_output",
        type=str,
        default="tsne_center_shift.png",
        help="Output path for center-shift difference figure (before->after)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="tsne_outputs",
        help="Directory to save all generated files",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional localnet checkpoint (.pt) to visualize adapter before/after residual features",
    )
    parser.add_argument(
        "--checkpoint_key",
        type=str,
        default="net",
        help="State-dict key in checkpoint for localnet weights",
    )
    parser.add_argument(
        "--report_stage",
        type=str,
        default="before",
        choices=["before", "after"],
        help=(
            "With --checkpoint, two distance histograms are saved (*_before / *_after); "
            "this chooses the stage for npz primary stats and the per-class distance table."
        ),
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
        if features.shape[0] > 1:
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


def load_adapter_from_checkpoint(
    checkpoint_path: str, checkpoint_key: str, feature_dim: int, device: torch.device
) -> torch.nn.Module:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and checkpoint_key in checkpoint:
        state_dict = checkpoint[checkpoint_key]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError(
            "Unsupported checkpoint format: expected dict or dict with a state_dict key."
        )

    adapter = model.localnet(len_feature=feature_dim).to(device)
    missing_keys, unexpected_keys = adapter.load_state_dict(state_dict, strict=False)
    if len(missing_keys) > 0 or len(unexpected_keys) > 0:
        print(
            f"[warn] checkpoint load mismatch | missing={missing_keys} unexpected={unexpected_keys}"
        )
    adapter.eval()
    return adapter


def load_reference_memory_by_class_from_checkpoint(
    checkpoint_path: str, device: torch.device
) -> Dict[int, np.ndarray]:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    raw = checkpoint.get("reference_memory_by_class")
    if raw is None:
        return {}

    out: Dict[int, np.ndarray] = {}
    for class_idx, mem in raw.items():
        cls = int(class_idx)
        mem_np = np.ascontiguousarray(np.asarray(mem, dtype=np.float32))
        if mem_np.size == 0:
            continue
        out[cls] = mem_np
    return out


def apply_adapter_to_residual(
    residual_features: np.ndarray, adapter: torch.nn.Module, device: torch.device
) -> np.ndarray:
    if residual_features.shape[0] == 0:
        return residual_features

    with torch.no_grad():
        x = torch.from_numpy(residual_features).float().to(device).unsqueeze(1)
        adapted, _ = adapter(x)
        adapted_np = adapted.squeeze(1).detach().cpu().numpy()
    return adapted_np


def parse_group_name(group_name: str) -> Tuple[str, str, str]:
    parts = str(group_name).split("-")
    if len(parts) >= 3 and parts[-1] in VALID_STAGES:
        stage = parts[-1]
        status = parts[-2]
        class_name = "-".join(parts[:-2])
        return class_name, status, stage
    class_name, status = str(group_name).rsplit("-", 1)
    return class_name, status, "before"


def _make_group_key(class_name: str, status: str, stage: str) -> str:
    return f"{class_name}-{status}-{stage}"


def build_groups(
    data_path: str,
    class_names: List[str],
    max_per_group: int,
    seed: int,
    batch_size: int,
    num_workers: int,
    use_cls_token: bool,
    num_reference_images: int,
    adapter: torch.nn.Module = None,
    device: torch.device = torch.device("cpu"),
    checkpoint_reference_memory_by_class: Dict[int, np.ndarray] = None,
    class_to_idx: Dict[str, int] = None,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    groups: Dict[str, np.ndarray] = {}
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    dino = torch.hub.load("facebookresearch/dino:main", "dino_vitb8").to(device)
    dino.eval()

    for class_name in tqdm.tqdm(class_names, desc="处理类别"):
        class_idx = class_to_idx.get(class_name) if class_to_idx is not None else None
        if (
            checkpoint_reference_memory_by_class is not None
            and class_idx is not None
            and class_idx in checkpoint_reference_memory_by_class
        ):
            reference_memory = checkpoint_reference_memory_by_class[class_idx]
        else:
            reference_memory = build_reference_memory(
                data_path=data_path,
                class_name=class_name,
                num_reference_images=num_reference_images,
                reference_batch_size=batch_size,
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

        groups[_make_group_key(class_name, "normal", "before")] = normal_residual
        groups[_make_group_key(class_name, "anomaly", "before")] = anomaly_residual

        if adapter is not None:
            normal_after = apply_adapter_to_residual(normal_residual, adapter, device)
            anomaly_after = apply_adapter_to_residual(anomaly_residual, adapter, device)
            groups[_make_group_key(class_name, "normal", "after")] = normal_after
            groups[_make_group_key(class_name, "anomaly", "after")] = anomaly_after

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
    *,
    filter_stage: Optional[str] = None,
    title: Optional[str] = None,
) -> None:
    if filter_stage is not None and filter_stage not in VALID_STAGES:
        raise ValueError(f"filter_stage must be one of {VALID_STAGES}, got {filter_stage}")
    class_names = []
    for group_name in group_names:
        class_name, _, _ = parse_group_name(str(group_name))
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
        group_name = str(group_name)
        class_name, status, stage = parse_group_name(group_name)
        if filter_stage is not None and stage != filter_stage:
            continue
        mask = group_ids == idx
        if not mask.any():
            continue
        base_color = class_color.get(class_name, (0.5, 0.5, 0.5))
        color = _shift_lightness(base_color, 1.25 if status == "normal" else 0.70)
        marker = "^" if status == "anomaly" else "o"
        # Single-stage figures use one alpha; combined plot (no filter) distinguishes before/after.
        if filter_stage is not None:
            alpha = 0.75
        else:
            alpha = 0.40 if stage == "before" else 0.85
        plt.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=18,
            alpha=alpha,
            c=[color],
            marker=marker,
            label=group_name,
        )

    plt.title(
        title
        if title is not None
        else "t-SNE of Residual Features (Reference Memory)"
    )
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    #plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_center_shift(
    embedding: np.ndarray,
    group_ids: np.ndarray,
    group_names: np.ndarray,
    output_path: str,
) -> None:
    class_names = []
    for group_name in group_names:
        class_name, _, _ = parse_group_name(str(group_name))
        if class_name not in class_names:
            class_names.append(class_name)

    cmap = plt.get_cmap("tab10")
    class_color = {name: to_rgb(cmap(i % 10)) for i, name in enumerate(class_names)}
    name_to_idx = {str(group_names[i]): i for i in range(len(group_names))}

    plt.figure(figsize=(8, 6))
    for class_name in class_names:
        for status in ("normal", "anomaly"):
            idx_before = name_to_idx.get(_make_group_key(class_name, status, "before"))
            idx_after = name_to_idx.get(_make_group_key(class_name, status, "after"))
            if idx_before is None or idx_after is None:
                continue

            emb_before = embedding[group_ids == idx_before]
            emb_after = embedding[group_ids == idx_after]
            if emb_before.shape[0] == 0 or emb_after.shape[0] == 0:
                continue

            center_before = emb_before.mean(axis=0)
            center_after = emb_after.mean(axis=0)
            color = class_color.get(class_name, (0.5, 0.5, 0.5))
            marker = "o" if status == "normal" else "^"

            plt.scatter(center_before[0], center_before[1], c=[color], marker=marker, s=36, alpha=0.45)
            plt.scatter(center_after[0], center_after[1], c=[color], marker=marker, s=42, alpha=0.95)
            plt.arrow(
                center_before[0],
                center_before[1],
                center_after[0] - center_before[0],
                center_after[1] - center_before[1],
                color=color,
                alpha=0.9,
                length_includes_head=True,
                head_width=0.2,
                head_length=0.3,
                linewidth=1.2,
            )

    plt.title("Center shift in t-SNE space (before -> after)")
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def build_binary_labels(
    group_ids: np.ndarray, group_names: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    point_group_names = group_names[group_ids]
    point_status = np.array(
        [parse_group_name(str(name))[1] for name in point_group_names], dtype=object
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
    stage: str,
) -> List[Tuple[str, float, float]]:
    """Mean distance to origin per dataset class (normal vs anomaly subgroups)."""
    names_arr = np.asarray(group_names)
    name_to_idx = {str(names_arr[i]): i for i in range(len(names_arr))}
    out: List[Tuple[str, float, float]] = []
    for cn in base_class_names:
        idx_n = name_to_idx.get(_make_group_key(cn, "normal", stage))
        idx_a = name_to_idx.get(_make_group_key(cn, "anomaly", stage))
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


def _group_indices_for_stage(group_names: np.ndarray, stage: str) -> np.ndarray:
    indices = [
        idx
        for idx, name in enumerate(group_names)
        if parse_group_name(str(name))[2] == stage
    ]
    return np.array(indices, dtype=np.int32)


def main() -> None:
    args = parse_args()
    class_names = list(dict.fromkeys(args.classes))
    if len(class_names) == 0:
        raise ValueError("--classes must contain at least one class name")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    adapter = None
    checkpoint_reference_memory_by_class = None
    class_to_idx = None
    if args.checkpoint is not None:
        # Infer feature dimension from current setting.
        dummy_dim = 1536 if args.use_cls_token else 768
        adapter = load_adapter_from_checkpoint(
            checkpoint_path=args.checkpoint,
            checkpoint_key=args.checkpoint_key,
            feature_dim=dummy_dim,
            device=device,
        )
        checkpoint_reference_memory_by_class = load_reference_memory_by_class_from_checkpoint(
            checkpoint_path=args.checkpoint,
            device=device,
        )
        if len(checkpoint_reference_memory_by_class) > 0:
            all_class_names = get_all_class_names("mvtec")
            class_to_idx = {name: idx for idx, name in enumerate(all_class_names)}
            print(
                f"Loaded reference_memory_by_class from checkpoint: {len(checkpoint_reference_memory_by_class)} classes"
            )
        else:
            print(
                "[warn] checkpoint has no reference_memory_by_class; fallback to building reference memory from dataset."
            )

    groups = build_groups(
        data_path=args.data_path,
        class_names=class_names,
        max_per_group=args.max_per_group,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_cls_token=args.use_cls_token,
        num_reference_images=args.num_reference_images,
        adapter=adapter,
        device=device,
        checkpoint_reference_memory_by_class=checkpoint_reference_memory_by_class,
        class_to_idx=class_to_idx,
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
    ckpt_tag = "_with_adapter" if adapter is not None else "_no_adapter"
    ref_suffix = f"_ref{args.num_reference_images}{ckpt_tag}"
    tsne_features_path = os.path.join(
        args.output_dir, os.path.basename(args.save_tsne_features)
    )
    save_npz_path = (
        os.path.join(args.output_dir, os.path.basename(args.save_npz))
        if args.save_npz is not None
        else None
    )

    embedding = run_tsne(all_features, perplexity=args.perplexity, seed=args.seed)
    if adapter is not None:
        tsne_before_path = os.path.join(
            args.output_dir, _append_suffix_to_stem(args.output, ref_suffix + "_before")
        )
        tsne_after_path = os.path.join(
            args.output_dir, _append_suffix_to_stem(args.output, ref_suffix + "_after")
        )
        tsne_diff_path = os.path.join(
            args.output_dir, _append_suffix_to_stem(args.difference_output, ref_suffix)
        )
        plot_embeddings(
            embedding,
            all_group_ids,
            group_names,
            tsne_before_path,
            filter_stage="before",
            title="t-SNE of residual features (before adapter)",
        )
        plot_embeddings(
            embedding,
            all_group_ids,
            group_names,
            tsne_after_path,
            filter_stage="after",
            title="t-SNE of residual features (after adapter)",
        )
        plot_center_shift(
            embedding=embedding,
            group_ids=all_group_ids,
            group_names=group_names,
            output_path=tsne_diff_path,
        )
        tsne_saved_paths = [tsne_before_path, tsne_after_path, tsne_diff_path]
    else:
        tsne_output_path = os.path.join(
            args.output_dir, _append_suffix_to_stem(args.output, ref_suffix)
        )
        plot_embeddings(
            embedding,
            all_group_ids,
            group_names,
            tsne_output_path,
        )
        tsne_saved_paths = [tsne_output_path]

    summary_group_indices = _group_indices_for_stage(group_names, args.report_stage)
    if summary_group_indices.size == 0:
        raise ValueError(f"No groups found for report stage: {args.report_stage}")

    summary_mask = np.isin(all_group_ids, summary_group_indices)
    summary_embedding = embedding[summary_mask]
    summary_group_ids = all_group_ids[summary_mask]

    binary_labels, status_labels = build_binary_labels(summary_group_ids, group_names)
    binary_labels_all, status_labels_all = build_binary_labels(all_group_ids, group_names)
    distance_saved_paths: List[str] = []
    stage_distance_means: Dict[str, Tuple[float, float]] = {}
    if adapter is not None:
        for stage in ("before", "after"):
            st_group_idx = _group_indices_for_stage(group_names, stage)
            st_mask = np.isin(all_group_ids, st_group_idx)
            st_embedding = embedding[st_mask]
            st_group_ids = all_group_ids[st_mask]
            st_binary, _ = build_binary_labels(st_group_ids, group_names)
            dist_path = os.path.join(
                args.output_dir,
                _append_suffix_to_stem(args.distance_output, ref_suffix + f"_{stage}"),
            )
            distances_st = plot_distance_distribution(st_embedding, st_binary, dist_path)
            n_m, a_m = summarize_distance_means(distances_st, st_binary)
            stage_distance_means[stage] = (n_m, a_m)
            distance_saved_paths.append(dist_path)
        distances = np.linalg.norm(summary_embedding, axis=1)
        normal_mean_distance, anomaly_mean_distance = summarize_distance_means(
            distances, binary_labels
        )
    else:
        distance_output_path = os.path.join(
            args.output_dir, _append_suffix_to_stem(args.distance_output, ref_suffix)
        )
        distances = plot_distance_distribution(
            summary_embedding, binary_labels, distance_output_path
        )
        distance_saved_paths = [distance_output_path]
        normal_mean_distance, anomaly_mean_distance = summarize_distance_means(
            distances, binary_labels
        )

    per_class_means = summarize_distance_means_per_base_class(
        distances, summary_group_ids, group_names, class_names, args.report_stage
    )

    savez_tsne: Dict[str, object] = {
        "embedding": embedding,
        "group_ids": all_group_ids,
        "group_names": group_names,
        "summary_group_ids": summary_group_ids,
        "summary_mask": summary_mask,
        "report_stage": args.report_stage,
        "binary_labels": binary_labels,
        "status_labels": status_labels,
        "binary_labels_all": binary_labels_all,
        "status_labels_all": status_labels_all,
        "distances_to_origin": distances,
        "mean_normal_distance_to_origin": normal_mean_distance,
        "mean_anomaly_distance_to_origin": anomaly_mean_distance,
    }
    if adapter is not None:
        nb, ab = stage_distance_means["before"]
        na, aa = stage_distance_means["after"]
        savez_tsne.update(
            {
                "mean_normal_distance_to_origin_before": nb,
                "mean_anomaly_distance_to_origin_before": ab,
                "mean_normal_distance_to_origin_after": na,
                "mean_anomaly_distance_to_origin_after": aa,
            }
        )
    np.savez(tsne_features_path, **savez_tsne)

    if save_npz_path is not None:
        np.savez(
            save_npz_path,
            embedding=embedding,
            group_ids=all_group_ids,
            group_names=group_names,
        )

    for path in tsne_saved_paths:
        print(f"Saved t-SNE figure to: {path}")
    for path in distance_saved_paths:
        print(f"Saved distance distribution figure to: {path}")
    if adapter is not None:
        nb, ab = stage_distance_means["before"]
        na, aa = stage_distance_means["after"]
        print(f"Mean distance to origin | before adapter | normal: {nb:.6f} | anomaly: {ab:.6f}")
        print(f"Mean distance to origin | after adapter  | normal: {na:.6f} | anomaly: {aa:.6f}")
    print(f"Report stage for primary distance summary / per-class table: {args.report_stage}")
    print(f"Mean distance to origin | normal (all classes): {normal_mean_distance:.6f}")
    print(f"Mean distance to origin | anomaly (all classes): {anomaly_mean_distance:.6f}")
    print(f"Mean distance to origin | per class ({args.report_stage}):")
    for cn, mn, ma in per_class_means:
        print(f"  {cn} | normal: {mn:.6f} | anomaly: {ma:.6f}")
    print(f"Saved t-SNE 2D features to: {tsne_features_path}")
    if save_npz_path is not None:
        print(f"Saved embedding data to: {save_npz_path}")


if __name__ == "__main__":
    main()
