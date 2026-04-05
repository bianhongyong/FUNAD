import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from matplotlib.colors import to_rgb
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Subset

from dataset import dataset_extract
from dataset.multiclass_feature_dataset import MVTEC_CLASS_NAMES, get_all_class_names


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
    parser = argparse.ArgumentParser(
        "DINO CLS-token reference matching + t-SNE (default: DINOv3)"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/media/honeywell/D/bhy/dataset/MVTec_overlap/MVTec_noisy10",
        help="Root path of overlapped MVTec dataset (same as training).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mvtec",
        choices=["mvtec", "visa"],
        help="Dataset name, used only for class-name list.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help=(
            "Subset of classes to use. "
            "If omitted, use all classes of the specified dataset."
        ),
    )
    parser.add_argument(
        "--num_reference_per_class",
        type=int,
        default=4,
        help="Number of reference images sampled from training set per class.",
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=5,
        help="Repeat sampling reference images multiple times to measure stability.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for feature extraction.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity.",
    )
    parser.add_argument(
        "--max_tsne_points_per_group",
        type=int,
        default=800,
        help="Max number of points per (class, is_reference) group for t-SNE.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for PyTorch.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="tsne_cls_token_outputs",
        help="Directory to save t-SNE figure and logs.",
    )
    parser.add_argument(
        "--tsne_figure",
        type=str,
        default="tsne_cls_token_reference_vs_test.png",
        help="Filename of t-SNE scatter plot.",
    )
    parser.add_argument(
        "--save_npz",
        type=str2bool,
        default="true",
        help="Whether to save raw t-SNE embedding / labels as npz.",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="dinov3",
        choices=["dinov3", "dinov2"],
        help="Vision backbone: dinov3 (torch.hub) or legacy dino_vitb8 (DINOv2).",
    )
    parser.add_argument(
        "--dinov3_hub_dir",
        type=str,
        default=None,
        help=(
            "Local facebookresearch/dinov3 hub directory (contains hubconf.py). "
            "If omitted and ~/.cache/torch/hub/facebookresearch_dinov3_main exists, "
            "uses it automatically (source=local, no GitHub)."
        ),
    )
    parser.add_argument(
        "--dinov3_model",
        type=str,
        default="dinov3_vitb16",
        help="torch.hub entry for DINOv3, e.g. dinov3_vitb16, dinov3_vits16.",
    )
    parser.add_argument(
        "--dinov3_weights",
        type=str,
        default=None,
        help="Optional local path or file:// URL for pretrained weights (DINOv3 hub API).",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    #np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_class_list(dataset_name: str, classes: List[str] = None) -> List[str]:
    if classes is not None and len(classes) > 0:
        return list(dict.fromkeys(classes))
    all_classes = get_all_class_names(dataset_name)
    return list(all_classes)


def _resolve_dinov3_hub_dir(explicit: Optional[str]) -> Optional[str]:
    """Prefer explicit --dinov3_hub_dir; else use torch hub default cache if valid."""
    if explicit and os.path.isdir(explicit):
        return explicit
    default_dir = os.path.expanduser(
        "~/.cache/torch/hub/facebookresearch_dinov3_main"
    )
    hubconf = os.path.join(default_dir, "hubconf.py")
    if os.path.isfile(hubconf):
        return default_dir
    return None


def _load_backbone(device: torch.device, args: argparse.Namespace) -> torch.nn.Module:
    if args.backbone == "dinov2":
        model = torch.hub.load("facebookresearch/dino:main", "dino_vitb8")
        model = model.to(device)
        model.eval()
        return model

    # DINOv3：torch.hub 会 import dino 内部 utils，与项目包 utils 冲突时需临时解除
    local_utils_module = sys.modules.get("utils")
    should_restore_utils = (
        local_utils_module is not None
        and os.path.abspath(getattr(local_utils_module, "__file__", "")).endswith(
            os.path.join("FUNAD", "utils.py")
        )
    )
    if should_restore_utils:
        del sys.modules["utils"]

    load_kw = {"pretrained": True}
    if args.dinov3_weights:
        load_kw["weights"] = args.dinov3_weights

    try:
        hub_dir = _resolve_dinov3_hub_dir(args.dinov3_hub_dir)
        if hub_dir:
            print(f"[DINOv3] loading from local hub: {hub_dir}")
            model = torch.hub.load(
                hub_dir,
                args.dinov3_model,
                source="local",
                **load_kw,
            )
        else:
            print("[DINOv3] loading from GitHub: facebookresearch/dinov3")
            model = torch.hub.load(
                "facebookresearch/dinov3",
                args.dinov3_model,
                **load_kw,
            )
    finally:
        if should_restore_utils:
            sys.modules["utils"] = local_utils_module

    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def extract_cls_tokens(
    images: torch.Tensor, model: torch.nn.Module, backbone: str
) -> torch.Tensor:
    """
    提取最后一层 CLS token（不拼接 patch）。

    images: [B, 3, H, W] ImageNet 归一化输入。
    返回: [B, C]
    """
    model.eval()
    if backbone == "dinov3":
        _patches, cls_tok = model.get_intermediate_layers(
            images, return_class_token=True
        )[0]
        return cls_tok
    feature = model.get_intermediate_layers(images)[0]
    return feature[:, 0, :]


def _build_train_dataset(
    data_path: str, dataset_name: str, class_name: str
) -> dataset_extract.MyDataset:
    # 与其它脚本保持一致：使用 dataset_extract.MyDataset
    ds = dataset_extract.MyDataset(
        dataset_path=data_path,
        dataset=dataset_name,
        class_name=class_name,
        is_train=True,
    )
    return ds


def _build_test_dataset(
    data_path: str, dataset_name: str, class_name: str
) -> dataset_extract.MyDataset:
    ds = dataset_extract.MyDataset(
        dataset_path=data_path,
        dataset=dataset_name,
        class_name=class_name,
        is_train=False,
    )
    return ds


def sample_reference_indices(
    train_dataset: dataset_extract.MyDataset,
    num_ref: int,
    rng: np.random.Generator,
) -> List[int]:
    # 参考自 self_train_ad_multiclass_residual 中的 noisy 过滤逻辑。
    candidates: List[int] = []
    for idx, path in enumerate(train_dataset.x):
        filename = os.path.basename(path).lower()
        if filename.startswith("noisy"):
            continue
        candidates.append(idx)

    if len(candidates) == 0:
        return []

    sample_n = min(max(1, num_ref), len(candidates))
    selected = rng.choice(np.array(candidates, dtype=np.int64), size=sample_n, replace=False)
    return sorted(selected.tolist())


def compute_class_prototypes_and_refs(
    data_path: str,
    dataset_name: str,
    class_names: List[str],
    num_ref_per_class: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    model: torch.nn.Module,
    backbone: str,
    rng: np.random.Generator,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, List[int]]]:
    """
    返回:
      - prototypes[class_name]: [C] CLS token 平均向量
      - ref_tokens[class_name]: [N_ref, C] 每个参考图像 CLS token
      - ref_indices[class_name]: 参考图像在 train_dataset.x 中的 index
    """
    prototypes: Dict[str, np.ndarray] = {}
    ref_tokens: Dict[str, np.ndarray] = {}
    ref_indices: Dict[str, List[int]] = {}

    for class_name in class_names:
        train_ds = _build_train_dataset(
            data_path=data_path, dataset_name=dataset_name, class_name=class_name
        )
        if len(train_ds) == 0:
            continue

        selected_ids = sample_reference_indices(
            train_dataset=train_ds, num_ref=num_ref_per_class, rng=rng
        )
        if len(selected_ids) == 0:
            continue
        ref_indices[class_name] = selected_ids

        subset = Subset(train_ds, selected_ids)
        loader = DataLoader(
            subset,
            batch_size=min(batch_size, len(subset)),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        cls_all = []
        for batch in loader:
            # train 模式 MyDataset 通常只返回图像张量.
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch
            images = images.to(device)
            cls_tokens = extract_cls_tokens(images, model, backbone)  # [b, C]
            cls_all.append(cls_tokens.detach().cpu().numpy())

        cls_all_np = np.concatenate(cls_all, axis=0)  # [N_ref, C]
        ref_tokens[class_name] = cls_all_np
        prototypes[class_name] = cls_all_np.mean(axis=0, keepdims=False)  # [C]

    return prototypes, ref_tokens, ref_indices


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    a: [N, C], b: [M, C]
    返回: [N, M]
    """
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True).clip(min=1e-12)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True).clip(min=1e-12)
    return a_norm @ b_norm.T


def evaluate_matching_accuracy(
    data_path: str,
    dataset_name: str,
    class_names: List[str],
    prototypes: Dict[str, np.ndarray],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    model: torch.nn.Module,
    backbone: str,
    ref_indices: Dict[str, List[int]],
) -> Tuple[float, float]:
    """
    在 (1) 训练集(排除参考库) + (2) 测试集中评估匹配准确率。

    返回:
      - train_acc: 训练集(不含参考)上的匹配精度
      - test_acc: 测试集上的匹配精度
    """
    proto_mat = np.stack([prototypes[c] for c in class_names], axis=0)  # [K, C]
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    def _eval_split(use_train: bool) -> float:
        correct = 0
        total = 0
        for class_name in class_names:
            if class_name not in prototypes:
                continue
            if use_train:
                ds = _build_train_dataset(data_path, dataset_name, class_name)
                drop_ids = set(ref_indices.get(class_name, []))
                keep_indices = [
                    i for i in range(len(ds)) if i not in drop_ids
                ]
                if len(keep_indices) == 0:
                    continue
                subset = Subset(ds, keep_indices)
            else:
                ds = _build_test_dataset(data_path, dataset_name, class_name)
                if len(ds) == 0:
                    continue
                subset = ds

            loader = DataLoader(
                subset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
            )

            for batch in loader:
                if use_train:
                    if isinstance(batch, (list, tuple)):
                        images = batch[0]
                    else:
                        images = batch
                else:
                    # test: (x, y, mask) => 取 x.
                    if isinstance(batch, (list, tuple)) and len(batch) >= 1:
                        images = batch[0]
                    else:
                        images = batch

                images = images.to(device)
                cls_tokens = extract_cls_tokens(images, model, backbone)  # [b, C]
                cls_np = cls_tokens.detach().cpu().numpy()  # [b, C]

                sims = cosine_similarity(cls_np, proto_mat)  # [b, K]
                pred_idx = sims.argmax(axis=1)  # [b]
                true_idx = class_to_idx[class_name]
                correct += int((pred_idx == true_idx).sum())
                total += cls_np.shape[0]

        return float(correct) / float(total) if total > 0 else 0.0

    train_acc = _eval_split(use_train=True)
    test_acc = _eval_split(use_train=False)
    return train_acc, test_acc


def run_tsne_and_plot(
    all_features: np.ndarray,
    all_class_ids: np.ndarray,
    all_is_ref: np.ndarray,
    class_names: List[str],
    perplexity: float,
    seed: int,
    output_dir: str,
    figure_name: str,
    save_npz: bool,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    n_samples = all_features.shape[0]
    if n_samples < 3:
        raise ValueError("t-SNE 需要至少 3 个样本。")

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
    embedding = tsne.fit_transform(all_features)  # [N, 2]

    cmap = plt.get_cmap("tab10")
    class_color = {name: to_rgb(cmap(i % 10)) for i, name in enumerate(class_names)}

    plt.figure(figsize=(8, 6))
    for class_idx, class_name in enumerate(class_names):
        mask_cls = all_class_ids == class_idx
        if not mask_cls.any():
            continue
        # 参考库点用方块，测试/验证图像用圆点。
        for is_ref, marker, alpha in ((1, "s", 0.9), (0, "o", 0.6)):
            mask = mask_cls & (all_is_ref == is_ref)
            if not mask.any():
                continue
            color = class_color.get(class_name, (0.5, 0.5, 0.5))
            label = f"{class_name} ({'ref' if is_ref == 1 else 'test'})"
            plt.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                s=18,
                alpha=alpha,
                c=[color],
                marker=marker,
                label=label,
            )

    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.title("CLS-token space: reference library vs test images")
    plt.legend(fontsize=8, markerscale=1.2)
    plt.tight_layout()

    fig_path = os.path.join(output_dir, figure_name)
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print(f"[t-SNE] saved figure to: {fig_path}")

    if save_npz:
        npz_path = os.path.join(output_dir, "tsne_cls_token_embedding.npz")
        np.savez(
            npz_path,
            embedding=embedding,
            class_ids=all_class_ids,
            is_reference=all_is_ref,
            class_names=np.array(class_names, dtype=object),
        )
        print(f"[t-SNE] saved embedding data to: {npz_path}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    class_names = build_class_list(args.dataset, args.classes)
    if len(class_names) == 0:
        raise ValueError("没有可用类别，请检查 --classes 或 dataset 设置。")

    print(f"Using classes: {class_names}")
    print(f"Backbone: {args.backbone}")
    model = _load_backbone(device, args)
    rng = np.random.default_rng(args.seed)

    train_acc_list: List[float] = []
    test_acc_list: List[float] = []

    # 为 t-SNE 收集一次完整的特征（使用第一次 trial 的参考库）。
    tsne_features_ref: List[np.ndarray] = []
    tsne_features_test: List[np.ndarray] = []
    tsne_class_ids_ref: List[int] = []
    tsne_class_ids_test: List[int] = []

    # ===== 多次重复：随机抽取参考库，计算匹配率 =====
    for trial in range(args.num_trials):
        print(f"\n==== Trial {trial + 1}/{args.num_trials} ====")
        prototypes, ref_tokens, ref_indices = compute_class_prototypes_and_refs(
            data_path=args.data_path,
            dataset_name=args.dataset,
            class_names=class_names,
            num_ref_per_class=args.num_reference_per_class,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            model=model,
            backbone=args.backbone,
            rng=rng,
        )

        train_acc, test_acc = evaluate_matching_accuracy(
            data_path=args.data_path,
            dataset_name=args.dataset,
            class_names=class_names,
            prototypes=prototypes,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            model=model,
            backbone=args.backbone,
            ref_indices=ref_indices,
        )

        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(
            f"[trial {trial + 1}] train (excluding refs) acc: {train_acc:.4f}, "
            f"test acc: {test_acc:.4f}"
        )

        if trial == 0:
            # 第一次 trial 的参考库用于 t-SNE。
            class_to_idx = {c: i for i, c in enumerate(class_names)}

            # 1) 参考库 CLS token 收集。
            for class_name, cls_tokens in ref_tokens.items():
                cls_idx = class_to_idx[class_name]
                # 随机下采样，避免点太多。
                if cls_tokens.shape[0] > args.max_tsne_points_per_group:
                    sel = rng.choice(
                        cls_tokens.shape[0],
                        size=args.max_tsne_points_per_group,
                        replace=False,
                    )
                    cls_sel = cls_tokens[sel]
                else:
                    cls_sel = cls_tokens
                tsne_features_ref.append(cls_sel)
                tsne_class_ids_ref.extend([cls_idx] * cls_sel.shape[0])

            # 2) 测试集图像 CLS token 收集。
            for class_name in class_names:
                if class_name not in prototypes:
                    continue
                test_ds = _build_test_dataset(
                    data_path=args.data_path,
                    dataset_name=args.dataset,
                    class_name=class_name,
                )
                if len(test_ds) == 0:
                    continue
                loader = DataLoader(
                    test_ds,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=use_cuda,
                )

                cls_tokens_list = []
                for batch in tqdm.tqdm(loader, desc=f"收集 t-SNE 测试 CLS: {class_name}"):
                    if isinstance(batch, (list, tuple)) and len(batch) >= 1:
                        images = batch[0]
                    else:
                        images = batch
                    images = images.to(device)
                    cls_tokens = extract_cls_tokens(images, model, args.backbone)
                    cls_tokens_list.append(cls_tokens.detach().cpu().numpy())

                if len(cls_tokens_list) == 0:
                    continue
                cls_all = np.concatenate(cls_tokens_list, axis=0)
                if cls_all.shape[0] > args.max_tsne_points_per_group:
                    sel = rng.choice(
                        cls_all.shape[0],
                        size=args.max_tsne_points_per_group,
                        replace=False,
                    )
                    cls_all = cls_all[sel]
                tsne_features_test.append(cls_all)
                tsne_class_ids_test.extend(
                    [class_to_idx[class_name]] * cls_all.shape[0]
                )

    # 打印多次 trial 的总体统计。
    if len(train_acc_list) > 0:
        print(
            "\n==== Summary over trials ====\n"
            f"Train (excluding refs) acc: mean={np.mean(train_acc_list):.4f}, "
            f"std={np.std(train_acc_list):.4f}"
        )
    if len(test_acc_list) > 0:
        print(
            f"Test acc: mean={np.mean(test_acc_list):.4f}, "
            f"std={np.std(test_acc_list):.4f}"
        )

    # ===== t-SNE 可视化：CLS token 空间中参考库 vs 测试图像 =====
    if len(tsne_features_ref) == 0 or len(tsne_features_test) == 0:
        print(
            "[t-SNE] 没有足够的数据用于可视化（参考库或测试特征为空），跳过 t-SNE 绘图。"
        )
        return

    features_ref = np.concatenate(tsne_features_ref, axis=0)
    features_test = np.concatenate(tsne_features_test, axis=0)
    class_ids_ref = np.asarray(tsne_class_ids_ref, dtype=np.int32)
    class_ids_test = np.asarray(tsne_class_ids_test, dtype=np.int32)

    all_features = np.concatenate([features_ref, features_test], axis=0)
    all_class_ids = np.concatenate([class_ids_ref, class_ids_test], axis=0)
    all_is_ref = np.concatenate(
        [np.ones_like(class_ids_ref), np.zeros_like(class_ids_test)], axis=0
    )

    run_tsne_and_plot(
        all_features=all_features,
        all_class_ids=all_class_ids,
        all_is_ref=all_is_ref,
        class_names=class_names,
        perplexity=args.perplexity,
        seed=args.seed,
        output_dir=args.output_dir,
        figure_name=args.tsne_figure,
        save_npz=bool(args.save_npz),
    )


if __name__ == "__main__":
    main()

