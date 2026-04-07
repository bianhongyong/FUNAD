import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader

try:
    import faiss

    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False
    faiss = None


_REPO_ROOT = Path(__file__).resolve().parent.parent
_root_str = str(_REPO_ROOT)
if _root_str not in sys.path:
    sys.path.insert(0, _root_str)

from dataset import dataset_extract
from dataset.multiclass_feature_dataset import MVTEC_CLASS_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "统计 MVTec 各类别互近邻对组成比例（Dinov3 非残差特征）"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/media/honeywell/D/bhy/dataset/MVTec",
        help="MVTec 数据集根目录",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="plot_outputs/mutual_nn_mvtec_raw_feature",
        help="结果输出目录",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=list(MVTEC_CLASS_NAMES),
        help="要统计的类别列表，默认全类别",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="推理 batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="dataloader workers")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--image_size", type=int, default=512, help="输入 resize 大小")
    parser.add_argument("--crop_size", type=int, default=448, help="输入 center crop 大小")
    parser.add_argument(
        "--dinov3_hub_dir",
        type=str,
        default=None,
        help="本地 dinov3 hub 目录（含 hubconf.py），不填则自动尝试 ~/.cache/torch/hub/... ",
    )
    parser.add_argument(
        "--dinov3_model",
        type=str,
        default="dinov3_vitb16",
        help="torch.hub 的 Dinov3 模型名",
    )
    parser.add_argument(
        "--dinov3_weights",
        type=str,
        default=None,
        help="可选本地权重路径或 file:// URL",
    )
    parser.add_argument(
        "--use_cls_token",
        action="store_true",
        help="是否拼接 cls token 到 patch 特征（默认不拼接）",
    )
    parser.add_argument(
        "--max_patches_per_class",
        type=int,
        default=0,
        help="每类参与统计的最大 patch 数，0 表示全量",
    )
    return parser.parse_args()


def _resolve_dinov3_hub_dir(explicit: Optional[str]) -> Optional[str]:
    if explicit and os.path.isdir(explicit):
        return explicit
    default_dir = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov3_main")
    hubconf = os.path.join(default_dir, "hubconf.py")
    if os.path.isfile(hubconf):
        return default_dir
    return None


def _load_dinov3(
    device: torch.device,
    dinov3_hub_dir: Optional[str],
    dinov3_model: str,
    dinov3_weights: Optional[str],
) -> torch.nn.Module:
    load_kw = {"pretrained": True}
    if dinov3_weights:
        load_kw["weights"] = dinov3_weights

    hub_dir = _resolve_dinov3_hub_dir(dinov3_hub_dir)
    if hub_dir:
        print(f"[DINOv3] load from local hub: {hub_dir}")
        dino = torch.hub.load(hub_dir, dinov3_model, source="local", **load_kw)
    else:
        print("[DINOv3] load from GitHub: facebookresearch/dinov3")
        dino = torch.hub.load("facebookresearch/dinov3", dinov3_model, **load_kw)
    dino = dino.to(device)
    dino.eval()
    return dino


def _extract_patch_tokens(
    images: torch.Tensor,
    dino: torch.nn.Module,
    use_cls_token: bool,
) -> torch.Tensor:
    with torch.no_grad():
        patch_tokens, cls_tok = dino.get_intermediate_layers(
            images, return_class_token=True
        )[0]
        if use_cls_token:
            cls_tokens = torch.repeat_interleave(
                cls_tok.unsqueeze(1), patch_tokens.shape[1], dim=1
            )
            patch_tokens = torch.cat([cls_tokens, patch_tokens], dim=-1)
    return patch_tokens


def _collect_test_patches_and_labels(
    data_path: str,
    class_name: str,
    dino: torch.nn.Module,
    use_cls_token: bool,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    resize: int,
    cropsize: int,
) -> Tuple[np.ndarray, np.ndarray]:
    test_set = dataset_extract.MyDataset(
        dataset_path=data_path,
        dataset="mvtec",
        class_name=class_name,
        is_train=False,
        resize=resize,
        cropsize=cropsize,
    )
    if len(test_set) == 0:
        raise ValueError(f"类别 {class_name} 的 test 为空。")

    loader = DataLoader(
        test_set,
        batch_size=batch_size,
        pin_memory=torch.cuda.is_available(),
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    feats_all: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []

    for images, _, masks in tqdm.tqdm(loader, desc=f"[{class_name}] extract test features"):
        images = images.to(device)
        patch_tokens = _extract_patch_tokens(images, dino, use_cls_token=use_cls_token)
        feat_np = patch_tokens.detach().cpu().numpy().reshape(-1, patch_tokens.shape[-1]).astype(
            np.float32
        )
        feats_all.append(feat_np)

        mask_np = masks.detach().cpu().numpy()
        num_patches = patch_tokens.shape[1]
        grid_size = int(np.sqrt(num_patches))
        patch_h = mask_np.shape[2] // grid_size
        patch_w = mask_np.shape[3] // grid_size
        patch_mask = mask_np.reshape(
            mask_np.shape[0], 1, grid_size, patch_h, grid_size, patch_w
        ).max(axis=(3, 5))
        patch_labels = (patch_mask.reshape(-1) > 0).astype(np.int32)
        labels_all.append(patch_labels)

    feat = np.ascontiguousarray(np.concatenate(feats_all, axis=0).astype(np.float32))
    lbl = np.ascontiguousarray(np.concatenate(labels_all, axis=0).astype(np.int32))
    return feat, lbl


def _mutual_nearest_pairs(features: np.ndarray) -> np.ndarray:
    n, dim = features.shape
    if n < 2:
        return np.zeros((0, 2), dtype=np.int64)

    if _HAS_FAISS:
        index = faiss.IndexFlatL2(dim)
        index.add(np.ascontiguousarray(features, dtype=np.float32))
        _, idx = index.search(np.ascontiguousarray(features, dtype=np.float32), 2)
        nn = idx[:, 1].astype(np.int64)
    else:
        x = torch.from_numpy(features)
        dist = torch.cdist(x, x)
        dist.fill_diagonal_(float("inf"))
        nn = torch.argmin(dist, dim=1).cpu().numpy().astype(np.int64)

    i = np.arange(n, dtype=np.int64)
    j = nn
    mutual_mask = i < j
    mutual_mask &= (nn[j] == i)
    pairs = np.stack([i[mutual_mask], j[mutual_mask]], axis=1)
    return pairs.astype(np.int64)


def _count_pair_types(
    pairs: np.ndarray,
    patch_labels: np.ndarray,
) -> Dict[str, float]:
    if pairs.shape[0] == 0:
        return {
            "count_nn": 0.0,
            "count_aa": 0.0,
            "count_an": 0.0,
            "ratio_nn": 0.0,
            "ratio_aa": 0.0,
            "ratio_an": 0.0,
            "num_pairs": 0.0,
        }

    li = patch_labels[pairs[:, 0]]
    lj = patch_labels[pairs[:, 1]]
    both_normal = np.logical_and(li == 0, lj == 0)
    both_anomaly = np.logical_and(li == 1, lj == 1)
    cross = np.logical_xor(li == 1, lj == 1)
    num_pairs = float(pairs.shape[0])
    count_nn = float(both_normal.sum())
    count_aa = float(both_anomaly.sum())
    count_an = float(cross.sum())
    return {
        "count_nn": count_nn,
        "count_aa": count_aa,
        "count_an": count_an,
        "ratio_nn": count_nn / num_pairs,
        "ratio_aa": count_aa / num_pairs,
        "ratio_an": count_an / num_pairs,
        "num_pairs": num_pairs,
    }


def _plot_class_ratio_bars(
    rows: List[Dict[str, float]],
    output_path: str,
) -> None:
    classes = [r["class_name"] for r in rows]
    ratio_nn = np.array([r["ratio_nn"] for r in rows], dtype=np.float32)
    ratio_aa = np.array([r["ratio_aa"] for r in rows], dtype=np.float32)
    ratio_an = np.array([r["ratio_an"] for r in rows], dtype=np.float32)

    x = np.arange(len(classes), dtype=np.float32)
    width = 0.26
    fig, ax = plt.subplots(figsize=(max(13, len(classes) * 0.9), 6))
    ax.bar(x - width, ratio_nn, width=width, color="#1f77b4", label="N-N")
    ax.bar(x, ratio_aa, width=width, color="#d62728", label="A-A")
    ax.bar(x + width, ratio_an, width=width, color="#2ca02c", label="A-N")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Ratio")
    ax.set_title("Mutual nearest-neighbor pair composition per MVTec class (raw DINOv3 feature)")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_overall_ratio_stacked(overall: Dict[str, float], output_path: str) -> None:
    vals = np.array(
        [overall["ratio_nn"], overall["ratio_aa"], overall["ratio_an"]], dtype=np.float32
    )
    labels = ["N-N", "A-A", "A-N"]
    colors = ["#1f77b4", "#d62728", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(6, 5))
    bottom = 0.0
    for v, name, c in zip(vals, labels, colors):
        ax.bar([0], [v], bottom=bottom, width=0.6, color=c, label=f"{name}: {v:.3f}")
        bottom += float(v)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([0])
    ax.set_xticklabels(["MVTec-all"])
    ax.set_ylabel("Ratio")
    ax.set_title("Overall mutual nearest-neighbor pair composition")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    class_names = list(dict.fromkeys(args.classes))
    dino = _load_dinov3(
        device=device,
        dinov3_hub_dir=args.dinov3_hub_dir,
        dinov3_model=args.dinov3_model,
        dinov3_weights=args.dinov3_weights,
    )

    rows: List[Dict[str, float]] = []
    total_count_nn = 0.0
    total_count_aa = 0.0
    total_count_an = 0.0
    total_num_pairs = 0.0

    for class_name in class_names:
        print(f"\n===== class: {class_name} =====")
        features, patch_labels = _collect_test_patches_and_labels(
            data_path=args.data_path,
            class_name=class_name,
            dino=dino,
            use_cls_token=args.use_cls_token,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            resize=args.image_size,
            cropsize=args.crop_size,
        )

        if args.max_patches_per_class > 0 and features.shape[0] > args.max_patches_per_class:
            keep = rng.choice(features.shape[0], size=args.max_patches_per_class, replace=False)
            features = features[keep]
            patch_labels = patch_labels[keep]
            print(f"[{class_name}] downsample patch count to {features.shape[0]}")

        pairs = _mutual_nearest_pairs(features)
        stat = _count_pair_types(pairs, patch_labels)
        row = {"class_name": class_name, **stat}
        rows.append(row)

        total_count_nn += stat["count_nn"]
        total_count_aa += stat["count_aa"]
        total_count_an += stat["count_an"]
        total_num_pairs += stat["num_pairs"]

        print(
            f"[{class_name}] pairs={int(stat['num_pairs'])} | "
            f"N-N={stat['ratio_nn']:.4f}, A-A={stat['ratio_aa']:.4f}, A-N={stat['ratio_an']:.4f}"
        )

    if total_num_pairs <= 0:
        overall = {
            "ratio_nn": 0.0,
            "ratio_aa": 0.0,
            "ratio_an": 0.0,
            "num_pairs": 0.0,
            "count_nn": 0.0,
            "count_aa": 0.0,
            "count_an": 0.0,
        }
    else:
        overall = {
            "ratio_nn": total_count_nn / total_num_pairs,
            "ratio_aa": total_count_aa / total_num_pairs,
            "ratio_an": total_count_an / total_num_pairs,
            "num_pairs": total_num_pairs,
            "count_nn": total_count_nn,
            "count_aa": total_count_aa,
            "count_an": total_count_an,
        }

    csv_path = os.path.join(args.output_dir, "mutual_nn_pair_ratio_mvtec_per_class.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("class_name,num_pairs,count_nn,count_aa,count_an,ratio_nn,ratio_aa,ratio_an\n")
        for r in rows:
            f.write(
                f"{r['class_name']},{int(r['num_pairs'])},{int(r['count_nn'])},{int(r['count_aa'])},"
                f"{int(r['count_an'])},{r['ratio_nn']:.8f},{r['ratio_aa']:.8f},{r['ratio_an']:.8f}\n"
            )
        f.write(
            f"ALL,{int(overall['num_pairs'])},{int(overall['count_nn'])},{int(overall['count_aa'])},"
            f"{int(overall['count_an'])},{overall['ratio_nn']:.8f},{overall['ratio_aa']:.8f},"
            f"{overall['ratio_an']:.8f}\n"
        )

    class_plot_path = os.path.join(args.output_dir, "mutual_nn_pair_ratio_mvtec_per_class.png")
    overall_plot_path = os.path.join(args.output_dir, "mutual_nn_pair_ratio_mvtec_overall.png")
    _plot_class_ratio_bars(rows, class_plot_path)
    _plot_overall_ratio_stacked(overall, overall_plot_path)

    print("\n===== Overall =====")
    print(
        f"pairs={int(overall['num_pairs'])} | "
        f"N-N={overall['ratio_nn']:.4f}, A-A={overall['ratio_aa']:.4f}, A-N={overall['ratio_an']:.4f}"
    )
    print(f"saved csv: {csv_path}")
    print(f"saved per-class figure: {class_plot_path}")
    print(f"saved overall figure: {overall_plot_path}")
    if not _HAS_FAISS:
        print("[warn] FAISS 不可用，已使用 torch.cdist 回退路径，速度可能较慢。")


if __name__ == "__main__":
    main()
