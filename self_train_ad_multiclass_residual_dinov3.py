import datetime
import os
import random
from re import A
import sys
import time
import warnings

import cv2
import faiss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as torch_mp
from PIL import Image
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import tqdm
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset

from dataset import dataset_extract, multiclass_feature_dataset
from src.train.epoch_precompute import precompute_pseudo_labels_multiclass_residual
from utils import evaluate as eval_utils
from utils.loss import (
    build_adaptive_threshold_map,
    compute_balanced_bce_loss,
    compute_oto_loss_multiclass,
)
from src.model import model
from dataset.multiclass_feature_dataset import MultiClassFeatureDataset, get_all_class_names
from utils.print import (
    print_epoch_losses,
    print_epoch_times,
    print_full_dataset_confusion_matrix_by_raw_score,
    print_selected_clean_ratio,
    print_selected_score_distribution_by_class,
)
import utils.train_utils as common_utils
from src.train.args_multiclass_residual_dinov3 import parse_args

warnings.filterwarnings("ignore")

try:
    # Avoid exhausting file descriptors when DataLoader workers share CPU tensors.
    torch_mp.set_sharing_strategy("file_system")
except (AttributeError, RuntimeError):
    warnings.warn("Failed to set torch multiprocessing sharing strategy.")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# --feature_model 与 torch.hub 入口名、本地 hub 目录的映射。
# hub_repo_dir 为 None 时表示不在表中写死路径，改由环境变量 DINOV3_HUB_DIR（若设置）
# 或默认 <torch.hub.get_dir()>/facebookresearch_dinov3_main 解析；若仍不可用则回退从 GitHub 拉取。
DINOV3_FEATURE_MODEL_REGISTRY = {
    "dinov3_vits16": {"hub_entry": "dinov3_vits16", "hub_repo_dir": "/home/honeywell/.cache/torch/hub/facebookresearch_dinov3_main"},
    "dinov3_vits16plus": {"hub_entry": "dinov3_vits16plus", "hub_repo_dir": "/home/honeywell/.cache/torch/hub/facebookresearch_dinov3_main"},
    "dinov3_vitb16": {"hub_entry": "dinov3_vitb16", "hub_repo_dir": "/home/honeywell/.cache/torch/hub/facebookresearch_dinov3_main"},
    "dinov3_vitl16": {"hub_entry": "dinov3_vitl16", "hub_repo_dir": "/home/honeywell/.cache/torch/hub/facebookresearch_dinov3_main"},
    "dinov3_vitl16plus": {"hub_entry": "dinov3_vitl16plus", "hub_repo_dir": "/home/honeywell/.cache/torch/hub/facebookresearch_dinov3_main"},
    "dinov3_vith16plus": {"hub_entry": "dinov3_vith16plus", "hub_repo_dir": "/home/honeywell/.cache/torch/hub/facebookresearch_dinov3_main"},
    "dinov3_vit7b16": {"hub_entry": "dinov3_vit7b16", "hub_repo_dir": "/home/honeywell/.cache/torch/hub/facebookresearch_dinov3_main"},
}

_FEATURE_MODEL_CHOICES = tuple(DINOV3_FEATURE_MODEL_REGISTRY.keys())

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
_FAISS_GPU_RESOURCES = None
_FAISS_USE_CPU_INDEX = False
_FAISS_GPU_TEMP_MEM_MB = 256
_LOG_STREAM_HOLDER = []


class _TeeStream:
    """Mirror writes to terminal and log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def _enable_print_logging(log_path: str):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_f = open(log_path, "a", encoding="utf-8")
    _LOG_STREAM_HOLDER.append(log_f)
    sys.stdout = _TeeStream(sys.__stdout__, log_f)
    sys.stderr = _TeeStream(sys.__stderr__, log_f)
    print(f"[log] mirrored stdout/stderr to: {log_path}")


def _print_args(args):
    print("[args] ----")
    for key in sorted(vars(args)):
        print(f"[args] {key}: {getattr(args, key)}")
    print("[args] ----")




def fix_seed(number):
    common_utils.fix_seed(number)


def _collect_rng_state():
    state = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "random": random.getstate(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state):
    if state is None:
        return
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["random"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _default_train_checkpoint_path(saved_dir: str, run_name: str) -> str:
    return os.path.join(saved_dir, run_name + "_train_checkpoint.pt")


def save_train_checkpoint(
    path: str,
    epoch_completed: int,
    iteration: int,
    localnet: nn.Module,
    localnet_optimizer: optim.Optimizer,
    onetoone_optimizer,
    best_mean: float,
    best_result_by_class: dict,
):
    """epoch_completed: last finished epoch index (0-based)."""
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
        "rng_state": _collect_rng_state(),
    }
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _torch_load_train_checkpoint(path: str):
    # PyTorch >= 2.6 defaults weights_only=True; full checkpoints include numpy RNG state etc.
    # Always load to CPU first so torch RNG state stays ByteTensor on CPU.
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


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
    _restore_rng_state(ckpt.get("rng_state"))
    return (
        int(ckpt["epoch"]),
        int(ckpt["iteration"]),
        float(ckpt.get("best_mean", -1.0)),
        ckpt.get("best_result_by_class") or {},
    )


def find_matching(id_array):
    return common_utils.find_matching(id_array)


def _is_valid_local_torch_hub_dir(path: str) -> bool:
    return bool(path) and os.path.isdir(path) and os.path.isfile(os.path.join(path, "hubconf.py"))


def _resolve_dinov3_local_hub_dir(feature_model: str):
    if feature_model not in DINOV3_FEATURE_MODEL_REGISTRY:
        raise KeyError(f"Unknown feature_model={feature_model!r}")
    entry = DINOV3_FEATURE_MODEL_REGISTRY[feature_model]
    explicit = entry.get("hub_repo_dir")
    if explicit:
        p = os.path.expanduser(str(explicit))
        if _is_valid_local_torch_hub_dir(p):
            return p
    env_dir = os.environ.get("DINOV3_HUB_DIR")
    if env_dir:
        p = os.path.expanduser(env_dir)
        if _is_valid_local_torch_hub_dir(p):
            return p
    default_dir = os.path.join(torch.hub.get_dir(), "facebookresearch_dinov3_main")
    if _is_valid_local_torch_hub_dir(default_dir):
        return default_dir
    legacy = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov3_main")
    if _is_valid_local_torch_hub_dir(legacy):
        return legacy
    return None


def build_feature_extractor(args):
    entry = DINOV3_FEATURE_MODEL_REGISTRY[args.feature_model]
    hub_entry = entry["hub_entry"]
    repo_dir = _resolve_dinov3_local_hub_dir(args.feature_model)

    # DINO imports `trunc_normal_` via `from utils import ...`.
    # This project also has `utils.py`, so temporarily unshadow it.
    local_utils_module = sys.modules.get("utils")
    should_restore_utils = (
        local_utils_module is not None
        and os.path.abspath(getattr(local_utils_module, "__file__", "")).endswith(
            os.path.join("FUNAD", "utils.py")
        )
    )
    if should_restore_utils:
        del sys.modules["utils"]
    try:
        if repo_dir is not None:
            print(f"[DINOv3] load {hub_entry} from local hub: {repo_dir}")
            feature_extractor = torch.hub.load(
                repo_dir,
                hub_entry,
                source="local",
                pretrained=True,
            )
        else:
            print(f"[DINOv3] load {hub_entry} from GitHub: facebookresearch/dinov3")
            feature_extractor = torch.hub.load(
                "facebookresearch/dinov3",
                hub_entry,
                pretrained=True,
            )
    finally:
        if should_restore_utils:
            sys.modules["utils"] = local_utils_module
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()
    return feature_extractor


def extract_feature_batch(input_tensor, feature_extractor, args, return_cls_token=False):
    def _resolve_dino_block_indices():
        # CLI uses paper-style 1-based layer ids; DINO internals use 0-based block indices.
        num_blocks = len(getattr(feature_extractor, "blocks", []))
        requested_layers = list(getattr(args, "dino_layer_indices", [22, 23, 24, 25, 26, 27, 28]))
        requested_blocks = sorted({int(layer_id) - 1 for layer_id in requested_layers})
        valid_blocks = [idx for idx in requested_blocks if 0 <= idx < num_blocks]
        if len(valid_blocks) > 0:
            return valid_blocks

        # Fallback for shallower backbones (e.g., ViT-B/16): use middle 7 blocks if available.
        if num_blocks <= 0:
            return [0]
        if num_blocks <= 7:
            return list(range(num_blocks))
        center = num_blocks // 2
        start = max(0, center - 3)
        end = min(num_blocks, start + 7)
        start = max(0, end - 7)
        return list(range(start, end))

    with torch.no_grad():
        feature_extractor.eval()
        cls_token = None
        # DINOv3: mean-pool selected middle layers (paper default: 22-28).
        selected_blocks = _resolve_dino_block_indices()
        selected_layers = feature_extractor.get_intermediate_layers(
            input_tensor,
            n=selected_blocks,
            return_class_token=True,
        )
        patch_tokens_list = [layer_patch for layer_patch, _layer_cls in selected_layers]
        cls_tokens_list = [_layer_cls for _layer_patch, _layer_cls in selected_layers]
        x_prenorm = torch.stack(patch_tokens_list, dim=0).mean(dim=0)
        cls_tok = torch.stack(cls_tokens_list, dim=0).mean(dim=0)
        x_norm = cls_tok
        cls_token = cls_tok

    if args.use_cls_token:
        x_norm = torch.repeat_interleave(x_norm.unsqueeze(1), x_prenorm.shape[1], dim=1)
        x_prenorm = torch.cat([x_norm, x_prenorm], dim=-1)
    if return_cls_token:
        return x_prenorm, cls_token
    return x_prenorm


def infer_feature_dim(feature_extractor, train_loader, args):
    for batch in train_loader:
        images = batch[0]
        images = images.to(device, non_blocking=True)
        features = extract_feature_batch(images, feature_extractor, args)
        return int(features.shape[-1])
    raise RuntimeError("训练集为空，无法推断特征维度。")


def infer_patch_mask_size(train_dataset, feature_extractor, args):
    if len(train_dataset.samples) == 0:
        raise RuntimeError("训练集为空，无法推断 patch_mask_size。")
    image_path, _ = train_dataset.samples[0]
    image = Image.open(image_path).convert("RGB")
    image = train_dataset.transform_x(image).unsqueeze(0).to(device, non_blocking=True)
    features = extract_feature_batch(image, feature_extractor, args)
    num_patches = int(features.shape[1])
    patch_mask_size = int(np.sqrt(num_patches))
    if patch_mask_size * patch_mask_size != num_patches:
        raise RuntimeError(
            f"无法从 patch 数 {num_patches} 推断方形 patch 网格，请检查模型与输入尺寸。"
        )
    return patch_mask_size


def select_reference_indices_by_class(train_dataset, num_classes, args):
    rng = np.random.default_rng(args.seed)
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
            if args.strict_clean_reference:
                raise RuntimeError(msg)
            warnings.warn(msg)
            continue

        sample_n = min(max(1, args.num_reference_images_per_class), len(cls_candidates))
        selected = rng.choice(np.array(cls_candidates), size=sample_n, replace=False)
        reference_indices_by_class[class_idx] = sorted(selected.tolist())

    return reference_indices_by_class


def build_reference_memory_bank(
    train_dataset, reference_indices_by_class, feature_extractor, args
):
    all_indices = []
    for class_idx in sorted(reference_indices_by_class.keys()):
        all_indices.extend(reference_indices_by_class[class_idx])

    if len(all_indices) == 0:
        raise RuntimeError("没有参考图可用于构建残差记忆库。")

    reference_subset = Subset(train_dataset, all_indices)
    loader_kwargs = build_loader_kwargs(args.num_workers, pin_memory=True, prefetch_factor=2)
    reference_loader = DataLoader(
        reference_subset,
        batch_size=max(1, min(args.batch_size, 16)),
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    memory_by_class = {class_idx: [] for class_idx in reference_indices_by_class.keys()}
    with torch.no_grad():
        feature_extractor.eval()
        for batch in reference_loader:
            images, class_idx = batch[0], batch[1]
            images = images.to(device, non_blocking=True)
            base_features = extract_feature_batch(images, feature_extractor, args)
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
        index_by_class[int(class_idx)] = _build_faiss_index(cls_memory)

    if len(index_by_class) == 0:
        raise RuntimeError("参考图特征提取失败，无法构建残差记忆库。")

    return memory_np_by_class, index_by_class


def move_reference_memory_to_gpu(reference_memory_by_class):
    reference_memory_gpu_by_class = {}
    for class_idx, memory_np in reference_memory_by_class.items():
        if memory_np is None or memory_np.shape[0] == 0:
            continue
        # Keep reference memory in fp32 for stable nearest-neighbor matching.
        reference_memory_gpu_by_class[int(class_idx)] = torch.as_tensor(
            memory_np, dtype=torch.float32, device=device
        )
    return reference_memory_gpu_by_class


def compute_residual_feature_batch(
    features, class_idx_batch, reference_memory_by_class, reference_index_by_class
):
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


def compute_distance(feature):
    return common_utils.compute_distance(
        feature,
        use_cuda=use_cuda,
        use_cpu_index=_FAISS_USE_CPU_INDEX,
        gpu_temp_mem_mb=_FAISS_GPU_TEMP_MEM_MB,
    )


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


def build_test_loader(args, class_name):
    test_set = dataset_extract.MyDataset(
        dataset_path=args.data_path,
        dataset=args.dataset,
        class_name=class_name,
        is_train=False,
        resize=args.image_size,
        cropsize=args.crop_size,
    )
    # Eval loader is created/destroyed per class; keep workers small and
    # non-persistent to avoid accumulating processes/resources.
    eval_num_workers = max(1, min(int(args.num_workers), 2))
    loader_kwargs = build_loader_kwargs(
        eval_num_workers,
        pin_memory=False,
        prefetch_factor=1,
        persistent_workers=False,
    )
    return DataLoader(
        test_set,
        batch_size=16,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )


def get_oto_loss(args):
    return common_utils.get_oto_loss(args.oto_loss, device)


def evaluate_epoch(
    localnet,
    feature_extractor,
    test_loader,
    args,
    class_idx_eval,
    reference_memory_by_class,
    reference_index_by_class,
):
    return eval_utils.evaluate_residual_multiclass_epoch(
        localnet=localnet,
        feature_extractor=feature_extractor,
        test_loader=test_loader,
        args=args,
        class_idx_eval=class_idx_eval,
        reference_memory_by_class=reference_memory_by_class,
        reference_index_by_class=reference_index_by_class,
        device=device,
        extract_feature_batch_fn=extract_feature_batch,
        compute_residual_feature_batch_fn=compute_residual_feature_batch,
    )


def _build_faiss_index(feature_np):
    return common_utils.build_faiss_index(
        feature_np,
        use_cuda=use_cuda,
        use_cpu_index=_FAISS_USE_CPU_INDEX,
        gpu_temp_mem_mb=_FAISS_GPU_TEMP_MEM_MB,
    )


def aggregate_image_scores(score_2d, topk_ratio):
    return common_utils.aggregate_image_scores(score_2d, topk_ratio)


def save_moe_expert_visualizations(epoch, class_expert_count, class_names, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    counts = np.asarray(class_expert_count, dtype=np.float64)
    if counts.ndim != 2:
        return

    num_classes, num_expert = counts.shape
    if len(class_names) == num_classes:
        row_labels = list(class_names)
    else:
        row_labels = [f"class_{idx}" for idx in range(num_classes)]
    expert_labels = [f"expert_{idx}" for idx in range(num_expert)]

    row_sum = counts.sum(axis=1, keepdims=True)
    pref = np.divide(counts, row_sum, out=np.zeros_like(counts), where=row_sum > 0)

    stem = f"epoch_{epoch + 1:03d}"
    pref_csv = os.path.join(save_dir, f"{stem}_class_expert_pref.csv")
    pref_png = os.path.join(save_dir, f"{stem}_class_expert_pref.png")
    util_csv = os.path.join(save_dir, f"{stem}_expert_utilization.csv")
    util_png = os.path.join(save_dir, f"{stem}_expert_utilization.png")

    pref_df = pd.DataFrame(pref, index=row_labels, columns=expert_labels)
    pref_df.to_csv(pref_csv, encoding="utf-8")

    util = counts.sum(axis=0)
    util_sum = float(util.sum())
    if util_sum > 0:
        util = util / util_sum
    util_df = pd.DataFrame({"expert": expert_labels, "utilization": util})
    util_df.to_csv(util_csv, index=False, encoding="utf-8")

    fig_h = max(4.0, 0.45 * num_classes)
    fig_w = max(6.0, 0.8 * num_expert)
    plt.figure(figsize=(fig_w, fig_h))
    im = plt.imshow(pref, aspect="auto")
    plt.colorbar(im, fraction=0.035, pad=0.02)
    plt.xticks(np.arange(num_expert), expert_labels, rotation=35, ha="right")
    plt.yticks(np.arange(num_classes), row_labels)
    plt.xlabel("Expert")
    plt.ylabel("Class")
    plt.title("Class-to-Expert Routing Preference")
    plt.tight_layout()
    plt.savefig(pref_png, dpi=220)
    plt.close()

    plt.figure(figsize=(max(6.0, 0.8 * num_expert), 4.0))
    plt.bar(expert_labels, util)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Expert")
    plt.ylabel("Utilization ratio")
    plt.title("Expert Routing Utilization")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(util_png, dpi=220)
    plt.close()


def _get_faiss_gpu_resources(temp_mem_mb=None):
    if temp_mem_mb is None:
        temp_mem_mb = _FAISS_GPU_TEMP_MEM_MB
    return common_utils._get_faiss_gpu_resources(temp_mem_mb)


def _update_topk_features(top_feat, top_dist, cand_feat, cand_dist, k):
    return common_utils.update_topk_features(top_feat, top_dist, cand_feat, cand_dist, k)


def train_one_epoch(
    args,
    epoch,
    localnet,
    feature_extractor,
    localnet_optimizer,
    onetoone_optimizer,
    localnet_criterion,
    l_loss,
    local_loader,
    mini_loader,
    iteration,
    num_classes,
    reference_memory_by_class,
    reference_index_by_class,
    moe_expert_vis_ctx=None,
):
    total_batch = len(local_loader)
    threshold = args.threshold

    local_loss = 0
    oto_loss = 0
    bce_loss = 0
    gate_aux_loss = 0
    memory_bank_time = 0.0
    pseudo_label_time = 0.0
    kl_loss_time = 0.0
    pseudo_normal_correct = 0
    pseudo_normal_total = 0
    pseudo_anomaly_correct = 0
    pseudo_anomaly_total = 0

    distance_map = None
    confident_feature_bank = None
    global_dim = None
    moe_vis_enabled = bool(
        moe_expert_vis_ctx is not None
        and moe_expert_vis_ctx.get("enabled", False)
        and args.use_moe_discriminator
    )
    class_expert_count = None
    if moe_vis_enabled:
        num_expert = int(moe_expert_vis_ctx["num_expert"])
        class_expert_count = np.zeros((num_classes, num_expert), dtype=np.float64)

    if threshold <= 1:
        distance_map, confident_feature_bank, global_dim, mb_time, pl_time = precompute_pseudo_labels_multiclass_residual(
            args=args,
            localnet=localnet,
            feature_extractor=feature_extractor,
            mini_loader=mini_loader,
            num_classes=num_classes,
            reference_memory_by_class=reference_memory_by_class,
            reference_index_by_class=reference_index_by_class,
            device=device,
            extract_feature_batch_fn=extract_feature_batch,
            compute_residual_feature_batch_fn=compute_residual_feature_batch,
            aggregate_image_scores_fn=aggregate_image_scores,
            build_faiss_index_fn=_build_faiss_index,
            update_topk_features_fn=_update_topk_features,
            print_selected_score_distribution_fn=print_selected_score_distribution_by_class,
            print_selected_clean_ratio_fn=print_selected_clean_ratio,
            print_confusion_matrix_fn=print_full_dataset_confusion_matrix_by_raw_score,
            epoch=epoch,
            pseudo_label_scorer=None,
        )
        memory_bank_time += mb_time
        pseudo_label_time += pl_time

    for batch_data in tqdm.tqdm(local_loader, f"| run | train | {epoch + 1} |"):
        images, class_idx, sample_idx = batch_data[0], batch_data[1], batch_data[2]
        class_idx = class_idx.to(device, non_blocking=True)
        class_idx_np = class_idx.detach().cpu().numpy().astype(np.int64)
        sample_idx = sample_idx.detach().cpu().numpy()
        batch = images.shape[0]

        if batch != args.batch_size:
            continue

        images = images.to(device, non_blocking=True)
        x, cls_token = extract_feature_batch(
            images, feature_extractor, args, return_cls_token=True
        )
        x = compute_residual_feature_batch(
            x,
            class_idx,
            reference_memory_by_class,
            reference_index_by_class,
        )
        dim = int(x.shape[-1]) if global_dim is None else int(global_dim)

        distance = np.zeros((batch, 784), dtype=np.float32)
        if threshold <= 1:
            distance = distance_map[sample_idx]
        _copy = None
        if (threshold <= 1) and args.gaussian:
            uncertain_mask = (distance > threshold) & (distance < args.noise_threshold)
            uncertain_mask = uncertain_mask.reshape(-1)
            x_std = x.detach().std(dim=0).max(dim=0)[0]
            _copy = x.detach().clone().reshape(-1, dim)
            n_uncertain = int(uncertain_mask.sum())

            if n_uncertain > 0:
                if args.std is None:
                    std = x_std
                    std = std.repeat_interleave(n_uncertain).reshape(-1, n_uncertain).T
                    _copy[uncertain_mask] += torch.normal(mean=0, std=std)
                else:
                    _copy[uncertain_mask] += torch.normal(
                        mean=0, std=args.std, size=_copy[uncertain_mask].shape
                    )

            _copy = _copy.reshape(-1, 784, dim)

        pseudo_label_assign_start = time.perf_counter()
        class_idx_for_oto = class_idx
        local_label = torch.zeros((args.batch_size, 784))
        if args.threshold <= 1:
            distance_mask = torch.as_tensor(
                distance > threshold, dtype=torch.bool
            )
            local_label[distance_mask] = 1
        pseudo_label_time += time.perf_counter() - pseudo_label_assign_start

        localnet.train()
        localnet_optimizer.zero_grad()
        if args.alternative and onetoone_optimizer is not None:
            onetoone_optimizer.zero_grad()

        x = x.to(device, non_blocking=True)
        local_label = local_label.to(device, non_blocking=True)

        gate_cls_token = cls_token if cls_token.shape[0] == x.shape[0] else None
        patch_class_idx = None
        if args.use_moe_discriminator and args.moe_hard_class_gate:
            token_per_image = int(x.shape[1]) if x.dim() >= 2 else 1
            patch_class_idx = (
                class_idx_for_oto.to(device=x.device, dtype=torch.long)
                .reshape(-1, 1, 1)
                .expand(-1, token_per_image, 1)
                .contiguous()
            )
        batch_feature, local_pred = localnet(
            x,
            cls_token=gate_cls_token,
            patch_class_idx=patch_class_idx,
        )
        if moe_vis_enabled:
            discriminator = getattr(localnet, "discriminator", None)
            get_stats = getattr(discriminator, "get_latest_gate_stats", None)
            gate_stats = get_stats() if callable(get_stats) else None
            if gate_stats is not None:
                gate_top_k_idx, _gate_score = gate_stats
                token_per_image = int(batch_feature.shape[1]) if batch_feature.dim() >= 2 else 1
                class_for_gate = class_idx_for_oto
                if class_for_gate.shape[0] == batch_feature.shape[0]:
                    token_labels = (
                        class_for_gate.detach().cpu().numpy().astype(np.int64).repeat(token_per_image)
                    )
                    gate_idx = gate_top_k_idx.detach().reshape(-1).cpu().numpy().astype(np.int64)
                    if token_labels.size > 0 and gate_idx.size % token_labels.size == 0:
                        rep = gate_idx.size // token_labels.size
                        if rep > 1:
                            token_labels = np.repeat(token_labels, rep)
                        valid = (
                            (token_labels >= 0)
                            & (token_labels < num_classes)
                            & (gate_idx >= 0)
                            & (gate_idx < class_expert_count.shape[1])
                        )
                        if np.any(valid):
                            np.add.at(
                                class_expert_count,
                                (token_labels[valid], gate_idx[valid]),
                                1.0,
                            )

        pred_for_loss = local_pred
        if (threshold <= 1) and args.gaussian:
            _copy = _copy.to(device, non_blocking=True)
            gaussian_cls_token = (
                cls_token if cls_token.shape[0] == _copy.shape[0] else None
            )
            gaussian_patch_class_idx = None
            if args.use_moe_discriminator and args.moe_hard_class_gate:
                token_per_image = int(_copy.shape[1]) if _copy.dim() >= 2 else 1
                gaussian_patch_class_idx = class_idx_for_oto.to(
                    device=_copy.device, dtype=torch.long
                ).reshape(-1, 1, 1).expand(
                    -1, token_per_image, 1
                )
                gaussian_patch_class_idx = gaussian_patch_class_idx.contiguous()
            gaussian_feature, gaussian_pred = localnet(
                _copy,
                cls_token=gaussian_cls_token,
                patch_class_idx=gaussian_patch_class_idx,
            )
            pred_for_loss = gaussian_pred
            
        if args.balancing:
            _loss = compute_balanced_bce_loss(
                localnet_criterion=localnet_criterion,
                pred=pred_for_loss,
                target=local_label,
            )
        else:
            _loss = localnet_criterion(pred_for_loss, local_label)

        with torch.no_grad():
            pred_bin = pred_for_loss.detach() >= 0.5
            target_bin = local_label >= 0.5
            normal_mask = ~target_bin
            anomaly_mask = target_bin
            if normal_mask.any().item():
                pseudo_normal_total += int(normal_mask.sum().item())
                pseudo_normal_correct += int((~pred_bin[normal_mask]).sum().item())
            if anomaly_mask.any().item():
                pseudo_anomaly_total += int(anomaly_mask.sum().item())
                pseudo_anomaly_correct += int(pred_bin[anomaly_mask].sum().item())

        if (iteration >= args.iter) and args.kl:
            kl_start = time.perf_counter()
            _l_loss = compute_oto_loss_multiclass(
                args=args,
                batch_feature=batch_feature,
                local_pred=local_pred,
                class_idx=class_idx_for_oto,
                l_loss=l_loss,
                compute_distance_fn=compute_distance,
                find_matching_fn=find_matching,
            )
            kl_loss_time += time.perf_counter() - kl_start
        else:
            _l_loss = torch.tensor(0.0, device=local_pred.device)

        if args.use_moe_discriminator:
            discriminator = getattr(localnet, "discriminator", None)
            if discriminator is not None and hasattr(discriminator, "get_loss"):
                _gate_aux_loss = discriminator.get_loss(
                    clear=True, reduction="mean", default=None
                )
                if _gate_aux_loss is None:
                    _gate_aux_loss = torch.tensor(0.0, device=local_pred.device)
            else:
                _gate_aux_loss = torch.tensor(0.0, device=local_pred.device)
        else:
            _gate_aux_loss = torch.tensor(0.0, device=local_pred.device)

        _local_loss = _loss if args.alternative else (_loss + args.weight * _l_loss)
        if args.use_moe_discriminator:
            _local_loss = _local_loss + args.gate_aux_weight * _gate_aux_loss

        _local_loss.backward()
        localnet_optimizer.step()

        if args.alternative and onetoone_optimizer is not None and _l_loss.requires_grad:
            _l_loss.backward()
            onetoone_optimizer.step()

        local_loss += _local_loss / total_batch
        bce_loss += _loss / total_batch
        oto_loss += _l_loss / total_batch
        gate_aux_loss += _gate_aux_loss / total_batch
        iteration += 1

    if moe_vis_enabled:
        save_interval = max(1, int(moe_expert_vis_ctx.get("save_interval", 1)))
        if (epoch + 1) % save_interval == 0:
            save_moe_expert_visualizations(
                epoch=epoch,
                class_expert_count=class_expert_count,
                class_names=moe_expert_vis_ctx.get("class_names", []),
                save_dir=moe_expert_vis_ctx["save_dir"],
            )

    local_loss_value = (
        local_loss.item() if torch.is_tensor(local_loss) else float(local_loss)
    )
    bce_loss_value = bce_loss.item() if torch.is_tensor(bce_loss) else float(bce_loss)
    oto_loss_value = oto_loss.item() if torch.is_tensor(oto_loss) else float(oto_loss)
    gate_aux_loss_value = (
        gate_aux_loss.item() if torch.is_tensor(gate_aux_loss) else float(gate_aux_loss)
    )
    pseudo_normal_acc = (
        float(pseudo_normal_correct) / float(pseudo_normal_total)
        if pseudo_normal_total > 0
        else 0.0
    )
    pseudo_anomaly_acc = (
        float(pseudo_anomaly_correct) / float(pseudo_anomaly_total)
        if pseudo_anomaly_total > 0
        else 0.0
    )

    return (
        local_loss_value,
        bce_loss_value,
        oto_loss_value,
        gate_aux_loss_value,
        iteration,
        memory_bank_time,
        pseudo_label_time,
        kl_loss_time,
        pseudo_normal_acc,
        pseudo_normal_correct,
        pseudo_normal_total,
        pseudo_anomaly_acc,
        pseudo_anomaly_correct,
        pseudo_anomaly_total,
    )


def main():
    torch.autograd.set_detect_anomaly(True)
    args = parse_args(_FEATURE_MODEL_CHOICES)
    global _FAISS_USE_CPU_INDEX, _FAISS_GPU_TEMP_MEM_MB
    _FAISS_USE_CPU_INDEX = bool(args.faiss_cpu_index)
    _FAISS_GPU_TEMP_MEM_MB = int(args.faiss_gpu_temp_mem_mb)
    fix_seed(args.seed)

    saved_dir = os.path.join(args.save_path, args.dataset, args.noise)
    os.makedirs(saved_dir, exist_ok=True)
    _enable_print_logging(os.path.join(saved_dir, "run_stdout.log"))
    _print_args(args)

    if args.synthetic:
        raise ValueError("当前多类脚本暂不支持 --synthetic。")

    class_names = get_all_class_names(args.dataset)
    args.class_names = class_names
    num_classes_runtime = len(class_names)
    train_dataset = MultiClassFeatureDataset(
        data_path=args.data_path,
        dataset_name=args.dataset,
        image_size=args.image_size,
        crop_size=args.crop_size,
        seed=args.seed,
        shuffle=True,
    )

    reference_indices_by_class = select_reference_indices_by_class(
        train_dataset, len(class_names), args
    )
    for class_idx, ref_ids in reference_indices_by_class.items():
        print(
            f"class {class_idx} | selected clean references: {len(ref_ids)} "
            f"(target={args.num_reference_images_per_class})"
        )

    feature_extractor = build_feature_extractor(args)
    inferred_patch_mask_size = infer_patch_mask_size(train_dataset, feature_extractor, args)
    train_dataset.patch_mask_size = inferred_patch_mask_size
    print(f"inferred patch_mask_size from feature extractor: {inferred_patch_mask_size}")
    reference_memory_by_class_cpu, reference_index_by_class = build_reference_memory_bank(
        train_dataset, reference_indices_by_class, feature_extractor, args
    )
    reference_memory_by_class = move_reference_memory_to_gpu(reference_memory_by_class_cpu)
    remove_reference_samples_from_dataset(train_dataset, reference_indices_by_class)
    print(f"train samples after removing references: {len(train_dataset)}")

    loader_kwargs = build_loader_kwargs(args.num_workers, pin_memory=True, prefetch_factor=1,persistent_workers=False)
    local_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        **loader_kwargs,
    )
    mini_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    feature_dim = infer_feature_dim(feature_extractor, mini_loader, args)

    moe_num_expert_runtime = (
        num_classes_runtime if args.moe_hard_class_gate else int(args.moe_num_expert)
    )
    moe_top_k_runtime = 1 if args.moe_hard_class_gate else int(args.moe_top_k)
    localnet = model.localnet(
        len_feature=feature_dim,
        use_moe_discriminator=args.use_moe_discriminator,
        moe_num_expert=moe_num_expert_runtime,
        moe_top_k=moe_top_k_runtime,
        moe_use_cls_token=args.moe_use_cls_token,
        moe_hard_class_gate=args.moe_hard_class_gate,
    ).to(device)
    if args.use_moe_discriminator:
        discriminator = getattr(localnet, "discriminator", None)
        if discriminator is not None and hasattr(discriminator, "enable_gate_stats"):
            discriminator.enable_gate_stats(bool(args.moe_expert_vis_enable))
    localnet_optimizer = optim.RMSprop(localnet.parameters(), lr=args.lr, momentum=0.2)
    onetoone_optimizer = optim.Adam(localnet.parameters(), lr=args.lr) if args.alternative else None
    localnet_criterion = nn.BCELoss().to(device)
    l_loss = get_oto_loss(args)

    run_name = (
        "gaussian_"
        + str(args.gaussian)
        + "_noise_"
        + args.noise
        + "_balancing_"
        + str(args.balancing)
        + "_oto_"
        + str(args.kl)
        + "_weight_"
        + str(args.weight)
        + "_multiclass_residual"
    )
    moe_expert_vis_ctx = {
        "enabled": bool(args.moe_expert_vis_enable),
        "save_interval": max(1, int(args.moe_expert_vis_interval)),
        "save_dir": os.path.join(saved_dir, args.moe_expert_vis_dirname, run_name),
        "class_names": class_names,
        "num_expert": int(moe_num_expert_runtime),
    }
    if moe_expert_vis_ctx["enabled"] and args.use_moe_discriminator:
        os.makedirs(moe_expert_vis_ctx["save_dir"], exist_ok=True)

    resume_path = (args.resume or "").strip()

    start_epoch = 0
    iteration = 0
    best_mean = -1.0
    best_result_by_class = {}

    if resume_path:
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"--resume path not found: {resume_path}")
        print(f"[resume] loading training checkpoint: {resume_path}")
        last_done, iteration, best_mean, best_result_by_class = load_train_checkpoint(
            resume_path,
            localnet,
            localnet_optimizer,
            onetoone_optimizer,
        )
        start_epoch = last_done + 1
        if start_epoch >= args.epoch:
            print(
                f"[resume] checkpoint epoch {last_done} finished; "
                f"--epoch {args.epoch} means no further epochs to run."
            )
        else:
            print(
                f"[resume] will continue from epoch {start_epoch + 1} "
                f"(completed through epoch {last_done + 1}), global_iteration={iteration}"
            )

    for epoch in range(start_epoch, args.epoch):
        (
            local_loss_value,
            bce_loss_value,
            oto_loss_value,
            gate_aux_loss_value,
            iteration,
            memory_bank_time,
            pseudo_label_time,
            kl_loss_time,
            pseudo_normal_acc,
            pseudo_normal_correct,
            pseudo_normal_total,
            pseudo_anomaly_acc,
            pseudo_anomaly_correct,
            pseudo_anomaly_total,
        ) = train_one_epoch(
            args=args,
            epoch=epoch,
            localnet=localnet,
            feature_extractor=feature_extractor,
            localnet_optimizer=localnet_optimizer,
            onetoone_optimizer=onetoone_optimizer,
            localnet_criterion=localnet_criterion,
            l_loss=l_loss,
            local_loader=local_loader,
            mini_loader=mini_loader,
            iteration=iteration,
            num_classes=len(class_names),
            reference_memory_by_class=reference_memory_by_class,
            reference_index_by_class=reference_index_by_class,
            moe_expert_vis_ctx=moe_expert_vis_ctx,
        )

        print_epoch_losses(
            epoch,
            local_loss_value,
            bce_loss_value,
            oto_loss_value,
            gate_aux_loss=gate_aux_loss_value,
        )
        print_epoch_times(epoch, memory_bank_time, pseudo_label_time, kl_loss_time)
        print(
            f"epoch {epoch + 1} | pseudo normal->normal acc@0.5: {pseudo_normal_acc:.4f} "
            f"({pseudo_normal_correct}/{pseudo_normal_total})"
        )
        print(
            f"epoch {epoch + 1} | pseudo anomaly->anomaly acc@0.5: {pseudo_anomaly_acc:.4f} "
            f"({pseudo_anomaly_correct}/{pseudo_anomaly_total})"
        )

        if (epoch + 1) % args.eval_interval == 0:
            eval_rows = []
            mean_img_list = []
            mean_pixel_list = []
            mean_ap_sp_list = []
            mean_f1_sp_list = []
            mean_ap_px_list = []
            mean_f1_px_list = []
            mean_aupro_px_list = []
            for class_idx_eval, class_name in enumerate(class_names):
                test_loader = build_test_loader(args, class_name)
                (
                    auroc,
                    ap_sp,
                    f1_sp,
                    pixel_auroc,
                    ap_px,
                    f1_px,
                    aupro_px,
                ) = evaluate_epoch(
                    localnet,
                    feature_extractor,
                    test_loader,
                    args,
                    class_idx_eval=class_idx_eval,
                    reference_memory_by_class=reference_memory_by_class,
                    reference_index_by_class=reference_index_by_class,
                )
                eval_rows.append(
                    [class_name, auroc, ap_sp, f1_sp, pixel_auroc, ap_px, f1_px, aupro_px]
                )
                mean_img_list.append(auroc)
                mean_pixel_list.append(pixel_auroc)
                mean_ap_sp_list.append(ap_sp)
                mean_f1_sp_list.append(f1_sp)
                mean_ap_px_list.append(ap_px)
                mean_f1_px_list.append(f1_px)
                mean_aupro_px_list.append(aupro_px)
                print(
                    (
                        f"epoch {epoch + 1} | {class_name} | auroc: {auroc:.5f}, ap_sp: {ap_sp:.5f}, "
                        f"f1_sp: {f1_sp:.5f}, pixel auroc: {pixel_auroc:.5f}, ap_px: {ap_px:.5f}, "
                        f"f1_px: {f1_px:.5f}, aupro_px: {aupro_px:.5f}"
                    )
                )
                del test_loader

            epoch_mean_img = float(np.mean(mean_img_list))
            epoch_mean_pixel = float(np.mean(mean_pixel_list))
            epoch_mean_ap_sp = float(np.mean(mean_ap_sp_list))
            epoch_mean_f1_sp = float(np.mean(mean_f1_sp_list))
            epoch_mean_ap_px = float(np.mean(mean_ap_px_list))
            epoch_mean_f1_px = float(np.mean(mean_f1_px_list))
            epoch_mean_aupro_px = float(np.mean(mean_aupro_px_list))
            epoch_mean = (epoch_mean_img + epoch_mean_pixel) / 2
            print(f"epoch {epoch + 1} | multiclass img mean: {epoch_mean_img:.5f}")
            print(f"epoch {epoch + 1} | multiclass pixel mean: {epoch_mean_pixel:.5f}")
            print(f"epoch {epoch + 1} | multiclass ap_sp mean: {epoch_mean_ap_sp:.5f}")
            print(f"epoch {epoch + 1} | multiclass f1_sp mean: {epoch_mean_f1_sp:.5f}")
            print(f"epoch {epoch + 1} | multiclass ap_px mean: {epoch_mean_ap_px:.5f}")
            print(f"epoch {epoch + 1} | multiclass f1_px mean: {epoch_mean_f1_px:.5f}")
            print(f"epoch {epoch + 1} | multiclass aupro_px mean: {epoch_mean_aupro_px:.5f}")

            if epoch_mean > best_mean:
                best_mean = epoch_mean
                best_result_by_class = {row[0]: tuple(row[1:]) for row in eval_rows}
                torch.save(
                    {
                        "net": localnet.state_dict(),
                        "reference_memory_by_class": reference_memory_by_class_cpu,
                        "reference_indices_by_class": reference_indices_by_class,
                    },
                    os.path.join(saved_dir, run_name + "_localnet.pt"),
                )

            if args.save_log:
                with open(os.path.join(saved_dir, "log.txt"), "a") as file:
                    file.write(
                        f"epoch {epoch + 1} | total loss: {local_loss_value:.6f} | bce loss: {bce_loss_value:.6f} | one-to-one loss: {oto_loss_value:.6f} | gate aux loss: {gate_aux_loss_value:.6f} | gate aux weight: {args.gate_aux_weight:.6f} | multiclass img mean: {epoch_mean_img:.5f} | multiclass pixel mean: {epoch_mean_pixel:.5f} | multiclass ap_sp mean: {epoch_mean_ap_sp:.5f} | multiclass f1_sp mean: {epoch_mean_f1_sp:.5f} | multiclass ap_px mean: {epoch_mean_ap_px:.5f} | multiclass f1_px mean: {epoch_mean_f1_px:.5f} | multiclass aupro_px mean: {epoch_mean_aupro_px:.5f}\n"
                    )

        ckpt_path = _default_train_checkpoint_path(saved_dir, run_name)
        save_train_checkpoint(
            ckpt_path,
            epoch_completed=epoch,
            iteration=iteration,
            localnet=localnet,
            localnet_optimizer=localnet_optimizer,
            onetoone_optimizer=onetoone_optimizer,
            best_mean=best_mean,
            best_result_by_class=best_result_by_class,
        )
        print(f"[checkpoint] saved training state -> {ckpt_path}")

    if len(best_result_by_class) == 0:
        raise RuntimeError("训练未产生可用评估结果，请检查数据与参数。")

    results = []
    for class_name in class_names:
        auroc, ap_sp, f1_sp, pixel_auroc, ap_px, f1_px, aupro_px = best_result_by_class[class_name]
        results.append([class_name, auroc, ap_sp, f1_sp, pixel_auroc, ap_px, f1_px, aupro_px])
        print(
            (
                f"best | {class_name} | img_auroc: {auroc:.5f} | ap_sp: {ap_sp:.5f} "
                f"| f1_sp: {f1_sp:.5f} | pixel_auroc: {pixel_auroc:.5f} "
                f"| ap_px: {ap_px:.5f} | f1_px: {f1_px:.5f} | aupro_px: {aupro_px:.5f}"
            )
        )

    df = pd.DataFrame(
        results,
        columns=[
            "class",
            "auroc_sp",
            "ap_sp",
            "f1_sp",
            "auroc_px",
            "ap_px",
            "f1_px",
            "aupro_px",
        ],
    )
    result_path = os.path.join(
        saved_dir,
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    os.makedirs(result_path, exist_ok=True)
    df.to_excel(os.path.join(result_path, run_name + "_result.xlsx"), index=False)


if __name__ == "__main__":
    main()