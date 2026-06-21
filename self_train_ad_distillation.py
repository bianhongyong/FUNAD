"""
Distillation Stage — 蒸馏训练脚本

借鉴 MeDS/dinomaly/step_2_distillation.py 的蒸馏范式，
用第一阶段 memory_score_generation_multiclass_residual.py 产出的
异常分数图作为 teacher 软监督，训练 student localnet 直接预测
per-patch anomaly scores [0,1]。

核心设计：
  - Teacher 信号：memory_scores.pth（余弦距离），按类别 min-max 归一化到 [0,1]
  - Student：localnet，默认使用基础 3 层 MLP 判别器
  - 可选 --use_hard_gate：启用 MoE hard-gate 模式，每类固定路由到专属 expert
  - 无残差：使用 identity_residual（原始 DINOv3 features 直接输入）
  - 损失：MSE (l2) 或 L1
  - 评估：复用 evaluate_residual_multiclass_epoch + identity_residual

Usage:
    python self_train_ad_distillation.py \
      --data_path /path/to/dataset \
      --dataset mvtec \
      --save_path ./output_distill \
      --memory_score_path ./memory_scores/memory_scores.pth \
      --feature_model dinov3_vitl16 \
      --loss_function l2 \
      --n_iters 500 \
      --batch_size 16

    # 启用 hard-gate MoE（每类专属 expert）:
    python self_train_ad_distillation.py \
      ... --use_hard_gate
"""

import argparse
import gc
import os
import random
import sys
import time
import warnings
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# 确保项目根目录在 sys.path 上
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from dataset.feature_extract import (
    _FEATURE_MODEL_CHOICES,
    build_dinov3_feature_extractor,
    extract_dinov3_feature_batch,
    infer_dinov3_feature_dim,
    infer_dinov3_patch_mask_size,
    resolve_dino_block_indices,
)
from dataset.multiclass_feature_dataset import (
    MultiClassFeatureDataset,
    get_all_class_names,
    DINO_CLASS_LAYER_INDICES,
)
from src.model import model
from utils import evaluate as eval_utils
from utils.evaluate import _build_patch_class_idx
import utils.train_utils as common_utils

warnings.filterwarnings("ignore")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


# ---------------------------------------------------------------------------
# Dataset：加载 teacher scores 并按类归一化到 [0, 1]
# ---------------------------------------------------------------------------

class DistillationDataset(MultiClassFeatureDataset):
    """蒸馏数据集。

    继承 MultiClassFeatureDataset，额外加载 Phase 1 产出的 memory_scores.pth，
    将每个类的原始余弦距离按 min-max 归一化到 [0, 1] 区间，作为蒸馏的软监督信号。
    """

    def __init__(
        self,
        memory_score_path: str,
        data_path: str,
        dataset_name: str,
        image_size: int = 256,
        crop_size: int = 224,
        patch_mask_size: int = 28,
        seed: int = 0,
        shuffle: bool = True,
        class_names_override: list = None,
    ):
        super().__init__(
            data_path=data_path,
            dataset_name=dataset_name,
            image_size=image_size,
            crop_size=crop_size,
            patch_mask_size=patch_mask_size,
            seed=seed,
            shuffle=shuffle,
            class_names_override=class_names_override,
        )

        # 加载 memory_scores.pth
        # 结构: dict[cls_name][filename] -> Tensor[P] (余弦距离)
        raw_scores = torch.load(memory_score_path, map_location="cpu")
        print(f"[DistillationDataset] loaded memory_scores.pth with {len(raw_scores)} classes")

        # 按类 min-max 归一化到 [0, 1]，建立 (class_idx, filename) -> scores 的映射
        self.teacher_scores = {}  # key: (class_idx, filename_stem) -> Tensor[P] in [0,1]

        for cls_name, cls_dict in raw_scores.items():
            if cls_name not in self.class_to_idx:
                print(f"  WARNING: class {cls_name} not in dataset classes, skipping")
                continue
            cls_idx = self.class_to_idx[cls_name]
            all_vals = torch.cat(list(cls_dict.values()))
            vmin = all_vals.min().item()
            vmax = all_vals.max().item()

            if vmax > vmin:
                for fname, vals in cls_dict.items():
                    normalized = (vals - vmin) / (vmax - vmin)
                    # 数值稳定裁剪
                    normalized = normalized.clamp(0.0, 1.0)
                    self.teacher_scores[(cls_idx, fname)] = normalized
            else:
                for fname, vals in cls_dict.items():
                    self.teacher_scores[(cls_idx, fname)] = torch.zeros_like(vals)

            print(f"  [{cls_name}] normalized {len(cls_dict)} images, "
                  f"range [{vmin:.4f}, {vmax:.4f}] -> [0, 1]")

        # 建立 index -> (class_idx, filename_stem) 的映射
        self.idx_to_key = []
        missing_count = 0
        for image_path, class_idx in self.samples:
            fname = os.path.splitext(os.path.basename(image_path))[0]
            key = (class_idx, fname)
            if key not in self.teacher_scores:
                missing_count += 1
            self.idx_to_key.append(key)

        if missing_count > 0:
            print(f"  WARNING: {missing_count}/{len(self.samples)} samples have no teacher score "
                  f"(will use zeros)")

        # 输出统计信息
        num_patches = self.patch_mask_size * self.patch_mask_size
        print(f"[DistillationDataset] ready: {len(self.samples)} samples, "
              f"{num_patches} patches per image, {len(self.teacher_scores)} teacher entries")

    def __getitem__(self, idx):
        image, class_idx, global_idx, patch_mask = super().__getitem__(idx)
        key = self.idx_to_key[idx]
        teacher_score = self.teacher_scores.get(key)
        if teacher_score is None:
            # 若找不到 teacher score（如 noisy* 文件在 Phase 1 中未出现），用零填充
            teacher_score = torch.zeros(
                self.patch_mask_size * self.patch_mask_size,
                dtype=torch.float32,
            )
        return image, class_idx, global_idx, teacher_score, patch_mask


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _adapt_extract_fn(images, feature_extractor, args, return_cls_token=False, class_indices=None):
    """Adapter 统一 extract_dinov3_feature_batch 的调用签名。"""
    return extract_dinov3_feature_batch(
        images, feature_extractor, args.class_layer_indices,
        class_indices=class_indices,
        use_cls_token=args.use_cls_token, return_cls_token=return_cls_token,
    )


def build_test_loader(args, class_name):
    """为指定类别构建测试 DataLoader。"""
    from dataset.dataset_extract import MyDataset
    test_set = MyDataset(
        dataset_path=args.data_path,
        dataset=args.dataset,
        class_name=class_name,
        is_train=False,
        resize=args.image_size,
        cropsize=args.crop_size,
    )
    eval_num_workers = max(1, min(int(args.num_workers), 2))
    loader_kwargs = common_utils.build_loader_kwargs(
        eval_num_workers,
        pin_memory=False,
        prefetch_factor=1,
        persistent_workers=False,
    )
    return DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )


def evaluate_distill_epoch(localnet, feature_extractor, test_loader, args, class_idx_eval, device):
    """对单个类别进行评估。

    复用 evaluate_residual_multiclass_epoch，但传入 identity_residual 和 dummy
    reference_memory（均为 None），相当于无残差评估。这样做的好处是能利用
    class_idx_eval 正确指定该类的 layer indices。
    """
    return eval_utils.evaluate_residual_multiclass_epoch(
        localnet=localnet,
        feature_extractor=feature_extractor,
        test_loader=test_loader,
        args=args,
        class_idx_eval=class_idx_eval,
        reference_memory_by_class=None,
        reference_index_by_class=None,
        device=device,
        extract_feature_batch_fn=_adapt_extract_fn,
        compute_residual_feature_batch_fn=common_utils.identity_residual,
    )


def save_checkpoint(localnet, optimizer, epoch, it, save_dir, is_best=False):
    """保存蒸馏训练 checkpoint。"""
    prefix = "best" if is_best else f"epoch_{epoch}"
    ckpt_path = os.path.join(save_dir, f"{prefix}_distill_checkpoint.pt")
    torch.save({
        "epoch": epoch,
        "iter": it,
        "model_state_dict": localnet.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, ckpt_path)
    print(f"[Checkpoint] saved -> {ckpt_path}")


# ---------------------------------------------------------------------------
# 主训练函数
# ---------------------------------------------------------------------------

def train(args):
    setup_seed(args.seed)
    saved_dir = args.save_path
    os.makedirs(saved_dir, exist_ok=True)

    print(f"[Distillation] device: {device}")
    print(f"[Distillation] feature_model: {args.feature_model}")
    print(f"[Distillation] loss_function: {args.loss_function}")
    print(f"[Distillation] n_iters: {args.n_iters}")
    print(f"[Distillation] batch_size: {args.batch_size}")
    print(f"[Distillation] lr: {args.lr}")
    print(f"[Distillation] memory_score_path: {args.memory_score_path}")

    # -------------------------------------------------------------------
    # Step 1: 获取类别信息
    # -------------------------------------------------------------------
    class_names = get_all_class_names(args.dataset)
    args.class_names = class_names
    num_classes = len(class_names)

    # 构建 per-class layer indices dict
    args.class_layer_indices = {
        i: DINO_CLASS_LAYER_INDICES.get(name, [-1])
        for i, name in enumerate(class_names)
    }
    _all_layer_values = sorted(set(
        idx for lst in args.class_layer_indices.values() for idx in lst
    ))
    print(f"[Distillation] classes ({num_classes}): {class_names}")

    # -------------------------------------------------------------------
    # Step 2: 构建蒸馏数据集
    # -------------------------------------------------------------------
    train_dataset = DistillationDataset(
        memory_score_path=args.memory_score_path,
        data_path=args.data_path,
        dataset_name=args.dataset,
        image_size=args.image_size,
        crop_size=args.crop_size,
        seed=args.seed,
        shuffle=True,
    )
    print(f"[Distillation] train samples: {len(train_dataset)}")

    # -------------------------------------------------------------------
    # Step 3: 构建 DINOv3 feature extractor（frozen backbone）
    # -------------------------------------------------------------------
    feature_extractor = build_dinov3_feature_extractor(args.feature_model, device)
    feature_extractor.eval()
    for p in feature_extractor.parameters():
        p.requires_grad = False

    # 推断 patch mask size 和 feature dim
    inferred_patch_size = infer_dinov3_patch_mask_size(
        train_dataset, feature_extractor, _all_layer_values, args.use_cls_token, device
    )
    train_dataset.patch_mask_size = inferred_patch_size
    print(f"[Distillation] inferred patch_mask_size: {inferred_patch_size}")

    _selected_blocks = resolve_dino_block_indices(feature_extractor, _all_layer_values)
    _display = [b + 1 for b in _selected_blocks]
    print(f"[DINOv3] aggregating layers {_display} (0-based blocks {_selected_blocks})")

    # -------------------------------------------------------------------
    # Step 4: 构建 DataLoader
    # -------------------------------------------------------------------
    loader_kwargs = common_utils.build_loader_kwargs(
        args.num_workers, pin_memory=True, prefetch_factor=1, persistent_workers=False,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )

    # 用一个小 batch 推断 feature dim
    mini_loader = DataLoader(
        train_dataset,
        batch_size=min(2, args.batch_size),
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    feature_dim = infer_dinov3_feature_dim(
        feature_extractor, mini_loader, _all_layer_values, args.use_cls_token, device
    )
    print(f"[Distillation] feature_dim: {feature_dim}")

    # 清理 mini_loader
    del mini_loader
    gc.collect()

    # -------------------------------------------------------------------
    # Step 5: 构建 Student localnet
    # -------------------------------------------------------------------
    # --use_hard_gate → MoE hard-gate（每类专属 expert，top_k=1，无 aux loss）
    # 默认 → 基础 3 层 MLP 判别器
    use_moe = args.use_hard_gate
    # 同步到 moe_hard_class_gate，供 evaluate_residual_multiclass_epoch 检测
    args.moe_hard_class_gate = use_moe
    args.use_moe_discriminator = use_moe
    student = model.localnet(
        len_feature=feature_dim,
        use_moe_discriminator=use_moe,
        moe_num_expert=num_classes if use_moe else 4,
        moe_top_k=1 if use_moe else 2,
        moe_use_cls_token=False,
        moe_hard_class_gate=use_moe,
        num_classes=num_classes,
        class_conditioned_adapter=True,
    ).to(device)

    # 统计参数量
    trainable_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in student.parameters())
    print(f"[Distillation] student parameters: {total_params:,} total, {trainable_params:,} trainable")

    # -------------------------------------------------------------------
    # Step 6: 优化器 & 损失函数
    # -------------------------------------------------------------------
    optimizer = torch.optim.RMSprop(student.parameters(), lr=args.lr, momentum=0.2)

    # -------------------------------------------------------------------
    # Step 7: 训练循环（蒸馏）
    # -------------------------------------------------------------------
    it = 0
    best_mean_auroc_sp = 0.0
    best_metrics = None

    print(f"\n{'=' * 60}")
    print(f"开始蒸馏训练 (n_iters={args.n_iters})")
    print(f"{'=' * 60}")

    for epoch in range(int(np.ceil(args.n_iters / len(train_loader)))):
        student.train()
        epoch_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False, ncols=100)
        for batch_data in pbar:
            images, class_idx, _, teacher_scores, _ = batch_data
            batch = images.shape[0]
            if batch != args.batch_size:
                continue

            images = images.to(device, non_blocking=True)
            class_idx = class_idx.to(device, non_blocking=True)
            teacher_scores = teacher_scores.float().to(device, non_blocking=True)

            # ---- Forward ----
            features, cls_token = extract_dinov3_feature_batch(
                images, feature_extractor, args.class_layer_indices,
                class_indices=class_idx,
                use_cls_token=args.use_cls_token, return_cls_token=True,
            )
            features = common_utils.identity_residual(
                features, class_idx, None, None
            )

            patch_class_idx = None
            if args.use_hard_gate:
                patch_class_idx = _build_patch_class_idx(class_idx, features)
            _, student_scores = student(
                features,
                cls_token=None,
                patch_class_idx=patch_class_idx,
                class_idx=class_idx,
            )

            if args.loss_function == 'l2':
                distill_loss = F.mse_loss(student_scores, teacher_scores)
            elif args.loss_function == 'l1':
                distill_loss = F.l1_loss(student_scores, teacher_scores)
            else:
                raise ValueError(f"Unsupported loss_function: {args.loss_function}")

            total_loss = distill_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=0.1)
            optimizer.step()

            epoch_losses.append(distill_loss.item())

            pbar.set_postfix(loss=f"{distill_loss.item():.6f}", it=f"{it}/{args.n_iters}")

            it += 1
            if it >= args.n_iters:
                break

        # ---- Epoch 结束 ----
        mean_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        print(f"[Distill] Epoch {epoch} | iter {it}/{args.n_iters} | "
              f"loss={mean_loss:.6f}")

        # ---- 评估 ----
        if (epoch + 1) % args.eval_interval == 0 or it >= args.n_iters:
            eval_start = time.perf_counter()
            print(f"\n{'=' * 60}")
            print(f"Evaluation at epoch {epoch}, iter {it}")
            print(f"{'=' * 60}")

            student.eval()
            metrics_rows = []
            for cls_name in class_names:
                test_loader = build_test_loader(args, cls_name)
                if len(test_loader.dataset) == 0:
                    print(f"  [{cls_name}] empty test set, skipping")
                    continue

                cls_idx = class_names.index(cls_name)
                results = evaluate_distill_epoch(
                    student, feature_extractor, test_loader, args,
                    class_idx_eval=cls_idx, device=device,
                )
                auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results

                metrics_rows.append({
                    "class": cls_name,
                    "auroc_sp": auroc_sp, "ap_sp": ap_sp, "f1_sp": f1_sp,
                    "auroc_px": auroc_px, "ap_px": ap_px, "f1_px": f1_px, "aupro_px": aupro_px,
                })
                print(f"  [{cls_name}] I-AUROC:{auroc_sp:.4f} I-AP:{ap_sp:.4f} I-F1:{f1_sp:.4f} | "
                      f"P-AUROC:{auroc_px:.4f} P-AP:{ap_px:.4f} P-F1:{f1_px:.4f} P-AUPRO:{aupro_px:.4f}")

                # 释放 test_loader 资源
                del test_loader
                gc.collect()
                if use_cuda:
                    torch.cuda.empty_cache()

            if metrics_rows:
                mean_auroc_sp = np.mean([r["auroc_sp"] for r in metrics_rows])
                mean_auroc_px = np.mean([r["auroc_px"] for r in metrics_rows])
                mean_aupro = np.mean([r["aupro_px"] for r in metrics_rows])
                print(f"\n  [Mean] I-AUROC:{mean_auroc_sp:.4f} P-AUROC:{mean_auroc_px:.4f} P-AUPRO:{mean_aupro:.4f}")

                # 保存最佳 checkpoint
                if mean_auroc_sp > best_mean_auroc_sp:
                    best_mean_auroc_sp = mean_auroc_sp
                    best_metrics = metrics_rows
                    save_checkpoint(student, optimizer, epoch, it, saved_dir, is_best=True)

                # 保存评估结果 CSV
                import pandas as pd
                df = pd.DataFrame(metrics_rows)
                csv_path = os.path.join(saved_dir, f"eval_epoch_{epoch}.csv")
                df.to_csv(csv_path, index=False)
                print(f"  Evaluation results saved -> {csv_path}")

            print(f"  Evaluation time: {time.perf_counter() - eval_start:.1f}s\n")
            student.train()

        if it >= args.n_iters:
            break

    # -------------------------------------------------------------------
    # 训练结束
    # -------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"蒸馏训练完成！总迭代 {it}，最佳 Mean I-AUROC: {best_mean_auroc_sp:.4f}")
    print(f"{'=' * 60}")

    if best_metrics:
        print("\n最佳 Checkpoint 评估结果（按类别）:")
        for r in best_metrics:
            print(f"  [{r['class']}] I-AUROC:{r['auroc_sp']:.4f} I-AP:{r['ap_sp']:.4f} "
                  f"I-F1:{r['f1_sp']:.4f} | P-AUROC:{r['auroc_px']:.4f} "
                  f"P-AP:{r['ap_px']:.4f} P-F1:{r['f1_px']:.4f} P-AUPRO:{r['aupro_px']:.4f}")

        mean_row = {k: np.mean([r[k] for r in best_metrics]) for k in
                     ["auroc_sp", "ap_sp", "f1_sp", "auroc_px", "ap_px", "f1_px", "aupro_px"]}
        print(f"\n  [Best Mean] I-AUROC:{mean_row['auroc_sp']:.4f} I-AP:{mean_row['ap_sp']:.4f} "
              f"I-F1:{mean_row['f1_sp']:.4f} | P-AUROC:{mean_row['auroc_px']:.4f} "
              f"P-AP:{mean_row['ap_px']:.4f} P-F1:{mean_row['f1_px']:.4f} P-AUPRO:{mean_row['aupro_px']:.4f}")

    # 保存最终 checkpoint
    save_checkpoint(student, optimizer, epoch, it, saved_dir, is_best=False)
    print(f"\n[Distillation] Done. All outputs saved to {saved_dir}")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        "self_train_ad_distillation",
        description="Distillation: 用 Phase 1 memory scores 作为 teacher 信号训练 student localnet",
    )
    # 数据相关
    parser.add_argument("--data_path", type=str,
                        default="/media/honeywell/D/bhy/dataset/MVTec_overlap/MVTec_noisy10")
    parser.add_argument("--dataset", type=str, default="mvtec", choices=["mvtec", "visa"])
    parser.add_argument("--save_path", type=str, default="./output_distill")
    parser.add_argument("--memory_score_path", type=str, required=True,
                        help="Path to memory_scores.pth from Phase 1")

    # 模型
    parser.add_argument("--feature_model", type=str, choices=_FEATURE_MODEL_CHOICES,
                        default="dinov3_vitb16")
    parser.add_argument("--use_cls_token", type=common_utils.str2bool, default="False")
    parser.add_argument("--use_hard_gate", action="store_true",
                        help="启用 MoE hard-gate 模式（每类固定路由到专属 expert），默认使用基础 MLP 判别器")

    # 训练
    parser.add_argument("--n_iters", type=int, default=2000,
                        help="Total training iterations (like MeDS n_iters)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--loss_function", type=str, choices=["l1", "l2"], default="l2",
                        help="Distillation loss: L1 or MSE (l2)")

    # 图像尺寸
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--crop_size", type=int, default=448)

    # 评估
    parser.add_argument("--eval_interval", type=int, default=5,
                        help="Evaluate every N epochs")
    parser.add_argument("--img_score_topk_ratio", type=float, default=0.01,
                        help="Image-level score uses mean of top-k patch scores")

    # 其他
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)

    return parser.parse_args()


def main():
    args = parse_args()
    # 补全 save_path（若只给了前缀，追加 dataset 名）
    if not args.save_path.endswith(args.dataset):
        args.save_path = os.path.join(args.save_path, args.dataset)
    os.makedirs(args.save_path, exist_ok=True)

    train(args)


if __name__ == "__main__":
    main()
