"""Minimal multiclass baseline: one frozen layer, residual, CRAS-inspired DAFS/BCE.

No feature fusion, class adapter, MoE, LOIO, spatial masks, OTO or pseudo labels.
Run with --help for training and --checkpoint <last.pt> for evaluation only.
"""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.model.simple_residual import ClassReferenceResidual, ResidualMLP
from src.train.simple_residual_training import (
    residual_training_batch, cras_bce_loss,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", required=True, help="Clean MVTec-layout dataset root.")
    parser.add_argument("--save_path", default="./save_results/simple_residual")
    parser.add_argument("--dataset", choices=["mvtec", "visa"], default="mvtec")
    parser.add_argument("--feature_model", default="dinov3_vitl16")
    parser.add_argument("--feature_layer", type=int, default=-1, help="1-based block; -1 = last.")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--crop_size", type=int, default=448)
    parser.add_argument("--num_reference_images_per_class", type=int, default=8)
    parser.add_argument("--noise_std", type=float, default=0.015,
                        help="Base Gaussian std (not variance); CRAS official default=0.015.")
    parser.add_argument("--synthesis_mode", choices=["cras", "gaussian"], default="cras",
                        help="CRAS distance-guided noise, or fixed-Gaussian control.")
    parser.add_argument("--cras_beta", type=float, default=0.3,
                        help="DAFS scale adjustment strength; CRAS official default=0.3.")
    parser.add_argument("--cras_eps", type=float, default=1e-8,
                        help="Numerical floor for zero residual norms.")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--eval_compute_pro", action="store_true")
    parser.add_argument("--img_score_topk_ratio", type=float, default=0.01)
    parser.add_argument("--query_chunk", type=int, default=512)
    parser.add_argument("--reference_chunk", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", default="", help="Evaluate a simple-baseline checkpoint only.")
    args = parser.parse_args()
    positive = ("epoch", "batch_size", "eval_interval", "query_chunk", "reference_chunk",
                "num_reference_images_per_class", "image_size", "crop_size", "lr", "noise_std", "cras_eps")
    for key in positive:
        value = getattr(args, key)
        if not np.isfinite(value) or value <= 0:
            parser.error(f"--{key} must be finite and positive")
    if args.num_workers < 0 or args.crop_size > args.image_size:
        parser.error("Require num_workers >= 0 and crop_size <= image_size")
    if not 0 < args.img_score_topk_ratio <= 1:
        parser.error("--img_score_topk_ratio must be in (0,1]")
    if not np.isfinite(args.cras_beta) or not 0 <= args.cras_beta <= 1:
        parser.error("--cras_beta must be finite and in [0,1]")
    return args


def split_references(dataset, count, seed):
    """Reserve normal images for references; never query a training image against itself."""
    rng = np.random.default_rng(seed)
    refs = {}
    for cls, name in enumerate(dataset.class_names):
        candidates = [i for i, (path, c) in enumerate(dataset.samples) if c == cls]
        if len(candidates) < 2:
            raise ValueError(f"{name}: need at least 2 normal images for reference/query separation.")
        refs[cls] = sorted(rng.choice(candidates, min(count, len(candidates) - 1), replace=False).tolist())
    reserved = {i for rows in refs.values() for i in rows}
    queries = [i for i in range(len(dataset.samples)) if i not in reserved]
    return refs, queries


def main():
    args = parse_args()
    checkpoint = None
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if checkpoint.get("format") != "simple_residual_v1":
            raise ValueError("Expected a simple_residual_v1 checkpoint, not a fusion checkpoint.")
        # Preprocessing, scoring and architecture must match the saved experiment.
        for key in ("dataset", "feature_model", "feature_layer", "image_size", "crop_size",
                    "img_score_topk_ratio", "seed", "noise_std"):
            setattr(args, key, checkpoint["config"][key])
        # v1 checkpoints produced before DAFS remain evaluable, with truthful logs.
        args.synthesis_mode = checkpoint["config"].get("synthesis_mode", "gaussian")
        args.cras_beta = checkpoint["config"].get("cras_beta", 0.0)
        args.cras_eps = checkpoint["config"].get("cras_eps", 1e-8)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Keep --help and CPU core tests independent of dataset/backbone dependencies.
    from dataset.feature_extract import build_dinov3_feature_extractor, resolve_dino_block_indices
    from dataset.multiclass_feature_dataset import MultiClassFeatureDataset, get_all_class_names
    from dataset.dataset_extract import MyDataset
    from utils import evaluate as eval_utils
    from utils.logging import enable_print_logging

    output = Path(args.save_path) / args.dataset / f"seed_{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    enable_print_logging(str(output / ("eval_stdout.log" if checkpoint else "run_stdout.log")))
    class_names = list(get_all_class_names(args.dataset))
    if checkpoint and class_names != checkpoint["class_names"]:
        raise ValueError("Checkpoint class order differs from the selected dataset.")
    backbone = build_dinov3_feature_extractor(args.feature_model, device)
    backbone.requires_grad_(False)
    block = resolve_dino_block_indices(backbone, [args.feature_layer])[0]
    args.feature_layer = block + 1
    args.use_moe_discriminator = False

    @torch.no_grad()
    def extract(images, *_unused, **_kwargs):
        backbone.eval()
        # Select exactly one layer; no stacking, averaging, weights or fusion module.
        selected = backbone.get_intermediate_layers(images, n=[block], return_class_token=True)
        return selected[0][0].float()

    def loader(dataset, shuffle=False, batch_size=None):
        return DataLoader(dataset, batch_size=batch_size or args.batch_size, shuffle=shuffle,
                          num_workers=args.num_workers, pin_memory=device.type == "cuda")

    if checkpoint:
        reference_file = Path(args.checkpoint).resolve().parent / checkpoint["reference_file"]
        saved_references = torch.load(reference_file, map_location="cpu", weights_only=False)
        references = saved_references["references"]
    else:
        dataset = MultiClassFeatureDataset(args.data_path, args.dataset, args.image_size,
                                           args.crop_size, seed=args.seed, shuffle=False)
        if any(Path(path).name.lower().startswith("noisy") for path, _ in dataset.samples):
            raise ValueError("This baseline expects clean train/good; noisy* files were found.")
        ref_ids, query_ids = split_references(dataset, args.num_reference_images_per_class, args.seed)
        references = {c: [] for c in ref_ids}
        reference_paths = {c: [dataset.samples[i][0] for i in ids] for c, ids in ref_ids.items()}
        with torch.no_grad():
            subset = Subset(dataset, [i for ids in ref_ids.values() for i in ids])
            for images, classes, *_ in loader(subset, batch_size=min(args.batch_size, 8)):
                features = extract(images.to(device)).cpu()
                for cls in classes.unique().tolist():
                    references[cls].append(features[classes == cls].flatten(0, 1))
        references = {c: torch.cat(rows) for c, rows in references.items()}
        train_loader = loader(Subset(dataset, query_ids), shuffle=True)
        print(f"[Data] {len(query_ids)} query images; references excluded from queries")

    residual_fn = ClassReferenceResidual(references, args.query_chunk, args.reference_chunk).to(device)
    detector = ResidualMLP(residual_fn.feature_dim).to(device)
    if checkpoint:
        detector.load_state_dict(checkpoint["net"], strict=True)
    print(f"[Baseline] {args.feature_model} block={args.feature_layer} | "
          f"C={residual_fn.feature_dim} | shared MLP | synthesis={args.synthesis_mode} "
          f"base_std={args.noise_std} beta={args.cras_beta} | loss=CRAS_BCE(1:1)")
    print(f"[Parameters] trainable={sum(p.numel() for p in detector.parameters()):,}")

    def evaluate(epoch):
        # Existing metric/postprocessing implementation, identical for train and standalone eval.
        detector.eval()
        results = []
        for cls, name in enumerate(class_names):
            test = MyDataset(dataset_path=args.data_path, dataset=args.dataset, class_name=name,
                             is_train=False, resize=args.image_size, cropsize=args.crop_size)
            metrics = eval_utils.evaluate_residual_multiclass_epoch(
                localnet=detector, feature_extractor=backbone, test_loader=loader(test), args=args,
                class_idx_eval=cls, reference_memory_by_class=None, reference_index_by_class=None,
                device=device, extract_feature_batch_fn=extract,
                compute_residual_feature_batch_fn=residual_fn, return_dict=True,
                compute_pro_metric=args.eval_compute_pro,
            )
            results.append({"class": name, **metrics})
            print(f"[Eval {epoch}] {name}: {metrics}")
        mean = {key: float(np.mean([r[key] for r in results]))
                for key in results[0] if key != "class" and results[0][key] is not None}
        results.append({"class": "macro_mean", **mean})
        path = output / f"{'eval_' if checkpoint else ''}metrics_epoch_{epoch:04d}.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"[Macro mean] {mean}")

    if checkpoint:
        evaluate(checkpoint["epoch"])
        return

    with (output / "config.json").open("w", encoding="utf-8") as stream:
        json.dump(vars(args), stream, indent=2)
    optimizer = torch.optim.RMSprop(detector.parameters(), lr=args.lr, momentum=0.2)
    # Save the fixed reference bank once, rather than duplicating it in every epoch checkpoint.
    reference_file = output / "references.pt"
    torch.save({"references": references, "reference_paths": reference_paths}, reference_file)
    for epoch in range(1, args.epoch + 1):
        detector.train()
        pos_total = neg_total = rms_total = steps = 0
        scale_total = perturbation_total = 0.0
        scale_min, scale_max = float("inf"), float("-inf")
        for images, classes, *_ in train_loader:
            residual = residual_fn(extract(images.to(device)), classes.to(device))
            features, _labels, stats = residual_training_batch(
                residual, args.noise_std, beta=args.cras_beta,
                mode=args.synthesis_mode, eps=args.cras_eps,
            )
            optimizer.zero_grad(set_to_none=True)
            loss, neg, pos = cras_bce_loss(detector.logits(features))
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss in epoch {epoch}")
            loss.backward()
            optimizer.step()
            pos_total += pos.item()
            neg_total += neg.item()
            rms_total += residual.square().mean().sqrt().item()
            scale_total += stats["scale_mean"]
            scale_min = min(scale_min, stats["scale_min"])
            scale_max = max(scale_max, stats["scale_max"])
            perturbation_total += stats["perturbation_rms"]
            steps += 1
        print(f"[Epoch {epoch}] normal_bce={neg_total / steps:.6f} "
              f"synthetic_bce={pos_total / steps:.6f} residual_rms={rms_total / steps:.6f} "
              f"noise_std={args.noise_std} scale_mean={scale_total / steps:.6f} "
              f"scale_range=[{scale_min:.6f},{scale_max:.6f}] "
              f"perturbation_rms={perturbation_total / steps:.6f}")
        # Always save the last model; test labels do not select a checkpoint.
        torch.save({"format": "simple_residual_v1", "epoch": epoch,
                    "config": vars(args), "class_names": class_names,
                    "synthesis_config": {"mode": args.synthesis_mode, "noise_std": args.noise_std,
                                         "beta": args.cras_beta, "eps": args.cras_eps},
                    "loss_config": {"name": "cras_bce", "normal_weight": 1.0,
                                    "synthetic_weight": 1.0, "reduction": "mean_per_group"},
                    "net": detector.state_dict(), "reference_file": reference_file.name},
                   output / "last.pt")
        if epoch % args.eval_interval == 0 or epoch == args.epoch:
            evaluate(epoch)


if __name__ == "__main__":
    main()
