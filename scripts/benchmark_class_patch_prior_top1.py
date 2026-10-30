"""Synthetic benchmark: prior_top1 vs dynamic_topk reference matching.

This measures only the fusion/matching stage with a fixed raw bank; it does
not include DINOv3 extraction, adapters/MoE, data loading or backward passes.

Example:
    python scripts/benchmark_class_patch_prior_top1.py \
        --num-classes 4 --bank-size 2000 --num-layers 8 --feature-dim 1024 \
        --batch-size 8 --patches 784 --topk 32 --iters 20
"""

import argparse
import os
import sys
import time

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.model.class_patch_fusion_adapter import ClassPatchFusionAdapter  # noqa: E402
import utils.train_utils as common_utils  # noqa: E402


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_ms(fn, warmup, iters, device):
    for _ in range(warmup):
        fn()
    _sync(device)
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync(device)
    return (time.perf_counter() - start) * 1000.0 / max(1, iters)


def parse_args():
    parser = argparse.ArgumentParser("benchmark_class_patch_prior_top1")
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--bank-size", type=int, default=2000,
                        help="Reference rows per class.")
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--feature-dim", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patches", type=int, default=784)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--query-chunk", type=int, default=256)
    parser.add_argument("--candidate-chunk", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto",
                        help="auto | cpu | cuda")
    parser.add_argument("--storage-dtype", type=str, default="float16",
                        choices=["float16", "float32"])
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    raw = {
        cls: torch.randn(args.bank_size, args.num_layers, args.feature_dim)
        for cls in range(args.num_classes)
    }
    if args.storage_dtype == "float16":
        raw = {cls: arr.half() for cls, arr in raw.items()}

    fusion = ClassPatchFusionAdapter(
        num_classes=args.num_classes,
        num_layers=args.num_layers,
        feature_dim=args.feature_dim,
        class_embedding_dim=32,
        gate_hidden_dim=64,
    ).to(device)
    with torch.no_grad():
        fusion.theta.normal_()
    cache = common_utils.FusedReferenceCache(
        raw,
        fusion,
        device,
        expected_num_layers=args.num_layers,
        expected_feature_dim=args.feature_dim,
        storage_dtype=args.storage_dtype,
        query_chunk_size=args.query_chunk,
        ref_chunk_size=4096,
        fuse_chunk_size=1024,
    )
    cache.refresh()

    class_idx = torch.arange(args.batch_size, device=device) % args.num_classes
    query = torch.randn(
        args.batch_size, args.patches, args.num_layers, args.feature_dim,
        device=device,
    )

    def run_prior():
        return common_utils.compute_class_patch_prior_top1_residual_batch(
            query,
            class_idx,
            cache,
            fusion,
            query_chunk_size=args.query_chunk,
            return_neighbor_ids=True,
        )

    def run_dynamic(topk):
        return common_utils.compute_class_patch_fused_residual_batch(
            query,
            class_idx,
            cache,
            fusion,
            topk=topk,
            query_chunk_size=args.query_chunk,
            candidate_chunk_size=args.candidate_chunk,
            return_neighbor_ids=True,
        )

    # Equivalence check under the same fresh cache/prior: prior_top1 is the
    # numerically same problem as dynamic_topk K=1.
    prior_residual, prior_ids = run_prior()
    old1_residual, old1_ids = run_dynamic(1)
    ids_equal = bool(torch.equal(prior_ids, old1_ids))
    residual_close = bool(
        torch.allclose(prior_residual, old1_residual, atol=1e-5)
    )

    timings = {
        "prior_top1": _time_ms(run_prior, args.warmup, args.iters, device),
        "dynamic_topk_k1": _time_ms(
            lambda: run_dynamic(1), args.warmup, args.iters, device
        ),
        f"dynamic_topk_k{args.topk}": _time_ms(
            lambda: run_dynamic(args.topk), args.warmup, args.iters, device
        ),
    }

    print(
        f"device={device} classes={args.num_classes} bank={args.bank_size}/class "
        f"L={args.num_layers} C={args.feature_dim} B={args.batch_size} "
        f"P={args.patches} storage={args.storage_dtype}"
    )
    print(f"K=1 ids equal to prior_top1: {ids_equal}")
    print(f"K=1 residual close to prior_top1: {residual_close}")
    for name, ms in timings.items():
        print(f"{name:>22}: {ms:8.2f} ms/batch")
    if torch.cuda.is_available() and device.type == "cuda":
        print(
            "peak CUDA memory: "
            f"{torch.cuda.max_memory_allocated(device) / (1024 ** 2):.1f} MiB"
        )


if __name__ == "__main__":
    main()
