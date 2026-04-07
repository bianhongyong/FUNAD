#!/usr/bin/env python3
"""Smoke test for FMoETransformerMLP forward inference."""

import argparse
from pathlib import Path
import sys

import torch

# Ensure repo root is importable when running this file directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model.moeblock.transformer import FMoETransformerMLP


def _run_case(
    model: FMoETransformerMLP, inp: torch.Tensor, case_name: str, expected_last_dim: int
) -> None:
    with torch.no_grad():
        out = model(inp)

    expected_shape = (*inp.shape[:-1], expected_last_dim)
    if out.shape != expected_shape:
        raise RuntimeError(
            f"[{case_name}] output shape mismatch: got {tuple(out.shape)}, "
            f"expected {tuple(expected_shape)}"
        )
    if not torch.isfinite(out).all():
        raise RuntimeError(f"[{case_name}] output contains NaN or Inf")

    print(f"[PASS] {case_name}: input {tuple(inp.shape)} -> output {tuple(out.shape)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check FMoETransformerMLP forward inference on random tensors."
    )
    parser.add_argument("--num-expert", type=int, default=4)
    parser.add_argument("--d-input", type=int, default=256)
    parser.add_argument("--d-hidden", type=int, default=1024)
    parser.add_argument("--d-output", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="auto: use cuda if available, else cpu",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available, cannot run with --device cuda", file=sys.stderr)
        return 1

    model = FMoETransformerMLP(
        num_expert=args.num_expert,
        d_input=args.d_input,
        d_hidden=args.d_hidden,
        d_output=args.d_output,
        top_k=args.top_k,
        world_size=1,
    ).to(device)
    model.eval()

    inp_2d = torch.randn(args.batch_size, args.d_input, device=device)
    inp_3d = torch.randn(args.batch_size, args.seq_len, args.d_input, device=device)

    _run_case(model, inp_2d, "2D_input", expected_last_dim=args.d_output)
    _run_case(model, inp_3d, "3D_input", expected_last_dim=args.d_output)

    print(f"All forward checks passed on device={device}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
