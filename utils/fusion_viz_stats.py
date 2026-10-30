"""Streaming statistics for class-patch fusion visualisation.

The visualisation entry point processes a full test split without caching
per-patch weights.  This module accumulates the only quantities the six figures
need:

* per-class, per-layer ``mean`` / ``std`` of the patch weights ``alpha[p, l]``;
* per-patch entropy ``H[p] = -sum_l alpha[p,l] log alpha[p,l]``;
* per-patch L1 distance to the class prior
  ``D[p] = sum_l |alpha[p,l] - alpha_global[class,l]|``;
* the dominant layer histogram.

Everything is float64 accumulation so the mean/std match a direct
``numpy.mean`` / ``numpy.std`` over the full set within numerical tolerance and
do not depend on batch size.

This module is deliberately free of torch / matplotlib imports (it accepts
anything with ``detach``/``cpu``/``numpy``) so it can be unit-tested on
hand-built arrays.  See ``tests/test_fusion_viz_stats.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

# Softmax rows in float32 are accurate well below this; a hard error above the
# tolerance catches wrong alpha sources (e.g. unnormalised logits) instead of
# silently plotting them.
ALPHA_SUM_TOL = 1e-5
WEIGHT_EPS = 1e-6
LOG_EPS = 1e-12


def to_numpy(value) -> np.ndarray:
    """Convert a torch tensor / list / ndarray to a numpy array without copy
    semantics beyond what the source requires."""
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def validate_global_prior(alpha_global, tol: float = ALPHA_SUM_TOL) -> np.ndarray:
    """Validate ``alpha^G`` as ``[C, L]`` probabilities in ``[0, 1]``."""
    arr = np.asarray(to_numpy(alpha_global), dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"alpha_global must be [C, L], got {arr.shape}")
    if arr.size == 0:
        raise ValueError("alpha_global is empty")
    if not np.isfinite(arr).all():
        raise ValueError("alpha_global contains non-finite values")
    if arr.min() < -WEIGHT_EPS or arr.max() > 1.0 + WEIGHT_EPS:
        raise ValueError(
            f"alpha_global outside [0, 1]: min={arr.min():.6g}, max={arr.max():.6g}"
        )
    sums = arr.sum(axis=-1)
    if np.abs(sums - 1.0).max() > tol:
        raise ValueError(
            "alpha_global rows must sum to 1; max |sum - 1| = "
            f"{np.abs(sums - 1.0).max():.3e}"
        )
    return arr


def validate_alpha(alpha, num_layers: Optional[int] = None,
                   tol: float = ALPHA_SUM_TOL) -> np.ndarray:
    """Validate patch weights as ``[B, P, L]`` probabilities in ``[0, 1]``."""
    arr = np.asarray(to_numpy(alpha), dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"alpha must be [B, P, L], got {arr.shape}")
    if num_layers is not None and arr.shape[-1] != int(num_layers):
        raise ValueError(
            f"alpha has {arr.shape[-1]} layers, expected {int(num_layers)}"
        )
    if arr.size == 0:
        raise ValueError("alpha is empty")
    if not np.isfinite(arr).all():
        raise ValueError("alpha contains non-finite values")
    if arr.min() < -WEIGHT_EPS or arr.max() > 1.0 + WEIGHT_EPS:
        raise ValueError(
            f"alpha outside [0, 1]: min={arr.min():.6g}, max={arr.max():.6g}"
        )
    sums = arr.sum(axis=-1)
    if np.abs(sums - 1.0).max() > tol:
        raise ValueError(
            "alpha rows must sum to 1; max |sum - 1| = "
            f"{np.abs(sums - 1.0).max():.3e}"
        )
    return arr


def normalize_prior(prior, batch: int, num_layers: int,
                    tol: float = ALPHA_SUM_TOL) -> np.ndarray:
    """Broadcast/validate the class prior to ``[batch, L]``."""
    arr = np.asarray(to_numpy(prior), dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[-1] != int(num_layers):
        raise ValueError(
            f"prior must be [B, L={int(num_layers)}] or [L], got {arr.shape}"
        )
    if arr.shape[0] == 1 and batch > 1:
        arr = np.repeat(arr, batch, axis=0)
    if arr.shape[0] != batch:
        raise ValueError(
            f"prior has {arr.shape[0]} rows but alpha has batch={batch}"
        )
    if not np.isfinite(arr).all():
        raise ValueError("prior contains non-finite values")
    sums = arr.sum(axis=-1)
    if np.abs(sums - 1.0).max() > tol:
        raise ValueError(
            "prior rows must sum to 1; max |sum - 1| = "
            f"{np.abs(sums - 1.0).max():.3e}"
        )
    return arr


def patch_entropy(alpha: np.ndarray) -> np.ndarray:
    """Per-patch entropy in nats; ``alpha`` is ``[..., L]``."""
    return -(alpha * np.log(np.clip(alpha, LOG_EPS, None))).sum(axis=-1)


def patch_l1_to_prior(alpha: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """Per-patch L1 distance ``[B, P]`` to a ``[B, L]`` class prior."""
    if alpha.ndim != 3:
        raise ValueError(f"alpha must be [B, P, L], got {alpha.shape}")
    prior = np.asarray(prior, dtype=np.float64)
    if prior.shape != (alpha.shape[0], alpha.shape[-1]):
        raise ValueError(
            f"prior shape {prior.shape} does not match alpha {alpha.shape}"
        )
    return np.abs(alpha - prior[:, None, :]).sum(axis=-1)


@dataclass
class ClassFusionStats:
    """Streaming per-class accumulator (all sums in float64)."""

    num_layers: int
    n_images: int = 0
    n_patches: int = 0
    alpha_sum: np.ndarray = field(init=False)
    alpha_sq_sum: np.ndarray = field(init=False)
    entropy_sum: float = 0.0
    l1_sum: float = 0.0
    l1_sq_sum: float = 0.0
    dominant_counts: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.num_layers = int(self.num_layers)
        if self.num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.alpha_sum = np.zeros(self.num_layers, dtype=np.float64)
        self.alpha_sq_sum = np.zeros(self.num_layers, dtype=np.float64)
        self.dominant_counts = np.zeros(self.num_layers, dtype=np.int64)

    def update(self, alpha, prior) -> None:
        """Accumulate one batch of ``[B, P, L]`` weights and ``[B, L]`` prior."""
        alpha = validate_alpha(alpha, self.num_layers)
        prior = normalize_prior(prior, alpha.shape[0], self.num_layers)

        self.alpha_sum += alpha.sum(axis=(0, 1))
        self.alpha_sq_sum += np.square(alpha).sum(axis=(0, 1))
        self.entropy_sum += float(patch_entropy(alpha).sum())
        l1 = patch_l1_to_prior(alpha, prior)
        self.l1_sum += float(l1.sum())
        self.l1_sq_sum += float(np.square(l1).sum())
        self.dominant_counts += np.bincount(
            alpha.argmax(axis=-1).reshape(-1), minlength=self.num_layers
        ).astype(np.int64, copy=False)
        self.n_images += int(alpha.shape[0])
        self.n_patches += int(alpha.shape[0] * alpha.shape[1])

    def result(self) -> Dict[str, object]:
        """Finalise the accumulated moments into plot-ready statistics."""
        if self.n_patches <= 0:
            raise RuntimeError("ClassFusionStats.result() called before update().")
        n = float(self.n_patches)
        mean = self.alpha_sum / n
        var = np.maximum(self.alpha_sq_sum / n - np.square(mean), 0.0)
        std = np.sqrt(var)

        l1_mean = self.l1_sum / n
        l1_var = max(self.l1_sq_sum / n - l1_mean * l1_mean, 0.0)
        entropy_mean = self.entropy_sum / n
        max_entropy = float(np.log(self.num_layers))

        # Cheap invariants: a violation means the caller mixed different
        # fusion schemas or fed unnormalised weights.
        if mean.min() < -WEIGHT_EPS or mean.max() > 1.0 + WEIGHT_EPS:
            raise RuntimeError(f"mean weight outside [0, 1]: {mean}")
        if np.abs(mean.sum() - 1.0) > 1e-4:
            raise RuntimeError(f"mean weights do not sum to 1: {mean.sum():.6f}")
        if entropy_mean < -1e-6 or entropy_mean > max_entropy + 1e-5:
            raise RuntimeError(
                f"mean entropy {entropy_mean:.6f} outside [0, ln L={max_entropy:.6f}]"
            )
        if l1_mean < -1e-6 or l1_mean > 2.0 + 1e-5:
            raise RuntimeError(f"mean L1 {l1_mean:.6f} outside [0, 2]")

        return {
            "n_images": int(self.n_images),
            "n_patches": int(self.n_patches),
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
            "entropy_mean": float(entropy_mean),
            "entropy_norm": float(entropy_mean / max_entropy) if max_entropy > 0 else 0.0,
            "l1_mean": float(l1_mean),
            "l1_std": float(np.sqrt(l1_var)),
            "dominant_counts": self.dominant_counts.copy(),
            "dominant_fraction": (
                self.dominant_counts.astype(np.float64) / n
            ),
        }


class FusionStatsCollector:
    """Per-class registry of :class:`ClassFusionStats`."""

    def __init__(self, num_layers: int):
        self.num_layers = int(num_layers)
        self._stats: Dict[int, ClassFusionStats] = {}

    def update(self, class_idx, alpha, prior) -> None:
        idx = int(class_idx)
        stats = self._stats.get(idx)
        if stats is None:
            stats = ClassFusionStats(self.num_layers)
            self._stats[idx] = stats
        stats.update(alpha, prior)

    def has(self, class_idx) -> bool:
        return int(class_idx) in self._stats

    def get(self, class_idx) -> ClassFusionStats:
        idx = int(class_idx)
        if idx not in self._stats:
            raise KeyError(f"no fusion statistics for class index {idx}")
        return self._stats[idx]

    def result(self, class_idx) -> Dict[str, object]:
        return self.get(class_idx).result()

    @property
    def class_indices(self) -> List[int]:
        return sorted(self._stats)

    def totals(self) -> Dict[str, int]:
        return {
            "n_images": sum(s.n_images for s in self._stats.values()),
            "n_patches": sum(s.n_patches for s in self._stats.values()),
        }


def stats_to_row(
    class_idx: int,
    class_name: str,
    result: Mapping[str, object],
    block_labels: Sequence[int],
    extra: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    """Flatten one class result into a CSV row keyed by real block numbers."""
    row: Dict[str, object] = {
        "class_index": int(class_idx),
        "class": str(class_name),
        "n_images": int(result["n_images"]),
        "n_patches": int(result["n_patches"]),
    }
    mean = np.asarray(result["mean"], dtype=np.float64)
    std = np.asarray(result["std"], dtype=np.float64)
    for pos, block in enumerate(block_labels):
        row[f"mean_block_{int(block)}"] = float(mean[pos])
    for pos, block in enumerate(block_labels):
        row[f"std_block_{int(block)}"] = float(std[pos])

    dominant_fraction = np.asarray(result["dominant_fraction"], dtype=np.float64)
    dominant_pos = int(np.argmax(dominant_fraction))
    row.update(
        {
            "entropy_mean": float(result["entropy_mean"]),
            "entropy_norm": float(result["entropy_norm"]),
            "l1_mean": float(result["l1_mean"]),
            "l1_std": float(result["l1_std"]),
            "dominant_block": int(block_labels[dominant_pos]),
            "dominant_fraction": float(dominant_fraction[dominant_pos]),
        }
    )
    if extra:
        row.update({str(k): v for k, v in extra.items()})
    return row
