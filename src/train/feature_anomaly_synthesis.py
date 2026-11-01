"""Feature-level anomaly synthesis for clean multiclass AD (Chapter 1).

Synthetic anomalies are generated in the **reference-residual space** after
class-patch layer fusion, i.e. on ``r = Q_fused - M_{j*}``, never in RGB image
space and never on raw backbone features.  This keeps the training signal in
exactly the representation the discriminator consumes and avoids running the
frozen backbone a second time.

Two pieces live here:

* :func:`calibrate_class_sigmas_loio` -- per-class perturbation scales
  ``sigma_c`` estimated with leave-one-image-out (LOIO) reference residuals.
  Every reference patch of the held-out image is excluded from the candidate
  bank, so near-self matches cannot shrink the scale.
* :class:`FeatureAnomalySynthesizer` -- ``r_syn = r + M * gamma * sigma_c * eps``
  with structured spatial masks (blob / line / spot / irregular).  The noise,
  mask and gamma carry no gradient; the original residual ``r`` keeps its graph
  so BCE gradients still reach the fusion module.

The module is deliberately free of learnable parameters and of any dependency
on the training script so it can be unit-tested in isolation.
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Tuple

import numpy as np
import torch
import torch.nn.functional as F

SPATIAL_MASK_TYPES = ("blob", "line", "spot", "irregular")
MASK_TYPE_CHOICES = SPATIAL_MASK_TYPES + ("independent",)


def patch_grid_side(num_patches: int) -> int:
    """Return the square grid side for ``num_patches`` (never hard-code 784)."""
    num_patches = int(num_patches)
    side = int(round(math.sqrt(num_patches)))
    if side * side != num_patches:
        raise ValueError(
            f"num_patches={num_patches} is not a perfect square; spatial masks "
            "require a square patch grid."
        )
    return side


def _coordinate_grid(side: int):
    coords = torch.linspace(-1.0, 1.0, int(side))
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    return yy, xx


def _topk_scores_to_mask(scores: torch.Tensor, area: int) -> torch.Tensor:
    side = int(scores.shape[0])
    flat = scores.reshape(-1)
    area = max(1, min(int(area), flat.numel()))
    indices = torch.topk(flat, k=area, largest=True, sorted=False).indices
    mask = torch.zeros(flat.numel(), dtype=torch.bool)
    mask[indices] = True
    return mask.reshape(side, side)


def _sample_score_map(side: int, mask_type: str, generator) -> torch.Tensor:
    """Score map whose top-k entries define one connected-ish anomaly region."""
    yy, xx = _coordinate_grid(side)
    jitter = 0.02 * torch.randn((side, side), generator=generator)

    if mask_type == "spot":
        cx = float(torch.empty(1).uniform_(-0.6, 0.6, generator=generator))
        cy = float(torch.empty(1).uniform_(-0.6, 0.6, generator=generator))
        return -((xx - cx) ** 2 + (yy - cy) ** 2) + jitter

    if mask_type == "blob":
        cx = float(torch.empty(1).uniform_(-0.5, 0.5, generator=generator))
        cy = float(torch.empty(1).uniform_(-0.5, 0.5, generator=generator))
        rx = float(torch.empty(1).uniform_(0.15, 0.7, generator=generator))
        ry = float(torch.empty(1).uniform_(0.15, 0.7, generator=generator))
        angle = float(torch.empty(1).uniform_(0.0, math.pi, generator=generator))
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        xr = (xx - cx) * cos_a + (yy - cy) * sin_a
        yr = -(xx - cx) * sin_a + (yy - cy) * cos_a
        return -((xr / rx) ** 2 + (yr / ry) ** 2) + jitter

    if mask_type == "line":
        # Closest band to a random infinite line -> a thin stripe; the top-k
        # area controls its thickness.
        angle = float(torch.empty(1).uniform_(0.0, math.pi, generator=generator))
        cx = float(torch.empty(1).uniform_(-0.3, 0.3, generator=generator))
        cy = float(torch.empty(1).uniform_(-0.3, 0.3, generator=generator))
        nx, ny = math.cos(angle), math.sin(angle)
        dist = ((xx - cx) * nx + (yy - cy) * ny).abs()
        return -dist + jitter

    if mask_type == "irregular":
        low = max(3, side // 4)
        field = torch.rand((1, 1, low, low), generator=generator)
        field = F.interpolate(
            field, size=(side, side), mode="bilinear", align_corners=False
        )
        kernel_1d = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0])
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        kernel = (kernel_2d / kernel_2d.sum()).view(1, 1, 5, 5)
        field = F.conv2d(F.pad(field, (2, 2, 2, 2), mode="replicate"), kernel)
        return field[0, 0] + 0.01 * torch.randn(
            (side, side), generator=generator
        )

    raise ValueError(
        f"unknown spatial mask type {mask_type!r}; expected one of "
        f"{SPATIAL_MASK_TYPES}."
    )


def sample_masks(
    batch_size: int,
    num_patches: int,
    mask_type: str = "mixed",
    area_ratio_range: Tuple[float, float] = (0.005, 0.10),
    device=None,
    generator=None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Sample one spatial mask per batch item.

    Args:
        batch_size: B.
        num_patches: P (must be a perfect square for structured masks).
        mask_type: ``mixed`` (uniform over blob/line/spot/irregular),
            ``independent`` (i.i.d. patch selection, no spatial structure), or
            one specific type.
        area_ratio_range: inclusive range of ``|M| / P``.
        device: output device (masks are sampled on CPU for determinism).
        generator: optional CPU ``torch.Generator``.

    Returns:
        ``(mask [B,P] bool, stats)``.  ``stats`` holds per-type counts and the
        realised area ratios.
    """
    batch_size = int(batch_size)
    num_patches = int(num_patches)
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if num_patches < 1:
        raise ValueError("num_patches must be >= 1")
    lo, hi = (float(area_ratio_range[0]), float(area_ratio_range[1]))
    if not 0.0 < lo <= hi <= 1.0:
        raise ValueError(
            f"area_ratio_range must satisfy 0 < lo <= hi <= 1, got {area_ratio_range}."
        )
    if mask_type not in ("mixed",) + MASK_TYPE_CHOICES:
        raise ValueError(
            f"mask_type={mask_type!r}; expected 'mixed', 'independent' or one "
            f"of {SPATIAL_MASK_TYPES}."
        )

    side = patch_grid_side(num_patches)
    output = torch.zeros(batch_size, num_patches, dtype=torch.bool)
    type_counts = {name: 0 for name in MASK_TYPE_CHOICES}
    area_ratios = []

    for row in range(batch_size):
        if mask_type == "mixed":
            choice = int(
                torch.randint(len(SPATIAL_MASK_TYPES), (1,), generator=generator)
            )
            current_type = SPATIAL_MASK_TYPES[choice]
        else:
            current_type = mask_type

        ratio = float(
            torch.empty(1).uniform_(lo, hi, generator=generator)
        )
        area = max(1, min(num_patches, int(round(ratio * num_patches))))

        if current_type == "independent":
            probs = torch.rand(num_patches, generator=generator)
            selected = torch.topk(
                probs, k=area, largest=False, sorted=False
            ).indices
            flat_mask = torch.zeros(num_patches, dtype=torch.bool)
            flat_mask[selected] = True
        else:
            scores = _sample_score_map(side, current_type, generator)
            flat_mask = _topk_scores_to_mask(scores, area).reshape(-1)

        output[row] = flat_mask
        type_counts[current_type] += 1
        area_ratios.append(float(flat_mask.float().mean().item()))

    if device is not None:
        output = output.to(device)

    stats = {
        "area_ratio_mean": float(np.mean(area_ratios)),
        "area_ratio_min": float(np.min(area_ratios)),
        "area_ratio_max": float(np.max(area_ratios)),
    }
    for name in MASK_TYPE_CHOICES:
        stats[f"mask_type_{name}_count"] = int(type_counts[name])
    return output, stats


def sample_gamma(
    batch_size: int,
    gamma_range: Tuple[float, float] = (0.5, 1.5),
    device=None,
    generator=None,
) -> torch.Tensor:
    """Per-sample perturbation strength ``gamma ~ U(lo, hi)``."""
    lo, hi = (float(gamma_range[0]), float(gamma_range[1]))
    if not 0.0 <= lo <= hi:
        raise ValueError(f"gamma_range must satisfy 0 <= lo <= hi, got {gamma_range}.")
    gamma = torch.empty(int(batch_size)).uniform_(lo, hi, generator=generator)
    return gamma if device is None else gamma.to(device)


def class_sigma_tensor(
    class_sigmas: Mapping[int, float],
    num_classes: int,
    device=None,
    eps: float = 1e-6,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build ``[num_classes]`` sigma values, rejecting missing/zero scales."""
    values = []
    for cls in range(int(num_classes)):
        if cls not in class_sigmas:
            raise KeyError(f"class_sigmas is missing class {cls}.")
        value = float(class_sigmas[cls])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"class {cls} sigma must be finite and > 0, got {value!r}."
            )
        values.append(max(value, float(eps)))
    return torch.tensor(values, dtype=dtype, device=device)


def synthesize_residual_batch(
    residual: torch.Tensor,
    class_idx: torch.Tensor,
    class_sigmas: Mapping[int, float],
    mask: torch.Tensor,
    gamma: torch.Tensor,
    sigma_scale: float = 1.0,
    generator=None,
) -> torch.Tensor:
    """``r_syn = r + M * gamma * sigma_c * eps``.

    ``residual`` keeps its autograd graph; ``eps``, ``mask`` and ``gamma`` do
    not.  Outside the mask the output is bit-identical to ``residual``.
    """
    if residual.dim() != 3:
        raise ValueError(f"residual must be [B,P,C], got {tuple(residual.shape)}")
    batch, num_patches, _ = residual.shape
    class_idx = torch.as_tensor(
        class_idx, dtype=torch.long, device=residual.device
    ).reshape(-1)
    if class_idx.numel() == 1 and batch > 1:
        class_idx = class_idx.expand(batch)
    if class_idx.numel() != batch:
        raise ValueError(
            f"class_idx has {class_idx.numel()} entries for batch {batch}."
        )
    if mask.shape != (batch, num_patches):
        raise ValueError(
            f"mask shape {tuple(mask.shape)} does not match {(batch, num_patches)}."
        )
    if gamma.shape != (batch,):
        raise ValueError(f"gamma shape {tuple(gamma.shape)} does not match ({batch},).")

    num_classes = max(int(k) for k in class_sigmas.keys()) + 1
    sigmas = class_sigma_tensor(
        class_sigmas, num_classes, device=residual.device, dtype=residual.dtype
    )
    sigma_b = sigmas.index_select(0, class_idx).view(batch, 1, 1)
    gamma_b = gamma.to(device=residual.device, dtype=residual.dtype).view(
        batch, 1, 1
    )
    mask_b = mask.to(device=residual.device, dtype=residual.dtype).unsqueeze(-1)

    if generator is None:
        noise = torch.randn_like(residual)
    else:
        noise = torch.randn(
            residual.shape, generator=generator, dtype=torch.float32
        ).to(device=residual.device, dtype=residual.dtype)

    perturbation = mask_b * (float(sigma_scale) * gamma_b * sigma_b * noise)
    return residual + perturbation


class FeatureAnomalySynthesizer:
    """Residual-space anomaly synthesis with per-class fixed scales."""

    def __init__(
        self,
        class_sigmas: Mapping[int, float],
        num_classes: int,
        mask_type: str = "mixed",
        area_ratio_range: Tuple[float, float] = (0.005, 0.10),
        gamma_range: Tuple[float, float] = (0.5, 1.5),
        sigma_scale: float = 1.0,
    ):
        if float(sigma_scale) <= 0:
            raise ValueError("sigma_scale must be > 0.")
        self.class_sigmas = {int(k): float(v) for k, v in class_sigmas.items()}
        self.num_classes = int(num_classes)
        self.mask_type = str(mask_type)
        self.area_ratio_range = (
            float(area_ratio_range[0]),
            float(area_ratio_range[1]),
        )
        self.gamma_range = (float(gamma_range[0]), float(gamma_range[1]))
        self.sigma_scale = float(sigma_scale)
        # Validate once so a bad scale fails before training starts.
        class_sigma_tensor(self.class_sigmas, self.num_classes)

    def __call__(
        self,
        residual: torch.Tensor,
        class_idx: torch.Tensor,
        generator=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        batch, num_patches, _ = residual.shape
        mask, mask_stats = sample_masks(
            batch_size=batch,
            num_patches=num_patches,
            mask_type=self.mask_type,
            area_ratio_range=self.area_ratio_range,
            device=residual.device,
            generator=generator,
        )
        gamma = sample_gamma(
            batch,
            gamma_range=self.gamma_range,
            device=residual.device,
            generator=generator,
        )
        synthetic = synthesize_residual_batch(
            residual,
            class_idx,
            self.class_sigmas,
            mask,
            gamma,
            sigma_scale=self.sigma_scale,
            generator=generator,
        )
        sigmas = class_sigma_tensor(
            self.class_sigmas,
            self.num_classes,
            device=residual.device,
            dtype=torch.float32,
        )
        cls = torch.as_tensor(
            class_idx, dtype=torch.long, device=residual.device
        ).reshape(-1)
        used_sigma = sigmas.index_select(0, cls)
        stats = dict(mask_stats)
        stats.update(
            {
                "gamma_mean": float(gamma.mean().item()),
                "gamma_std": float(gamma.std(unbiased=False).item()),
                "gamma_min": float(gamma.min().item()),
                "gamma_max": float(gamma.max().item()),
                "sigma_mean": float(used_sigma.mean().item()),
                "sigma_min": float(used_sigma.min().item()),
                "sigma_max": float(used_sigma.max().item()),
            }
        )
        return synthetic, mask, stats


def build_synthetic_training_batch(
    residual: torch.Tensor,
    class_idx: torch.Tensor,
    synthesizer: FeatureAnomalySynthesizer,
    generator=None,
):
    """Build ``X=[r; r_syn]`` and ``Y=[0; M]`` for Balanced BCE.

    Returns:
        features ``[2B,P,C]``, labels ``[2B,P]`` float, classes ``[2B]``,
        synthetic mask ``[B,P]`` bool, stats dict.
    """
    synthetic, mask, stats = synthesizer(
        residual, class_idx, generator=generator
    )
    batch, num_patches, _ = residual.shape
    labels = torch.cat(
        [torch.zeros_like(mask, dtype=torch.float32), mask.to(torch.float32)],
        dim=0,
    )
    features = torch.cat([residual, synthetic], dim=0)
    cls = torch.as_tensor(
        class_idx, dtype=torch.long, device=residual.device
    ).reshape(-1)
    if cls.numel() == 1 and batch > 1:
        cls = cls.expand(batch)
    classes = torch.cat([cls, cls], dim=0)
    return features, labels, classes, mask, stats


# ---------------------------------------------------------------------------
# LOIO scale calibration
# ---------------------------------------------------------------------------


def _fuse_rows(
    fusion,
    raw_rows: torch.Tensor,
    alpha: torch.Tensor,
    device,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """Fuse ``[N,L,C]`` raw rows with one ``[L]`` alpha, chunked to the device."""
    outputs = []
    for start in range(0, raw_rows.shape[0], max(1, int(chunk_size))):
        chunk = raw_rows[start:start + chunk_size].to(
            device=device, dtype=torch.float32
        )
        if hasattr(fusion, "fuse_query"):
            # 3D input expects ``[N,L]`` / ``[1,L]`` alpha.
            alpha_in = alpha.unsqueeze(0) if alpha.dim() == 1 else alpha
            fused = fusion.fuse_query(chunk, alpha_in)
        else:
            # ClassConditionedLayerFusion exposes ``norm`` + ``forward`` only.
            x = fusion.norm(chunk)
            alpha_in = alpha.reshape(-1)
            fused = torch.einsum("nlc,l->nc", x, alpha_in.to(x.dtype))
        outputs.append(fused)
    if not outputs:
        return torch.empty(
            (0, int(fusion.feature_dim)), dtype=torch.float32, device=device
        )
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)


def _chunked_nearest_ids(
    query: torch.Tensor,
    reference: torch.Tensor,
    query_chunk_size: int = 2048,
    ref_chunk_size: int = 4096,
) -> torch.Tensor:
    if reference.shape[0] == 0:
        raise RuntimeError("Cannot search an empty LOIO reference set.")
    query = query.detach().to(dtype=torch.float32)
    reference = reference.detach().to(dtype=torch.float32)
    out = torch.empty(query.shape[0], dtype=torch.long, device=query.device)
    for q_start in range(0, query.shape[0], max(1, int(query_chunk_size))):
        q = query[q_start:q_start + query_chunk_size]
        best_dist = torch.full((q.shape[0],), float("inf"), device=q.device)
        best_idx = torch.zeros(q.shape[0], dtype=torch.long, device=q.device)
        for r_start in range(0, reference.shape[0], max(1, int(ref_chunk_size))):
            r = reference[r_start:r_start + ref_chunk_size]
            dist = torch.cdist(q, r, p=2.0)
            min_dist, min_idx = dist.min(dim=1)
            better = min_dist < best_dist
            best_dist = torch.where(better, min_dist, best_dist)
            best_idx = torch.where(better, min_idx + r_start, best_idx)
        out[q_start:q_start + q.shape[0]] = best_idx
    return out


def calibrate_class_sigmas_loio(
    raw_memory_by_class: Mapping[int, np.ndarray],
    reference_image_ids_by_class: Mapping[int, np.ndarray],
    fusion,
    device,
    quantile: float = 0.75,
    residual_chunk_size: int = 4096,
    query_chunk_size: int = 2048,
    ref_chunk_size: int = 4096,
    sigma_min: float = 1e-6,
) -> Tuple[Dict[int, float], Dict[int, dict]]:
    """Estimate one fixed ``sigma_c`` per class from LOIO reference residuals.

    For every reference image, all of its patches are removed from the
    candidate bank, the remaining raw rows are fused with the class prior
    ``alpha^G_c``, and each held-out patch is matched to its unique nearest
    candidate.  The residual L2 norms are pooled per class and

        ``sigma_c = Q_quantile(||r||_2) / sqrt(C)``

    is stored.  No gradients are computed and the fusion module's train/eval
    mode is restored on exit.

    Args:
        raw_memory_by_class: ``{cls: [N_c, L, C]}`` raw reference bank.
        reference_image_ids_by_class: ``{cls: [N_c]}`` dataset index of the
            image every patch row came from.
        fusion: class-patch or class-conditioned fusion module.
        device: torch device.
        quantile: robust quantile, default 0.75 per the chapter-1 plan.

    Returns:
        ``(sigmas, stats)``: ``sigmas[cls]`` float, ``stats[cls]`` diagnostics.
    """
    if not 0.0 <= float(quantile) <= 1.0:
        raise ValueError(f"quantile must be in [0, 1], got {quantile!r}.")

    was_training = bool(fusion.training)
    fusion.eval()
    sigmas: Dict[int, float] = {}
    stats: Dict[int, dict] = {}
    try:
        with torch.no_grad():
            for cls in sorted(int(c) for c in raw_memory_by_class.keys()):
                raw = torch.as_tensor(np.asarray(raw_memory_by_class[cls]))
                if raw.dim() != 3:
                    raise ValueError(
                        f"class {cls} raw bank must be [N,L,C], got {tuple(raw.shape)}"
                    )
                image_ids = torch.as_tensor(
                    np.asarray(reference_image_ids_by_class[cls]), dtype=torch.long
                ).reshape(-1)
                if image_ids.numel() != raw.shape[0]:
                    raise ValueError(
                        f"class {cls}: {image_ids.numel()} image ids for "
                        f"{raw.shape[0]} patch rows."
                    )
                unique_images = torch.unique(image_ids)
                if unique_images.numel() < 2:
                    raise RuntimeError(
                        f"class {cls} has only {unique_images.numel()} reference "
                        "image(s); LOIO calibration needs at least 2. "
                        "Increase --num_reference_images_per_class."
                    )

                alpha = fusion.get_layer_weights(cls)
                if alpha.dim() == 2:
                    if alpha.shape[0] != 1:
                        raise ValueError(
                            f"class {cls}: expected one prior row, got "
                            f"{tuple(alpha.shape)}."
                        )
                    alpha = alpha[0]
                alpha = alpha.detach().to(device=device, dtype=torch.float32)
                fused = _fuse_rows(
                    fusion,
                    raw,
                    alpha,
                    device=device,
                    chunk_size=residual_chunk_size,
                )
                feature_dim = int(fused.shape[-1])

                norm_chunks = []
                for image_id in unique_images.tolist():
                    held_out = image_ids == int(image_id)
                    query = fused[held_out.to(fused.device)]
                    candidates = fused[(~held_out).to(fused.device)]
                    nearest = _chunked_nearest_ids(
                        query,
                        candidates,
                        query_chunk_size=query_chunk_size,
                        ref_chunk_size=ref_chunk_size,
                    )
                    residual = query - candidates.index_select(0, nearest)
                    norm_chunks.append(
                        residual.norm(dim=-1).cpu().numpy().astype(np.float64)
                    )

                pooled = np.concatenate(norm_chunks, axis=0)
                q_value = float(np.quantile(pooled, float(quantile)))
                sigma = max(q_value / math.sqrt(feature_dim), float(sigma_min))
                sigmas[cls] = float(sigma)
                stats[cls] = {
                    "num_images": int(unique_images.numel()),
                    "num_patches": int(raw.shape[0]),
                    "residual_norm_quantile": q_value,
                    "sigma": float(sigma),
                    "quantile": float(quantile),
                    "feature_dim": feature_dim,
                }
    finally:
        fusion.train(was_training)

    return sigmas, stats
