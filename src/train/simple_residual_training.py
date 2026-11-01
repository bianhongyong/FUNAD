"""CRAS-inspired synthesis/BCE for the residual-only baseline.

Source: https://github.com/cqylunlun/CRAS/blob/main/cras.py
        CRAS._train_discriminator, synthesis and loss computation.

For a fixed matched reference m, (q + alpha*g) - m = r + alpha*g.
We apply that perturbation to r directly; the detector still sees only r,
not the [q, r] concatenation used by the complete CRAS model.
"""

import math

import torch
from torch.nn import functional as F


def residual_training_batch(residual, noise_std=0.015, beta=0.3, mode="cras", eps=1e-8):
    """Return [r; r_syn], [0; 1] and detached perturbation diagnostics.

    CRAS DAFS, Eq. (4)-(6):
        ratio = ||g|| / ||r||
        alpha = 1 + beta * (ratio / mean(ratio) - 1)
        r_syn = r + alpha*g,  g ~ N(0, noise_std**2 I).

    As in the official code, mean(ratio) includes all B*P patches in the
    minibatch, across classes. Only the channel dimension is norm-reduced.
    Epsilon floors extend the formula to zero residuals. beta is restricted
    to [0,1] to keep perturbation scales nonnegative. No extra scale clipping.
    All synthesis statistics/noise are detached; r itself keeps its graph.
    mode='gaussian' is the original fixed-noise control (alpha=1).
    """
    if residual.ndim != 3 or residual.numel() == 0 or not residual.is_floating_point():
        raise ValueError("residual must be a nonempty floating [B,P,C] tensor.")
    if not math.isfinite(noise_std) or noise_std <= 0:
        raise ValueError("noise_std must be finite and > 0.")
    if not math.isfinite(beta) or not 0 <= beta <= 1:
        raise ValueError("beta must be finite and in [0,1].")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and > 0.")
    if mode not in ("cras", "gaussian"):
        raise ValueError("mode must be 'cras' or 'gaussian'.")

    with torch.no_grad():
        # Norms and normalization stay in float32 even if the input uses fp16.
        noise = torch.randn_like(residual, dtype=torch.float32) * float(noise_std)
        scale = torch.ones_like(noise[..., 0])
        if mode == "cras":
            distance = residual.detach().float().norm(dim=-1).clamp_min(eps)
            ratio = noise.norm(dim=-1) / distance
            scale = 1.0 + float(beta) * (ratio / ratio.mean().clamp_min(eps) - 1.0)
        perturbation = (scale.unsqueeze(-1) * noise).to(residual.dtype)
        stats = {"scale_mean": scale.mean().item(), "scale_min": scale.min().item(),
                 "scale_max": scale.max().item(),
                 "perturbation_rms": perturbation.float().square().mean().sqrt().item()}

    features = torch.cat((residual, residual + perturbation), dim=0)
    labels = residual.new_zeros(features.shape[:2])
    labels[residual.shape[0]:] = 1
    return features, labels, stats


def cras_bce_loss(logits):
    """Official CRAS objective: mean normal BCE + mean synthetic BCE (1:1).

    Input order is [normal; synthetic]. BCEWithLogits is mathematically
    equivalent to the official Sigmoid + BCELoss and is numerically stable.
    Returns (total, normal, synthetic); no focal, OTO or auxiliary loss.
    """
    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[0] % 2 or logits.shape[1] == 0:
        raise ValueError("Expected nonempty [2B,P] logits ordered [normal; synthetic].")
    normal, synthetic = logits.chunk(2, dim=0)
    normal_loss = F.binary_cross_entropy_with_logits(normal, torch.zeros_like(normal))
    synthetic_loss = F.binary_cross_entropy_with_logits(synthetic, torch.ones_like(synthetic))
    return normal_loss + synthetic_loss, normal_loss, synthetic_loss
