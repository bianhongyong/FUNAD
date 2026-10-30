"""Matplotlib figures for the class-patch fusion visualisation entry point.

Six figures, all built from plain numpy arrays:

1. ``fig1_global_prior_heatmap``  class x DINOv3-block class prior ``alpha^G``
2. ``fig2_patch_mean_std``        per-class patch weight mean +/- std
3. ``fig3_dominant_layer``        dominant block per patch (discrete map)
4. ``fig4_patch_entropy``         per-patch weight entropy
5. ``fig5_l1_to_prior_by_class``  per-class mean patch-to-prior L1
6. ``fig6_fusion_detection``      RGB | GT | score | alpha | L1 | overlay

Colour rules (static, print-oriented):

* sequential quantities use one hue, light -> dark, always with a colourbar;
* the dominant-layer map is nominal: fixed categorical hues in the same order
  for every case, plus a discrete legend, so identity never depends on hue
  alone;
* ranges are fixed by the caller (``[0, 1]``, ``[0, ln L]``, ``[0, 2]``) - no
  per-image min-max stretch, so panels remain comparable.

The module never touches checkpoints or model state.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import (
    BoundaryNorm,
    LinearSegmentedColormap,
    ListedColormap,
    to_rgba_array,
)
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Palette / chrome
# ---------------------------------------------------------------------------
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

# Fixed categorical order (validated 8-slot order).  Never cycled: past eight
# layers we fall back to an ordinal blue ramp instead of generating hues.
CATEGORICAL = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
]

# Single-hue sequential ramps (light -> dark).
BLUE_RAMP = LinearSegmentedColormap.from_list(
    "fusion_blue",
    ["#f4f8fe", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"],
)
ORANGE_RAMP = plt.get_cmap("Oranges")
VIOLET_RAMP = plt.get_cmap("Purples")
RED_RAMP = plt.get_cmap("Reds")


def _truncate(cmap, lo: float = 0.05, hi: float = 0.95) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        f"{cmap.name}_trunc", cmap(np.linspace(lo, hi, 256))
    )


ORANGE_RAMP = _truncate(ORANGE_RAMP)
VIOLET_RAMP = _truncate(VIOLET_RAMP)
RED_RAMP = _truncate(RED_RAMP)


def configure_matplotlib() -> None:
    """Global rcParams shared by every figure in the run."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.labelcolor": INK_SECONDARY,
            "text.color": INK_PRIMARY,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.grid": False,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _style_axes(ax, *, grid_axis: Optional[str] = None) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_MUTED, length=2.5, width=0.7)
    if grid_axis:
        ax.grid(
            True, axis=grid_axis, color=GRID, linewidth=0.6, linestyle="-", zorder=0
        )
        ax.set_axisbelow(True)


def _save(fig, out_dir: str, stem: str, formats: Sequence[str], dpi: int) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for fmt in formats:
        fmt = str(fmt).lower().lstrip(".")
        path = os.path.join(out_dir, f"{stem}.{fmt}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        paths.append(path)
    return paths


def _add_footer(fig, note: Optional[str], subtitle: Optional[str]) -> None:
    """Place notes below the axes; ``bbox_inches='tight'`` expands to include them.

    Using one text block keeps the note and the provenance line from colliding
    with the x-axis label or with each other.
    """
    lines = [line for line in (note, subtitle) if line]
    if not lines:
        return
    fig.text(
        0.5, -0.03, "\n".join(lines), ha="center", va="top",
        fontsize=7.5, color=INK_SECONDARY, linespacing=1.35,
    )


def layer_colormap(num_layers: int) -> ListedColormap:
    """Fixed layer -> colour mapping, identical for every case figure."""
    num_layers = int(num_layers)
    if num_layers < 1:
        raise ValueError("num_layers must be >= 1")
    if num_layers <= len(CATEGORICAL):
        return ListedColormap(
            to_rgba_array(CATEGORICAL[:num_layers]), name="fusion_layers"
        )
    # Layers are ordered; beyond eight categories use an ordinal ramp instead
    # of inventing new hues.
    return ListedColormap(
        BLUE_RAMP(np.linspace(0.25, 0.95, num_layers)), name="fusion_layers_ordinal"
    )


# ---------------------------------------------------------------------------
# Fig.1 - class x layer global prior
# ---------------------------------------------------------------------------
def plot_global_prior(
    alpha_global: np.ndarray,
    class_names: Sequence[str],
    block_labels: Sequence[int],
    pinned: Sequence[bool],
    figures_dir: str,
    *,
    stem: str = "fig1_global_prior_heatmap",
    formats: Sequence[str] = ("png", "pdf"),
    dpi: int = 200,
    subtitle: Optional[str] = None,
) -> List[str]:
    """Heatmap of ``alpha^G`` with real 1-based DINOv3 block numbers."""
    alpha = np.asarray(alpha_global, dtype=np.float64)
    n_classes, n_layers = alpha.shape
    if n_classes != len(class_names) or n_layers != len(block_labels):
        raise ValueError(
            f"alpha_global {alpha.shape} does not match "
            f"{len(class_names)} classes / {len(block_labels)} blocks"
        )

    width = max(5.6, 0.62 * n_layers + 3.2)
    height = max(3.2, 0.30 * n_classes + 1.6)
    fig, ax = plt.subplots(figsize=(width, height))
    im = ax.imshow(alpha, vmin=0.0, vmax=1.0, cmap=BLUE_RAMP, aspect="auto")
    _style_axes(ax)

    ax.set_xticks(np.arange(n_layers))
    ax.set_xticklabels([str(int(b)) for b in block_labels])
    ylabels = [
        f"{name} *" if is_pinned else str(name)
        for name, is_pinned in zip(class_names, pinned)
    ]
    ax.set_yticks(np.arange(n_classes))
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("DINOv3 block (1-based)")
    ax.set_ylabel("class")

    if n_classes * n_layers <= 240:
        for r in range(n_classes):
            for c in range(n_layers):
                value = alpha[r, c]
                ax.text(
                    c, r, f"{value:.2f}",
                    ha="center", va="center", fontsize=6.5,
                    color="white" if value > 0.55 else INK_SECONDARY,
                )

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("class-prior weight α$^G$")
    cbar.outline.set_edgecolor(AXIS)
    cbar.ax.tick_params(colors=INK_MUTED, labelsize=7.5)

    ax.set_title("Class × DINOv3 block global prior", pad=10)
    note = "colour fixed to [0, 1]; each row sums to 1"
    if any(pinned):
        note += "\n* prior pinned to the last candidate block (not learned)"
    fig.tight_layout()
    _add_footer(fig, note, subtitle)
    paths = _save(fig, figures_dir, stem, formats, dpi)
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# Fig.2 - per-class patch weight mean +/- std
# ---------------------------------------------------------------------------
def plot_patch_mean_std(
    class_names: Sequence[str],
    block_labels: Sequence[int],
    means: np.ndarray,
    stds: np.ndarray,
    priors: np.ndarray,
    counts: Sequence[Tuple[int, int]],
    figures_dir: str,
    *,
    stem: str = "fig2_patch_mean_std",
    formats: Sequence[str] = ("png", "pdf"),
    dpi: int = 200,
    subtitle: Optional[str] = None,
) -> List[str]:
    """Small multiples: patch mean with patch-std error bars vs class prior.

    Error bars are the standard deviation across patches, explicitly **not** a
    confidence interval for the mean.
    """
    means = np.asarray(means, dtype=np.float64)
    stds = np.asarray(stds, dtype=np.float64)
    priors = np.asarray(priors, dtype=np.float64)
    n_classes = len(class_names)
    n_layers = len(block_labels)
    if means.shape != (n_classes, n_layers):
        raise ValueError(f"means {means.shape} != {(n_classes, n_layers)}")

    ncols = min(3, max(1, n_classes))
    nrows = int(np.ceil(n_classes / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.6 * ncols, 2.8 * nrows + 0.8),
        sharey=True, squeeze=False,
    )
    x = np.arange(n_layers)

    for idx in range(nrows * ncols):
        ax = axes[idx // ncols, idx % ncols]
        if idx >= n_classes:
            ax.axis("off")
            continue
        _style_axes(ax, grid_axis="y")
        ax.errorbar(
            x, means[idx], yerr=stds[idx],
            fmt="o-", color=CATEGORICAL[0], ecolor=CATEGORICAL[0],
            elinewidth=0.9, capsize=2.4, capthick=0.9,
            markersize=4, linewidth=1.5, label="patch mean ± std",
        )
        ax.plot(
            x, priors[idx],
            color=INK_SECONDARY, linewidth=1.2, marker="o", markersize=3.4,
            markerfacecolor=SURFACE, label="class prior α$^G$",
        )
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(b)) for b in block_labels])
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(str(class_names[idx]))
        n_images, n_patches = counts[idx]
        ax.text(
            0.02, 0.90, f"{n_images} images · {n_patches:,} patches",
            transform=ax.transAxes, fontsize=7, color=INK_MUTED, va="top",
        )
        if idx // ncols == nrows - 1:
            ax.set_xlabel("DINOv3 block (1-based)")
        if idx % ncols == 0:
            ax.set_ylabel("layer weight")

    handles = [
        Line2D([0], [0], color=CATEGORICAL[0], marker="o", linewidth=1.5,
               markersize=4, label="patch mean ± std"),
        Line2D([0], [0], color=INK_SECONDARY, marker="o", linewidth=1.2,
               markersize=3.4, markerfacecolor=SURFACE, label="class prior α$^G$"),
    ]
    fig.legend(handles=handles, loc="upper right", ncol=2, bbox_to_anchor=(0.99, 0.99))
    fig.suptitle(
        "Patch-conditioned layer weights: mean ± std across patches",
        x=0.01, ha="left", y=1.0,
    )
    note = "error bars = std across patches (not a confidence interval)"
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    _add_footer(fig, note, subtitle)
    paths = _save(fig, figures_dir, stem, formats, dpi)
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# Fig.3 - dominant layer map
# ---------------------------------------------------------------------------
def plot_dominant_layer(
    rgb: np.ndarray,
    dominant_grid: np.ndarray,
    block_labels: Sequence[int],
    figures_dir: str,
    *,
    stem: str,
    formats: Sequence[str] = ("png", "pdf"),
    dpi: int = 200,
    subtitle: Optional[str] = None,
) -> List[str]:
    """RGB + discrete dominant-block map (nearest-neighbour, no smoothing)."""
    dominant_grid = np.asarray(dominant_grid)
    if dominant_grid.ndim != 2:
        raise ValueError(f"dominant_grid must be [H, W], got {dominant_grid.shape}")
    n_layers = len(block_labels)
    if dominant_grid.max() >= n_layers:
        raise ValueError(
            f"dominant layer {dominant_grid.max()} outside {n_layers} candidates"
        )

    cmap = layer_colormap(n_layers)
    norm = BoundaryNorm(np.arange(-0.5, n_layers + 0.5), n_layers)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4))
    axes[0].imshow(rgb)
    axes[0].set_title("RGB (CenterCrop)")
    axes[0].axis("off")

    im = axes[1].imshow(
        dominant_grid, cmap=cmap, norm=norm, interpolation="nearest"
    )
    axes[1].set_title("Dominant DINOv3 block per patch")
    axes[1].axis("off")

    cbar = fig.colorbar(
        im, ax=axes[1], fraction=0.046, pad=0.02,
        ticks=np.arange(n_layers), boundaries=np.arange(-0.5, n_layers + 0.5),
    )
    cbar.ax.set_yticklabels([str(int(b)) for b in block_labels])
    cbar.set_label("dominant block")
    cbar.outline.set_edgecolor(AXIS)
    cbar.ax.tick_params(colors=INK_MUTED, labelsize=7.5)

    counts = np.bincount(dominant_grid.reshape(-1), minlength=n_layers)
    top = int(np.argmax(counts))
    frac = counts[top] / max(1, counts.sum())
    fig.suptitle(
        f"Dominant block {int(block_labels[top])} on {frac:.1%} of patches",
        fontsize=11,
    )
    note = (
        "discrete map, nearest-neighbour upsampling; colours are fixed "
        "per block across all cases"
    )
    if frac >= 0.999:
        note += "\nsingle-colour map = one block dominates every patch (reported as-is)"
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    _add_footer(fig, note, subtitle)
    paths = _save(fig, figures_dir, stem, formats, dpi)
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# Fig.4 - patch entropy
# ---------------------------------------------------------------------------
def plot_patch_entropy(
    rgb: np.ndarray,
    entropy_grid: np.ndarray,
    num_layers: int,
    figures_dir: str,
    *,
    stem: str,
    formats: Sequence[str] = ("png", "pdf"),
    dpi: int = 200,
    subtitle: Optional[str] = None,
) -> List[str]:
    """RGB + entropy with a **fixed** ``[0, ln L]`` range.

    ``alpha`` sums to 1 per patch, so ``H >= 0``; the upper bound is ``ln L``.
    """
    entropy_grid = np.asarray(entropy_grid, dtype=np.float64)
    if entropy_grid.ndim != 2:
        raise ValueError(f"entropy_grid must be [H, W], got {entropy_grid.shape}")
    num_layers = int(num_layers)
    if num_layers < 1:
        raise ValueError("num_layers must be >= 1")
    max_entropy = float(np.log(num_layers))
    if entropy_grid.min() < -1e-6 or entropy_grid.max() > max_entropy + 1e-5:
        raise ValueError(
            f"entropy outside [0, ln L={max_entropy:.4f}]: "
            f"[{entropy_grid.min():.4g}, {entropy_grid.max():.4g}]"
        )
    if max_entropy <= 0:
        max_entropy = 1.0  # single candidate layer: constant zero map

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.4))
    axes[0].imshow(rgb)
    axes[0].set_title("RGB (CenterCrop)")
    axes[0].axis("off")

    im = axes[1].imshow(
        entropy_grid, cmap=ORANGE_RAMP, vmin=0.0, vmax=max_entropy,
        interpolation="nearest",
    )
    axes[1].set_title(
        f"Patch weight entropy (mean {entropy_grid.mean():.3f} nats)"
    )
    axes[1].axis("off")
    cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.02)
    cbar.set_label("H (nats), fixed [0, ln L]")
    cbar.outline.set_edgecolor(AXIS)
    cbar.ax.tick_params(colors=INK_MUTED, labelsize=7.5)

    norm_entropy = entropy_grid / max_entropy
    im2 = axes[2].imshow(
        norm_entropy, cmap=ORANGE_RAMP, vmin=0.0, vmax=1.0,
        interpolation="nearest",
    )
    axes[2].set_title(
        f"Normalised entropy (mean {norm_entropy.mean():.3f} of ln L)"
    )
    axes[2].axis("off")
    cbar2 = fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.02)
    cbar2.set_label("H / ln L")
    cbar2.outline.set_edgecolor(AXIS)
    cbar2.ax.tick_params(colors=INK_MUTED, labelsize=7.5)

    fig.suptitle("How concentrated are the per-patch fusion weights?", fontsize=11)
    note = (
        "low entropy = weights concentrated; high entropy = spread out. "
        "Entropy is not anomaly confidence."
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    _add_footer(fig, note, subtitle)
    paths = _save(fig, figures_dir, stem, formats, dpi)
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# Fig.5 - per-class L1 to prior
# ---------------------------------------------------------------------------
def plot_l1_by_class(
    class_names: Sequence[str],
    l1_mean: Sequence[float],
    l1_std: Sequence[float],
    figures_dir: str,
    *,
    stem: str = "fig5_l1_to_prior_by_class",
    formats: Sequence[str] = ("png", "pdf"),
    dpi: int = 200,
    subtitle: Optional[str] = None,
) -> List[str]:
    """Horizontal bar chart, fixed ``[0, 2]`` range, in checkpoint class order."""
    l1_mean = np.asarray(l1_mean, dtype=np.float64)
    l1_std = np.asarray(l1_std, dtype=np.float64)
    n = len(class_names)
    if l1_mean.shape != (n,) or l1_std.shape != (n,):
        raise ValueError("l1_mean / l1_std must match class_names length")

    height = max(3.0, 0.32 * n + 1.4)
    fig, ax = plt.subplots(figsize=(6.4, height))
    y = np.arange(n)
    ax.barh(
        y, l1_mean, height=0.62, color=CATEGORICAL[0],
        xerr=l1_std, error_kw=dict(ecolor=INK_SECONDARY, elinewidth=0.9,
                                   capsize=2.2, capthick=0.9),
    )
    _style_axes(ax, grid_axis="x")
    ax.set_yticks(y)
    ax.set_yticklabels([str(name) for name in class_names])
    ax.invert_yaxis()
    ax.set_xlim(0.0, 2.0)
    ax.set_xlabel("mean patch-to-prior L1 distance  Σ$_l$ |α$_{p,l}$ − α$^G_l$|")

    for yi, value in zip(y, l1_mean):
        ax.text(
            min(value + 0.03, 1.94), yi, f"{value:.3f}",
            va="center", fontsize=7.5, color=INK_SECONDARY,
        )

    ax.set_title("Deviation of patch weights from the class prior", pad=10)
    note = (
        "bar = mean over patches; whisker = std of the per-patch L1. "
        "High L1 alone is not evidence of patch-level dynamics -\n"
        "read together with Fig.2 (std) and Fig.3 (spatial map)."
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _add_footer(fig, note, subtitle)
    paths = _save(fig, figures_dir, stem, formats, dpi)
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# Fig.6 - detection + fusion case panel
# ---------------------------------------------------------------------------
def _nearest_resize_2d(grid: np.ndarray, height: int, width: int) -> np.ndarray:
    """Integer-ratio-safe nearest-neighbour resize for a 2D grid."""
    ph, pw = grid.shape
    if (ph, pw) == (height, width):
        return grid
    ys = np.minimum((np.arange(height) * ph) // max(1, height), ph - 1)
    xs = np.minimum((np.arange(width) * pw) // max(1, width), pw - 1)
    return grid[ys][:, xs]


def _alpha_overlay(rgb: np.ndarray, alpha_map: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float64) / 255.0
    alpha = np.clip(np.asarray(alpha_map, dtype=np.float64), 0.0, 1.0)
    alpha = _nearest_resize_2d(alpha, rgb.shape[0], rgb.shape[1])
    tint = BLUE_RAMP(alpha)[..., :3]
    return np.clip(0.55 * rgb + 0.45 * tint, 0.0, 1.0)


def plot_fusion_detection(
    rgb: np.ndarray,
    gt_mask: np.ndarray,
    score_map: np.ndarray,
    alpha_map: np.ndarray,
    l1_map: np.ndarray,
    figures_dir: str,
    *,
    stem: str,
    class_name: str,
    case_id: str,
    label: int,
    image_score: float,
    focus_block: int,
    score_min: float = 0.0,
    score_max: float = 1.0,
    formats: Sequence[str] = ("png", "pdf"),
    dpi: int = 200,
    subtitle: Optional[str] = None,
) -> List[str]:
    """Six-panel case figure: image, GT, score, alpha, L1, alpha overlay."""
    gt = np.asarray(gt_mask).squeeze()
    score = np.asarray(score_map, dtype=np.float64).squeeze()
    alpha = np.asarray(alpha_map, dtype=np.float64).squeeze()
    l1 = np.asarray(l1_map, dtype=np.float64).squeeze()
    for name, arr in (("gt", gt), ("score", score), ("alpha", alpha), ("l1", l1)):
        if arr.ndim != 2:
            raise ValueError(f"{name} must be 2D after squeeze, got {arr.shape}")

    fig, axes = plt.subplots(1, 6, figsize=(23.0, 4.5))

    axes[0].imshow(rgb)
    axes[0].set_title("RGB")
    axes[0].axis("off")

    axes[1].imshow(gt, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    axes[1].set_title("GT mask (nearest)")
    axes[1].axis("off")

    im_score = axes[2].imshow(
        score, cmap=RED_RAMP, vmin=float(score_min), vmax=float(score_max),
        interpolation="bilinear",
    )
    axes[2].set_title("Anomaly score map")
    axes[2].axis("off")
    cbar_score = fig.colorbar(im_score, ax=axes[2], fraction=0.046, pad=0.02)
    cbar_score.set_label("score (fixed range)")
    cbar_score.outline.set_edgecolor(AXIS)
    cbar_score.ax.tick_params(colors=INK_MUTED, labelsize=7.5)

    im_alpha = axes[3].imshow(
        alpha, cmap=BLUE_RAMP, vmin=0.0, vmax=1.0, interpolation="nearest"
    )
    axes[3].set_title(f"α map, block {int(focus_block)}")
    axes[3].axis("off")
    cbar_alpha = fig.colorbar(im_alpha, ax=axes[3], fraction=0.046, pad=0.02)
    cbar_alpha.set_label("weight α$_{p,l}$")
    cbar_alpha.outline.set_edgecolor(AXIS)
    cbar_alpha.ax.tick_params(colors=INK_MUTED, labelsize=7.5)

    im_l1 = axes[4].imshow(
        l1, cmap=VIOLET_RAMP, vmin=0.0, vmax=2.0, interpolation="nearest"
    )
    axes[4].set_title("Patch-to-prior L1")
    axes[4].axis("off")
    cbar_l1 = fig.colorbar(im_l1, ax=axes[4], fraction=0.046, pad=0.02)
    cbar_l1.set_label("L1 distance (fixed [0, 2])")
    cbar_l1.outline.set_edgecolor(AXIS)
    cbar_l1.ax.tick_params(colors=INK_MUTED, labelsize=7.5)

    axes[5].imshow(_alpha_overlay(rgb, alpha))
    axes[5].set_title("α overlay on image")
    axes[5].axis("off")

    gt_label = "anomaly" if int(label) == 1 else "good"
    fig.suptitle(
        f"{class_name} · {case_id} · GT={gt_label} · image score={image_score:.3f} "
        f"· focus block {int(focus_block)}",
        fontsize=11,
    )
    note = (
        "score map = original inference post-processing (resize + Gaussian); "
        "alpha/L1 upsampled with nearest-neighbour.\n"
        "Alpha values are fusion weights, not causal evidence for the anomaly."
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    _add_footer(fig, note, subtitle)
    paths = _save(fig, figures_dir, stem, formats, dpi)
    plt.close(fig)
    return paths
