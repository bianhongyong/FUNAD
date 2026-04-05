"""Diagnostics for class-wise memory banks (e.g. after greedy coreset)."""

from __future__ import annotations

from typing import Mapping, Optional, Sequence, Union

import numpy as np


def _flatten_gt_patch_anomaly_flags(
    selected_image_indices: np.ndarray,
    gt_patch_masks: np.ndarray,
) -> np.ndarray:
    """Stack GT anomaly flags for patches in loader order for one class.

    ``gt_patch_masks[img]`` is flattened row-wise; True/1 means anomaly patch.
    """
    selected_image_indices = np.asarray(selected_image_indices, dtype=np.int64)
    if selected_image_indices.size == 0:
        return np.zeros((0,), dtype=bool)
    parts = [
        gt_patch_masks[int(i)].ravel().astype(bool) for i in selected_image_indices
    ]
    return np.concatenate(parts, axis=0)


def print_greedy_memory_bank_anomaly_stats(
    *,
    num_classes: int,
    selected_images_by_class: Mapping[int, np.ndarray],
    gt_patch_masks: np.ndarray,
    reduced_features_by_class: Mapping[int, np.ndarray],
    coreset_indices_by_class: Mapping[int, np.ndarray],
    class_names: Optional[Union[Sequence[str], Dict[int, str]]] = None,
    log_prefix: str = "[MemoryBank-Greedy]",
    epoch: Optional[int] = None,
) -> None:
    """Print per-class anomaly patch count and ratio in the bank after greedy sampling.

    Anomaly is defined by GT patch mask (True on defect pixels). The memory bank rows
    are a subset of patches from selected images; ``coreset_indices_by_class`` maps
    each kept row to its index in the pre-coreset patch list for that class.
    """
    epoch_str = f" epoch={epoch}" if epoch is not None else ""

    def _name(cls: int) -> str:
        if class_names is None:
            return str(cls)
        if isinstance(class_names, dict):
            return str(class_names.get(cls, cls))
        if 0 <= cls < len(class_names):
            return str(class_names[cls])
        return str(cls)

    lines = [f"{log_prefix} GT anomaly patches in memory bank (after greedy){epoch_str}"]

    total_kept = 0
    total_anom = 0

    for cls in range(num_classes):
        if cls not in reduced_features_by_class:
            continue
        feats = reduced_features_by_class[cls]
        if feats is None or feats.shape[0] == 0:
            lines.append(f"  class {_name(cls)}: no features in bank")
            continue

        sel = selected_images_by_class.get(cls)
        if sel is None or np.asarray(sel).size == 0:
            lines.append(f"  class {_name(cls)}: missing selected images; skip stats")
            continue

        patch_anomaly = _flatten_gt_patch_anomaly_flags(sel, gt_patch_masks)
        pre_rows = int(patch_anomaly.shape[0])
        idx = np.asarray(coreset_indices_by_class[cls], dtype=np.int64)
        kept_total = int(feats.shape[0])

        if idx.size != kept_total:
            lines.append(
                f"  class {_name(cls)}: coreset idx len {idx.size} != bank rows {kept_total}; skip"
            )
            continue
        if pre_rows == 0:
            lines.append(f"  class {_name(cls)}: zero pre-coreset patches; skip")
            continue
        if idx.size and (idx.min() < 0 or idx.max() >= pre_rows):
            lines.append(
                f"  class {_name(cls)}: coreset indices out of range [0,{pre_rows}); skip"
            )
            continue

        kept_anomaly = int(patch_anomaly[idx].sum())
        ratio = float(kept_anomaly) / float(kept_total) if kept_total > 0 else 0.0
        total_kept += kept_total
        total_anom += kept_anomaly

        lines.append(
            f"  class {_name(cls)}: anomaly_patches={kept_anomaly}/{kept_total} "
            f"({100.0 * ratio:.2f}%)"
        )

    if total_kept > 0:
        r_all = float(total_anom) / float(total_kept)
        lines.append(
            f"  all classes: anomaly_patches={total_anom}/{total_kept} ({100.0 * r_all:.2f}%)"
        )

    print("\n".join(lines))

