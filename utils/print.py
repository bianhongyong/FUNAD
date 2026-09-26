import os

import numpy as np


def print_epoch_losses(epoch, loss, bce_loss, oto_loss, gate_aux_loss=None):
    if gate_aux_loss is None:
        print(
            "epoch %d | loss: %.6f, bce loss: %.6f, one-to-one loss: %.6f"
            % (epoch + 1, loss, bce_loss, oto_loss)
        )
        return
    print(
        "epoch %d | loss: %.6f, bce loss: %.6f, one-to-one loss: %.6f, gate aux loss: %.6f"
        % (epoch + 1, loss, bce_loss, oto_loss, gate_aux_loss)
    )


def print_epoch_times(epoch, memory_bank_time, pseudo_label_time, kl_loss_time):
    print(
        "epoch %d | memory bank: %.4fs, pseudo label: %.4fs, kl: %.4fs"
        % (epoch + 1, memory_bank_time, pseudo_label_time, kl_loss_time)
    )


def print_selected_score_distribution_by_class(
    selected_indices: np.ndarray, class_stack: np.ndarray, image_scores: np.ndarray
):
    if selected_indices.shape[0] == 0:
        print("[Phase 2/3] selected score distribution (raw): no selected samples.")
        return

    print("[Phase 2/3] selected score distribution by class (raw, before normalization):")
    selected_classes = np.unique(class_stack[selected_indices]).tolist()
    for class_idx in selected_classes:
        cls = int(class_idx)
        cls_selected = selected_indices[class_stack[selected_indices] == cls]
        cls_scores = image_scores[cls_selected]
        if cls_scores.size == 0:
            continue
        q25, q50, q75 = np.quantile(cls_scores, [0.25, 0.5, 0.75])
        print(
            "  class %d | n=%d | min=%.6f | q25=%.6f | median=%.6f | q75=%.6f | max=%.6f | mean=%.6f | std=%.6f"
            % (
                cls,
                cls_scores.size,
                float(cls_scores.min()),
                float(q25),
                float(q50),
                float(q75),
                float(cls_scores.max()),
                float(cls_scores.mean()),
                float(cls_scores.std()),
            )
        )


def print_selected_clean_ratio(selected_indices: np.ndarray, dataset):
    if selected_indices.shape[0] == 0:
        print("[Phase 2/3] selected image clean ratio: no selected samples.")
        return

    if not hasattr(dataset, "samples"):
        print("[Phase 2/3] selected image clean ratio: dataset has no samples metadata.")
        return

    selected_noisy = 0
    selected_total = 0
    samples = dataset.samples
    max_index = len(samples) - 1
    noisy_per_class = {}

    for idx in selected_indices.tolist():
        idx_int = int(idx)
        if idx_int < 0 or idx_int > max_index:
            continue
        path, _class_idx = samples[idx_int]
        filename = os.path.basename(path).lower()
        if "noisy" in filename:
            selected_noisy += 1
            cls_int = int(_class_idx)
            noisy_per_class[cls_int] = noisy_per_class.get(cls_int, 0) + 1
        selected_total += 1

    if selected_total == 0:
        print("[Phase 2/3] selected image clean ratio: no valid sample metadata found.")
        return

    selected_clean = selected_total - selected_noisy
    clean_ratio = selected_clean / selected_total
    noisy_ratio = selected_noisy / selected_total
    print(
        "[Phase 2/3] selected image clean ratio | clean=%d/%d (%.2f%%) | noisy=%d/%d (%.2f%%)"
        % (
            selected_clean,
            selected_total,
            clean_ratio * 100.0,
            selected_noisy,
            selected_total,
            noisy_ratio * 100.0,
        )
    )

    if noisy_per_class:
        # Sort by class index for stable, readable output.
        parts = []
        for cls_int in sorted(noisy_per_class.keys()):
            parts.append(f"class={cls_int}: {noisy_per_class[cls_int]}")
        print(
            "[Phase 2/3] selected noisy image breakdown by class | "
            + " | ".join(parts)
        )


def print_full_dataset_confusion_matrix_by_raw_score(
    dataset_size: int = None,
    image_scores: np.ndarray = None,
    class_stack: np.ndarray = None,
    dataset=None,
    num_classes: int = None,
    threshold: float = 0.5,
    selected_indices: np.ndarray = None,
    gt_patch_masks: np.ndarray = None,
    image_norm_scores: np.ndarray = None,
):
    # Backward compatible adapter:
    # - old call style: dataset_size/image_scores/class_stack/dataset/num_classes/threshold
    # - new call style from multiclass precompute:
    #   selected_indices/class_stack/gt_patch_masks/image_norm_scores
    if (
        image_scores is None
        and image_norm_scores is not None
        and class_stack is not None
        and selected_indices is not None
    ):
        image_scores = np.asarray(image_norm_scores, dtype=np.float32)
        class_stack = np.asarray(class_stack, dtype=np.int64)
        selected_indices = np.asarray(selected_indices, dtype=np.int64)
        if selected_indices.size == 0:
            print(
                "[Phase 1] confusion matrix (raw score, selected set): empty selected_indices."
            )
            return
        if image_scores.shape[0] != class_stack.shape[0]:
            print(
                "[Phase 1] confusion matrix (raw score, selected set): "
                "image_norm_scores/class_stack size mismatch."
            )
            return

        n_cls = int(class_stack.max()) + 1 if class_stack.size > 0 else 0
        if n_cls <= 0:
            print(
                "[Phase 1] confusion matrix (raw score, selected set): no valid classes."
            )
            return

        # If gt_patch_masks provided, use patch-derived anomaly label; otherwise fall back
        # to score thresholding on selected set as a weak self-consistency metric.
        gt_image_is_anomaly = None
        if gt_patch_masks is not None:
            gt_patch_masks = np.asarray(gt_patch_masks)
            if (
                gt_patch_masks.ndim == 2
                and gt_patch_masks.shape[0] == image_scores.shape[0]
            ):
                gt_image_is_anomaly = (
                    gt_patch_masks.reshape(gt_patch_masks.shape[0], -1).sum(axis=1) > 0
                )

        def _init_counter():
            return {"tp": 0, "fp": 0, "tn": 0, "fn": 0}

        per_class = {i: _init_counter() for i in range(n_cls)}
        overall = _init_counter()

        for idx_int in selected_indices.tolist():
            if idx_int < 0 or idx_int >= image_scores.shape[0]:
                continue
            cls = int(class_stack[idx_int])
            if cls < 0 or cls >= n_cls:
                continue

            score = float(image_scores[idx_int])
            pred_anomaly = score > float(threshold)
            if gt_image_is_anomaly is None:
                actual_anomaly = pred_anomaly
            else:
                actual_anomaly = bool(gt_image_is_anomaly[idx_int])

            bucket = per_class[cls]
            if pred_anomaly and actual_anomaly:
                bucket["tp"] += 1
                overall["tp"] += 1
            elif pred_anomaly and (not actual_anomaly):
                bucket["fp"] += 1
                overall["fp"] += 1
            elif (not pred_anomaly) and (not actual_anomaly):
                bucket["tn"] += 1
                overall["tn"] += 1
            else:
                bucket["fn"] += 1
                overall["fn"] += 1

        def _safe_div(num, den):
            return float(num) / float(den) if den > 0 else 0.0

        def _metrics(c):
            tp, fp, tn, fn = c["tp"], c["fp"], c["tn"], c["fn"]
            support = tp + fp + tn + fn
            acc = _safe_div(tp + tn, support)
            prec = _safe_div(tp, tp + fp)
            rec = _safe_div(tp, tp + fn)
            f1 = _safe_div(2 * prec * rec, prec + rec) if (prec + rec) > 0 else 0.0
            return support, acc, prec, rec, f1

        print(
            "[Phase 1] confusion matrix by class (selected set, normalized score; "
            f"threshold={float(threshold):.3f}, pred: score>thr => anomaly)"
        )
        print(
            "CM_BY_CLASS|class_idx|class_name|support|tp|fp|tn|fn|accuracy|precision|recall|f1"
        )
        for cls in range(n_cls):
            c = per_class[cls]
            support, acc, prec, rec, f1 = _metrics(c)
            print(
                "CM_BY_CLASS|%d|%s|%d|%d|%d|%d|%d|%.4f|%.4f|%.4f|%.4f"
                % (
                    cls,
                    f"class_{cls}",
                    support,
                    c["tp"],
                    c["fp"],
                    c["tn"],
                    c["fn"],
                    acc,
                    prec,
                    rec,
                    f1,
                )
            )
        s, acc, prec, rec, f1 = _metrics(overall)
        print(
            "CM_OVERALL|support=%d|tp=%d|fp=%d|tn=%d|fn=%d|accuracy=%.4f|precision=%.4f|recall=%.4f|f1=%.4f"
            % (s, overall["tp"], overall["fp"], overall["tn"], overall["fn"], acc, prec, rec, f1)
        )
        return

    if (
        dataset_size is None
        or image_scores is None
        or class_stack is None
        or dataset is None
        or num_classes is None
    ):
        raise TypeError(
            "print_full_dataset_confusion_matrix_by_raw_score: invalid arguments. "
            "Expected either old signature "
            "(dataset_size, image_scores, class_stack, dataset, num_classes[, threshold]) "
            "or new signature "
            "(selected_indices=..., class_stack=..., gt_patch_masks=..., image_norm_scores=...)."
        )
    if not hasattr(dataset, "samples"):
        print(
            "[Phase 1] confusion matrix (raw score, full set): dataset has no samples metadata."
        )
        return

    samples = dataset.samples
    n = min(int(dataset_size), len(samples), int(image_scores.shape[0]))
    class_names = (
        list(dataset.class_names)
        if hasattr(dataset, "class_names") and len(dataset.class_names) >= num_classes
        else [f"class_{i}" for i in range(num_classes)]
    )

    def _init_counter():
        return {"tp": 0, "fp": 0, "tn": 0, "fn": 0}

    per_class = {i: _init_counter() for i in range(num_classes)}
    overall = _init_counter()

    for idx_int in range(n):
        path, _class_idx = samples[idx_int]
        cls = int(class_stack[idx_int])
        if cls < 0 or cls >= num_classes:
            continue

        filename = os.path.basename(path).lower()
        actual_anomaly = "noisy" in filename
        score = float(image_scores[idx_int])
        pred_anomaly = score > float(threshold)

        bucket = per_class[cls]
        if pred_anomaly and actual_anomaly:
            bucket["tp"] += 1
            overall["tp"] += 1
        elif pred_anomaly and (not actual_anomaly):
            bucket["fp"] += 1
            overall["fp"] += 1
        elif (not pred_anomaly) and (not actual_anomaly):
            bucket["tn"] += 1
            overall["tn"] += 1
        else:
            bucket["fn"] += 1
            overall["fn"] += 1

    def _safe_div(num, den):
        return float(num) / float(den) if den > 0 else 0.0

    def _metrics(c):
        tp, fp, tn, fn = c["tp"], c["fp"], c["tn"], c["fn"]
        support = tp + fp + tn + fn
        acc = _safe_div(tp + tn, support)
        prec = _safe_div(tp, tp + fp)
        rec = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * prec * rec, prec + rec) if (prec + rec) > 0 else 0.0
        return support, acc, prec, rec, f1

    print(
        "[Phase 1] confusion matrix by class (raw score, full set, before bank filter; "
        "threshold=%.3f, pred: score>thr => anomaly)"
        % float(threshold)
    )
    print(
        "CM_BY_CLASS|class_idx|class_name|support|tp|fp|tn|fn|accuracy|precision|recall|f1"
    )
    for cls in range(num_classes):
        c = per_class[cls]
        support, acc, prec, rec, f1 = _metrics(c)
        print(
            "CM_BY_CLASS|%d|%s|%d|%d|%d|%d|%d|%.6f|%.6f|%.6f|%.6f"
            % (
                cls,
                str(class_names[cls]),
                support,
                c["tp"],
                c["fp"],
                c["tn"],
                c["fn"],
                acc,
                prec,
                rec,
                f1,
            )
        )

    support, acc, prec, rec, f1 = _metrics(overall)
    print(
        "CM_OVERALL|all|all|%d|%d|%d|%d|%d|%.6f|%.6f|%.6f|%.6f"
        % (
            support,
            overall["tp"],
            overall["fp"],
            overall["tn"],
            overall["fn"],
            acc,
            prec,
            rec,
            f1,
        )
    )
