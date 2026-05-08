"""Visualization utilities for training diagnostics."""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_moe_expert_visualizations(epoch, class_expert_count, class_names, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    counts = np.asarray(class_expert_count, dtype=np.float64)
    if counts.ndim != 2:
        return

    num_classes, num_expert = counts.shape
    if len(class_names) == num_classes:
        row_labels = list(class_names)
    else:
        row_labels = [f"class_{idx}" for idx in range(num_classes)]
    expert_labels = [f"expert_{idx}" for idx in range(num_expert)]

    row_sum = counts.sum(axis=1, keepdims=True)
    pref = np.divide(counts, row_sum, out=np.zeros_like(counts), where=row_sum > 0)

    stem = f"epoch_{epoch + 1:03d}"
    pref_csv = os.path.join(save_dir, f"{stem}_class_expert_pref.csv")
    pref_png = os.path.join(save_dir, f"{stem}_class_expert_pref.png")
    util_csv = os.path.join(save_dir, f"{stem}_expert_utilization.csv")
    util_png = os.path.join(save_dir, f"{stem}_expert_utilization.png")

    pref_df = pd.DataFrame(pref, index=row_labels, columns=expert_labels)
    pref_df.to_csv(pref_csv, encoding="utf-8")

    util = counts.sum(axis=0)
    util_sum = float(util.sum())
    if util_sum > 0:
        util = util / util_sum
    util_df = pd.DataFrame({"expert": expert_labels, "utilization": util})
    util_df.to_csv(util_csv, index=False, encoding="utf-8")

    fig_h = max(4.0, 0.45 * num_classes)
    fig_w = max(6.0, 0.8 * num_expert)
    plt.figure(figsize=(fig_w, fig_h))
    im = plt.imshow(pref, aspect="auto")
    plt.colorbar(im, fraction=0.035, pad=0.02)
    plt.xticks(np.arange(num_expert), expert_labels, rotation=35, ha="right")
    plt.yticks(np.arange(num_classes), row_labels)
    plt.xlabel("Expert")
    plt.ylabel("Class")
    plt.title("Class-to-Expert Routing Preference")
    plt.tight_layout()
    plt.savefig(pref_png, dpi=220)
    plt.close()

    plt.figure(figsize=(max(6.0, 0.8 * num_expert), 4.0))
    plt.bar(expert_labels, util)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Expert")
    plt.ylabel("Utilization ratio")
    plt.title("Expert Routing Utilization")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(util_png, dpi=220)
    plt.close()
