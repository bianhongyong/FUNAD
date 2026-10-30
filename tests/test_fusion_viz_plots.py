"""Smoke tests for the six fusion visualisation figures.

These exercise every plotting function with tiny synthetic arrays so a broken
signature / colour norm / save path is caught without a checkpoint.

Run from the repository root:
    python -m unittest tests.test_fusion_viz_plots -v
"""

import os
import tempfile
import unittest

import matplotlib

matplotlib.use("Agg")

import numpy as np

from utils.fusion_viz_plots import (
    configure_matplotlib,
    layer_colormap,
    plot_dominant_layer,
    plot_fusion_detection,
    plot_global_prior,
    plot_l1_by_class,
    plot_patch_entropy,
    plot_patch_mean_std,
)


BLOCKS = [3, 6, 9]
CLASSES = ["bottle", "cable", "capsule"]


class PlotSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure_matplotlib()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.figures = os.path.join(self.tmp.name, "figures")
        os.makedirs(self.figures, exist_ok=True)

    def tearDown(self):
        self.tmp.cleanup()

    def _assert_saved(self, paths, stem):
        self.assertTrue(paths)
        for path in paths:
            self.assertTrue(os.path.isfile(path), path)
            self.assertTrue(os.path.basename(path).startswith(stem))

    def test_fig1_global_prior(self):
        alpha = np.array(
            [[0.6, 0.3, 0.1], [0.2, 0.2, 0.6], [0.0, 0.0, 1.0]]
        )
        paths = plot_global_prior(
            alpha, CLASSES, BLOCKS, [False, False, True], self.figures,
            formats=("png",),
        )
        self._assert_saved(paths, "fig1_global_prior_heatmap")

    def test_fig2_mean_std(self):
        means = np.array([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]])
        stds = np.full_like(means, 0.05)
        priors = np.array([[0.4, 0.4, 0.2], [0.2, 0.4, 0.4]])
        paths = plot_patch_mean_std(
            CLASSES[:2], BLOCKS, means, stds, priors, [(4, 100), (5, 125)],
            self.figures, formats=("png",),
        )
        self._assert_saved(paths, "fig2_patch_mean_std")

    def test_fig3_dominant_layer(self):
        rng = np.random.default_rng(0)
        rgb = rng.integers(0, 255, size=(16, 16, 3), dtype=np.uint8)
        dominant = np.zeros((4, 4), dtype=int)
        dominant[:, :2] = 2
        dominant[0, 0] = 0
        paths = plot_dominant_layer(
            rgb, dominant, BLOCKS, self.figures,
            stem="fig3_dominant_layer_bottle_good_0", formats=("png",),
        )
        self._assert_saved(paths, "fig3_dominant_layer_bottle_good_0")

    def test_fig4_entropy(self):
        rng = np.random.default_rng(1)
        rgb = rng.integers(0, 255, size=(16, 16, 3), dtype=np.uint8)
        entropy = rng.uniform(0.0, np.log(3.0), size=(4, 4))
        paths = plot_patch_entropy(
            rgb, entropy, 3, self.figures,
            stem="fig4_patch_entropy_bottle_good_0", formats=("png",),
        )
        self._assert_saved(paths, "fig4_patch_entropy_bottle_good_0")

    def test_fig4_rejects_out_of_range_entropy(self):
        rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        entropy = np.full((2, 2), np.log(3.0) + 0.5)
        with self.assertRaises(ValueError):
            plot_patch_entropy(
                rgb, entropy, 3, self.figures,
                stem="bad", formats=("png",),
            )

    def test_fig5_l1(self):
        paths = plot_l1_by_class(
            CLASSES, [0.5, 0.9, 1.3], [0.2, 0.1, 0.3], self.figures,
            formats=("png",),
        )
        self._assert_saved(paths, "fig5_l1_to_prior_by_class")

    def test_fig6_detection_panel(self):
        rng = np.random.default_rng(2)
        rgb = rng.integers(0, 255, size=(16, 16, 3), dtype=np.uint8)
        gt = np.zeros((16, 16), dtype=np.uint8)
        gt[4:10, 5:11] = 1
        score = rng.uniform(0.0, 1.0, size=(16, 16))
        alpha = rng.uniform(0.0, 1.0, size=(4, 4))
        l1 = rng.uniform(0.0, 2.0, size=(4, 4))
        paths = plot_fusion_detection(
            rgb, gt, score, alpha, l1, self.figures,
            stem="fig6_fusion_detection_bottle_anomaly_0",
            class_name="bottle",
            case_id="anomaly_0",
            label=1,
            image_score=0.93,
            focus_block=9,
            formats=("png",),
        )
        self._assert_saved(paths, "fig6_fusion_detection_bottle_anomaly_0")

    def test_layer_colormap_is_stable(self):
        cmap_a = layer_colormap(3)
        cmap_b = layer_colormap(3)
        np.testing.assert_allclose(cmap_a.colors, cmap_b.colors)
        self.assertEqual(len(cmap_a.colors), 3)
        # > 8 layers still returns exactly L colours (ordinal fallback).
        self.assertEqual(len(layer_colormap(12).colors), 12)


if __name__ == "__main__":
    unittest.main()
