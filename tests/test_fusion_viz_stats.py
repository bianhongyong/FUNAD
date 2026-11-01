"""Tests for the streaming statistics used by the fusion visualisation.

Run from the repository root:
    python -m unittest tests.test_fusion_viz_stats -v
"""

import unittest

import numpy as np

from utils.fusion_viz_stats import (
    ClassFusionStats,
    FusionStatsCollector,
    normalize_prior,
    patch_entropy,
    patch_l1_to_prior,
    stats_to_row,
    validate_alpha,
    validate_global_prior,
)


def _toy_alpha():
    # [B=2, P=2, L=3], rows sum to 1
    return np.array(
        [
            [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]],
            [[0.3, 0.3, 0.4], [0.5, 0.25, 0.25]],
        ],
        dtype=np.float64,
    )


def _toy_prior():
    return np.array([[0.6, 0.3, 0.1], [0.2, 0.2, 0.6]], dtype=np.float64)


class StreamingStatsTest(unittest.TestCase):
    def test_matches_direct_numpy(self):
        alpha = _toy_alpha()
        prior = _toy_prior()

        stats = ClassFusionStats(num_layers=3)
        stats.update(alpha, prior)
        result = stats.result()

        n = alpha.shape[0] * alpha.shape[1]
        mean = alpha.sum(axis=(0, 1)) / n
        std = np.sqrt(np.square(alpha).sum(axis=(0, 1)) / n - np.square(mean))
        entropy = patch_entropy(alpha)
        l1 = patch_l1_to_prior(alpha, prior)

        np.testing.assert_allclose(result["mean"], mean, atol=1e-7)
        np.testing.assert_allclose(result["std"], std, atol=1e-7)
        self.assertAlmostEqual(result["entropy_mean"], float(entropy.mean()), places=7)
        self.assertAlmostEqual(result["l1_mean"], float(l1.mean()), places=7)
        self.assertAlmostEqual(result["l1_std"], float(l1.std()), places=7)
        self.assertEqual(result["n_images"], 2)
        self.assertEqual(result["n_patches"], 4)

        expected_dominant = np.bincount(alpha.argmax(-1).ravel(), minlength=3) / 4
        np.testing.assert_allclose(result["dominant_fraction"], expected_dominant)

    def test_batch_size_invariance(self):
        alpha = _toy_alpha()
        prior = _toy_prior()

        whole = ClassFusionStats(num_layers=3)
        whole.update(alpha, prior)

        streamed = ClassFusionStats(num_layers=3)
        for i in range(alpha.shape[0]):
            for j in range(alpha.shape[1]):
                streamed.update(alpha[i : i + 1, j : j + 1], prior[i : i + 1])

        a = whole.result()
        b = streamed.result()
        np.testing.assert_allclose(a["mean"], b["mean"], atol=1e-12)
        np.testing.assert_allclose(a["std"], b["std"], atol=1e-12)
        self.assertAlmostEqual(a["entropy_mean"], b["entropy_mean"], places=12)
        self.assertAlmostEqual(a["l1_mean"], b["l1_mean"], places=12)

    def test_single_layer_is_deterministic(self):
        stats = ClassFusionStats(num_layers=1)
        stats.update(np.ones((2, 3, 1)), np.ones((2, 1)))
        result = stats.result()
        np.testing.assert_allclose(result["mean"], [1.0])
        np.testing.assert_allclose(result["std"], [0.0])
        self.assertEqual(result["entropy_mean"], 0.0)
        self.assertEqual(result["l1_mean"], 0.0)

    def test_collector_tracks_classes(self):
        collector = FusionStatsCollector(num_layers=3)
        collector.update(0, _toy_alpha(), _toy_prior())
        collector.update(1, _toy_alpha(), _toy_prior())
        self.assertEqual(collector.class_indices, [0, 1])
        self.assertEqual(collector.totals(), {"n_images": 4, "n_patches": 8})
        self.assertTrue(collector.has(0))
        self.assertFalse(collector.has(2))

    def test_result_before_update_raises(self):
        stats = ClassFusionStats(num_layers=3)
        with self.assertRaises(RuntimeError):
            stats.result()


class ValidationTest(unittest.TestCase):
    def test_rejects_unnormalised_alpha(self):
        alpha = np.full((1, 2, 3), 0.5)
        with self.assertRaises(ValueError):
            validate_alpha(alpha, 3)

    def test_rejects_wrong_layer_count(self):
        with self.assertRaises(ValueError):
            validate_alpha(_toy_alpha(), num_layers=4)

    def test_rejects_negative_weight(self):
        alpha = _toy_alpha()
        alpha[0, 0, 0] = -0.1
        with self.assertRaises(ValueError):
            validate_alpha(alpha, 3)

    def test_rejects_non_finite(self):
        alpha = _toy_alpha()
        alpha[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            validate_alpha(alpha, 3)

    def test_prior_broadcast(self):
        prior = np.array([0.6, 0.3, 0.1])
        broadcast = normalize_prior(prior, batch=4, num_layers=3)
        self.assertEqual(broadcast.shape, (4, 3))
        np.testing.assert_allclose(broadcast[0], broadcast[3])

    def test_prior_shape_mismatch(self):
        with self.assertRaises(ValueError):
            normalize_prior(np.array([[0.5, 0.5]]), batch=2, num_layers=3)

    def test_global_prior_accepts_one_hot(self):
        alpha = np.array([[0.0, 1.0], [0.25, 0.75]])
        validated = validate_global_prior(alpha)
        np.testing.assert_allclose(validated.sum(axis=1), 1.0)

    def test_global_prior_row_sum(self):
        with self.assertRaises(ValueError):
            validate_global_prior(np.array([[0.4, 0.4]]))

    def test_confidence_interval_free_l1_bounds(self):
        # Worst case: disjoint supports -> L1 == 2
        alpha = np.array([[[1.0, 0.0]]])
        prior = np.array([[0.0, 1.0]])
        l1 = patch_l1_to_prior(alpha, prior)
        np.testing.assert_allclose(l1, [[2.0]])


class RowFormattingTest(unittest.TestCase):
    def test_rows_use_real_block_numbers(self):
        stats = ClassFusionStats(num_layers=3)
        stats.update(_toy_alpha(), _toy_prior())
        row = stats_to_row(
            0, "bottle", stats.result(), block_labels=[3, 6, 9],
            extra={"lambda": 0.1},
        )
        self.assertEqual(row["class"], "bottle")
        self.assertIn("mean_block_3", row)
        self.assertIn("std_block_9", row)
        self.assertNotIn("mean_block_2", row)
        self.assertEqual(row["lambda"], 0.1)
        self.assertIn(row["dominant_block"], (3, 6, 9))


if __name__ == "__main__":
    unittest.main()
