"""Tests for chapter-1 feature-level anomaly synthesis.

Run from the repository root:
    python -m unittest tests.test_feature_anomaly_synthesis -v
"""

import unittest

import numpy as np
import torch

from src.model.class_patch_fusion_adapter import ClassPatchFusionAdapter
from src.train.feature_anomaly_synthesis import (
    FeatureAnomalySynthesizer,
    build_synthetic_training_batch,
    calibrate_class_sigmas_loio,
    class_sigma_tensor,
    patch_grid_side,
    sample_masks,
    synthesize_residual_batch,
)
import utils.train_utils as common_utils
from utils.loss import compute_balanced_bce_components


def _fusion(num_classes=2, num_layers=3, feature_dim=4):
    return ClassPatchFusionAdapter(
        num_classes=num_classes,
        num_layers=num_layers,
        feature_dim=feature_dim,
        class_embedding_dim=4,
        gate_hidden_dim=8,
        gate_demean_delta=True,
    )


class MaskSamplingTest(unittest.TestCase):
    def test_all_mask_types_shape_and_area(self):
        num_patches = 28 * 28
        for mask_type in ("blob", "line", "spot", "irregular", "mixed", "independent"):
            generator = torch.Generator().manual_seed(0)
            mask, stats = sample_masks(
                batch_size=4,
                num_patches=num_patches,
                mask_type=mask_type,
                area_ratio_range=(0.005, 0.10),
                generator=generator,
            )
            self.assertEqual(tuple(mask.shape), (4, num_patches))
            self.assertEqual(mask.dtype, torch.bool)
            ratios = mask.float().mean(dim=1)
            self.assertTrue(bool((ratios >= 0.005 - 1e-3).all()))
            self.assertTrue(bool((ratios <= 0.10 + 1e-3).all()))
            self.assertAlmostEqual(
                stats["area_ratio_mean"], float(ratios.mean()), places=6
            )
            total = sum(
                stats[f"mask_type_{name}_count"]
                for name in ("blob", "line", "spot", "irregular", "independent")
            )
            self.assertEqual(total, 4)
            if mask_type != "mixed":
                self.assertEqual(
                    stats[f"mask_type_{mask_type}_count"], 4
                )

    def test_non_square_patch_count_rejected(self):
        with self.assertRaises(ValueError):
            patch_grid_side(785)
        with self.assertRaises(ValueError):
            sample_masks(1, 785, mask_type="blob")


class ResidualSynthesisTest(unittest.TestCase):
    def test_outside_mask_is_unchanged_and_gradient_flows(self):
        torch.manual_seed(0)
        residual = torch.randn(2, 16, 4, requires_grad=True)
        mask = torch.zeros(2, 16, dtype=torch.bool)
        mask[0, 0:4] = True
        mask[1, 8:12] = True
        gamma = torch.tensor([1.0, 1.0])
        synthetic = synthesize_residual_batch(
            residual,
            torch.tensor([0, 1]),
            {0: 1.0, 1: 2.0},
            mask,
            gamma,
        )
        self.assertTrue(torch.equal(synthetic[~mask], residual.detach()[~mask]))
        self.assertFalse(
            torch.equal(synthetic[mask], residual.detach()[mask])
        )
        self.assertTrue(synthetic.requires_grad)

        synthetic[mask].sum().backward()
        self.assertIsNotNone(residual.grad)
        self.assertTrue(torch.equal(residual.grad[~mask], torch.zeros_like(
            residual.grad[~mask]
        )))
        self.assertTrue(torch.allclose(
            residual.grad[mask], torch.ones_like(residual.grad[mask])
        ))

    def test_class_sigma_is_applied_and_stats_match(self):
        residual = torch.zeros(3, 16, 4, requires_grad=True)
        mask = torch.zeros(3, 16, dtype=torch.bool)
        mask[:, :2] = True
        syn = synthesize_residual_batch(
            residual,
            torch.tensor([0, 1, 1]),
            {0: 1.0, 1: 3.0},
            mask,
            torch.tensor([1.0, 1.0, 1.0]),
            sigma_scale=0.5,
        )
        # With a zero residual, the inside values are exactly the perturbation;
        # their per-row RMS must respect the class sigma ratio (ignoring noise).
        self.assertGreater(float(syn[0][mask[0]].norm()), 0.0)
        # Class-1 rows share the same sigma, so their perturbation scales agree
        # in distribution.  The exact assertion is the sigma tensor itself:
        sigmas = class_sigma_tensor({0: 1.0, 1: 3.0}, 2)
        self.assertTrue(torch.allclose(sigmas, torch.tensor([1.0, 3.0])))

    def test_synthesizer_stats_and_training_batch_labels(self):
        torch.manual_seed(0)
        residual = torch.randn(2, 16, 4, requires_grad=True)
        synthesizer = FeatureAnomalySynthesizer(
            class_sigmas={0: 1.0, 1: 1.0},
            num_classes=2,
            mask_type="spot",
            area_ratio_range=(0.05, 0.05),
            gamma_range=(0.5, 1.5),
        )
        features, labels, classes, mask, stats = build_synthetic_training_batch(
            residual, torch.tensor([0, 1]), synthesizer
        )
        self.assertEqual(tuple(features.shape), (4, 16, 4))
        self.assertEqual(tuple(labels.shape), (4, 16))
        self.assertEqual(tuple(classes.shape), (4,))
        self.assertTrue(bool((labels[:2] == 0).all()))
        self.assertTrue(torch.equal(labels[2:], mask.to(torch.float32)))
        self.assertTrue(0.5 <= stats["gamma_min"] <= stats["gamma_max"] <= 1.5)
        self.assertEqual(stats["mask_type_spot_count"], 2)
        self.assertTrue(features.requires_grad)


class LoioCalibrationTest(unittest.TestCase):
    def _bank(self, feature_dim=4):
        # Two images x two patches; every patch of an image is identical.
        image0 = torch.zeros(2, 2, feature_dim)
        image1 = torch.ones(2, 2, feature_dim)
        raw = torch.cat([image0, image1], dim=0).numpy().astype(np.float32)
        image_ids = np.array([10, 10, 11, 11], dtype=np.int64)
        return raw, image_ids

    def test_excludes_whole_held_out_image(self):
        raw, image_ids = self._bank()
        fusion = _fusion(num_classes=1, num_layers=2, feature_dim=4)
        sigmas, stats = calibrate_class_sigmas_loio(
            {0: raw}, {0: image_ids}, fusion, torch.device("cpu")
        )
        # ||0 - 1||_2 = sqrt(4) = 2; sigma = 2 / sqrt(C=4) = 1.
        self.assertAlmostEqual(sigmas[0], 1.0, places=6)
        self.assertEqual(stats[0]["num_images"], 2)
        self.assertEqual(stats[0]["num_patches"], 4)
        # If self-matching were not excluded the residuals would be zero.
        self.assertGreater(sigmas[0], 1e-3)

    def test_single_reference_image_rejected(self):
        raw = np.zeros((2, 2, 4), dtype=np.float32)
        fusion = _fusion(num_classes=1, num_layers=2, feature_dim=4)
        with self.assertRaises(RuntimeError):
            calibrate_class_sigmas_loio(
                {0: raw}, {0: np.array([7, 7])}, fusion, torch.device("cpu")
            )


class _FakeReferenceDataset(torch.utils.data.Dataset):
    def __init__(self, classes):
        self.classes = list(classes)

    def __len__(self):
        return len(self.classes)

    def __getitem__(self, idx):
        class_idx = torch.tensor(self.classes[idx], dtype=torch.long)
        image = torch.zeros(3, 4, 4)
        return image, class_idx, idx, torch.zeros(16)


class ImageIdMetadataTest(unittest.TestCase):
    def test_bank_returns_patch_to_image_ids(self):
        dataset = _FakeReferenceDataset([0, 0, 1])
        reference_indices = {0: [0, 1], 1: [2]}

        def extract(images, class_indices=None):
            batch = images.shape[0]
            return torch.zeros(batch, 4, 2, 3)

        bank, info, image_ids = common_utils.build_multilayer_reference_memory_bank(
            dataset,
            reference_indices,
            extract_multilayer_fn=extract,
            batch_size=2,
            num_workers=0,
            device=torch.device("cpu"),
            storage_dtype="float32",
            return_image_ids=True,
        )
        self.assertEqual(bank[0].shape, (8, 2, 3))
        self.assertEqual(bank[1].shape, (4, 2, 3))
        self.assertEqual(image_ids[0].tolist(), [0, 0, 0, 0, 1, 1, 1, 1])
        self.assertEqual(image_ids[1].tolist(), [2, 2, 2, 2])
        self.assertEqual(info["num_layers"], 2)
        self.assertEqual(info["feature_dim"], 3)


class BalancedBceComponentsTest(unittest.TestCase):
    def test_components_reduce_each_group(self):
        criterion = torch.nn.BCELoss()
        pred = torch.tensor([[0.9, 0.1, 0.8, 0.2]])
        target = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
        pos, neg = compute_balanced_bce_components(criterion, pred, target)
        expected_pos = criterion(
            pred[target == 1], target[target == 1]
        )
        expected_neg = criterion(
            pred[target == 0], target[target == 0]
        )
        self.assertTrue(torch.allclose(pos, expected_pos))
        self.assertTrue(torch.allclose(neg, expected_neg))


if __name__ == "__main__":
    unittest.main()
