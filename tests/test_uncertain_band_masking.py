"""Tests for dropping the uncertain pseudo-label band from training.

When the Gaussian noise branch is disabled, patches with
``threshold < distance < noise_threshold`` must not contribute to the BCE or
the one-to-one (OTO) loss.

Run from the repository root:
    python -m unittest tests.test_uncertain_band_masking -v
"""

import unittest

import numpy as np
import torch
import torch.nn as nn

from utils.loss import compute_oto_loss_multiclass
from utils.train_utils import build_uncertain_patch_mask


class UncertainPatchMaskTest(unittest.TestCase):
    def test_band_is_exclusive_of_threshold_and_noise_threshold(self):
        distance = np.array([[0.10, 0.15, 0.40, 0.80, 0.95]], dtype=np.float32)
        threshold_map = np.full_like(distance, 0.15)

        mask = build_uncertain_patch_mask(
            distance=distance,
            threshold_map=threshold_map,
            noise_threshold=0.8,
        )

        # 0.15 is not > threshold and 0.8 is not < noise_threshold.
        self.assertEqual(
            mask.tolist(),
            [[False, False, True, False, False]],
        )

    def test_per_patch_threshold_map(self):
        distance = np.array([[0.20, 0.20, 0.70, 0.70]], dtype=np.float32)
        threshold_map = np.array([[0.10, 0.30, 0.30, 0.90]], dtype=np.float32)

        mask = build_uncertain_patch_mask(distance, threshold_map, 0.8)

        self.assertEqual(mask.tolist(), [[True, False, True, False]])

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            build_uncertain_patch_mask(
                np.zeros((2, 3), dtype=np.float32),
                np.zeros((1, 3), dtype=np.float32),
                0.8,
            )


class OtomaskTest(unittest.TestCase):
    def _run(self, keep_patch_mask):
        class _Args:
            oto_loss = "mse"

        batch_feature = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
                [[4.0, 4.0], [5.0, 5.0], [6.0, 6.0], [7.0, 7.0]],
            ],
            dtype=torch.float32,
            requires_grad=True,
        )
        local_pred = torch.tensor(
            [[0.10, 0.20, 0.30, 0.40], [0.50, 0.60, 0.70, 0.80]],
            dtype=torch.float32,
            requires_grad=True,
        )
        class_idx = torch.tensor([0, 0])

        seen_feature_counts = []

        def compute_distance_fn(feature_np):
            seen_feature_counts.append(int(feature_np.shape[0]))
            # Every patch matches its next neighbour (mutual pairs).
            return None, np.arange(feature_np.shape[0])

        def find_matching_fn(id_array):
            n = int(id_array.shape[0])
            src = np.arange(0, n - 1, 2)
            dst = src + 1
            return [src, dst]

        loss = compute_oto_loss_multiclass(
            args=_Args(),
            batch_feature=batch_feature,
            local_pred=local_pred,
            class_idx=class_idx,
            l_loss=nn.MSELoss(),
            compute_distance_fn=compute_distance_fn,
            find_matching_fn=find_matching_fn,
            keep_patch_mask=keep_patch_mask,
        )
        if loss.requires_grad:
            loss.backward()
            grad = local_pred.grad
        else:
            # Every class had fewer than two kept patches: the helper returns a
            # constant zero and the training loop skips the OTO backward.
            grad = torch.zeros_like(local_pred)
        return loss, seen_feature_counts, grad

    def test_without_mask_all_patches_are_used(self):
        loss, counts, grad = self._run(keep_patch_mask=None)

        self.assertEqual(counts, [8])
        self.assertGreater(loss.item(), 0.0)
        self.assertTrue(torch.all(grad != 0))

    def test_mask_drops_patches_and_their_gradients(self):
        keep = torch.tensor(
            [[True, True, True, True], [False, False, False, False]]
        )
        loss, counts, grad = self._run(keep_patch_mask=keep)

        self.assertEqual(counts, [4])
        self.assertGreater(loss.item(), 0.0)
        # Only the kept image contributes gradients.
        self.assertTrue(torch.all(grad[0] != 0))
        self.assertTrue(torch.all(grad[1] == 0))

    def test_mask_with_fewer_than_two_patches_in_a_class_is_skipped(self):
        keep = torch.tensor(
            [[True, False, False, False], [False, False, False, False]]
        )
        loss, counts, grad = self._run(keep_patch_mask=keep)

        self.assertEqual(counts, [])
        self.assertEqual(loss.item(), 0.0)
        self.assertTrue(torch.all(grad == 0))


if __name__ == "__main__":
    unittest.main()
