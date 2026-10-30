"""Tests for per-class pinned last-layer class prior (e.g. screw).

For ``no_layer_fusion_classes``:
  * the class prior (used for reference matching) is exactly one-hot on the
    last candidate layer, with no learned inter-layer fusion weights;
  * the patch gate may still deviate per patch around that fixed prior.

Run from the repository root:
    python -m unittest tests.test_class_patch_last_layer_prior -v
"""

import unittest

import torch

from src.model.class_patch_fusion_adapter import ClassPatchFusionAdapter
import utils.train_utils as common_utils


def _make_fusion(num_classes=3, num_layers=3, feature_dim=4):
    return ClassPatchFusionAdapter(
        num_classes=num_classes,
        num_layers=num_layers,
        feature_dim=feature_dim,
        class_embedding_dim=4,
        gate_hidden_dim=8,
        no_layer_fusion_classes=[1],
        last_layer_prior_logit=8.0,
    )


class PinnedLastLayerPriorTest(unittest.TestCase):
    def test_global_weights_are_exact_one_hot_for_pinned_class(self):
        fusion = _make_fusion()
        idx = torch.tensor([0, 1, 2])
        alpha = fusion.global_weights(idx)
        self.assertTrue(
            torch.equal(alpha[1], torch.tensor([0.0, 0.0, 1.0]))
        )
        # Unpinned classes keep the regular uniform theta=0 softmax.
        self.assertTrue(
            torch.allclose(alpha[0], torch.full((3,), 1.0 / 3.0))
        )
        self.assertTrue(
            torch.allclose(alpha[2], torch.full((3,), 1.0 / 3.0))
        )

    def test_get_layer_weights_reports_one_hot(self):
        fusion = _make_fusion()
        alpha = fusion.get_layer_weights()
        self.assertTrue(
            torch.equal(alpha[1], torch.tensor([0.0, 0.0, 1.0]))
        )
        self.assertEqual(float(fusion.weight_entropy(1)), 0.0)

    def test_patch_weights_start_at_last_layer(self):
        fusion = _make_fusion()
        query = torch.randn(1, 5, 3, 4)
        alpha = fusion.patch_weights(query, torch.tensor([1]))
        self.assertTrue(torch.all(alpha[..., -1] > 0.99))
        self.assertTrue(torch.all(alpha[..., :2] < 1e-2))
        self.assertTrue(
            torch.allclose(alpha.sum(dim=-1), torch.ones(1, 5), atol=1e-6)
        )

    def test_patch_gate_can_deviate_and_gets_gradient(self):
        fusion = _make_fusion()
        query = torch.randn(1, 5, 3, 4)
        base = fusion.patch_weights(query, torch.tensor([1])).detach()

        with torch.no_grad():
            fusion.gate.delta_head.weight.normal_(std=0.5)
            fusion.gate.delta_head.bias.normal_(std=0.5)
        alpha = fusion.patch_weights(query, torch.tensor([1]))
        self.assertFalse(torch.allclose(base, alpha.detach()))

        alpha.sum().backward()
        # Pinned theta row is not part of the graph.
        self.assertTrue(
            fusion.theta.grad is None
            or torch.count_nonzero(fusion.theta.grad[1]) == 0
        )
        self.assertIsNotNone(fusion.gate.delta_head.weight.grad)
        self.assertGreater(
            float(fusion.gate.delta_head.weight.grad.abs().sum()), 0.0
        )

    def test_matching_uses_last_layer_only(self):
        torch.manual_seed(0)
        num_layers, feature_dim = 3, 4
        raw = {
            0: torch.randn(7, num_layers, feature_dim),
            1: torch.randn(5, num_layers, feature_dim) + 10.0,
        }
        fusion = ClassPatchFusionAdapter(
            num_classes=2,
            num_layers=num_layers,
            feature_dim=feature_dim,
            class_embedding_dim=4,
            gate_hidden_dim=8,
            no_layer_fusion_classes=[1],
            last_layer_prior_logit=8.0,
        )
        with torch.no_grad():
            fusion.theta.normal_()
        cache = common_utils.FusedReferenceCache(
            raw,
            fusion,
            torch.device("cpu"),
            expected_num_layers=num_layers,
            expected_feature_dim=feature_dim,
            query_chunk_size=4,
            ref_chunk_size=3,
        )
        cache.refresh()

        query = torch.randn(1, 6, num_layers, feature_dim)
        idx = torch.tensor([1])
        _, ids = common_utils.compute_class_patch_prior_top1_residual_batch(
            query, idx, cache, fusion, return_neighbor_ids=True
        )
        expected = torch.cdist(
            query[0, :, -1, :],
            raw[1][:, -1, :].to(torch.float32),
        ).argmin(dim=1)
        self.assertTrue(torch.equal(ids[0], expected))

    def test_state_dict_is_unchanged(self):
        plain = ClassPatchFusionAdapter(
            num_classes=3, num_layers=3, feature_dim=4,
            class_embedding_dim=4, gate_hidden_dim=8,
        )
        pinned = _make_fusion()
        self.assertEqual(
            set(plain.state_dict().keys()), set(pinned.state_dict().keys())
        )
        pinned.load_state_dict(plain.state_dict(), strict=True)


if __name__ == "__main__":
    unittest.main()
