"""Unit tests for the V2 class-and-patch fusion module (HCPAF).

Run from the repository root:
    python -m unittest tests.test_class_patch_fusion_adapter -v
"""

import unittest

import torch

from src.model.class_patch_fusion_adapter import ClassPatchFusionAdapter
from src.model.layer_fusion import ClassConditionedLayerFusion
from src.model.model import localnet


def _make_fusion(**kwargs):
    defaults = dict(
        num_classes=3,
        num_layers=4,
        feature_dim=6,
        norm_type="none",
        class_embedding_dim=5,
        gate_hidden_dim=8,
        gate_norm="layer_norm",
        gate_lambda_init=0.1,
    )
    defaults.update(kwargs)
    return ClassPatchFusionAdapter(**defaults)


def _randomise_delta_head(fusion, std=0.05):
    with torch.no_grad():
        fusion.gate.delta_head.weight.normal_(std=std)
        fusion.gate.delta_head.bias.normal_(std=std)


class PatchWeightsTest(unittest.TestCase):
    def test_zero_delta_equals_global_prior(self):
        torch.manual_seed(0)
        fusion = _make_fusion()
        with torch.no_grad():
            fusion.theta.copy_(torch.randn_like(fusion.theta))
        raw = torch.randn(2, 5, 4, 6)
        class_idx = torch.tensor([0, 2])
        patch_alpha = fusion.patch_weights(raw, class_idx)
        global_alpha = fusion.global_weights(class_idx)
        self.assertEqual(tuple(patch_alpha.shape), (2, 5, 4))
        self.assertTrue(
            torch.allclose(
                patch_alpha, global_alpha.unsqueeze(1).expand(-1, 5, -1), atol=1e-7
            )
        )

    def test_alpha_is_simplex_and_finite(self):
        torch.manual_seed(1)
        fusion = _make_fusion()
        _randomise_delta_head(fusion)
        raw = torch.randn(3, 7, 4, 6)
        alpha = fusion.patch_weights(raw, torch.tensor([0, 1, 2]))
        self.assertTrue(torch.isfinite(alpha).all())
        self.assertTrue((alpha >= 0).all())
        self.assertTrue(
            torch.allclose(alpha.sum(dim=-1), torch.ones(3, 7), atol=1e-6)
        )

    def test_patches_and_classes_get_different_weights(self):
        torch.manual_seed(2)
        fusion = _make_fusion()
        _randomise_delta_head(fusion, std=0.5)
        raw = torch.randn(1, 6, 4, 6)
        alpha_a = fusion.patch_weights(raw, torch.tensor([0]))[0]
        alpha_b = fusion.patch_weights(raw, torch.tensor([1]))[0]
        self.assertFalse(torch.allclose(alpha_a, alpha_b))
        self.assertFalse(torch.allclose(alpha_a[0], alpha_a[1]))

    def test_same_class_does_not_force_identical_patches(self):
        torch.manual_seed(3)
        fusion = _make_fusion()
        _randomise_delta_head(fusion, std=0.5)
        raw = torch.randn(2, 5, 4, 6)
        alpha = fusion.patch_weights(raw, torch.tensor([1, 1]))
        self.assertFalse(torch.allclose(alpha[0, 0], alpha[0, 1]))
        self.assertFalse(torch.allclose(alpha[0, 0], alpha[1, 0]))

    def test_single_patch_single_layer(self):
        fusion = _make_fusion(num_layers=1, feature_dim=3)
        raw = torch.randn(2, 1, 1, 3)
        alpha = fusion.patch_weights(raw, torch.tensor([0, 2]))
        self.assertEqual(tuple(alpha.shape), (2, 1, 1))
        self.assertTrue(torch.allclose(alpha, torch.ones_like(alpha)))

    def test_mixed_class_batch(self):
        torch.manual_seed(4)
        fusion = _make_fusion()
        _randomise_delta_head(fusion)
        raw = torch.randn(4, 3, 4, 6)
        idx = torch.tensor([0, 1, 2, 0])
        alpha = fusion.patch_weights(raw, idx)
        prior = fusion.global_weights(idx)
        self.assertEqual(tuple(alpha.shape), (4, 3, 4))
        self.assertTrue(torch.isfinite(prior).all())

    def test_gradients_reach_theta_gate_and_lambda(self):
        torch.manual_seed(5)
        fusion = _make_fusion()
        _randomise_delta_head(fusion, std=0.1)
        raw = torch.randn(2, 4, 4, 6)
        alpha = fusion.patch_weights(raw, torch.tensor([0, 1]))
        fused = fusion.fuse_query(raw, alpha)
        fused.square().mean().backward()

        expected = [
            ("theta", fusion.theta),
            ("gate.proj.weight", fusion.gate.proj.weight),
            ("gate.class_proj.weight", fusion.gate.class_proj.weight),
            ("gate.delta_head.weight", fusion.gate.delta_head.weight),
            ("class_embedding.weight", fusion.class_embedding.weight),
            ("lambda_raw", fusion.lambda_raw),
        ]
        for name, param in expected:
            self.assertIsNotNone(param.grad, name)
            self.assertGreater(float(param.grad.abs().sum()), 0.0, name)

    def test_no_gradient_to_absent_class_rows(self):
        fusion = _make_fusion()
        _randomise_delta_head(fusion)
        alpha = fusion.patch_weights(
            torch.randn(1, 3, 4, 6), torch.tensor([1])
        )
        fusion.fuse_query(torch.randn(1, 3, 4, 6), alpha).square().mean().backward()
        self.assertGreater(float(fusion.theta.grad[1].abs().sum()), 0.0)
        self.assertTrue(
            torch.allclose(
                fusion.theta.grad[[0, 2]],
                torch.zeros_like(fusion.theta.grad[[0, 2]]),
            )
        )


class FusionApiTest(unittest.TestCase):
    def test_fuse_query_matches_v1_with_same_theta(self):
        torch.manual_seed(6)
        v2 = _make_fusion(norm_type="none")
        v1 = ClassConditionedLayerFusion(
            num_classes=3, num_layers=4, feature_dim=6, norm_type="none"
        )
        with torch.no_grad():
            v1.theta.copy_(torch.randn_like(v1.theta))
            v2.theta.copy_(v1.theta)
        raw = torch.randn(2, 5, 4, 6)
        idx = torch.tensor([0, 2])
        self.assertTrue(torch.allclose(v2(raw, idx), v1(raw, idx), atol=1e-7))
        self.assertTrue(
            torch.allclose(
                v2(raw, idx),
                v2.fuse_query(raw, v2.global_weights(idx)),
                atol=1e-7,
            )
        )

    def test_reference_bank_forward_shape(self):
        fusion = _make_fusion()
        ref = torch.randn(9, 4, 6)
        out = fusion(ref, 1)
        self.assertEqual(tuple(out.shape), (9, 6))

    def test_fuse_refs_for_query_same_alpha(self):
        torch.manual_seed(7)
        fusion = _make_fusion()
        raw_refs = torch.randn(2, 3, 4, 6)
        alpha = torch.softmax(torch.randn(2, 4), dim=-1)
        fused = fusion.fuse_refs_for_query(raw_refs, alpha)
        self.assertEqual(tuple(fused.shape), (2, 3, 6))
        expected = torch.einsum("mklc,ml->mkc", fusion.norm(raw_refs), alpha)
        self.assertTrue(torch.allclose(fused, expected, atol=1e-6))

        alpha_k = torch.softmax(torch.randn(2, 3, 4), dim=-1)
        fused_k = fusion.fuse_refs_for_query(raw_refs, alpha_k)
        expected_k = torch.einsum("mklc,mkl->mkc", fusion.norm(raw_refs), alpha_k)
        self.assertTrue(torch.allclose(fused_k, expected_k, atol=1e-6))


class AdapterTest(unittest.TestCase):
    def _adapter_fusion(self):
        return _make_fusion(enable_adapter=True, adapter_hidden_dim=7)

    def test_beta_initial_value(self):
        fusion = self._adapter_fusion()
        beta = fusion.get_beta()
        self.assertTrue(
            torch.allclose(beta, torch.full_like(beta, 0.15), atol=1e-2)
        )

    def test_near_identity_when_beta_is_small(self):
        fusion = self._adapter_fusion()
        with torch.no_grad():
            fusion.beta_head.bias.fill_(-20.0)
        residual = torch.randn(2, 5, 6)
        adapted = fusion.adapt_residual(residual, torch.tensor([0, 1]))
        self.assertTrue(torch.allclose(adapted, residual, atol=1e-6))

    def test_adapter_changes_features_and_has_gradients(self):
        torch.manual_seed(8)
        fusion = self._adapter_fusion()
        with torch.no_grad():
            fusion.beta_head.bias.fill_(5.0)  # beta ~ 1
            fusion.adapter_up.weight.normal_(std=0.5)
        residual = torch.randn(2, 5, 6)
        adapted = fusion.adapt_residual(residual, torch.tensor([0, 2]))
        self.assertFalse(torch.allclose(adapted, residual))
        adapted.square().mean().backward()

        for name, param in [
            ("beta_head.weight", fusion.beta_head.weight),
            ("beta_head.bias", fusion.beta_head.bias),
            ("film.weight", fusion.film.weight),
            ("film.bias", fusion.film.bias),
            ("adapter_down.weight", fusion.adapter_down.weight),
            ("adapter_up.weight", fusion.adapter_up.weight),
            ("class_embedding.weight", fusion.class_embedding.weight),
        ]:
            self.assertIsNotNone(param.grad, name)
            self.assertGreater(float(param.grad.abs().sum()), 0.0, name)

    def test_rho_diagnostics(self):
        fusion = self._adapter_fusion()
        fusion.reset_adapter_stats()
        with torch.no_grad():
            fusion.beta_head.bias.fill_(2.0)
        fusion.adapt_residual(torch.randn(2, 4, 6), torch.tensor([0, 1]))
        rho, count = fusion.adapter_rho()
        self.assertEqual(tuple(rho.shape), (3,))
        self.assertGreaterEqual(float(rho[0]), 0.0)
        self.assertGreaterEqual(float(rho[1]), 0.0)
        self.assertEqual(float(count[0]), 1.0)
        self.assertEqual(float(count[1]), 1.0)


class EndToEndV2CTest(unittest.TestCase):
    def test_residual_matching_to_adapter_backward_and_step(self):
        import utils.train_utils as common_utils

        torch.manual_seed(12)
        fusion = _make_fusion(
            num_classes=2,
            num_layers=3,
            feature_dim=8,
            enable_adapter=True,
            adapter_hidden_dim=7,
        )
        _randomise_delta_head(fusion, std=0.2)
        raw = {0: torch.randn(9, 3, 8)}
        cache = common_utils.FusedReferenceCache(
            raw, fusion, torch.device("cpu"), query_chunk_size=4, ref_chunk_size=3
        )
        cache.refresh()

        net = localnet(
            len_feature=8,
            use_moe_discriminator=False,
            num_classes=2,
            class_conditioned_adapter=True,
            feature_adapter_mode="class_residual",
        )
        net.set_feature_transform_adapter(fusion)

        raw_query = torch.randn(1, 5, 3, 8)
        class_idx = torch.tensor([0])
        residual = common_utils.compute_class_patch_fused_residual_batch(
            raw_query,
            class_idx,
            cache,
            fusion,
            topk=4,
            query_chunk_size=2,
            candidate_chunk_size=2,
        )
        features, score = net(residual, class_idx=class_idx)
        self.assertEqual(tuple(features.shape), (1, 5, 8))
        self.assertEqual(tuple(score.shape), (1, 5))

        loss = torch.nn.functional.binary_cross_entropy(
            score, torch.zeros_like(score)
        )
        loss.backward()
        self.assertGreater(float(fusion.theta.grad.abs().sum()), 0.0)
        self.assertGreater(
            float(fusion.gate.delta_head.weight.grad.abs().sum()), 0.0
        )
        self.assertGreater(float(fusion.beta_head.weight.grad.abs().sum()), 0.0)

        optimizer = torch.optim.SGD(
            list(net.parameters()) + list(fusion.parameters()), lr=0.1
        )
        before = fusion.theta.detach().clone()
        optimizer.step()
        self.assertFalse(torch.allclose(before, fusion.theta.detach()))


class LocalnetAdapterModeTest(unittest.TestCase):
    def _localnet(self, mode):
        return localnet(
            len_feature=8,
            use_moe_discriminator=False,
            num_classes=3,
            class_conditioned_adapter=True,
            feature_adapter_mode=mode,
        )

    def test_class_residual_skips_legacy_adaptor_and_uses_transform_once(self):
        torch.manual_seed(9)
        net = self._localnet("class_residual")
        fusion = _make_fusion(
            feature_dim=8, enable_adapter=True, adapter_hidden_dim=7
        )
        net.set_feature_transform_adapter(fusion)
        self.assertFalse(
            any(
                name.startswith("_feature_transform_adapter")
                for name, _ in net.named_parameters()
            )
        )
        net_params = set(net.parameters())
        self.assertTrue(
            all(p not in net_params for p in fusion.parameters())
        )

        calls = {"n": 0}
        original = fusion.adapt_residual

        def counted(residual, class_idx):
            calls["n"] += 1
            return original(residual, class_idx)

        fusion.adapt_residual = counted

        def _legacy_should_not_run(*_args, **_kwargs):
            raise AssertionError("legacy adaptor must be skipped in V2-C")

        net.adaptor.forward = _legacy_should_not_run

        x = torch.randn(2, 5, 8)
        class_idx = torch.tensor([0, 1])
        features, score = net(x, class_idx=class_idx)
        self.assertEqual(calls["n"], 1)
        self.assertEqual(tuple(features.shape), (2, 5, 8))
        self.assertEqual(tuple(score.shape), (2, 5))

    def test_legacy_mode_does_not_require_transform(self):
        net = self._localnet("legacy")
        x = torch.randn(2, 5, 8)
        features, score = net(x, class_idx=torch.tensor([0, 1]))
        self.assertEqual(tuple(features.shape), (2, 5, 8))
        self.assertEqual(tuple(score.shape), (2, 5))

    def test_invalid_mode_rejected(self):
        with self.assertRaises(ValueError):
            self._localnet("unknown")


if __name__ == "__main__":
    unittest.main()
