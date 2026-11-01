"""Unit tests for learned DINOv3 layer fusion (P0 pipeline).

Run from the repository root:
    python -m unittest tests.test_layer_fusion -v
"""

import os
import tempfile
import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

from dataset.feature_extract import (
    extract_dinov3_feature_batch,
    extract_dinov3_multilayer_feature_batch,
)
from src.model.layer_fusion import ClassConditionedLayerFusion
import utils.train_utils as common_utils


class _TinyReferenceDataset(Dataset):
    """Two images; the first pixel encodes the image id for extraction tests."""

    def __len__(self):
        return 2

    def __getitem__(self, index):
        image = torch.full((1, 1, 1), float(index))
        class_idx = torch.tensor(0, dtype=torch.long)
        return image, class_idx


def _fake_multilayer_extract(images, class_indices=None):
    batch = images.shape[0]
    image_ids = images.reshape(batch, -1)[:, 0].view(batch, 1, 1, 1)
    layers = (torch.arange(3, dtype=torch.float32) * 10.0).view(1, 1, 3, 1)
    return (image_ids * 100.0 + layers).expand(batch, 2, 3, 1)


class _FakeBackbone(nn.Module):
    """Deterministic stand-in exposing the DINOv3 intermediate-layer API."""

    def __init__(self, num_blocks=4, dim=3, num_patches=5):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Identity() for _ in range(num_blocks)])
        self.dim = int(dim)
        self.num_patches = int(num_patches)

    def get_intermediate_layers(self, x, n, return_class_token=True):
        batch = x.shape[0]
        image_stat = x.mean(dim=(1, 2, 3)).view(batch, 1, 1)
        outputs = []
        for block in n:
            base = float(int(block) + 1)
            patch = base + image_stat.expand(-1, self.num_patches, self.dim)
            cls = base + image_stat.reshape(batch, 1).expand(-1, self.dim)
            outputs.append((patch, cls))
        return tuple(outputs)


class MultilayerExtractionTest(unittest.TestCase):
    def test_layer_mean_matches_legacy_and_sorted_order(self):
        torch.manual_seed(9)
        extractor = _FakeBackbone(num_blocks=4)
        images = torch.randn(2, 3, 8, 8)
        layers = [1, 3, 4]

        raw = extract_dinov3_multilayer_feature_batch(images, extractor, layers)
        legacy = extract_dinov3_feature_batch(images, extractor, layers)
        self.assertEqual(tuple(raw.shape), (2, 5, 3, 3))
        self.assertTrue(torch.allclose(raw.mean(dim=2), legacy, atol=1e-6))

        raw_unordered = extract_dinov3_multilayer_feature_batch(
            images, extractor, [4, 1, 3, 3]
        )
        self.assertTrue(torch.allclose(raw, raw_unordered, atol=1e-6))

        single_raw = extract_dinov3_multilayer_feature_batch(images, extractor, [2])
        single_legacy = extract_dinov3_feature_batch(images, extractor, [2])
        self.assertEqual(tuple(single_raw.shape), (2, 5, 1, 3))
        self.assertTrue(
            torch.allclose(single_raw[:, :, 0], single_legacy, atol=1e-6)
        )

    def test_cls_token_matches_legacy_layer_mean(self):
        torch.manual_seed(10)
        extractor = _FakeBackbone(num_blocks=4)
        images = torch.randn(2, 3, 8, 8)
        layers = [2, 4]
        raw, cls_token = extract_dinov3_multilayer_feature_batch(
            images, extractor, layers, return_cls_token=True
        )
        legacy, legacy_cls = extract_dinov3_feature_batch(
            images, extractor, layers, return_cls_token=True
        )
        self.assertTrue(torch.allclose(raw.mean(dim=2), legacy, atol=1e-6))
        self.assertTrue(torch.allclose(cls_token, legacy_cls, atol=1e-6))


class FusionModuleTest(unittest.TestCase):
    def test_uniform_weights_equal_layer_mean(self):
        torch.manual_seed(0)
        x = torch.randn(2, 5, 4, 7)
        fusion = ClassConditionedLayerFusion(
            num_classes=2, num_layers=4, feature_dim=7
        )
        self.assertEqual(fusion.norm_type, "none")
        class_idx = torch.tensor([0, 1])
        out = fusion(x, class_idx)
        self.assertEqual(tuple(out.shape), (2, 5, 7))
        self.assertTrue(torch.allclose(out, x.mean(dim=2), atol=1e-6))

    def test_single_layer_identity(self):
        x = torch.randn(3, 4, 1, 6)
        fusion = ClassConditionedLayerFusion(
            num_classes=3, num_layers=1, feature_dim=6, norm_type="none"
        )
        out = fusion(x, torch.tensor([0, 1, 2]))
        self.assertTrue(torch.allclose(out, x[:, :, 0], atol=1e-6))

    def test_class_conditioned_weights_differ(self):
        fusion = ClassConditionedLayerFusion(
            num_classes=2, num_layers=3, feature_dim=2, norm_type="none"
        )
        with torch.no_grad():
            fusion.theta[0] = torch.tensor([4.0, 0.0, -4.0])
            fusion.theta[1] = torch.tensor([-4.0, 0.0, 4.0])
        alpha = fusion.get_layer_weights()
        self.assertFalse(torch.allclose(alpha[0], alpha[1]))
        self.assertTrue(torch.allclose(alpha.sum(dim=-1), torch.ones(2), atol=1e-6))

    def test_shared_weights_shape(self):
        fusion = ClassConditionedLayerFusion(
            num_classes=3,
            num_layers=2,
            feature_dim=4,
            norm_type="none",
            shared_weights=True,
        )
        self.assertEqual(tuple(fusion.theta.shape), (1, 2))
        out = fusion(torch.randn(3, 5, 2, 4), torch.tensor([0, 1, 2]))
        self.assertEqual(tuple(out.shape), (3, 5, 4))
        self.assertEqual(tuple(fusion.get_layer_weights().shape), (1, 2))

    def test_gradients_reach_theta_and_input(self):
        torch.manual_seed(1)
        x = torch.randn(2, 4, 3, 5, requires_grad=True)
        fusion = ClassConditionedLayerFusion(
            num_classes=2, num_layers=3, feature_dim=5, norm_type="none"
        )
        out = fusion(x, torch.tensor([0, 1]))
        out.square().mean().backward()
        self.assertIsNotNone(x.grad)
        self.assertGreater(float(x.grad.abs().sum()), 0.0)
        self.assertIsNotNone(fusion.theta.grad)
        self.assertGreater(float(fusion.theta.grad.abs().sum()), 0.0)

        optimizer = optim.SGD(fusion.parameters(), lr=0.5)
        before = fusion.theta.detach().clone()
        optimizer.step()
        self.assertFalse(torch.allclose(before, fusion.theta.detach()))

    def test_absent_class_rows_get_no_gradient(self):
        torch.manual_seed(11)
        fusion = ClassConditionedLayerFusion(
            num_classes=3, num_layers=2, feature_dim=4, norm_type="none"
        )
        out = fusion(torch.randn(1, 4, 2, 4), torch.tensor([0]))
        out.square().mean().backward()
        self.assertGreater(float(fusion.theta.grad[0].abs().sum()), 0.0)
        self.assertTrue(
            torch.allclose(
                fusion.theta.grad[1:], torch.zeros_like(fusion.theta.grad[1:])
            )
        )


class RawBankTest(unittest.TestCase):
    def test_layers_stay_aligned_per_patch(self):
        dataset = _TinyReferenceDataset()
        raw, info = common_utils.build_multilayer_reference_memory_bank(
            dataset,
            {0: [0, 1]},
            extract_multilayer_fn=_fake_multilayer_extract,
            batch_size=2,
            num_workers=0,
            device=torch.device("cpu"),
            storage_dtype="float32",
        )
        memory = np.asarray(raw[0])
        self.assertEqual(memory.shape, (4, 3, 1))
        # First two rows come from image 0, next two from image 1; each row's
        # three layers must all belong to the same source image/patch.
        self.assertTrue(np.all(memory[0:2, :, 0] == np.array([0, 10, 20])))
        self.assertTrue(np.all(memory[2:4, :, 0] == np.array([100, 110, 120])))
        self.assertEqual(info["num_layers"], 3)
        self.assertEqual(info["feature_dim"], 1)


class FusedCacheTest(unittest.TestCase):
    @staticmethod
    def _make_cache(theta_row=None):
        # Layer 0 and layer 1 swap the two reference clusters, so changing
        # theta flips the nearest reference and exposes cache staleness.
        raw = {
            0: torch.tensor(
                [
                    [[0.0, 0.0], [10.0, 0.0]],
                    [[10.0, 0.0], [0.0, 0.0]],
                ],
                dtype=torch.float32,
            )
        }
        fusion = ClassConditionedLayerFusion(
            num_classes=1, num_layers=2, feature_dim=2, norm_type="none"
        )
        if theta_row is not None:
            with torch.no_grad():
                fusion.theta[0] = torch.tensor(theta_row, dtype=torch.float32)
        cache = common_utils.FusedReferenceCache(raw, fusion, torch.device("cpu"))
        cache.refresh(step=0, epoch=0)
        return fusion, cache

    def test_known_nearest_and_staleness(self):
        fusion, cache = self._make_cache(theta_row=[2.2, -2.2])
        query = torch.tensor([[[2.0, 0.0]]])
        class_idx = torch.tensor([0])
        ids = cache.nearest_ids(query, class_idx)
        self.assertEqual(int(ids.item()), 0)

        # Change theta without refreshing: the cached fused bank is stale.
        with torch.no_grad():
            fusion.theta[0] = torch.tensor([-2.2, 2.2])
        stale_ids = cache.nearest_ids(query, class_idx)
        self.assertEqual(int(stale_ids.item()), 0)
        self.assertGreater(cache.nn_mismatch_rate(query, class_idx), 0.5)

        cache.refresh(step=1, epoch=0)
        fresh_ids = cache.nearest_ids(query, class_idx)
        self.assertEqual(int(fresh_ids.item()), 1)
        self.assertAlmostEqual(cache.nn_mismatch_rate(query, class_idx), 0.0)

    def test_refresh_schedule(self):
        self.assertTrue(
            common_utils.should_refresh_reference_cache("epoch", 2, epoch=4)
        )
        self.assertFalse(
            common_utils.should_refresh_reference_cache("epoch", 2, epoch=3)
        )
        self.assertTrue(
            common_utils.should_refresh_reference_cache("step", 1, step=7)
        )
        self.assertFalse(
            common_utils.should_refresh_reference_cache("step", 5, step=7)
        )
        with self.assertRaises(ValueError):
            common_utils.should_refresh_reference_cache("batch", 1, step=1)


class FusedResidualTest(unittest.TestCase):
    def test_residual_value_and_gradients_from_both_sides(self):
        torch.manual_seed(2)
        raw_ref = {
            0: torch.tensor(
                [
                    [[0.0, 0.0], [1.0, 1.0]],
                    [[5.0, 5.0], [6.0, 6.0]],
                ],
                dtype=torch.float32,
            )
        }
        fusion = ClassConditionedLayerFusion(
            num_classes=1, num_layers=2, feature_dim=2, norm_type="none"
        )
        cache = common_utils.FusedReferenceCache(raw_ref, fusion, torch.device("cpu"))
        cache.refresh()

        raw_query = torch.randn(2, 4, 2, 2, requires_grad=True)
        class_idx = torch.tensor([0, 0])
        residual = common_utils.compute_fused_residual_batch(
            raw_query, class_idx, cache, fusion
        )
        loss = residual.square().mean()
        loss.backward()

        self.assertIsNotNone(raw_query.grad)
        self.assertGreater(float(raw_query.grad.abs().sum()), 0.0)
        self.assertIsNotNone(fusion.theta.grad)
        self.assertGreater(float(fusion.theta.grad.abs().sum()), 0.0)

    def test_residual_is_query_minus_selected_raw_reference(self):
        raw_ref = {
            0: torch.tensor(
                [[[1.0, 2.0]], [[10.0, 20.0]]], dtype=torch.float32
            )
        }
        fusion = ClassConditionedLayerFusion(
            num_classes=1, num_layers=1, feature_dim=2, norm_type="none"
        )
        cache = common_utils.FusedReferenceCache(raw_ref, fusion, torch.device("cpu"))
        cache.refresh()
        raw_query = torch.tensor([[[[9.0, 19.0]], [[2.0, 3.0]]]])
        residual = common_utils.compute_fused_residual_batch(
            raw_query, torch.tensor([0]), cache, fusion
        ).detach()
        expected = torch.tensor([[[-1.0, -1.0], [1.0, 1.0]]])
        self.assertTrue(torch.allclose(residual, expected, atol=1e-6))


class GaussianBceFusionGradTest(unittest.TestCase):
    def test_bce_only_theta_grad_and_reference_side_contribution(self):
        torch.manual_seed(5)
        raw_ref = {
            0: torch.randn(6, 2, 3),
        }
        fusion = ClassConditionedLayerFusion(
            num_classes=1, num_layers=2, feature_dim=3, norm_type="none"
        )
        cache = common_utils.FusedReferenceCache(raw_ref, fusion, torch.device("cpu"))
        cache.refresh()

        raw_query = torch.randn(2, 4, 2, 3)
        class_idx = torch.tensor([0, 0])
        residual = common_utils.compute_fused_residual_batch(
            raw_query, class_idx, cache, fusion
        )
        x_perturbed = common_utils.apply_gaussian_feature_noise(
            residual,
            distance=np.zeros(8, dtype=np.float32),
            threshold_map=np.zeros(8, dtype=np.float32),
            noise_threshold=1.0,
            std=0.05,
        )
        head = nn.Linear(3, 1)
        target = torch.zeros(2, 4, 1)
        bce = nn.BCEWithLogitsLoss()
        bce(head(x_perturbed), target).backward()

        self.assertIsNotNone(fusion.theta.grad)
        grad_with_ref_side = fusion.theta.grad.detach().clone()
        self.assertTrue(torch.isfinite(grad_with_ref_side).all())
        self.assertGreater(float(grad_with_ref_side.abs().sum()), 0.0)

        # Wrong variant: subtract a detached cached fused reference.  The
        # query side still contributes, but the reference side is gone, so the
        # gradient must differ from the correct re-fused reference path.
        fusion.zero_grad()
        fused_query = fusion(raw_query, class_idx)
        ids = cache.nearest_ids(fused_query.detach(), class_idx)
        cached_ref = cache.fused_by_class[0][ids.reshape(-1)]
        cached_ref = cached_ref.reshape(2, 4, 3).detach()
        wrong_residual = fused_query - cached_ref
        wrong_perturbed = common_utils.apply_gaussian_feature_noise(
            wrong_residual,
            distance=np.zeros(8, dtype=np.float32),
            threshold_map=np.zeros(8, dtype=np.float32),
            noise_threshold=1.0,
            std=0.05,
        )
        bce(head(wrong_perturbed), target).backward()
        grad_query_only = fusion.theta.grad.detach().clone()
        self.assertFalse(torch.allclose(grad_with_ref_side, grad_query_only))


class GaussianGradTest(unittest.TestCase):
    def test_2d_numpy_distance_map(self):
        x = torch.ones(2, 4, 3, requires_grad=True)
        out = common_utils.apply_gaussian_feature_noise(
            x,
            distance=np.zeros((2, 4), dtype=np.float32),
            threshold_map=np.zeros((2, 4), dtype=np.float32),
            noise_threshold=1.0,
            std=0.1,
        )
        self.assertEqual(tuple(out.shape), (2, 4, 3))
        out.sum().backward()
        self.assertTrue(torch.allclose(x.grad, torch.ones_like(x)))

    def test_noise_does_not_detach_x(self):
        x = torch.ones(2, 4, 3, requires_grad=True)
        distance = np.zeros(8, dtype=np.float32)
        threshold = np.zeros(8, dtype=np.float32)
        out = common_utils.apply_gaussian_feature_noise(
            x,
            distance=distance,
            threshold_map=threshold,
            noise_threshold=1.0,
            std=0.1,
        )
        self.assertTrue(out.requires_grad)
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.allclose(x.grad, torch.ones_like(x)))


class SidecarTest(unittest.TestCase):
    def test_save_load_and_hash_validation(self):
        raw = {
            0: np.zeros((3, 2, 4), dtype=np.float16),
            1: np.ones((2, 2, 4), dtype=np.float16),
        }
        metadata = {
            "feature_model": "dinov3_vitl16",
            "candidate_layers": [3, 6],
            "storage_dtype": "float16",
        }
        with tempfile.TemporaryDirectory() as tmp:
            manifest = common_utils.save_raw_reference_bank(
                raw, metadata, tmp, prefix="test_bank"
            )
            self.assertFalse(os.path.isabs(manifest["path"]))
            loaded, loaded_meta = common_utils.load_raw_reference_bank(manifest, tmp)
            self.assertEqual(set(loaded.keys()), {0, 1})
            self.assertEqual(loaded_meta["candidate_layers"], [3, 6])
            self.assertTrue(
                np.array_equal(loaded[0], raw[0].astype(np.float16, copy=False))
            )

            bad_hash = dict(manifest)
            bad_hash["sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                common_utils.load_raw_reference_bank(bad_hash, tmp)

            traversal = dict(manifest)
            traversal["path"] = os.path.join("..", manifest["path"])
            with self.assertRaises(ValueError):
                common_utils.load_raw_reference_bank(traversal, tmp)

            mismatch = dict(manifest)
            mismatch["candidate_layers"] = [1, 2]
            with self.assertRaises(ValueError):
                common_utils.load_raw_reference_bank(mismatch, tmp)


class CheckpointTest(unittest.TestCase):
    def _optimizer(self, localnet, fusion):
        params = list(localnet.parameters()) + list(fusion.parameters())
        return optim.SGD(params, lr=0.01)

    def test_fusion_state_roundtrip_and_legacy_compat(self):
        torch.manual_seed(3)
        localnet = nn.Linear(4, 4)
        fusion = ClassConditionedLayerFusion(
            num_classes=2, num_layers=3, feature_dim=4, norm_type="none"
        )
        optimizer = self._optimizer(localnet, fusion)
        with torch.no_grad():
            fusion.theta.copy_(torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]))

        extra = {
            "fusion_state": fusion.state_dict(),
            "fusion_config": {"fusion_mode": "learned_class"},
            "raw_reference_manifest": {"path": "bank.pt", "sha256": "abc"},
            "fusion_cache_metadata": {"version": 1},
            "reference_memory_format": "multilayer_raw_v1",
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ckpt.pt")
            common_utils.save_train_checkpoint(
                path,
                epoch_completed=1,
                iteration=7,
                localnet=localnet,
                localnet_optimizer=optimizer,
                onetoone_optimizer=None,
                best_mean=0.5,
                best_result_by_class={},
                extra=extra,
            )
            info = common_utils.inspect_train_checkpoint(path)
            self.assertTrue(info["has_fusion_state"])
            self.assertEqual(info["reference_memory_format"], "multilayer_raw_v1")

            fresh_localnet = nn.Linear(4, 4)
            fresh_fusion = ClassConditionedLayerFusion(
                num_classes=2, num_layers=3, feature_dim=4, norm_type="none"
            )
            fresh_optimizer = self._optimizer(fresh_localnet, fresh_fusion)
            loaded = common_utils.load_train_checkpoint(
                path,
                fresh_localnet,
                fresh_optimizer,
                None,
                fusion=fresh_fusion,
                return_metadata=True,
            )
            self.assertEqual(loaded[0], 1)
            self.assertEqual(loaded[1], 7)
            self.assertEqual(loaded[6]["fusion_config"]["fusion_mode"], "learned_class")
            self.assertTrue(
                torch.allclose(fusion.theta, fresh_fusion.theta)
            )

            # Fusion parameters must remain in the same optimizer slots so the
            # resumed optimizer state is meaningful.
            self.assertIs(
                fresh_optimizer.param_groups[0]["params"][-1],
                fresh_fusion.theta,
            )

            # Legacy checkpoints only contain localnet parameters and a
            # localnet-only optimizer state.
            legacy_path = os.path.join(tmp, "legacy.pt")
            legacy_saved_localnet = nn.Linear(4, 4)
            legacy_saved_optimizer = optim.SGD(
                legacy_saved_localnet.parameters(), lr=0.01
            )
            common_utils.save_train_checkpoint(
                legacy_path,
                epoch_completed=0,
                iteration=1,
                localnet=legacy_saved_localnet,
                localnet_optimizer=legacy_saved_optimizer,
                onetoone_optimizer=None,
                best_mean=-1.0,
                best_result_by_class={},
            )
            self.assertFalse(
                common_utils.inspect_train_checkpoint(legacy_path)["has_fusion_state"]
            )
            legacy_localnet = nn.Linear(4, 4)
            legacy_optimizer = optim.SGD(legacy_localnet.parameters(), lr=0.01)
            legacy_loaded = common_utils.load_train_checkpoint(
                legacy_path, legacy_localnet, legacy_optimizer, None
            )
            self.assertEqual(legacy_loaded[0], 0)
            other_localnet = nn.Linear(4, 4)
            other_optimizer = optim.SGD(other_localnet.parameters(), lr=0.01)
            with self.assertRaises(RuntimeError):
                common_utils.load_train_checkpoint(
                    legacy_path,
                    other_localnet,
                    other_optimizer,
                    None,
                    fusion=fresh_fusion,
                )


if __name__ == "__main__":
    unittest.main()
