"""Tests for prior_top1 class-prior matching + patch-only residual (schema v3).

Run from the repository root:
    python -m unittest tests.test_class_patch_prior_top1 -v
"""

import os
import tempfile
import unittest

import torch

from src.model.class_patch_fusion_adapter import ClassPatchFusionAdapter
from src.model.layer_fusion import ClassConditionedLayerFusion
import utils.train_utils as common_utils


def _make_fusion(
    num_classes=2,
    num_layers=3,
    feature_dim=4,
    norm_type="none",
    norm_affine=False,
):
    return ClassPatchFusionAdapter(
        num_classes=num_classes,
        num_layers=num_layers,
        feature_dim=feature_dim,
        norm_type=norm_type,
        norm_affine=norm_affine,
        class_embedding_dim=4,
        gate_hidden_dim=8,
    )


def _randomise_delta_head(fusion, std=0.2):
    with torch.no_grad():
        fusion.gate.delta_head.weight.normal_(std=std)
        fusion.gate.delta_head.bias.normal_(std=std)


def _randomise_theta(fusion, seed=0):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        fusion.theta.copy_(
            torch.randn(fusion.theta.shape, generator=generator)
        )


def _make_bank_and_cache(
    num_layers=3, feature_dim=4, norm_type="none", norm_affine=False,
    storage_dtype=None,
):
    torch.manual_seed(0)
    raw = {
        0: torch.randn(7, num_layers, feature_dim),
        1: torch.randn(5, num_layers, feature_dim) + 10.0,
    }
    if storage_dtype is not None:
        raw = {k: v.to(storage_dtype) for k, v in raw.items()}
    fusion = _make_fusion(
        num_layers=num_layers,
        feature_dim=feature_dim,
        norm_type=norm_type,
        norm_affine=norm_affine,
    )
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
    return raw, fusion, cache


def _bruteforce_prior_ids(raw_query, class_idx, fusion, raw, dtype=torch.float32):
    """Exact class-prior Top-1 ids, independent of FusedReferenceCache.search."""
    best = torch.empty(raw_query.shape[0], raw_query.shape[1], dtype=torch.long)
    for i in range(raw_query.shape[0]):
        cls = int(class_idx[i])
        alpha = fusion.get_layer_weights(cls).detach()
        q = fusion.fuse_query(raw_query[i], alpha)  # [P,C]
        ref = fusion.fuse_query(raw[cls].to(dtype=dtype), alpha)  # [N,C]
        best[i] = torch.cdist(q, ref).argmin(dim=1)
    return best


class PriorTop1MatchTest(unittest.TestCase):
    def test_ids_match_independent_bruteforce(self):
        raw, fusion, cache = _make_bank_and_cache()
        _randomise_theta(fusion)
        cache.refresh()
        _randomise_delta_head(fusion)
        query = torch.randn(2, 9, 3, 4)
        idx = torch.tensor([0, 1])
        _, ids = common_utils.compute_class_patch_prior_top1_residual_batch(
            query, idx, cache, fusion, return_neighbor_ids=True
        )
        expected = _bruteforce_prior_ids(query, idx, fusion, raw)
        self.assertTrue(torch.equal(ids, expected))

    def test_layer_norm_matching(self):
        raw, fusion, cache = _make_bank_and_cache(norm_type="layer_norm")
        _randomise_theta(fusion)
        cache.refresh()
        _randomise_delta_head(fusion)
        query = torch.randn(2, 6, 3, 4)
        idx = torch.tensor([0, 1])
        _, ids = common_utils.compute_class_patch_prior_top1_residual_batch(
            query, idx, cache, fusion, return_neighbor_ids=True
        )
        expected = _bruteforce_prior_ids(query, idx, fusion, raw)
        self.assertTrue(torch.equal(ids, expected))

    def test_gate_never_changes_matched_ids(self):
        raw, fusion, cache = _make_bank_and_cache()
        _randomise_theta(fusion)
        cache.refresh()
        query = torch.randn(2, 5, 3, 4)
        idx = torch.tensor([0, 1])
        base_residual, base_ids = (
            common_utils.compute_class_patch_prior_top1_residual_batch(
                query, idx, cache, fusion, return_neighbor_ids=True
            )
        )

        _randomise_delta_head(fusion, std=0.5)
        with torch.no_grad():
            fusion.lambda_raw.add_(1.5)
        new_residual, new_ids = (
            common_utils.compute_class_patch_prior_top1_residual_batch(
                query, idx, cache, fusion, return_neighbor_ids=True
            )
        )
        self.assertTrue(torch.equal(base_ids, new_ids))
        self.assertFalse(torch.allclose(base_residual, new_residual))

    def test_mixed_class_batch_uses_each_own_bank(self):
        raw, fusion, cache = _make_bank_and_cache()
        _randomise_theta(fusion)
        cache.refresh()
        query = torch.randn(4, 5, 3, 4)
        idx = torch.tensor([0, 1, 0, 1])
        _, ids = common_utils.compute_class_patch_prior_top1_residual_batch(
            query, idx, cache, fusion, return_neighbor_ids=True
        )
        for row, cls in enumerate(idx.tolist()):
            self.assertGreaterEqual(int(ids[row].min()), 0)
            self.assertLess(int(ids[row].max()), raw[cls].shape[0])


class PriorTop1ResidualTest(unittest.TestCase):
    def test_residual_uses_one_alpha_on_both_sides(self):
        raw, fusion, cache = _make_bank_and_cache()
        _randomise_theta(fusion)
        cache.refresh()
        _randomise_delta_head(fusion)
        query = torch.randn(2, 5, 3, 4)
        idx = torch.tensor([0, 1])
        residual, ids = common_utils.compute_class_patch_prior_top1_residual_batch(
            query, idx, cache, fusion, return_neighbor_ids=True
        )
        alpha = fusion.patch_weights(query, idx).detach()
        for i in range(2):
            cls = int(idx[i])
            q_fused = fusion.fuse_query(query[i], alpha[i])
            ref = raw[cls][ids[i]].to(torch.float32)
            ref_fused = fusion.fuse_query(ref, alpha[i])
            self.assertTrue(
                torch.allclose(residual[i], q_fused - ref_fused, atol=1e-6)
            )

    def test_zero_delta_matches_v1_prior_residual(self):
        raw, fusion, cache = _make_bank_and_cache()
        _randomise_theta(fusion)
        cache.refresh()
        v1 = ClassConditionedLayerFusion(
            num_classes=fusion.num_classes,
            num_layers=fusion.num_layers,
            feature_dim=fusion.feature_dim,
            norm_type=fusion.norm_type,
        )
        with torch.no_grad():
            v1.theta.copy_(fusion.theta)
        v1_cache = common_utils.FusedReferenceCache(
            raw, v1, torch.device("cpu"), query_chunk_size=4, ref_chunk_size=3
        )
        v1_cache.refresh()

        query = torch.randn(2, 5, 3, 4)
        idx = torch.tensor([0, 1])
        new_residual = common_utils.compute_class_patch_prior_top1_residual_batch(
            query, idx, cache, fusion
        )
        v1_residual = common_utils.compute_fused_residual_batch(
            query, idx, v1_cache, v1
        )
        self.assertTrue(torch.allclose(new_residual, v1_residual, atol=1e-6))

    def test_matches_legacy_dynamic_topk_k1(self):
        raw, fusion, cache = _make_bank_and_cache()
        _randomise_theta(fusion)
        cache.refresh()
        _randomise_delta_head(fusion)
        query = torch.randn(2, 5, 3, 4)
        idx = torch.tensor([0, 1])
        new_residual, new_ids = (
            common_utils.compute_class_patch_prior_top1_residual_batch(
                query, idx, cache, fusion, return_neighbor_ids=True
            )
        )
        old_residual, old_ids = (
            common_utils.compute_class_patch_fused_residual_batch(
                query,
                idx,
                cache,
                fusion,
                topk=1,
                query_chunk_size=2,
                candidate_chunk_size=1,
                return_neighbor_ids=True,
            )
        )
        self.assertTrue(torch.equal(new_ids, old_ids))
        self.assertTrue(torch.allclose(new_residual, old_residual, atol=1e-6))

    def test_chunking_is_consistent(self):
        raw, fusion, cache = _make_bank_and_cache()
        _randomise_theta(fusion)
        cache.refresh()
        _randomise_delta_head(fusion)
        query = torch.randn(2, 6, 3, 4)
        idx = torch.tensor([0, 1])
        base_residual, base_ids = (
            common_utils.compute_class_patch_prior_top1_residual_batch(
                query, idx, cache, fusion, query_chunk_size=6,
                return_neighbor_ids=True,
            )
        )
        for chunk in (1, 2, 3):
            residual, ids = (
                common_utils.compute_class_patch_prior_top1_residual_batch(
                    query, idx, cache, fusion, query_chunk_size=chunk,
                    return_neighbor_ids=True,
                )
            )
            self.assertTrue(torch.equal(base_ids, ids))
            self.assertTrue(torch.allclose(base_residual, residual, atol=1e-6))

    def test_gradients_reach_query_and_reference_sides(self):
        raw, fusion, cache = _make_bank_and_cache()
        _randomise_theta(fusion)
        cache.refresh()
        _randomise_delta_head(fusion, std=0.3)
        query = torch.randn(2, 4, 3, 4)
        idx = torch.tensor([0, 1])
        residual = common_utils.compute_class_patch_prior_top1_residual_batch(
            query, idx, cache, fusion
        )
        residual.square().mean().backward()
        for name, param in [
            ("theta", fusion.theta),
            ("gate.delta_head.weight", fusion.gate.delta_head.weight),
            ("gate.proj.weight", fusion.gate.proj.weight),
            ("class_embedding.weight", fusion.class_embedding.weight),
            ("lambda_raw", fusion.lambda_raw),
        ]:
            self.assertIsNotNone(param.grad, name)
            self.assertGreater(float(param.grad.abs().sum()), 0.0, name)

    def test_eval_no_grad_matches_training_path(self):
        raw, fusion, cache = _make_bank_and_cache()
        _randomise_theta(fusion)
        cache.refresh()
        _randomise_delta_head(fusion)
        query = torch.randn(2, 5, 3, 4)
        idx = torch.tensor([0, 1])
        train_residual = common_utils.compute_class_patch_prior_top1_residual_batch(
            query, idx, cache, fusion
        )
        with torch.no_grad():
            eval_residual = (
                common_utils.compute_class_patch_prior_top1_residual_batch(
                    query, idx, cache, fusion
                )
            )
        self.assertTrue(torch.allclose(train_residual, eval_residual, atol=1e-7))

    def test_fp16_raw_bank_matches_fp32_matching(self):
        raw16, fusion16, cache16 = _make_bank_and_cache(
            storage_dtype=torch.float16
        )
        _randomise_theta(fusion16)
        cache16.refresh()
        _randomise_delta_head(fusion16)
        query = torch.randn(2, 5, 3, 4)
        idx = torch.tensor([0, 1])
        residual16, ids16 = (
            common_utils.compute_class_patch_prior_top1_residual_batch(
                query, idx, cache16, fusion16, return_neighbor_ids=True
            )
        )
        # Brute force in float32 against the same fp16 storage.
        expected = _bruteforce_prior_ids(
            query, idx, fusion16,
            {k: v.float() for k, v in raw16.items()},
        )
        self.assertTrue(torch.equal(ids16, expected))
        self.assertTrue(torch.isfinite(residual16).all())

    def test_single_layer_duplicate_ids(self):
        raw = {0: torch.randn(1, 1, 4)}
        fusion = _make_fusion(num_classes=1, num_layers=1, feature_dim=4)
        cache = common_utils.FusedReferenceCache(
            raw, fusion, torch.device("cpu"), query_chunk_size=2, ref_chunk_size=1
        )
        cache.refresh()
        query = torch.randn(1, 5, 1, 4)
        idx = torch.tensor([0])
        residual, ids = common_utils.compute_class_patch_prior_top1_residual_batch(
            query, idx, cache, fusion, return_neighbor_ids=True
        )
        self.assertEqual(tuple(residual.shape), (1, 5, 4))
        self.assertTrue(torch.equal(ids, torch.zeros_like(ids)))

    def test_empty_class_raises(self):
        raw = {0: torch.randn(4, 3, 4)}
        fusion = _make_fusion()
        cache = common_utils.FusedReferenceCache(
            raw, fusion, torch.device("cpu")
        )
        cache.refresh()
        with self.assertRaises(KeyError):
            common_utils.compute_class_patch_prior_top1_residual_batch(
                torch.randn(1, 3, 3, 4), torch.tensor([1]), cache, fusion
            )

    def test_timing_and_diagnostics(self):
        raw, fusion, cache = _make_bank_and_cache()
        _randomise_theta(fusion)
        cache.refresh()
        _randomise_delta_head(fusion)
        query = torch.randn(2, 4, 3, 4)
        idx = torch.tensor([0, 1])
        timing = {}
        residual, ids, diagnostics = (
            common_utils.compute_class_patch_prior_top1_residual_batch(
                query,
                idx,
                cache,
                fusion,
                return_neighbor_ids=True,
                return_diagnostics=True,
                timing=timing,
            )
        )
        self.assertEqual(tuple(residual.shape), (2, 4, 4))
        self.assertEqual(tuple(ids.shape), (2, 4))
        self.assertEqual(tuple(diagnostics["alpha"].shape), (2, 4, 3))
        self.assertTrue(torch.equal(diagnostics["nn_ids"], ids))
        self.assertEqual(diagnostics["match_mode"], "prior_top1")
        for key in (
            "coarse_projection_ms",
            "nn_search_ms",
            "patch_gate_ms",
            "ref_gather_ms",
            "residual_fusion_ms",
            "total_ms",
        ):
            self.assertIn(key, timing)
            self.assertGreaterEqual(timing[key], 0.0)


class PriorSnapshotTest(unittest.TestCase):
    def test_stale_theta_uses_refresh_snapshot_on_both_sides(self):
        raw, fusion, cache = _make_bank_and_cache()
        _randomise_theta(fusion, seed=1)
        cache.refresh()
        query = torch.randn(2, 5, 3, 4)
        idx = torch.tensor([0, 1])

        expected_at_refresh = _bruteforce_prior_ids(query, idx, fusion, raw)
        _, ids_refresh = common_utils.compute_class_patch_prior_top1_residual_batch(
            query, idx, cache, fusion, return_neighbor_ids=True
        )
        self.assertTrue(torch.equal(ids_refresh, expected_at_refresh))

        # theta moves without refresh: the cached bank and the snapshot stay
        # at the old prior, and the query must use that same snapshot.
        with torch.no_grad():
            fusion.theta.add_(0.75)
        expected_stale = _bruteforce_prior_ids_from_snapshot(
            query, idx, fusion, raw, cache
        )
        _, ids_stale = common_utils.compute_class_patch_prior_top1_residual_batch(
            query, idx, cache, fusion, return_neighbor_ids=True
        )
        self.assertTrue(torch.equal(ids_stale, expected_stale))
        self.assertTrue(torch.equal(ids_stale, ids_refresh))

        # After an explicit refresh both sides advance to the new prior.
        cache.refresh()
        expected_after = _bruteforce_prior_ids(query, idx, fusion, raw)
        _, ids_after = common_utils.compute_class_patch_prior_top1_residual_batch(
            query, idx, cache, fusion, return_neighbor_ids=True
        )
        self.assertTrue(torch.equal(ids_after, expected_after))


    def test_affine_norm_snapshot_is_used_after_mutation(self):
        raw, fusion, cache = _make_bank_and_cache(
            norm_type="layer_norm", norm_affine=True
        )
        _randomise_theta(fusion)
        with torch.no_grad():
            fusion.norm.weight.normal_()
            fusion.norm.bias.normal_()
        cache.refresh()
        query = torch.randn(2, 5, 3, 4)
        idx = torch.tensor([0, 1])
        snapshot = cache._norm_affine_snapshot
        self.assertIsNotNone(snapshot)

        # Move the live affine params; the cached bank is still at the
        # snapshot, so the query projection must use the snapshot too.
        with torch.no_grad():
            fusion.norm.weight.mul_(-2.0)
            fusion.norm.bias.add_(1.0)

        expected = torch.empty(2, 5, dtype=torch.long)
        for i in range(2):
            cls = int(idx[i])
            alpha = cache.prior_alpha_at_refresh[cls].unsqueeze(0)
            q = torch.nn.functional.layer_norm(
                query[i], snapshot["normalized_shape"],
                snapshot["weight"], snapshot["bias"], snapshot["eps"],
            )
            ref = torch.nn.functional.layer_norm(
                raw[cls].to(torch.float32), snapshot["normalized_shape"],
                snapshot["weight"], snapshot["bias"], snapshot["eps"],
            )
            q_vec = torch.einsum("plc,l->pc", q, alpha[0])
            r_vec = torch.einsum("nlc,l->nc", ref, alpha[0])
            expected[i] = torch.cdist(q_vec, r_vec).argmin(dim=1)

        _, ids = common_utils.compute_class_patch_prior_top1_residual_batch(
            query, idx, cache, fusion, return_neighbor_ids=True
        )
        self.assertTrue(torch.equal(ids, expected))


def _training_args(**overrides):
    """Minimal Namespace for the training-script schema helpers."""
    import argparse

    values = dict(
        fusion_mode="class_patch",
        fusion_temperature=1.0,
        fusion_norm="none",
        fusion_norm_affine=False,
        feature_model="dinov3_vitb16",
        use_cls_token=False,
        dataset="mvtec",
        image_size=512,
        crop_size=448,
        reference_refresh_unit="epoch",
        reference_refresh_interval=1,
        reference_storage_dtype="float16",
        num_reference_images_per_class=8,
        strict_clean_reference=False,
        seed=0,
        feature_adapter_mode="legacy",
        fusion_match_mode="prior_top1",
        fusion_query_chunk=256,
        fusion_topk=32,
        fusion_candidate_chunk=8,
        gate_embedding_dim=32,
        gate_hidden_dim=64,
        gate_norm="layer_norm",
        gate_lambda_init=0.1,
        gate_use_context=False,
        gate_demean_delta=True,
        adapter_hidden_dim=128,
        adapter_beta_init=0.15,
        fusion_no_layer_fusion_class_names=[],
        fusion_last_layer_prior_logit=8.0,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


class CheckpointSchemaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import self_train_ad_multiclass_dinov3 as train_script

        cls.train = train_script
        cls.class_names = ["a"]
        cls.candidate_layers = [8]

    def _build(self, **overrides):
        args = _training_args(**overrides)
        return self.train._build_fusion_config(
            args, self.class_names, self.candidate_layers, feature_dim=64
        )

    def test_prior_top1_schema3_does_not_persist_topk(self):
        cfg = self._build(fusion_match_mode="prior_top1")
        self.assertEqual(cfg["schema_version"], 3)
        self.assertEqual(cfg["match_mode"], "prior_top1")
        self.assertNotIn("fusion_topk", cfg)
        self.assertNotIn("fusion_candidate_chunk", cfg)

    def test_dynamic_topk_schema3_persists_topk(self):
        cfg = self._build(fusion_match_mode="dynamic_topk", fusion_topk=16)
        self.assertEqual(cfg["schema_version"], 3)
        self.assertEqual(cfg["match_mode"], "dynamic_topk")
        self.assertEqual(cfg["fusion_topk"], 16)

    def _legacy_schema2_cfg(self):
        cfg = self._build(fusion_match_mode="dynamic_topk")
        cfg = dict(cfg)
        cfg["schema_version"] = 2
        cfg.pop("match_mode")
        return cfg

    def test_schema2_is_rejected_as_prior_top1_resume(self):
        args = _training_args(fusion_match_mode="prior_top1")
        with self.assertRaisesRegex(RuntimeError, "dynamic_topk"):
            self.train._validate_resume_fusion_config(
                self._legacy_schema2_cfg(),
                args,
                self.class_names,
                self.candidate_layers,
            )

    def test_schema2_resumes_as_dynamic_topk(self):
        args = _training_args(fusion_match_mode="dynamic_topk")
        self.train._validate_resume_fusion_config(
            self._legacy_schema2_cfg(),
            args,
            self.class_names,
            self.candidate_layers,
        )

    def test_schema3_mode_mismatch_is_rejected(self):
        cfg = self._build(fusion_match_mode="prior_top1")
        args = _training_args(fusion_match_mode="dynamic_topk")
        with self.assertRaisesRegex(RuntimeError, "match_mode"):
            self.train._validate_resume_fusion_config(
                cfg, args, self.class_names, self.candidate_layers
            )

    def test_weights_only_init_can_switch_match_mode(self):
        args = _training_args(fusion_match_mode="prior_top1")
        self.train._validate_resume_fusion_config(
            self._legacy_schema2_cfg(),
            args,
            self.class_names,
            self.candidate_layers,
            ignore_match_mode=True,
        )

    def test_no_layer_fusion_classes_roundtrip(self):
        cfg = self._build(
            fusion_no_layer_fusion_class_names=["a"],
            fusion_last_layer_prior_logit=6.0,
        )
        self.assertEqual(cfg["no_layer_fusion_classes"], ["a"])
        self.assertEqual(cfg["last_layer_prior_logit"], 6.0)
        args = _training_args(
            fusion_no_layer_fusion_class_names=["a"],
            fusion_last_layer_prior_logit=6.0,
        )
        self.train._validate_resume_fusion_config(
            cfg, args, self.class_names, self.candidate_layers
        )

    def test_no_layer_fusion_classes_mismatch_is_rejected(self):
        cfg = self._build(
            fusion_no_layer_fusion_class_names=["a"],
            fusion_last_layer_prior_logit=8.0,
        )
        args = _training_args(fusion_no_layer_fusion_class_names=[])
        with self.assertRaisesRegex(
            RuntimeError, "no_layer_fusion_classes"
        ):
            self.train._validate_resume_fusion_config(
                cfg, args, self.class_names, self.candidate_layers
            )

    def test_missing_no_layer_fields_resume_as_unpinned(self):
        cfg = dict(self._build(fusion_match_mode="prior_top1"))
        cfg.pop("no_layer_fusion_classes")
        cfg.pop("last_layer_prior_logit")
        args = _training_args(fusion_no_layer_fusion_class_names=[])
        self.train._validate_resume_fusion_config(
            cfg, args, self.class_names, self.candidate_layers
        )


class InferenceMatchModeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.inference.infer_multiclass_residual import (
            resolve_fusion_match_mode,
        )

        cls.resolve = staticmethod(resolve_fusion_match_mode)

    def test_schema2_is_dynamic_topk(self):
        self.assertEqual(
            self.resolve(
                {"fusion_mode": "class_patch", "schema_version": 2,
                 "fusion_topk": 32}
            ),
            "dynamic_topk",
        )

    def test_schema3_reads_explicit_mode(self):
        self.assertEqual(
            self.resolve(
                {"fusion_mode": "class_patch", "schema_version": 3,
                 "match_mode": "prior_top1"}
            ),
            "prior_top1",
        )
        self.assertEqual(
            self.resolve(
                {"fusion_mode": "class_patch", "schema_version": 3,
                 "match_mode": "dynamic_topk", "fusion_topk": 8}
            ),
            "dynamic_topk",
        )

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(RuntimeError):
            self.resolve(
                {"fusion_mode": "class_patch", "schema_version": 3,
                 "match_mode": "bogus"}
            )


class WeightsOnlyCheckpointTest(unittest.TestCase):
    def _save(self, directory, net, fusion):
        path = os.path.join(directory, "checkpoint.pt")
        torch.save(
            {
                "net": net.state_dict(),
                "fusion_state": fusion.state_dict() if fusion is not None else None,
                "epoch": 7,
                "iteration": 123,
            },
            path,
        )
        return path

    def test_loads_net_and_fusion_without_training_state(self):
        torch.manual_seed(3)
        net = torch.nn.Linear(4, 2)
        fusion = _make_fusion()
        with tempfile.TemporaryDirectory() as tmp:
            path = self._save(tmp, net, fusion)
            net2 = torch.nn.Linear(4, 2)
            fusion2 = _make_fusion()
            info = common_utils.load_weights_only_checkpoint(
                path, net2, fusion=fusion2
            )
        for key, value in net.state_dict().items():
            self.assertTrue(torch.equal(value, net2.state_dict()[key]))
        for key, value in fusion.state_dict().items():
            self.assertTrue(torch.equal(value, fusion2.state_dict()[key]))
        self.assertEqual(info["epoch"], 7)
        self.assertEqual(info["iteration"], 123)

    def test_fusion_presence_must_match(self):
        net = torch.nn.Linear(4, 2)
        fusion = _make_fusion()
        with tempfile.TemporaryDirectory() as tmp:
            path = self._save(tmp, net, fusion)
            with self.assertRaises(RuntimeError):
                common_utils.load_weights_only_checkpoint(path, net)
            path_plain = self._save(tmp, net, None)
            with self.assertRaises(RuntimeError):
                common_utils.load_weights_only_checkpoint(
                    path_plain, net, fusion=fusion
                )


def _bruteforce_prior_ids_from_snapshot(query, idx, fusion, raw, cache):
    best = torch.empty(query.shape[0], query.shape[1], dtype=torch.long)
    for i in range(query.shape[0]):
        cls = int(idx[i])
        alpha = cache.prior_alpha_at_refresh[cls].unsqueeze(0)
        q = fusion.fuse_query(query[i], alpha)
        ref = fusion.fuse_query(raw[cls].to(torch.float32), alpha)
        best[i] = torch.cdist(q, ref).argmin(dim=1)
    return best


if __name__ == "__main__":
    unittest.main()
