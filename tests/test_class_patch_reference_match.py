"""Tests for query-conditioned Top-K reference matching (V2).

Run from the repository root:
    python -m unittest tests.test_class_patch_reference_match -v
"""

import unittest
import warnings

import torch

from src.model.class_patch_fusion_adapter import ClassPatchFusionAdapter
from src.model.layer_fusion import ClassConditionedLayerFusion
import utils.train_utils as common_utils


def _make_fusion(num_classes=2, num_layers=3, feature_dim=4, norm_type="none"):
    return ClassPatchFusionAdapter(
        num_classes=num_classes,
        num_layers=num_layers,
        feature_dim=feature_dim,
        norm_type=norm_type,
        class_embedding_dim=4,
        gate_hidden_dim=8,
    )


def _randomise_delta_head(fusion, std=0.2):
    with torch.no_grad():
        fusion.gate.delta_head.weight.normal_(std=std)
        fusion.gate.delta_head.bias.normal_(std=std)


def _make_bank_and_cache(num_layers=3, feature_dim=4, norm_type="none"):
    torch.manual_seed(0)
    raw = {
        0: torch.randn(7, num_layers, feature_dim),
        1: torch.randn(5, num_layers, feature_dim) + 10.0,
    }
    fusion = _make_fusion(
        num_layers=num_layers, feature_dim=feature_dim, norm_type=norm_type
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


def _bruteforce_dynamic_ids(raw_query, class_idx, fusion, raw, dtype=torch.float32):
    """Exact per-query weighted NN, computed independently of train_utils."""
    alpha = fusion.patch_weights(raw_query, class_idx).detach()
    best = torch.empty(raw_query.shape[0], raw_query.shape[1], dtype=torch.long)
    for i in range(raw_query.shape[0]):
        cls = int(class_idx[i])
        q = fusion.fuse_query(raw_query[i], alpha[i])  # [P,C]
        ref = raw[cls].to(dtype=dtype)
        ref_fused = torch.einsum("pl,rlc->prc", alpha[i], fusion.norm(ref))
        dist = (q.unsqueeze(1) - ref_fused).pow(2).sum(dim=-1)
        best[i] = dist.argmin(dim=1)
    return best


class TopKSearchTest(unittest.TestCase):
    def test_topk1_matches_nearest_ids(self):
        _, fusion, cache = _make_bank_and_cache()
        query = torch.randn(2, 9, 4)
        idx = torch.tensor([0, 1])
        nearest = cache.nearest_ids(query, idx)
        topk = cache.nearest_topk_ids(query, idx, 1)
        self.assertEqual(tuple(topk.shape), (2, 9, 1))
        self.assertTrue(torch.equal(nearest, topk[..., 0]))

    def test_topk_is_sorted_across_ref_chunks(self):
        raw, fusion, cache = _make_bank_and_cache()
        raw_q = torch.randn(6, 3, 4)
        query = raw_q.unsqueeze(0)
        idx = torch.tensor([0])
        alpha = fusion.global_weights(idx)
        fused = fusion.fuse_query(raw_q, alpha)
        coarse = fusion.fuse_query(query, alpha)
        ids = cache.nearest_topk_ids(coarse, idx, 5)[0]
        ref = cache.fused_by_class[0]
        for patch in range(query.shape[1]):
            distances = (ref - fused[patch]).pow(2).sum(dim=-1)
            expected = torch.argsort(distances)[:5]
            self.assertTrue(torch.equal(ids[patch], expected))
            # Sorted ascending by distance.
            selected = distances[ids[patch]]
            self.assertTrue(torch.all(selected[1:] >= selected[:-1]))

    def test_k_equals_full_bank_uses_a_small_ref_chunk(self):
        raw, fusion, cache = _make_bank_and_cache()
        for cls, bank_size in [(0, 7), (1, 5)]:
            raw_q = torch.randn(4, 3, 4)
            query = raw_q.unsqueeze(0)
            idx = torch.tensor([cls])
            prior = fusion.global_weights(idx)
            q = fusion.fuse_query(raw_q, prior)
            full = cache.nearest_topk_ids(
                fusion.fuse_query(query, prior), idx, bank_size
            )
            self.assertEqual(full.shape[-1], bank_size)
            ref = cache.fused_by_class[cls]
            dist = torch.cdist(q, ref)
            expected = torch.argsort(dist, dim=1)
            self.assertTrue(torch.equal(full[0], expected))

    def test_mixed_classes_search_their_own_bank(self):
        raw, fusion, cache = _make_bank_and_cache()
        query = torch.randn(2, 4, 4)
        idx = torch.tensor([0, 1])
        ids = cache.nearest_topk_ids(query, idx, 3)
        for i, cls in enumerate([0, 1]):
            bank_size = raw[cls].shape[0]
            self.assertTrue(int(ids[i].min()) >= 0)
            self.assertTrue(int(ids[i].max()) < bank_size)

    def test_empty_class_errors(self):
        raw = {0: torch.randn(4, 3, 4)}
        fusion = _make_fusion()
        cache = common_utils.FusedReferenceCache(raw, fusion, torch.device("cpu"))
        cache.refresh()
        with self.assertRaises(KeyError):
            cache.nearest_topk_ids(torch.randn(1, 3, 4), torch.tensor([1]), 2)

    def test_k_is_clamped_to_bank_size(self):
        raw, fusion, cache = _make_bank_and_cache()
        query = torch.randn(1, 3, 4)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ids = cache.nearest_topk_ids(query, torch.tensor([0]), 99)
        self.assertEqual(ids.shape[-1], 7)
        self.assertTrue(any("clamped" in str(w.message) for w in caught))


def _prior_only(fusion):
    """Return a V1 fusion with the same theta (helper for prior-only search)."""
    v1 = ClassConditionedLayerFusion(
        num_classes=fusion.num_classes,
        num_layers=fusion.num_layers,
        feature_dim=fusion.feature_dim,
        norm_type=fusion.norm_type,
    )
    with torch.no_grad():
        v1.theta.copy_(fusion.theta)
    return v1


class DynamicResidualTest(unittest.TestCase):
    def test_query_and_selected_reference_share_alpha(self):
        raw, fusion, cache = _make_bank_and_cache()
        _randomise_delta_head(fusion)
        query = torch.randn(2, 5, 3, 4)
        idx = torch.tensor([0, 1])
        residual, best_ids = common_utils.compute_class_patch_fused_residual_batch(
            query,
            idx,
            cache,
            fusion,
            topk=5,
            query_chunk_size=2,
            candidate_chunk_size=2,
            return_neighbor_ids=True,
        )
        alpha = fusion.patch_weights(query, idx).detach()
        for i in range(2):
            cls = int(idx[i])
            q_fused = fusion.fuse_query(query[i], alpha[i]).detach()
            ref = raw[cls][best_ids[i]].to(torch.float32)
            ref_fused = fusion.fuse_query(ref, alpha[i])
            self.assertTrue(
                torch.allclose(residual[i], q_fused - ref_fused, atol=1e-6)
            )

    def test_full_bank_matches_bruteforce(self):
        raw, fusion, cache = _make_bank_and_cache()
        _randomise_delta_head(fusion)
        for cls, bank_size in [(0, 7), (1, 5)]:
            query = torch.randn(1, 4, 3, 4)
            idx = torch.tensor([cls])
            _, best = common_utils.compute_class_patch_fused_residual_batch(
                query,
                idx,
                cache,
                fusion,
                topk=bank_size,
                return_neighbor_ids=True,
            )
            expected = _bruteforce_dynamic_ids(query, idx, fusion, raw)
            self.assertTrue(torch.equal(best, expected))

    def test_layer_norm_fusion_matches_bruteforce(self):
        raw, fusion, cache = _make_bank_and_cache(norm_type="layer_norm")
        _randomise_delta_head(fusion)
        query = torch.randn(1, 4, 3, 4)
        idx = torch.tensor([0])
        _, best = common_utils.compute_class_patch_fused_residual_batch(
            query,
            idx,
            cache,
            fusion,
            topk=7,
            return_neighbor_ids=True,
        )
        expected = _bruteforce_dynamic_ids(query, idx, fusion, raw)
        self.assertTrue(torch.equal(best, expected))

    def test_k1_zero_delta_reproduces_v1_residual(self):
        raw, fusion, cache = _make_bank_and_cache()
        with torch.no_grad():
            fusion.theta.copy_(torch.randn_like(fusion.theta))
        # Coarse ids must come from the same theta as the V1 reference.
        cache.refresh()
        v1 = _prior_only(fusion)
        v1_cache = common_utils.FusedReferenceCache(
            raw, v1, torch.device("cpu"), query_chunk_size=4, ref_chunk_size=3
        )
        v1_cache.refresh()

        query = torch.randn(2, 5, 3, 4)
        idx = torch.tensor([0, 1])
        v2_residual = common_utils.compute_class_patch_fused_residual_batch(
            query,
            idx,
            cache,
            fusion,
            topk=1,
            query_chunk_size=2,
            candidate_chunk_size=1,
        )
        v1_residual = common_utils.compute_fused_residual_batch(
            query, idx, v1_cache, v1
        )
        self.assertTrue(torch.allclose(v2_residual, v1_residual, atol=1e-6))

    def test_gradients_from_both_query_and_reference_sides(self):
        raw, fusion, cache = _make_bank_and_cache()
        _randomise_delta_head(fusion, std=0.3)
        query = torch.randn(2, 4, 3, 4)
        idx = torch.tensor([0, 1])
        residual, best = common_utils.compute_class_patch_fused_residual_batch(
            query,
            idx,
            cache,
            fusion,
            topk=4,
            return_neighbor_ids=True,
        )
        residual.square().mean().backward()

        for name, param in [
            ("theta", fusion.theta),
            ("gate.delta_head.weight", fusion.gate.delta_head.weight),
            ("class_embedding.weight", fusion.class_embedding.weight),
            ("lambda_raw", fusion.lambda_raw),
        ]:
            self.assertIsNotNone(param.grad, name)
            self.assertGreater(float(param.grad.abs().sum()), 0.0, name)
        grad_both_sides = fusion.theta.grad.detach().clone()

        # Wrong variant: detach the selected-reference fusion.  The query side
        # still contributes, but the reference-side alpha gradient is gone.
        fusion.zero_grad()
        alpha = fusion.patch_weights(query, idx)
        q_fused = fusion.fuse_query(query.reshape(-1, 3, 4), alpha.reshape(-1, 3))
        selected = torch.cat(
            [
                raw[0][best[0].reshape(-1)],
                raw[1][best[1].reshape(-1)],
            ],
            dim=0,
        )
        selected_fused = fusion.fuse_query(selected, alpha.reshape(-1, 3)).detach()
        wrong = q_fused - selected_fused
        wrong.square().mean().backward()
        self.assertFalse(
            torch.allclose(grad_both_sides, fusion.theta.grad.detach(), atol=1e-8)
        )

    def test_chunking_is_consistent(self):
        raw, fusion, cache = _make_bank_and_cache()
        _randomise_delta_head(fusion)
        query = torch.randn(2, 6, 3, 4)
        idx = torch.tensor([0, 1])
        base = common_utils.compute_class_patch_fused_residual_batch(
            query,
            idx,
            cache,
            fusion,
            topk=5,
            query_chunk_size=6,
            candidate_chunk_size=5,
            return_neighbor_ids=True,
        )
        for q_chunk, k_chunk in [(1, 1), (2, 2), (3, 3)]:
            other = common_utils.compute_class_patch_fused_residual_batch(
                query,
                idx,
                cache,
                fusion,
                topk=5,
                query_chunk_size=q_chunk,
                candidate_chunk_size=k_chunk,
                return_neighbor_ids=True,
            )
            self.assertTrue(torch.equal(base[1], other[1]))
            self.assertTrue(torch.allclose(base[0], other[0], atol=1e-6))

    def test_topk_recall_is_one_for_full_bank(self):
        raw, fusion, cache = _make_bank_and_cache()
        _randomise_delta_head(fusion)
        for cls, bank_size in [(0, 7), (1, 5)]:
            query = torch.randn(1, 4, 3, 4)
            idx = torch.tensor([cls])
            result = common_utils.topk_recall_vs_exact(
                query, idx, cache, fusion, topk=bank_size, sample_size=8
            )
            self.assertEqual(result["topk"], bank_size)
            self.assertAlmostEqual(result["recall"], 1.0)
            self.assertGreater(result["num_samples"], 0)

    def test_exact_dynamic_nearest_ids_matches_bruteforce(self):
        raw, fusion, cache = _make_bank_and_cache()
        _randomise_delta_head(fusion)
        query = torch.randn(2, 3, 3, 4)
        idx = torch.tensor([0, 1])
        exact = common_utils.exact_dynamic_nearest_ids(query, idx, cache, fusion)
        expected = _bruteforce_dynamic_ids(query, idx, fusion, raw)
        self.assertTrue(torch.equal(exact, expected))


if __name__ == "__main__":
    unittest.main()
