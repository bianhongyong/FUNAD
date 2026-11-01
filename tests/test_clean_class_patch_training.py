"""Integration tests for the chapter-1 clean class-patch training step.

Run from the repository root:
    python -m unittest tests.test_clean_class_patch_training -v

These tests do not touch a dataset or DINOv3.  They build a synthetic raw
reference bank, run the exact ``prior_top1`` residual used by the trainer,
synthesize anomalies, and check that Balanced BCE backpropagates into both the
fusion module and the legacy discriminator.
"""

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from self_train_ad_multiclass_dinov3_clean import reconstruct_reference_image_ids
from src.model import model
from src.model.class_patch_fusion_adapter import ClassPatchFusionAdapter
from src.train.feature_anomaly_synthesis import (
    FeatureAnomalySynthesizer,
    build_synthetic_training_batch,
)
import utils.train_utils as common_utils
from utils import evaluate as eval_utils
from utils.loss import compute_balanced_bce_components, compute_balanced_bce_loss


def _build_pipeline(seed=0):
    torch.manual_seed(seed)
    num_classes, num_layers, feature_dim = 2, 3, 8
    raw = {
        0: torch.randn(6, num_layers, feature_dim),
        1: torch.randn(6, num_layers, feature_dim) + 5.0,
    }
    fusion = ClassPatchFusionAdapter(
        num_classes=num_classes,
        num_layers=num_layers,
        feature_dim=feature_dim,
        class_embedding_dim=8,
        gate_hidden_dim=16,
        gate_lambda_init=0.2,
    )
    with torch.no_grad():
        fusion.theta.normal_(std=0.5)
        fusion.gate.delta_head.weight.normal_(std=0.1)
        fusion.gate.delta_head.bias.normal_(std=0.1)
    cache = common_utils.FusedReferenceCache(
        raw,
        fusion,
        torch.device("cpu"),
        expected_num_layers=num_layers,
        expected_feature_dim=feature_dim,
        query_chunk_size=8,
        ref_chunk_size=4,
    )
    cache.refresh()
    return raw, fusion, cache


class CleanClassPatchStepTest(unittest.TestCase):
    def test_prior_top1_synthesis_bce_backpropagates(self):
        raw, fusion, cache = _build_pipeline()
        query = torch.randn(2, 16, 3, 8)
        class_idx = torch.tensor([0, 1])

        residual, ids = common_utils.compute_class_patch_prior_top1_residual_batch(
            query, class_idx, cache, fusion, return_neighbor_ids=True
        )
        self.assertEqual(tuple(residual.shape), (2, 16, 8))
        for row, cls in enumerate(class_idx.tolist()):
            self.assertGreaterEqual(int(ids[row].min()), 0)
            self.assertLess(int(ids[row].max()), raw[cls].shape[0])

        synthesizer = FeatureAnomalySynthesizer(
            class_sigmas={0: 1.0, 1: 1.0},
            num_classes=2,
            mask_type="blob",
            area_ratio_range=(0.05, 0.05),
            gamma_range=(0.5, 1.5),
        )
        features, labels, classes, mask, stats = build_synthetic_training_batch(
            residual, class_idx, synthesizer
        )
        self.assertEqual(tuple(features.shape), (4, 16, 8))
        self.assertTrue(torch.equal(labels[2:], mask.to(torch.float32)))

        localnet = model.localnet(
            len_feature=8,
            use_moe_discriminator=False,
            num_classes=2,
            class_conditioned_adapter=True,
        )
        criterion = torch.nn.BCELoss()
        _, pred = localnet(features, class_idx=classes)
        pos_loss, neg_loss = compute_balanced_bce_components(
            criterion, pred, labels
        )
        loss = pos_loss + neg_loss
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

        self.assertIsNotNone(fusion.theta.grad)
        self.assertGreater(float(fusion.theta.grad.norm()), 0.0)
        self.assertTrue(torch.isfinite(fusion.theta.grad).all())

        discr_grad = localnet.discriminator[0].weight.grad
        self.assertIsNotNone(discr_grad)
        self.assertEqual(int(torch.isfinite(discr_grad).all()), 1)
        adapter_grads = [
            p.grad for p in localnet.adaptor.parameters() if p.grad is not None
        ]
        self.assertGreater(len(adapter_grads), 0)

    def test_loss_equals_sum_of_components(self):
        criterion = torch.nn.BCELoss()
        pred = torch.rand(3, 4)
        target = (torch.rand(3, 4) > 0.5).float()
        pos, neg = compute_balanced_bce_components(criterion, pred, target)
        total = compute_balanced_bce_loss(criterion, pred, target)
        self.assertTrue(torch.allclose(total, pos + neg))

    def test_mask_outside_residual_is_unchanged(self):
        _, fusion, cache = _build_pipeline(seed=3)
        query = torch.randn(2, 16, 3, 8)
        class_idx = torch.tensor([0, 1])
        residual = common_utils.compute_class_patch_prior_top1_residual_batch(
            query, class_idx, cache, fusion
        )
        synthesizer = FeatureAnomalySynthesizer(
            class_sigmas={0: 1.0, 1: 1.0},
            num_classes=2,
            mask_type="irregular",
            area_ratio_range=(0.05, 0.10),
            gamma_range=(1.0, 1.0),
        )
        features, _, _, mask, _ = build_synthetic_training_batch(
            residual, class_idx, synthesizer
        )
        synthetic = features[2:]
        self.assertTrue(torch.equal(synthetic[~mask], residual.detach()[~mask]))
        self.assertFalse(torch.equal(synthetic[mask], residual.detach()[mask]))


class MoEHardClassTrainingTest(unittest.TestCase):
    def test_hard_class_routing_forward_and_grad(self):
        if not torch.cuda.is_available():
            self.skipTest("FMoE uses a CUDA extension and needs a CUDA device.")
        device = torch.device("cuda")
        localnet = model.localnet(
            len_feature=8,
            use_moe_discriminator=True,
            moe_num_expert=2,
            moe_top_k=1,
            moe_hard_class_gate=True,
            num_classes=2,
            class_conditioned_adapter=True,
        ).to(device)
        features = torch.randn(
            4, 16, 8, requires_grad=True, device=device
        )
        classes = torch.tensor([0, 1, 0, 1], device=device)
        patch_class_idx = (
            classes.reshape(-1, 1, 1)
            .expand(-1, features.shape[1], 1)
            .contiguous()
        )
        _, pred = localnet(
            features,
            patch_class_idx=patch_class_idx,
            class_idx=classes,
        )
        self.assertEqual(tuple(pred.shape), (4, 16))
        self.assertIsNone(
            localnet.discriminator.get_loss(
                clear=True, reduction="mean", default=None
            )
        )
        pred.mean().backward()
        self.assertIsNotNone(features.grad)
        self.assertTrue(torch.isfinite(features.grad).all())


class ReferenceImageIdReconstructionTest(unittest.TestCase):
    def test_rebuilds_patch_to_image_mapping(self):
        raw = {
            0: np.zeros((8, 2, 3), dtype=np.float32),
            1: np.zeros((4, 2, 3), dtype=np.float32),
        }
        refs = {0: [0, 1], 1: [2]}
        image_ids = reconstruct_reference_image_ids(raw, refs)
        self.assertEqual(image_ids[0].tolist(), [0, 0, 0, 0, 1, 1, 1, 1])
        self.assertEqual(image_ids[1].tolist(), [2, 2, 2, 2])


class _FakeEvalLocalnet(torch.nn.Module):
    def forward(self, x, patch_class_idx=None, class_idx=None):
        return x, torch.full(
            (x.shape[0], x.shape[1]), 0.5, device=x.device
        )


class _FakeEvalExtractor(torch.nn.Module):
    pass


class _FakeEvalLoader:
    def __iter__(self):
        images = torch.zeros(2, 3, 8, 8)
        labels = torch.tensor([0, 1])
        masks = torch.zeros(2, 1, 8, 8)
        yield images, labels, masks


class EvaluateResidualReshapeTest(unittest.TestCase):
    def test_dynamic_patch_grid_reshape(self):
        """Regression: reshape(-1, side, -1) raised "one unknown dimension"."""
        metrics = eval_utils.evaluate_residual_multiclass_epoch(
            localnet=_FakeEvalLocalnet(),
            feature_extractor=_FakeEvalExtractor(),
            test_loader=_FakeEvalLoader(),
            args=SimpleNamespace(
                img_score_topk_ratio=0.01, use_moe_discriminator=False
            ),
            class_idx_eval=0,
            reference_memory_by_class=None,
            reference_index_by_class=None,
            device=torch.device("cpu"),
            extract_feature_batch_fn=lambda images, fe, a, class_indices=None: (
                torch.zeros(images.shape[0], 784, 4)
            ),
            compute_residual_feature_batch_fn=(
                lambda features, cls, ref_mem, ref_idx: features
            ),
            return_dict=True,
        )
        for key in (
            "image_auroc",
            "pixel_auroc",
            "image_ap",
            "pixel_ap",
            "pixel_f1_max",
        ):
            self.assertIn(key, metrics)
        self.assertEqual(metrics["image_auroc"], 0.5)


class ExtendedMetricTest(unittest.TestCase):
    def test_returns_pixel_ap_and_f1_columns(self):
        label_gt = [0, 1]
        img_map = [0.1, 0.9]
        mask_gt = torch.zeros(2, 1, 4, 4)
        mask_gt[1, 0, 1:3, 1:3] = 1.0
        seg_map = torch.zeros(2, 1, 4, 4)
        seg_map[1, 0, 1:3, 1:3] = 0.9
        seg_map[0] = 0.05
        metrics = eval_utils._finalize_metrics_extended(
            label_gt,
            img_map,
            mask_gt.numpy(),
            seg_map.numpy(),
            compute_pro_metric=True,
        )
        for key in (
            "image_auroc",
            "pixel_auroc",
            "image_ap",
            "pixel_ap",
            "image_f1_max",
            "pixel_f1_max",
            "pixel_aupro",
        ):
            self.assertIn(key, metrics)
        self.assertAlmostEqual(metrics["image_auroc"], 1.0)
        self.assertGreater(metrics["pixel_ap"], 0.9)
        self.assertGreater(metrics["pixel_f1_max"], 0.9)
        self.assertIsNotNone(metrics["pixel_aupro"])


if __name__ == "__main__":
    unittest.main()
