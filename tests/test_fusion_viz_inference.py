"""Opt-in end-to-end test for the class-patch fusion visualisation entry point.

The test needs a real class_patch checkpoint, its raw-bank sidecar and a test
dataset, so it is skipped unless the environment is provided::

    FUNAD_FUSION_VIZ_CHECKPOINT=/path/to/*_fusion_class_patch_*_localnet.pt \
    FUNAD_FUSION_VIZ_DATA=/path/to/MVTec \
    FUNAD_FUSION_VIZ_CLASS=bottle \
    python -m unittest tests.test_fusion_viz_inference -v

Optional:
    FUNAD_FUSION_VIZ_DATASET=mvtec|visa
    FUNAD_FUSION_VIZ_MOE_CLS_TOKEN=True|False   (must match training)
    FUNAD_FUSION_VIZ_IMAGE_SIZE=512
    FUNAD_FUSION_VIZ_CROP_SIZE=448

Run from the repository root.
"""

import glob
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
import warnings

import numpy as np

from src.inference.visualize_class_patch_fusion import select_cases

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

CHECKPOINT = os.environ.get("FUNAD_FUSION_VIZ_CHECKPOINT", "")
DATA_PATH = os.environ.get("FUNAD_FUSION_VIZ_DATA", "")
CLASS_NAME = os.environ.get("FUNAD_FUSION_VIZ_CLASS", "bottle")
DATASET = os.environ.get("FUNAD_FUSION_VIZ_DATASET", "mvtec")
MOE_USE_CLS_TOKEN = os.environ.get("FUNAD_FUSION_VIZ_MOE_CLS_TOKEN", "True")
IMAGE_SIZE = os.environ.get("FUNAD_FUSION_VIZ_IMAGE_SIZE", "512")
CROP_SIZE = os.environ.get("FUNAD_FUSION_VIZ_CROP_SIZE", "448")


class _FakeTestSet:
    """Minimal stand-in for ``MyDataset``: 5 good followed by 4 anomalies."""

    def __init__(self):
        self.x = [f"/data/bottle/test/good/{i:03d}.png" for i in range(5)] + [
            f"/data/bottle/test/broken_large/{i:03d}.png" for i in range(4)
        ]
        self.y = [0] * 5 + [1] * 4


class CaseSelectionTest(unittest.TestCase):
    """Deterministic case selection must never swap labels or pick silently."""

    def setUp(self):
        self.base = dict(
            case_types=["good", "anomaly"],
            cases_per_class=2,
            seed=7,
            case_selection="first",
            case_paths=None,
        )

    def _args(self, **overrides):
        return types.SimpleNamespace(**{**self.base, **overrides})

    def test_first_and_seeded_random_keep_labels(self):
        for mode in ("first", "seeded_random"):
            selected = select_cases(
                _FakeTestSet(), "bottle", self._args(case_selection=mode)
            )
            self.assertEqual(len(selected), 4)
            for meta in selected.values():
                expected = 1 if meta["tag"] == "anomaly" else 0
                self.assertEqual(meta["label"], expected)
        first = select_cases(
            _FakeTestSet(), "bottle", self._args(case_selection="first")
        )
        self.assertEqual(sorted(first), [0, 1, 5, 6])

    def test_seeded_random_is_reproducible(self):
        a = select_cases(
            _FakeTestSet(), "bottle",
            self._args(case_selection="seeded_random"),
        )
        b = select_cases(
            _FakeTestSet(), "bottle",
            self._args(case_selection="seeded_random"),
        )
        self.assertEqual(sorted(a), sorted(b))

    def test_explicit_absolute_path_is_exact(self):
        selected = select_cases(
            _FakeTestSet(), "bottle",
            self._args(
                case_selection="explicit",
                case_paths=["/data/bottle/test/good/003.png"],
            ),
        )
        self.assertEqual(sorted(selected), [3])

    def test_explicit_relative_suffix(self):
        selected = select_cases(
            _FakeTestSet(), "bottle",
            self._args(
                case_selection="explicit",
                case_paths=["bottle/test/good/003.png"],
            ),
        )
        self.assertEqual(sorted(selected), [3])

    def test_explicit_label_mismatch_is_skipped(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            selected = select_cases(
                _FakeTestSet(), "bottle",
                self._args(
                    case_selection="explicit",
                    case_types=["anomaly"],
                    case_paths=["/data/bottle/test/good/003.png"],
                ),
            )
        self.assertEqual(selected, {})
        self.assertGreaterEqual(len(caught), 1)


@unittest.skipUnless(
    CHECKPOINT and DATA_PATH and os.path.isfile(CHECKPOINT),
    "set FUNAD_FUSION_VIZ_CHECKPOINT and FUNAD_FUSION_VIZ_DATA to run",
)
class FusionVizIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.output_dir = cls.tmp.name
        before = os.stat(CHECKPOINT)
        cls.checkpoint_size_before = before.st_size
        cls.checkpoint_mtime_before = before.st_mtime_ns
        cmd = [
            sys.executable,
            "src/inference/visualize_class_patch_fusion.py",
            "--checkpoint_path", CHECKPOINT,
            "--data_path", DATA_PATH,
            "--dataset", DATASET,
            "--output_dir", cls.output_dir,
            "--batch_size", "4",
            "--num_workers", "0",
            "--stat_class_names", CLASS_NAME,
            "--fig2_class_names", CLASS_NAME,
            "--case_class_names", CLASS_NAME,
            "--case_types", "good", "anomaly",
            "--cases_per_class", "1",
            "--max_images_per_class", "2",
            "--moe_use_cls_token", MOE_USE_CLS_TOKEN,
            "--image_size", IMAGE_SIZE,
            "--crop_size", CROP_SIZE,
            "--export_formats", "png",
            "--hash_checkpoint", "False",
        ]
        cls.proc = subprocess.run(
            cmd, cwd=REPO_ROOT, capture_output=True, text=True
        )
        if cls.proc.returncode != 0:
            raise AssertionError(
                "visualisation command failed:\n"
                f"stdout:\n{cls.proc.stdout}\n"
                f"stderr:\n{cls.proc.stderr}"
            )

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_checkpoint_not_modified(self):
        after = os.stat(CHECKPOINT)
        self.assertEqual(after.st_size, self.checkpoint_size_before)
        self.assertEqual(after.st_mtime_ns, self.checkpoint_mtime_before)

    def test_metadata_and_tables(self):
        metadata_path = os.path.join(self.output_dir, "metadata.json")
        self.assertTrue(os.path.isfile(metadata_path))
        with open(metadata_path, encoding="utf-8") as handle:
            metadata = json.load(handle)

        self.assertEqual(metadata["schema"]["fusion_mode"], "class_patch")
        self.assertIn(metadata["schema"]["match_mode"], ("prior_top1", "dynamic_topk"))
        self.assertEqual(metadata["statistics"]["class_names"], [CLASS_NAME])
        self.assertGreaterEqual(metadata["cases"]["rendered"], 1)

        prior_csv = os.path.join(self.output_dir, "tables", "global_prior.csv")
        self.assertTrue(os.path.isfile(prior_csv))
        with open(prior_csv, encoding="utf-8") as handle:
            header = handle.readline().strip().split(",")
        block_cols = [c for c in header if c.startswith("block_")]
        self.assertTrue(block_cols)

        import pandas as pd

        prior = pd.read_csv(prior_csv)
        np.testing.assert_allclose(
            prior[block_cols].sum(axis=1).to_numpy(), 1.0, atol=1e-5
        )
        self.assertTrue(((prior[block_cols] >= 0) & (prior[block_cols] <= 1)).all().all())

        stats_csv = os.path.join(self.output_dir, "tables", "patch_statistics.csv")
        self.assertTrue(os.path.isfile(stats_csv))
        stats = pd.read_csv(stats_csv)
        self.assertEqual(stats["class"].tolist(), [CLASS_NAME])

    def test_case_arrays_are_valid(self):
        npz_paths = sorted(
            glob.glob(os.path.join(self.output_dir, "cases", "*", "*.npz"))
        )
        self.assertGreaterEqual(len(npz_paths), 1)
        for path in npz_paths:
            with np.load(path, allow_pickle=False) as payload:
                alpha = payload["alpha_patch"]
                prior = payload["alpha_global"]
                entropy = payload["entropy"]
                l1 = payload["l1"]
                gt = payload["gt_mask"]
                score = payload["score_map"]

                self.assertEqual(alpha.ndim, 2)
                self.assertEqual(alpha.shape[1], prior.shape[0])
                np.testing.assert_allclose(alpha.sum(axis=1), 1.0, atol=1e-5)
                self.assertGreaterEqual(float(alpha.min()), 0.0)
                self.assertLessEqual(float(alpha.max()), 1.0)

                max_entropy = float(np.log(alpha.shape[1]))
                self.assertGreaterEqual(float(entropy.min()), -1e-6)
                self.assertLessEqual(float(entropy.max()), max_entropy + 1e-5)
                self.assertGreaterEqual(float(l1.min()), -1e-6)
                self.assertLessEqual(float(l1.max()), 2.0 + 1e-5)

                self.assertEqual(gt.squeeze().shape, score.shape)
                self.assertTrue(np.isfinite(score).all())
                self.assertTrue(os.path.isfile(str(payload["src_path"])))

    def test_figures_exist(self):
        figures = glob.glob(os.path.join(self.output_dir, "figures", "*.png"))
        names = {os.path.basename(p) for p in figures}
        self.assertTrue(any(n.startswith("fig1_global_prior_heatmap") for n in names))
        self.assertTrue(any(n.startswith("fig2_patch_mean_std") for n in names))
        self.assertTrue(any(n.startswith("fig5_l1_to_prior_by_class") for n in names))
        self.assertTrue(any(n.startswith("fig3_dominant_layer_") for n in names))
        self.assertTrue(any(n.startswith("fig4_patch_entropy_") for n in names))
        self.assertTrue(any(n.startswith("fig6_fusion_detection_") for n in names))


if __name__ == "__main__":
    unittest.main()
