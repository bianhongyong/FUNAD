"""CPU tests; no dataset, torchvision, FAISS, DINO weights or CUDA required."""

import io
import sys
import types
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.model.simple_residual import ClassReferenceResidual, ResidualMLP
from src.train.simple_residual_training import residual_training_batch, cras_bce_loss
from self_train_ad_multiclass_residual_simple import split_references
import self_train_ad_multiclass_residual_simple as trainer


def test_chunked_lookup_is_class_local_and_does_not_mutate_query():
    references = {0: torch.tensor([[0., 0.], [5., 5.], [2., 3.]]),
                  1: torch.tensor([[10., 10.], [20., 20.]])}
    query = torch.tensor([[[10., 10.], [2., 2.]], [[0., 0.], [19., 20.]]])
    original = query.clone()
    classes = torch.tensor([0, 1])
    bank = ClassReferenceResidual(references, query_chunk=1, reference_chunk=1)
    actual = bank(query, classes)
    for row, cls in enumerate(classes.tolist()):
        ids = torch.cdist(query[row], references[cls]).argmin(dim=1)
        expected = query[row] - references[cls][ids]
        torch.testing.assert_close(actual[row], expected)
    assert torch.equal(query, original)
    with pytest.raises(KeyError):
        bank(query[:1], torch.tensor([2]))


def test_references_are_excluded_without_emptying_small_classes():
    dataset = SimpleNamespace(class_names=["a", "b"],
                              samples=[("a0", 0), ("a1", 0), ("b0", 1), ("b1", 1), ("b2", 1)])
    refs, queries = split_references(dataset, count=8, seed=3)
    reserved = {i for ids in refs.values() for i in ids}
    assert reserved.isdisjoint(queries)
    assert reserved | set(queries) == set(range(5))
    assert {dataset.samples[i][1] for i in queries} == {0, 1}
    assert split_references(dataset, 8, 3) == (refs, queries)
    dataset.samples = [("a0", 0), ("b0", 1), ("b1", 1)]
    with pytest.raises(ValueError, match="at least 2"):
        split_references(dataset, 8, 3)


def test_synthesis_head_update_and_checkpoint_predictions():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    bank = ClassReferenceResidual({0: torch.randn(5, 8), 1: torch.randn(7, 8)}, 3, 2)
    residual = bank(torch.randn(2, 9, 8), torch.tensor([0, 1]))
    features, labels, stats = residual_training_batch(residual, noise_std=0.1)
    assert torch.equal(features[:2], residual)
    assert labels[:2].eq(0).all() and labels[2:].eq(1).all()
    assert (features[2:] - residual).abs().sum() > 0
    detector = ResidualMLP(8)
    optimizer = torch.optim.RMSprop(detector.parameters(), lr=2e-5, momentum=0.2)
    before = detector.discriminator[0].weight.detach().clone()
    loss, _, _ = cras_bce_loss(detector.logits(features))
    loss.backward()
    optimizer.step()
    assert not torch.equal(before, detector.discriminator[0].weight)
    assert not list(bank.parameters())
    assert all(value.grad is None for value in bank.buffers())

    buffer = io.BytesIO()
    torch.save({"bank": bank.state_dict(), "net": detector.state_dict()}, buffer)
    buffer.seek(0)
    restored = torch.load(buffer, weights_only=True)
    clone = ResidualMLP(8)
    clone.load_state_dict(restored["net"], strict=True)
    torch.testing.assert_close(clone(residual)[1], detector(residual)[1])
    with pytest.raises(ValueError):
        residual_training_batch(residual, 0)


def test_dafs_matches_official_code_and_preserves_residual_gradient():
    # Compare to the official flattened [B*P,C] calculation, without zero-norm guards.
    residual = torch.tensor([[[1., 0.], [2., 0.]], [[3., 0.], [4., 0.]]], requires_grad=True)
    torch.manual_seed(23)
    noise = torch.normal(0., 0.015, residual.reshape(-1, 2).shape)
    scale = noise.norm(dim=1) / residual.detach().reshape(-1, 2).norm(dim=1)
    scale = (scale / scale.mean() - 1) * 0.3 + 1
    expected = residual.detach() + (scale.unsqueeze(1) * noise).reshape_as(residual)
    torch.manual_seed(23)
    features, _, stats = residual_training_batch(residual)
    torch.testing.assert_close(features[2:], expected, rtol=1e-6, atol=1e-7)
    assert stats["scale_mean"] == pytest.approx(1.0, abs=1e-6)
    features[2:].sum().backward()
    torch.testing.assert_close(residual.grad, torch.ones_like(residual))


def test_dafs_handles_zero_residuals_and_has_fixed_gaussian_control():
    residual = torch.zeros(2, 4, 8)
    features, _, stats = residual_training_batch(residual)
    assert torch.isfinite(features).all()
    assert stats["scale_min"] >= 0
    assert stats["scale_mean"] == pytest.approx(1.0, abs=1e-6)
    torch.manual_seed(13)
    disabled, _, _ = residual_training_batch(residual, beta=0.0)
    torch.manual_seed(13)
    fixed, _, _ = residual_training_batch(residual, mode="gaussian")
    assert torch.equal(disabled, fixed)
    for invalid in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError):
            residual_training_batch(residual, beta=invalid)


def test_cras_bce_matches_official_objective_and_handles_extreme_logits():
    logits = torch.tensor([[-1., 0., 2.], [0.5, -2., 3.],
                           [1., 0.4, -0.2], [2., -0.8, 0.7]])
    loss, normal, synthetic = cras_bce_loss(logits)
    official = torch.nn.BCELoss()(logits[:2].sigmoid(), torch.zeros_like(logits[:2]))
    official += torch.nn.BCELoss()(logits[2:].sigmoid(), torch.ones_like(logits[2:]))
    torch.testing.assert_close(loss, official)
    torch.testing.assert_close(loss, normal + synthetic)
    extreme = torch.tensor([[1000.], [-1000.]], requires_grad=True)
    total, _, _ = cras_bce_loss(extreme)
    total.backward()
    assert torch.isfinite(total) and torch.isfinite(extreme.grad).all()


def test_training_entry_and_standalone_evaluation_round_trip(monkeypatch, tmp_path):
    """Exercise the real main loop/checkpoint sidecar with a tiny frozen fake backbone."""
    class Backbone(nn.Module):
        def get_intermediate_layers(self, images, n, return_class_token):
            patches = images.flatten(2).transpose(1, 2)
            return [(patches, patches.mean(dim=1))]

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, *_args, **_kwargs):
            self.class_names = ["a", "b"]
            self.samples = [(f"{c}_{i}.png", c) for c in range(2) for i in range(3)]

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return torch.full((3, 2, 2), idx / 10.), self.samples[idx][1], idx, torch.zeros(4)

    package = types.ModuleType("dataset")
    package.__path__ = []
    feature = types.ModuleType("dataset.feature_extract")
    feature.build_dinov3_feature_extractor = lambda *_: Backbone()
    feature.resolve_dino_block_indices = lambda *_: [0]
    data = types.ModuleType("dataset.multiclass_feature_dataset")
    data.MultiClassFeatureDataset = Dataset
    data.get_all_class_names = lambda _: ["a", "b"]
    extract = types.ModuleType("dataset.dataset_extract")
    extract.MyDataset = Dataset
    evaluation = types.ModuleType("utils.evaluate")
    predictions = []

    def evaluate(**kwargs):
        images = torch.full((1, 3, 2, 2), 0.15)
        features = kwargs["extract_feature_batch_fn"](images)
        residual = kwargs["compute_residual_feature_batch_fn"](
            features, torch.tensor([kwargs["class_idx_eval"]]))
        with torch.no_grad():
            scores = kwargs["localnet"](residual)[1]
        predictions.append(scores.clone())
        return {"image_auroc": 0.5, "pixel_ap": float(scores.mean()), "pixel_aupro": None}

    evaluation.evaluate_residual_multiclass_epoch = evaluate
    logging = types.ModuleType("utils.logging")
    logging.enable_print_logging = lambda _: None
    for name, module in {"dataset": package, "dataset.feature_extract": feature,
                         "dataset.multiclass_feature_dataset": data,
                         "dataset.dataset_extract": extract, "utils.evaluate": evaluation,
                         "utils.logging": logging}.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    options = ["baseline", "--data_path", str(tmp_path), "--save_path", str(tmp_path / "out"),
               "--epoch", "1", "--batch_size", "2", "--num_workers", "0",
               "--image_size", "2", "--crop_size", "2", "--noise_std", "0.1"]
    monkeypatch.setattr(sys, "argv", options)
    trainer.main()
    saved = tmp_path / "out" / "mvtec" / "seed_0"
    assert (saved / "last.pt").is_file()
    assert (saved / "references.pt").is_file()
    assert (saved / "metrics_epoch_0001.csv").is_file()
    metadata = torch.load(saved / "last.pt", weights_only=False)
    assert metadata["synthesis_config"]["mode"] == "cras"
    assert metadata["synthesis_config"]["beta"] == 0.3
    assert metadata["loss_config"]["normal_weight"] == 1.0
    assert metadata["loss_config"]["synthetic_weight"] == 1.0
    monkeypatch.setattr(sys, "argv", options + ["--checkpoint", str(saved / "last.pt")])
    trainer.main()
    assert (saved / "eval_metrics_epoch_0001.csv").is_file()
    for first, reloaded in zip(predictions[:2], predictions[2:]):
        torch.testing.assert_close(first, reloaded, rtol=0, atol=0)
