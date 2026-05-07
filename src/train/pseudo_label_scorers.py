"""Pseudo-label patch scoring: k-NN (FAISS) vs. multivariate Gaussian (Mahalanobis)."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn


def postprocess_faiss_distances(distance: np.ndarray, k_number: int) -> np.ndarray:
    """
    distance: shape (N, k). Distances below 1e-4 are treated as 0 (self / duplicate).
    Returns shape (N,): per row, mean of neighbors whose distance is > 0.
    If all k neighbors are 0, returns 0 for that row.
    """
    _ = k_number
    distance = np.asarray(distance, dtype=np.float32)
    distance[distance < 1e-4] = 0
    mask = distance > 0
    sums = (distance * mask).sum(axis=1)
    counts = mask.sum(axis=1)
    return np.where(counts > 0, sums / counts.astype(np.float32), 0.0).astype(np.float32)


class MultiVariateGaussian(nn.Module):
    """SoftPatch-style multivariate Gaussian per patch index; use patch=1 for a single global Gaussian."""

    def __init__(self, n_features: int, n_patches: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(n_features, n_patches))
        self.register_buffer("inv_covariance", torch.eye(n_features).unsqueeze(0).repeat(n_patches, 1, 1))
        self.mean: Tensor
        self.inv_covariance: Tensor

    @staticmethod
    def _cov(
        observations: Tensor,
        rowvar: bool = False,
        bias: bool = False,
        ddof: Optional[int] = None,
        aweights: Tensor = None,
    ) -> Tensor:
        if observations.dim() == 1:
            observations = observations.view(-1, 1)
        if rowvar and observations.shape[0] != 1:
            observations = observations.t()
        if ddof is None:
            ddof = 0 if bias else 1
        weights = aweights
        weights_sum: Any
        if weights is not None:
            if not torch.is_tensor(weights):
                weights = torch.tensor(weights, dtype=torch.float)
            weights_sum = torch.sum(weights)
            avg = torch.sum(observations * (weights / weights_sum)[:, None], 0)
        else:
            avg = torch.mean(observations, 0)
        if weights is None:
            fact = observations.shape[0] - ddof
        elif ddof == 0:
            fact = weights_sum
        elif aweights is None:
            fact = weights_sum - ddof
        else:
            fact = weights_sum - ddof * torch.sum(weights * weights) / weights_sum
        observations_m = observations.sub(avg.expand_as(observations))
        if weights is None:
            x_transposed = observations_m.t()
        else:
            x_transposed = torch.mm(torch.diag(weights), observations_m).t()
        covariance = torch.mm(x_transposed, observations_m)
        covariance = covariance / fact
        return covariance.squeeze()

    def forward(self, embedding: Tensor) -> List[Tensor]:
        device = embedding.device
        patch, _, channel = embedding.shape
        embedding_vectors = embedding.permute(1, 2, 0)
        self.mean = torch.mean(embedding_vectors, dim=0)
        covariance = torch.zeros(size=(channel, channel, patch), device=device)
        identity = torch.eye(channel).to(device)
        for i in range(patch):
            covariance[:, :, i] = self._cov(embedding_vectors[:, :, i], rowvar=False) + 0.01 * identity
        self.inv_covariance = torch.linalg.inv(covariance.permute(2, 0, 1))
        return [self.mean, self.inv_covariance]

    def fit(self, embedding: Tensor) -> List[Tensor]:
        return self.forward(embedding)


def mahalanobis_distance(
    embedding: torch.Tensor,
    mean: torch.Tensor,
    inv_covariance: torch.Tensor,
) -> torch.Tensor:
    embedding = embedding.permute(1, 2, 0)
    delta = (embedding - mean).permute(2, 0, 1)
    distances_sq = (torch.matmul(delta, inv_covariance) * delta).sum(2)
    return torch.sqrt(distances_sq)


def fit_global_gaussian(representative: torch.Tensor, ridge: float = 0.01):
    """representative: [N, D]. Unbiased cov when N>1; ridge on diagonal."""
    n, d = representative.shape
    mu = representative.mean(dim=0)
    if n <= 1:
        inv_cov = torch.eye(d, device=representative.device, dtype=representative.dtype) / ridge
        return mu, inv_cov
    centered = representative - mu
    denom = max(n - 1, 1)
    cov = centered.t().mm(centered) / denom
    cov = cov + ridge * torch.eye(d, device=cov.device, dtype=cov.dtype)
    inv_cov = torch.linalg.inv(cov)
    return mu, inv_cov


def mahalanobis_global(queries: torch.Tensor, mu: torch.Tensor, inv_cov: torch.Tensor) -> torch.Tensor:
    """queries: [M, D], mu: [D], inv_cov: [D, D] -> [M]"""
    delta = queries - mu
    m = (delta @ inv_cov) * delta
    return torch.sqrt(m.sum(dim=-1))


class PseudoLabelScorer(ABC):
    @abstractmethod
    def fit_class(self, cls: int, features: np.ndarray) -> None:
        """features: float32 [N, D]."""

    @abstractmethod
    def has_class(self, cls: int) -> bool:
        ...

    @abstractmethod
    def score_patches(self, cls: int, queries: np.ndarray) -> np.ndarray:
        """queries: [M, D] C-contiguous float -> [M] float32, larger = farther from normal bank."""


class NNPseudoLabelScorer(PseudoLabelScorer):
    def __init__(self, build_faiss_index_fn: Callable[[np.ndarray], Any], k_number: int):
        self._build_faiss_index_fn = build_faiss_index_fn
        self._k_number = int(k_number)
        self._index_by_class: Dict[int, Any] = {}

    def fit_class(self, cls: int, features: np.ndarray) -> None:
        feats = np.asarray(features, dtype=np.float32)
        if feats.size == 0:
            return
        self._index_by_class[int(cls)] = self._build_faiss_index_fn(feats)

    def has_class(self, cls: int) -> bool:
        return int(cls) in self._index_by_class

    def score_patches(self, cls: int, queries: np.ndarray) -> np.ndarray:
        idx = self._index_by_class[int(cls)]
        q = np.ascontiguousarray(np.asarray(queries, dtype=np.float32))
        distance, _ = idx.search(q, k=self._k_number)
        return postprocess_faiss_distances(distance, self._k_number)


class MahalanobisPseudoLabelScorer(PseudoLabelScorer):
    """Per-class single Gaussian on memory bank; Mahalanobis distance per patch."""

    def __init__(
        self,
        device: torch.device,
        ridge: float = 0.01,
        fallback_ridge: float = 0.1,
        gaussian_feature_dim: int = 128,
    ):
        self._device = device
        self._ridge = float(ridge)
        self._fallback_ridge = float(fallback_ridge)
        self._gaussian_dim = int(gaussian_feature_dim)
        self._mu: Dict[int, Tensor] = {}
        self._inv_cov: Dict[int, Tensor] = {}
        self._mapper_by_class: Dict[int, nn.Linear] = {}

    def _project_if_needed(self, cls: int, t: Tensor) -> Tensor:
        """When D is large, linearly project to gaussian_feature_dim before Gaussian fit/score."""
        d = int(t.shape[1])
        if self._gaussian_dim <= 0 or d <= self._gaussian_dim:
            return t
        c = int(cls)
        if c not in self._mapper_by_class:
            mapper = nn.Linear(d, self._gaussian_dim, bias=False).to(self._device)
            self._mapper_by_class[c] = mapper
        return self._mapper_by_class[c](t)

    def fit_class(self, cls: int, features: np.ndarray) -> None:
        feats = np.asarray(features, dtype=np.float32)
        if feats.size == 0:
            return
        c = int(cls)
        t = torch.from_numpy(feats).to(device=self._device, dtype=torch.float32)
        t = self._project_if_needed(c, t)
        n, d = t.shape
        mu: Tensor
        inv_cov: Tensor
        if n <= 2:
            mu, inv_cov = fit_global_gaussian(t, ridge=self._ridge)
        else:
            emb = t.unsqueeze(0)
            try:
                gaussian = MultiVariateGaussian(d, 1).to(self._device)
                mean_b, inv_b = gaussian.fit(emb)
                if not torch.isfinite(inv_b).all() or not torch.isfinite(mean_b).all():
                    raise RuntimeError("non-finite gaussian stats")
                mu = mean_b.squeeze(-1)
                inv_cov = inv_b.squeeze(0)
            except Exception:
                mu, inv_cov = fit_global_gaussian(t, ridge=self._ridge)
                if not torch.isfinite(inv_cov).all():
                    warnings.warn(
                        f"[MahalanobisPseudoLabelScorer] class {c}: fallback inv non-finite at ridge={self._ridge}, "
                        f"retrying ridge={self._fallback_ridge}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    mu, inv_cov = fit_global_gaussian(t, ridge=self._fallback_ridge)
        if not torch.isfinite(inv_cov).all():
            warnings.warn(
                f"[MahalanobisPseudoLabelScorer] class {c}: non-finite inv_cov, using isotropic fallback",
                RuntimeWarning,
                stacklevel=2,
            )
            d_dim = int(mu.numel())
            inv_cov = torch.eye(d_dim, device=self._device, dtype=mu.dtype) / self._fallback_ridge
        self._mu[c] = mu.detach()
        self._inv_cov[c] = inv_cov.detach()

    def has_class(self, cls: int) -> bool:
        return int(cls) in self._mu

    def score_patches(self, cls: int, queries: np.ndarray) -> np.ndarray:
        c = int(cls)
        q = torch.from_numpy(np.ascontiguousarray(np.asarray(queries, dtype=np.float32))).to(
            device=self._device, dtype=torch.float32
        )
        q = self._project_if_needed(c, q)
        dist = mahalanobis_global(q, self._mu[c], self._inv_cov[c])
        return dist.detach().cpu().numpy().astype(np.float32)


class PCAPseudoLabelScorer(PseudoLabelScorer):
    """Per-class PCA reconstruction scorer: larger residual means more anomalous."""

    def __init__(
        self,
        pca_dim: int = 0,
        pca_ev: float = 0.99,
        eps: float = 1e-6,
    ):
        self._pca_dim = int(pca_dim)
        self._pca_ev = float(pca_ev)
        self._eps = float(eps)
        self._mu: Dict[int, Tensor] = {}
        self._components: Dict[int, Tensor] = {}
        self._eigvals: Dict[int, Tensor] = {}

    def _select_k(self, eigvals_desc: Tensor) -> int:
        d = int(eigvals_desc.numel())
        if d <= 0:
            return 1
        if self._pca_dim > 0:
            return max(1, min(self._pca_dim, d))
        ev_target = min(max(self._pca_ev, 0.0), 1.0)
        total = float(eigvals_desc.sum().item())
        if total <= self._eps:
            return 1
        cumsum = torch.cumsum(eigvals_desc, dim=0)
        ratio = cumsum / (total + self._eps)
        idx = int(torch.searchsorted(ratio, torch.tensor(ev_target, device=ratio.device)).item())
        return max(1, min(idx + 1, d))

    def _fit_pca(self, features: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # features: [N, D]
        mu = features.mean(dim=0)
        centered = features - mu
        n = int(features.shape[0])
        if n <= 1:
            d = int(features.shape[1])
            components = torch.eye(d, device=features.device, dtype=features.dtype)[:, :1]
            eigvals = torch.ones(1, device=features.device, dtype=features.dtype)
            return mu, components, eigvals

        cov = centered.t().mm(centered) / float(max(n - 1, 1))
        eigvals, eigvecs = torch.linalg.eigh(cov)
        eigvals = torch.clamp(eigvals, min=self._eps)
        eigvals_desc = torch.flip(eigvals, dims=[0])
        eigvecs_desc = torch.flip(eigvecs, dims=[1])
        k = self._select_k(eigvals_desc)
        components = eigvecs_desc[:, :k]
        eigvals_k = eigvals_desc[:k]
        return mu, components, eigvals_k

    def fit_class(self, cls: int, features: np.ndarray) -> None:
        feats = np.asarray(features, dtype=np.float32)
        if feats.size == 0:
            return
        c = int(cls)
        t = torch.from_numpy(feats)
        mu, components, eigvals = self._fit_pca(t)
        self._mu[c] = mu.detach()
        self._components[c] = components.detach()
        self._eigvals[c] = eigvals.detach()

    def has_class(self, cls: int) -> bool:
        c = int(cls)
        return c in self._mu and c in self._components

    def score_patches(self, cls: int, queries: np.ndarray) -> np.ndarray:
        c = int(cls)
        q = torch.from_numpy(np.ascontiguousarray(np.asarray(queries, dtype=np.float32)))
        mu = self._mu[c]
        comp = self._components[c]
        centered = q - mu
        proj = centered.mm(comp).mm(comp.t())
        residual = centered - proj
        scores = torch.sum(residual * residual, dim=1)
        return scores.detach().cpu().numpy().astype(np.float32)


class PseudoLabelScorerFactory:
    """根据 mode 字符串（如 "nn"、"nn+mahalanobis+pca"）创建打分器列表。"""

    @staticmethod
    def create(
        mode: str,
        args,
        device: torch.device,
        build_faiss_index_fn,
    ) -> List[PseudoLabelScorer]:
        names = [s.strip() for s in str(mode).lower().split("+")]
        scorers: List[PseudoLabelScorer] = []
        for name in names:
            if name == "nn":
                scorers.append(NNPseudoLabelScorer(build_faiss_index_fn, args.k_number))
            elif name == "mahalanobis":
                scorers.append(
                    
                    MahalanobisPseudoLabelScorer(
                        device,
                        gaussian_feature_dim=int(getattr(args, "pseudo_label_mahalanobis_dim", 128)),
                    )
                )
            elif name == "pca":
                scorers.append(
                    PCAPseudoLabelScorer(
                        pca_dim=int(getattr(args, "pseudo_label_pca_dim", 0)),
                        pca_ev=float(getattr(args, "pseudo_label_pca_ev", 0.99)),
                        eps=float(getattr(args, "pseudo_label_pca_eps", 1e-6)),
                    )
                )
            else:
                raise ValueError(
                    f"Unknown pseudo-label scorer '{name}'. "
                    f"Valid options: nn, mahalanobis, pca (combinable with +, e.g. nn+mahalanobis)."
                )
        if not scorers:
            raise ValueError("No scorer selected (empty --pseudo_label_scoring).")
        return scorers
