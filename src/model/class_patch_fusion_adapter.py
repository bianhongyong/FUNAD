"""Class-and-patch adaptive DINOv3 layer fusion + class-strength adapter.

V2 of the learnable fusion design (HCPAF).  It keeps the V1 class prior
``theta[c]`` but lets every query patch predict a small layer-weight offset:

    alpha_{c,p,l} = softmax_l((theta_{c,l} + lambda_c * delta_{c,p,l}) / tau)

The delta head is zero-initialised, so a fresh V2 model is numerically
identical to V1 ``learned_class``.  The class embedding that drives the patch
gate is shared with the optional class-strength adapter (V2-C):

    z = r + beta_c * H_phi(r, e_c)

Only one fusion/adaptation instance is created; the training script owns it
and the reference runtime keeps a non-owning pointer (see model.localnet).

Reference matching invariant: the query patch and every candidate reference
row are projected with the **same** ``alpha_{c,p,:}``.  This module only fuses
given weights; the Top-K reranking lives in ``utils.train_utils``.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inverse_softplus(value: float, eps: float = 1e-6) -> float:
    value = max(float(value), eps)
    return math.log(math.expm1(value))


def _resolve_norm(norm_type: str, feature_dim: int, norm_affine: bool):
    if norm_type == "layer_norm":
        return nn.LayerNorm(feature_dim, elementwise_affine=bool(norm_affine))
    if norm_type in ("none", None):
        return nn.Identity()
    raise ValueError(
        f"Unsupported norm_type={norm_type!r}; expected 'layer_norm' or 'none'."
    )


class PatchGateEncoder(nn.Module):
    """Encode per-layer query features into a per-patch layer-offset vector.

    ``delta`` is produced from a lightweight layer-axis attention over a shared
    C -> hidden projection.  The gate has its own input normalization so raw
    feature scales do not dominate the softmax; this is independent of the
    fusion feature norm (which keeps the V1 ``none`` default).

    The last delta layer is zero-initialised to guarantee V1 equivalence at
    initialisation.
    """

    def __init__(
        self,
        num_layers: int,
        feature_dim: int,
        class_embedding_dim: int,
        hidden_dim: int,
        norm_type: str = "layer_norm",
        use_context: bool = False,
    ):
        super().__init__()
        if int(num_layers) < 1:
            raise ValueError("num_layers must be >= 1")
        if int(feature_dim) < 1:
            raise ValueError("feature_dim must be >= 1")
        if int(class_embedding_dim) < 1:
            raise ValueError("class_embedding_dim must be >= 1")
        if int(hidden_dim) < 1:
            raise ValueError("hidden_dim must be >= 1")

        self.num_layers = int(num_layers)
        self.feature_dim = int(feature_dim)
        self.class_embedding_dim = int(class_embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.norm = _resolve_norm(norm_type, self.feature_dim, norm_affine=False)
        self.norm_type = "none" if norm_type in ("none", None) else "layer_norm"

        self.proj = nn.Linear(self.feature_dim, self.hidden_dim)
        self.layer_embedding = nn.Parameter(
            torch.zeros(self.num_layers, self.hidden_dim)
        )
        self.layer_score = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.class_proj = nn.Linear(self.class_embedding_dim, self.hidden_dim)
        self.post = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.use_context = bool(use_context)
        if self.use_context:
            self.context_conv = nn.Conv2d(
                self.hidden_dim,
                self.hidden_dim,
                kernel_size=3,
                padding=1,
                groups=self.hidden_dim,
            )
            nn.init.zeros_(self.context_conv.weight)
            nn.init.zeros_(self.context_conv.bias)
        else:
            self.context_conv = None

        self.delta_head = nn.Linear(self.hidden_dim, self.num_layers)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def forward(self, raw_query: torch.Tensor, class_embedding: torch.Tensor) -> torch.Tensor:
        """Return ``delta`` with shape ``raw_query.shape[:-2] + (L,)``.

        Args:
            raw_query: ``[N, L, C]`` or ``[B, P, L, C]``.
            class_embedding: ``[N, De]`` or ``[B, De]``.
        """
        if raw_query.dim() not in (3, 4):
            raise ValueError(
                f"patch gate expects [N,L,C] or [B,P,L,C], got {tuple(raw_query.shape)}"
            )
        if raw_query.shape[-2] != self.num_layers:
            raise ValueError(
                f"expected {self.num_layers} layers, got {raw_query.shape[-2]}"
            )
        if raw_query.shape[-1] != self.feature_dim:
            raise ValueError(
                f"expected feature_dim={self.feature_dim}, got {raw_query.shape[-1]}"
            )

        lead_shape = raw_query.shape[:-2]
        x = self.norm(raw_query)
        h = self.proj(x)
        h = h + self.layer_embedding.view(*([1] * len(lead_shape)), self.num_layers, self.hidden_dim)
        scores = self.layer_score(h).squeeze(-1)  # [..., L]
        layer_weights = torch.softmax(scores, dim=-1)
        h = (h * layer_weights.unsqueeze(-1)).sum(dim=-2)  # [..., d]

        if self.context_conv is not None:
            if raw_query.dim() != 4:
                raise ValueError("gate context requires [B,P,L,C] input.")
            batch, patches, _, _ = raw_query.shape
            side = int(round(math.sqrt(patches)))
            if side * side != patches:
                raise ValueError(
                    f"gate context requires a square patch grid, got P={patches}."
                )
            grid = h.transpose(1, 2).reshape(batch, self.hidden_dim, side, side)
            h = h + self.context_conv(grid).reshape(
                batch, self.hidden_dim, patches
            ).transpose(1, 2)

        if class_embedding.dim() == 2 and h.dim() == 3:
            class_embedding = class_embedding.unsqueeze(1)
        if class_embedding.shape != h.shape[:-1] + (self.class_embedding_dim,):
            raise ValueError(
                "class_embedding shape does not match gate output: "
                f"{tuple(class_embedding.shape)} vs {tuple(h.shape)}."
            )
        h = h + self.class_proj(class_embedding)
        h = self.post(h)
        return self.delta_head(h)

    def extra_repr(self) -> str:
        return (
            f"num_layers={self.num_layers}, feature_dim={self.feature_dim}, "
            f"hidden_dim={self.hidden_dim}, norm={self.norm_type}, "
            f"use_context={self.use_context}"
        )


class ClassPatchFusionAdapter(nn.Module):
    """Class prior + patch gate fusion with an optional class-strength adapter.

    V2-A uses this module as a drop-in replacement for
    ``ClassConditionedLayerFusion`` (legacy adapter still applies in localnet).
    V2-C additionally calls :meth:`adapt_residual`; localnet must then skip its
    own legacy adaptor so features are adapted exactly once.
    """

    def __init__(
        self,
        num_classes: int,
        num_layers: int,
        feature_dim: int,
        temperature: float = 1.0,
        norm_type: str = "none",
        norm_affine: bool = False,
        class_embedding_dim: int = 32,
        gate_hidden_dim: int = 64,
        gate_norm: str = "layer_norm",
        gate_lambda_init: float = 0.1,
        gate_use_context: bool = False,
        gate_demean_delta: bool = True,
        enable_adapter: bool = False,
        adapter_hidden_dim: int = 128,
        adapter_beta_init: float = 0.15,
    ):
        super().__init__()
        if int(num_classes) < 1:
            raise ValueError("num_classes must be >= 1")
        if int(num_layers) < 1:
            raise ValueError("num_layers must be >= 1")
        if int(feature_dim) < 1:
            raise ValueError("feature_dim must be >= 1")
        if float(temperature) <= 0:
            raise ValueError("temperature must be > 0")
        if float(gate_lambda_init) < 0:
            raise ValueError("gate_lambda_init must be >= 0")
        if not 0.0 < float(adapter_beta_init) < 1.0:
            raise ValueError("adapter_beta_init must be in (0, 1)")

        self.num_classes = int(num_classes)
        self.num_layers = int(num_layers)
        self.feature_dim = int(feature_dim)
        self.shared_weights = False
        self._temperature = float(temperature)
        self.gate_use_context = bool(gate_use_context)
        self.gate_demean_delta = bool(gate_demean_delta)
        self.enable_adapter = bool(enable_adapter)
        self.adapter_hidden_dim = int(adapter_hidden_dim)

        # Class prior identical to V1 so alpha^G can start from a V1 run.
        self.theta = nn.Parameter(torch.zeros(self.num_classes, self.num_layers))

        self.class_embedding = nn.Embedding(self.num_classes, int(class_embedding_dim))
        nn.init.trunc_normal_(self.class_embedding.weight, std=0.02)

        self.gate = PatchGateEncoder(
            num_layers=self.num_layers,
            feature_dim=self.feature_dim,
            class_embedding_dim=int(class_embedding_dim),
            hidden_dim=int(gate_hidden_dim),
            norm_type=gate_norm,
            use_context=self.gate_use_context,
        )

        # softplus(a_c) > 0; small positive start so the gate is not blocked.
        self.lambda_raw = nn.Parameter(
            torch.full(
                (self.num_classes,),
                _inverse_softplus(gate_lambda_init),
            )
        )

        # Fusion feature norm (V1-compatible, default none).
        self.norm = _resolve_norm(norm_type, self.feature_dim, norm_affine)
        self.norm_type = "none" if norm_type in ("none", None) else "layer_norm"

        if self.enable_adapter:
            self.adapter_norm = nn.LayerNorm(self.feature_dim)
            self.film = nn.Linear(int(class_embedding_dim), 2 * self.feature_dim)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
            self.beta_head = nn.Linear(int(class_embedding_dim), 1)
            # Small non-zero weight keeps the shared class embedding on the
            # autograd path from step 1 while beta stays ~ adapter_beta_init.
            nn.init.normal_(self.beta_head.weight, std=1e-2)
            beta = float(adapter_beta_init)
            nn.init.constant_(self.beta_head.bias, math.log(beta / (1.0 - beta)))
            self.adapter_down = nn.Linear(self.feature_dim, self.adapter_hidden_dim)
            self.adapter_up = nn.Linear(self.adapter_hidden_dim, self.feature_dim)
            # Small non-zero init so beta receives a gradient on the first step
            # while H stays close to zero.
            nn.init.normal_(self.adapter_up.weight, std=1e-3)
            nn.init.zeros_(self.adapter_up.bias)
            self._rho_sum = None
            self._rho_count = None
        else:
            self.adapter_norm = None
            self.film = None
            self.beta_head = None
            self.adapter_down = None
            self.adapter_up = None
            self._rho_sum = None
            self._rho_count = None

    # ------------------------------------------------------------------
    # class prior / diagnostics
    # ------------------------------------------------------------------
    @property
    def temperature(self) -> float:
        return self._temperature

    def set_temperature(self, temperature: float) -> None:
        temperature = float(temperature)
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self._temperature = temperature

    def _normalize_class_index(self, class_idx) -> torch.Tensor:
        if isinstance(class_idx, torch.Tensor):
            idx = class_idx.to(device=self.theta.device, dtype=torch.long)
        else:
            idx = torch.as_tensor(class_idx, device=self.theta.device, dtype=torch.long)
        if idx.dim() == 0:
            idx = idx.reshape(1)
        return idx.reshape(-1)

    def _expand_class_index(self, class_idx, batch_size: int) -> torch.Tensor:
        idx = self._normalize_class_index(class_idx)
        if idx.numel() == 1 and batch_size > 1:
            idx = idx.expand(batch_size)
        if idx.numel() != batch_size:
            raise ValueError(
                f"class_idx has {idx.numel()} entries but batch size is {batch_size}."
            )
        return idx

    def global_weights(self, class_idx, batch_size: Optional[int] = None) -> torch.Tensor:
        """Class-prior weights ``alpha^G`` with shape ``[B, L]``."""
        idx = self._normalize_class_index(class_idx)
        if batch_size is not None:
            idx = self._expand_class_index(class_idx, int(batch_size))
        return torch.softmax(
            self.theta.index_select(0, idx) / self._temperature, dim=-1
        )

    def get_layer_weights(self, class_idx=None) -> torch.Tensor:
        """Detached class-prior weights ``[num_classes, L]`` (V1 API)."""
        with torch.no_grad():
            alpha = torch.softmax(self.theta.detach() / self._temperature, dim=-1)
            if class_idx is None:
                return alpha
            idx = self._normalize_class_index(class_idx)
            return alpha.index_select(0, idx)

    def weight_entropy(self, class_idx=None) -> torch.Tensor:
        alpha = self.get_layer_weights(class_idx)
        return -(alpha * torch.log(alpha.clamp_min(1e-12))).sum(dim=-1)

    def get_patch_lambda(self) -> torch.Tensor:
        """Detached per-class gate amplitude ``lambda_c``."""
        return F.softplus(self.lambda_raw.detach())

    def get_beta(self) -> torch.Tensor:
        """Detached per-class adapter strength beta (zeros when disabled)."""
        if not self.enable_adapter:
            return torch.zeros(self.num_classes, device=self.theta.device)
        with torch.no_grad():
            e = self.class_embedding.weight
            return torch.sigmoid(self.beta_head(e).squeeze(-1))

    # ------------------------------------------------------------------
    # patch-conditioned weights
    # ------------------------------------------------------------------
    def patch_weights(self, raw_query: torch.Tensor, class_idx) -> torch.Tensor:
        """Return ``alpha_{c,p,:}`` with shape ``[B, P, L]`` (keeps grad).

        Args:
            raw_query: ``[B, P, L, C]`` query features (normally no grad).
            class_idx: ``[B]`` or a scalar broadcast to the batch.
        """
        if raw_query.dim() != 4:
            raise ValueError(
                f"patch_weights expects [B,P,L,C], got {tuple(raw_query.shape)}"
            )
        batch, patches, layers, dim = raw_query.shape
        if layers != self.num_layers:
            raise ValueError(f"expected {self.num_layers} layers, got {layers}")
        if dim != self.feature_dim:
            raise ValueError(f"expected feature_dim={self.feature_dim}, got {dim}")
        idx = self._expand_class_index(class_idx, batch)
        e = self.class_embedding(idx)  # [B, De]

        if self.gate_use_context:
            delta = self.gate(raw_query, e)
        else:
            flat = raw_query.reshape(batch * patches, layers, dim)
            flat_e = e.unsqueeze(1).expand(batch, patches, e.shape[-1]).reshape(
                batch * patches, e.shape[-1]
            )
            chunk = 4096
            deltas = [
                self.gate(flat[start:start + chunk], flat_e[start:start + chunk])
                for start in range(0, flat.shape[0], chunk)
            ]
            delta = (
                deltas[0] if len(deltas) == 1 else torch.cat(deltas, dim=0)
            ).reshape(batch, patches, layers)

        if self.gate_demean_delta:
            # Softmax ignores a common per-patch shift; centering keeps the
            # logged delta comparable across layers without changing alpha.
            delta = delta - delta.mean(dim=-1, keepdim=True)

        theta_rows = self.theta.index_select(0, idx)  # [B, L]
        lam = F.softplus(self.lambda_raw.index_select(0, idx)).view(batch, 1, 1)
        logits = theta_rows.unsqueeze(1) + lam * delta
        return torch.softmax(logits / self._temperature, dim=-1)

    # ------------------------------------------------------------------
    # fusion
    # ------------------------------------------------------------------
    def fuse_query(self, raw: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """Fuse ``[B, P, L, C]`` or ``[N, L, C]`` with given weights.

        ``alpha`` is ``[B, P, L]`` for 4D input, or ``[N, L]`` / ``[1, L]`` for
        3D input.  Gradients flow both to ``raw`` and to ``alpha``.
        """
        if raw.dim() not in (3, 4):
            raise ValueError(
                f"fuse_query expects [B,P,L,C] or [N,L,C], got {tuple(raw.shape)}"
            )
        if raw.shape[-2] != self.num_layers:
            raise ValueError(
                f"expected {self.num_layers} layers on dim -2, got {raw.shape[-2]}"
            )
        if raw.shape[-1] != self.feature_dim:
            raise ValueError(
                f"expected feature_dim={self.feature_dim}, got {raw.shape[-1]}"
            )

        x = self.norm(raw)
        if alpha.dtype != x.dtype:
            x = x.to(alpha.dtype)

        if x.dim() == 4:
            if alpha.dim() == 2:
                if alpha.shape[0] == 1 and x.shape[0] > 1:
                    alpha = alpha.expand(x.shape[0], -1)
                alpha = alpha.unsqueeze(1).expand(-1, x.shape[1], -1)
            elif alpha.dim() == 3 and alpha.shape[1] == 1:
                alpha = alpha.expand(-1, x.shape[1], -1)
            if alpha.shape != (x.shape[0], x.shape[1], self.num_layers):
                raise ValueError(
                    f"alpha shape {tuple(alpha.shape)} does not match "
                    f"{tuple(x.shape[:2])} + ({self.num_layers},)."
                )
            return torch.einsum("bplc,bpl->bpc", x, alpha)

        if alpha.dim() == 3:
            if alpha.shape[0] == x.shape[0]:
                alpha = alpha[:, 0]
            else:
                raise ValueError(f"3D input expects [N,L] alpha, got {tuple(alpha.shape)}.")
        if alpha.dim() == 2 and alpha.shape[0] == 1 and x.shape[0] > 1:
            alpha = alpha.expand(x.shape[0], -1)
        if alpha.shape != (x.shape[0], self.num_layers):
            raise ValueError(
                f"alpha shape {tuple(alpha.shape)} does not match "
                f"{x.shape[0]} rows and {self.num_layers} layers."
            )
        return torch.einsum("nlc,nl->nc", x, alpha)

    def fuse_refs_for_query(
        self, raw_refs: torch.Tensor, alpha: torch.Tensor
    ) -> torch.Tensor:
        """Fuse ``[M, K, L, C]`` reference candidates with per-query weights.

        ``alpha`` may be ``[M, K, L]`` (one weight per candidate) or ``[M, L]``
        (the same query weight broadcast over its K candidates).
        """
        if raw_refs.dim() != 4:
            raise ValueError(
                f"fuse_refs_for_query expects [M,K,L,C], got {tuple(raw_refs.shape)}"
            )
        m, k, layers, dim = raw_refs.shape
        if alpha.dim() == 2:
            alpha = alpha.unsqueeze(1).expand(m, k, layers)
        if alpha.shape != (m, k, layers):
            raise ValueError(
                f"alpha shape {tuple(alpha.shape)} does not match {(m, k, layers)}."
            )
        fused = self.fuse_query(
            raw_refs.reshape(m * k, layers, dim), alpha.reshape(m * k, layers)
        )
        return fused.reshape(m, k, dim)

    # ------------------------------------------------------------------
    # class-strength adapter (V2-C)
    # ------------------------------------------------------------------
    def reset_adapter_stats(self, device=None) -> None:
        if not self.enable_adapter:
            return
        target_device = torch.device(device) if device is not None else self.theta.device
        self._rho_sum = torch.zeros(self.num_classes, device=target_device)
        self._rho_count = torch.zeros(self.num_classes, device=target_device)

    def adapter_rho(self):
        """Return ``(rho_per_class, count_per_class)`` since the last reset."""
        if not self.enable_adapter:
            return None
        if self._rho_sum is None or self._rho_count is None:
            zeros = torch.zeros(self.num_classes, device=self.theta.device)
            return zeros.detach().cpu(), zeros.long().detach().cpu()
        rho = self._rho_sum / self._rho_count.clamp_min(1.0)
        return rho.detach().cpu(), self._rho_count.long().detach().cpu()

    def adapt_residual(self, residual: torch.Tensor, class_idx) -> torch.Tensor:
        """Apply ``z = r + beta_c * H_phi(r, e_c)`` once.

        Args:
            residual: ``[B, P, C]``.
            class_idx: ``[B]`` (or scalar broadcast).
        """
        if not self.enable_adapter:
            raise RuntimeError(
                "adapt_residual was called but this ClassPatchFusionAdapter was "
                "built with enable_adapter=False (V2-A)."
            )
        if residual.dim() != 3:
            raise ValueError(
                f"adapt_residual expects [B,P,C], got {tuple(residual.shape)}"
            )
        if residual.shape[-1] != self.feature_dim:
            raise ValueError(
                f"expected feature_dim={self.feature_dim}, got {residual.shape[-1]}"
            )
        batch = residual.shape[0]
        idx = self._expand_class_index(class_idx, batch)
        e = self.class_embedding(idx)  # [B, De]

        beta = torch.sigmoid(self.beta_head(e)).view(batch, 1, 1)
        gamma_shift = self.film(e)
        gamma = gamma_shift[:, : self.feature_dim].view(batch, 1, self.feature_dim)
        shift = gamma_shift[:, self.feature_dim:].view(batch, 1, self.feature_dim)

        u = self.adapter_norm(residual)
        u = (1.0 + gamma) * u + shift
        hidden = F.gelu(self.adapter_down(u))
        correction = self.adapter_up(hidden)
        adapted = residual + beta * correction

        with torch.no_grad():
            diff = (adapted - residual).detach().norm(dim=-1)
            base = residual.detach().norm(dim=-1).clamp_min(1e-12)
            rho = (diff / base).mean(dim=1)
            if (
                self._rho_sum is None
                or self._rho_count is None
                or self._rho_sum.device != residual.device
            ):
                self.reset_adapter_stats(residual.device)
            self._rho_sum.index_add_(
                0, idx, rho.to(dtype=self._rho_sum.dtype)
            )
            self._rho_count.index_add_(
                0, idx, torch.ones_like(rho, dtype=self._rho_count.dtype)
            )
        return adapted

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, class_idx) -> torch.Tensor:
        """Coarse fusion with the class prior (used by the fused cache)."""
        if x.dim() == 4:
            alpha = self.global_weights(class_idx, batch_size=x.shape[0])
        elif x.dim() == 3:
            alpha = self.global_weights(class_idx)
        else:
            raise ValueError(
                f"Expected input [B,P,L,C] or [N,L,C], got shape {tuple(x.shape)}."
            )
        return self.fuse_query(x, alpha)

    def extra_repr(self) -> str:
        return (
            f"num_classes={self.num_classes}, num_layers={self.num_layers}, "
            f"feature_dim={self.feature_dim}, temperature={self._temperature}, "
            f"norm={self.norm_type}, gate_hidden={self.gate.hidden_dim}, "
            f"gate_norm={self.gate.norm_type}, use_context={self.gate_use_context}, "
            f"enable_adapter={self.enable_adapter}"
        )
