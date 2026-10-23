"""Class-conditioned learnable DINOv3 layer fusion.

P0 module: one softmax weight vector per class, shared across patches and
applied identically to query features and raw reference features.  The module
never changes the feature dimension C, so the existing Class Adapter / MoE
input shape is preserved.
"""

import torch
import torch.nn as nn


class ClassConditionedLayerFusion(nn.Module):
    """Fuse raw per-layer patch features with class-conditioned softmax weights.

    Args:
        num_classes: number of dataset classes.
        num_layers: L, number of candidate DINOv3 layers (same order as the
            ``[..., L, C]`` input).
        feature_dim: C, per-layer feature dimension (used only by the optional
            LayerNorm).
        temperature: softmax temperature tau; must be > 0.
        norm_type: ``"layer_norm"`` (default) or ``"none"``.
        norm_affine: affine flag for LayerNorm.  Default False so the norm adds
            no per-layer trainable transform that could mask the weights.
        shared_weights: if True, all classes share one weight vector (E1
            ablation).  Otherwise weights are class-specific (E2, default).
    """

    def __init__(
        self,
        num_classes: int,
        num_layers: int,
        feature_dim: int,
        temperature: float = 1.0,
        norm_type: str = "layer_norm",
        norm_affine: bool = False,
        shared_weights: bool = False,
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

        self.num_classes = int(num_classes)
        self.num_layers = int(num_layers)
        self.feature_dim = int(feature_dim)
        self.shared_weights = bool(shared_weights)
        self._temperature = float(temperature)

        # theta[c, l] logits.  All-zero init == uniform layer weights.
        theta_shape = (1, self.num_layers) if self.shared_weights else (
            self.num_classes, self.num_layers,
        )
        self.theta = nn.Parameter(torch.zeros(theta_shape))

        if norm_type == "layer_norm":
            self.norm = nn.LayerNorm(
                self.feature_dim, elementwise_affine=bool(norm_affine)
            )
        elif norm_type in ("none", None):
            self.norm = nn.Identity()
        else:
            raise ValueError(
                f"Unsupported norm_type={norm_type!r}; expected 'layer_norm' or 'none'."
            )
        self.norm_type = "none" if norm_type in ("none", None) else "layer_norm"

    @property
    def temperature(self) -> float:
        return self._temperature

    def set_temperature(self, temperature: float) -> None:
        temperature = float(temperature)
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self._temperature = temperature

    def get_layer_weights(self, class_idx=None) -> torch.Tensor:
        """Return detached layer weights ``[num_classes, L]`` (or ``[1, L]``).

        Args:
            class_idx: optional class index / tensor selecting rows.  When
                omitted, all rows are returned (shared mode always returns one
                row).
        """
        with torch.no_grad():
            logits = self.theta.detach()
            alpha = torch.softmax(logits / self._temperature, dim=-1)
            if class_idx is None or self.shared_weights:
                return alpha
            idx = self._normalize_class_index(class_idx, fused_query_dim=None)
            return alpha.index_select(0, idx.to(alpha.device))

    def weight_entropy(self, class_idx=None) -> torch.Tensor:
        """Shannon entropy of the layer weights, in nats."""
        alpha = self.get_layer_weights(class_idx)
        return -(alpha * torch.log(alpha.clamp_min(1e-12))).sum(dim=-1)

    def _normalize_class_index(self, class_idx, fused_query_dim):
        if isinstance(class_idx, torch.Tensor):
            idx = class_idx.to(device=self.theta.device, dtype=torch.long)
        else:
            idx = torch.as_tensor(class_idx, device=self.theta.device, dtype=torch.long)
        if idx.dim() == 0:
            idx = idx.reshape(1)
        return idx.reshape(-1)

    def _alpha_for(self, class_idx, batch_size, device, dtype):
        """Return alpha for broadcasting against a 4D input: ``[B, L]``."""
        if self.shared_weights:
            alpha = torch.softmax(self.theta[0] / self._temperature, dim=-1)
            return alpha.unsqueeze(0).expand(batch_size, -1).to(dtype=dtype, device=device)

        idx = self._normalize_class_index(class_idx, fused_query_dim=None)
        if idx.numel() == 1 and batch_size > 1:
            idx = idx.expand(batch_size)
        if idx.numel() != batch_size:
            raise ValueError(
                f"class_idx has {idx.numel()} entries but input batch size is {batch_size}."
            )
        alpha = torch.softmax(self.theta.index_select(0, idx) / self._temperature, dim=-1)
        return alpha.to(dtype=dtype, device=device)

    def forward(self, x: torch.Tensor, class_idx) -> torch.Tensor:
        """Fuse layers.

        Args:
            x: ``[B, P, L, C]`` query features, or ``[N, L, C]`` reference
                features.
            class_idx: ``[B]`` for queries, or a scalar class index for a
                reference bank of one class.
        Returns:
            ``[B, P, C]`` or ``[N, C]``.
        """
        if x.dim() not in (3, 4):
            raise ValueError(
                f"Expected input [B,P,L,C] or [N,L,C], got shape {tuple(x.shape)}."
            )
        if x.shape[-2] != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} candidate layers on dim -2, "
                f"got {x.shape[-2]}."
            )
        if x.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Expected feature_dim={self.feature_dim}, got {x.shape[-1]}."
            )

        x = self.norm(x)

        if x.dim() == 4:
            alpha = self._alpha_for(class_idx, x.shape[0], x.device, x.dtype)
            return torch.einsum("bplc,bl->bpc", x, alpha)

        # Reference bank: one class, [N, L, C].
        alpha = self._alpha_for(class_idx, 1, x.device, x.dtype)[0]
        return torch.einsum("nlc,l->nc", x, alpha)

    def extra_repr(self) -> str:
        return (
            f"num_classes={self.num_classes}, num_layers={self.num_layers}, "
            f"feature_dim={self.feature_dim}, temperature={self._temperature}, "
            f"norm={self.norm_type}, shared_weights={self.shared_weights}"
        )
