"""Single-layer reference residuals and a shared MLP; no fusion or adapters."""

import torch
from torch import nn


class ClassReferenceResidual(nn.Module):
    """Exact class-local nearest neighbors, with bounded distance-matrix memory."""

    def __init__(self, references, query_chunk=512, reference_chunk=2048):
        super().__init__()
        if not references or min(query_chunk, reference_chunk) < 1:
            raise ValueError("References and positive chunk sizes are required.")
        self.query_chunk = int(query_chunk)
        self.reference_chunk = int(reference_chunk)
        self.class_ids = tuple(sorted(int(c) for c in references))
        dimensions = set()
        for cls in self.class_ids:
            bank = torch.as_tensor(references[cls], dtype=torch.float32).detach().clone()
            if bank.ndim != 2 or min(bank.shape) < 1 or not torch.isfinite(bank).all():
                raise ValueError(f"Class {cls}: expected finite nonempty [N,C] references.")
            dimensions.add(bank.shape[1])
            self.register_buffer(f"reference_{cls}", bank)
        if len(dimensions) != 1:
            raise ValueError("All reference banks must have the same feature dimension.")
        self.feature_dim = dimensions.pop()

    def reference_dict(self):
        return {c: getattr(self, f"reference_{c}") for c in self.class_ids}

    @torch.no_grad()
    def forward(self, features, class_idx, *_unused):
        if features.ndim != 3 or features.shape[-1] != self.feature_dim:
            raise ValueError("Features must have shape [B,P,C] matching the reference banks.")
        class_idx = torch.as_tensor(class_idx, device=features.device).reshape(-1)
        if class_idx.numel() != features.shape[0]:
            raise ValueError("One class index is required per image.")
        result = torch.empty_like(features)
        for cls in torch.unique(class_idx).tolist():
            if cls not in self.class_ids:
                raise KeyError(f"No normal reference bank for class {cls}.")
            mask = class_idx == cls
            query = features[mask].reshape(-1, self.feature_dim).float()
            bank = getattr(self, f"reference_{cls}")
            chunks = []
            for start in range(0, query.shape[0], self.query_chunk):
                q = query[start:start + self.query_chunk]
                best = torch.full((len(q),), float("inf"), device=q.device)
                indices = torch.zeros(len(q), dtype=torch.long, device=q.device)
                for ref_start in range(0, len(bank), self.reference_chunk):
                    distances = torch.cdist(q, bank[ref_start:ref_start + self.reference_chunk])
                    values, ids = distances.min(dim=1)
                    better = values < best
                    indices = torch.where(better, ids + ref_start, indices)
                    best = torch.minimum(best, values)
                chunks.append(q - bank.index_select(0, indices))
            result[mask] = torch.cat(chunks).reshape_as(features[mask]).to(features.dtype)
        return result


class ResidualMLP(nn.Module):
    """Shared patch classifier. Class identity is used only for reference lookup."""

    def __init__(self, feature_dim):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(feature_dim, 1024), nn.GELU(),
            nn.Linear(1024, 128), nn.GELU(), nn.Linear(128, 1),
        )

    def logits(self, residual):
        return self.discriminator(residual).squeeze(-1)

    def forward(self, residual, **_unused):
        # Same evaluation interface as localnet, with no feature adapter.
        return residual, self.logits(residual).sigmoid()
