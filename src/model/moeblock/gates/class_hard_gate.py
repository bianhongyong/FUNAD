r"""
Hard gate that routes tokens by class id.
"""
import torch

from .base_gate import BaseGate


class ClassHardGate(BaseGate):
    r"""
    Deterministic routing gate.
    Each token is dispatched to the expert with the same index as its class id.
    """

    def __init__(self, d_model, num_expert, world_size, top_k=1):
        super().__init__(num_expert, world_size)
        if world_size != 1:
            raise ValueError("ClassHardGate currently supports world_size == 1 only.")
        if top_k != 1:
            raise ValueError("ClassHardGate requires top_k == 1.")
        self.top_k = 1

    def _to_class_ids(self, inp: torch.Tensor) -> torch.Tensor:
        if inp is None:
            raise ValueError("ClassHardGate expects class-id input tensor, got None.")

        if not torch.is_tensor(inp):
            inp = torch.as_tensor(inp)

        if inp.dim() == 0:
            class_ids = inp.view(1)
        elif inp.dim() == 1:
            class_ids = inp
        else:
            # Accept shapes like [N, 1] or [B, P, 1] after flatten.
            class_ids = inp.reshape(-1, inp.shape[-1])[:, 0]

        class_ids = class_ids.to(dtype=torch.long, device=inp.device).reshape(-1)
        return class_ids.contiguous()

    def forward(self, inp):
        class_ids = self._to_class_ids(inp)
        if class_ids.numel() == 0:
            raise ValueError("ClassHardGate received empty class-id tensor.")

        # Validate on CPU to force synchronization and surface deterministic errors.
        class_ids_cpu = class_ids.detach().cpu()
        if torch.any(class_ids_cpu < 0) or torch.any(class_ids_cpu >= self.tot_expert):
            min_id = int(class_ids_cpu.min().item())
            max_id = int(class_ids_cpu.max().item())
            raise ValueError(
                f"class id out of range: [{min_id}, {max_id}], valid [0, {self.tot_expert - 1}]"
            )

        top1_idx = class_ids.reshape(-1, 1).contiguous()
        top1_score = torch.ones(
            (class_ids.shape[0], 1),
            dtype=torch.float32,
            device=class_ids.device,
        ).contiguous()
        self.set_loss(None)
        return top1_idx, top1_score
