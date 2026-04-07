r"""
Adaption to act as the MLP layer using an MoE MLP layer in transformer.
"""
import torch
import torch.nn as nn
from .layers import FMoE
from .linear import FMoELinear
from .fastermoe.config import switch_from_env


class _Expert(nn.Module):
    r"""
    An expert using 2 FMoELinear modules to speed up the computation of experts
    within one worker.
    """

    def __init__(self, num_expert, d_input, d_hidden, d_output, activation, rank=0):
        super().__init__()
        self.htoh4 = FMoELinear(num_expert, d_input, d_hidden, bias=True, rank=rank)
        self.h4toh = FMoELinear(num_expert, d_hidden, d_output, bias=True, rank=rank)
        self.activation = activation

    def forward(self, inp, fwd_expert_count):
        r"""
        First expand input to 4h (the hidden size is variable, but is called h4
        for convenience). Then perform activation. Finally shirink back to h.
        """
        x = self.htoh4(inp, fwd_expert_count)
        x = self.activation(x)
        x = self.h4toh(x, fwd_expert_count)
        return x


class FMoETransformerMLP(FMoE):
    r"""
    A complete MoE MLP module in a Transformer block.
    * `activation` is the activation function to be used in MLP in each expert.
    * `d_hidden` is the dimension of the MLP layer.
    """

    def __init__(
        self,
        num_expert=16,
        d_input=256,
        d_hidden=1024,
        d_output=256,
        activation=torch.nn.GELU(),
        expert_dp_comm="none",
        expert_rank=0,
        **kwargs
    ):
        def one_expert(_):
            return _Expert(
                1,
                d_input=d_input,
                d_hidden=d_hidden,
                d_output=d_output,
                activation=activation,
                rank=expert_rank,
            )
        
        expert = one_expert
        self.d_output = d_output
        super().__init__(num_expert=num_expert, d_model=d_input, expert=expert, **kwargs)
        self.mark_parallel_comm(expert_dp_comm)

    def forward(self, inp: torch.Tensor, cls_token: torch.Tensor = None):
        r"""
        This module wraps up the FMoE module with reshape, residual and layer
        normalization.
        """
        original_shape = inp.shape
        inp = inp.reshape(-1, self.d_model)
        gate_inp = None
        if cls_token is not None:
            if cls_token.dim() == len(original_shape) - 1:
                gate_inp = cls_token.unsqueeze(1).expand(
                    *original_shape[:-1], cls_token.shape[-1]
                ).reshape(-1, cls_token.shape[-1])
            else:
                gate_inp = cls_token.reshape(-1, cls_token.shape[-1])
            if gate_inp.shape[0] != inp.shape[0]:
                raise ValueError(
                    f"cls_token token count mismatch: {gate_inp.shape[0]} vs {inp.shape[0]}"
                )
        output = super().forward(inp, cls_token=gate_inp)
        if len(original_shape) == 1:
            return output.reshape(self.d_output)
        return output.reshape(*original_shape[:-1], self.d_output)
