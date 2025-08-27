
from jaxtyping import Int, Float
from fancy_einsum import einsum

import torch
from torch import Tensor
from torch.nn import init, Module, Parameter

class Linear(Module):
    """
    Linear map without a bias.
    """
    def __init__(self, d_in, d_out, device=None, dtype=None):
        super().__init__()
        self.weight = Parameter(
            data = init.trunc_normal_(
                torch.zeros(d_out, d_in, device=device, dtype=dtype),
                mean=0,
                std=2/(d_in + d_out),
                a=-6/(d_in + d_out),
                b=6/(d_in + d_out),
            )
        )

    def forward(self, x: Float[Tensor, " ... d_in"]):
        y = einsum(" ... d_in, d_out d_in -> ... d_out", x, self.weight)
        return y


