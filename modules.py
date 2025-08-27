
import einops
from typing import Optional
from jaxtyping import Int, Float, Bool
from fancy_einsum import einsum

import numpy as np
import torch
from torch import Tensor
from torch.nn import init, Module, Parameter, Sigmoid

class Linear(Module):
    """
    Linear map without a bias.
    """
    def __init__(self, d_in: int, d_out: int, device=None, dtype=None):
        super().__init__()
        self.in_features=d_in
        self.out_features=d_out
        self.weight = Parameter(
            data = init.trunc_normal_(
                torch.zeros(d_out, d_in, device=device, dtype=dtype),
                mean=0,
                std=2/(d_in + d_out),
                a=-6/(d_in + d_out),
                b=6/(d_in + d_out),
            )
        )
        self.device=self.weight.data.device
        self.dtype=self.weight.data.dtype


    def forward(self, x: Float[Tensor, " ... d_in"]):
        y = einsum(" ... d_in, d_out d_in -> ... d_out", x, self.weight)
        return y


class Embedding(Module):
    def __init__(self, vocab_size: int, d_model: int, device=None, dtype=None):
        super().__init__()
        self.vocab_size=vocab_size
        self.d_model=d_model
        self.weight = Parameter(
            data = init.trunc_normal_(
                torch.zeros(vocab_size, d_model, device=device, dtype=dtype),
                mean=0,
                std=1,
                a=-3,
                b=3,
            )
        ) 
        self.device=self.weight.data.device
        self.dtype=self.weight.data.dtype

    def forward(self, x: Int[Tensor, "..."]):
        y = self.weight[x, :]
        return y



class RMSNorm(Module):
    def __init__(self, d_model: int, eps: float=1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model=d_model
        self.eps=eps
        self.gains = Parameter(
            data=torch.ones(d_model, device=device, dtype=dtype)
        )
        self.device=self.gains.data.device
        self.dtype=self.gains.data.dtype

    def forward(self, x: Float[Tensor, "... d_model"]):
        # upcast for numerical stability
        original_dtype = x.dtype
        x = x.float()
        x_gain = einsum("... d_model, d_model -> ... d_model", x, self.gains.float())

        squared_x = einsum("... d_model, ... d_model -> ...", x, x)
        rms_norm = torch.sqrt(squared_x / self.d_model) + self.eps

        y = x_gain / rms_norm[..., None]
        return y.to(original_dtype)


class Swish(Module):
    """
    Applies the Swish activation function pointwise.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: Float[Tensor, "..."]):
        return x * Sigmoid()(x)



class GatedLinearUnit(Module):
    def __init__(self, activation_function: Module, d_model: int, d_mlp: int|None=None, device=None, dtype=None):
        super().__init__()

        if d_mlp is None:
            d_mlp = (8 * d_model // (3 * 64)) * 64
        self.d_model = d_model
        self.d_mlp = d_mlp

        self.W_up = Linear(d_in=d_model, d_out=d_mlp, device=device, dtype=dtype)
        self.W_gate = Linear(d_in=d_model, d_out=d_mlp, device=device, dtype=dtype)
        self.W_down = Linear(d_in=d_mlp, d_out=d_model, device=device, dtype=dtype)
        self.act = activation_function

    def forward(self, x: Float[Tensor, "... d_model"]):
        y = self.W_down(
            self.W_up(x) * self.act(self.W_gate(x))
        )
        return y
    


class RotaryPositionalEmbedding(Module):
    def __init__(self, theta: float, d_qk: int, max_seq_len: int, device=None):
        super().__init__()
        assert d_qk % 2 == 0, "query/key dimension has to be even"

        self.theta=theta
        self.d_qk=d_qk
        self.max_seq_len=max_seq_len
        
        rows = torch.arange(max_seq_len, device=device)
        columns = torch.tensor([
            self.theta ** (-(2*j) / self.d_qk) for j in range(self.d_qk // 2)
        ], device=device)

        thetas = rows[:, None] * columns[None, :]

        self.register_buffer("buffer_sin", torch.sin(thetas), persistent=False)
        self.register_buffer("buffer_cos", torch.cos(thetas), persistent=False)


    def forward(self, x: Float[Tensor, "... seq_len d_qk"], token_positions: Int[Tensor, "... seq_len"]):
        print("x shape:", x.shape)
        print("positions shape:", token_positions.shape)
        sliced_sin: Float[Tensor, "... seq_len half_d_qk"] = self.buffer_sin[token_positions, :]
        sliced_cos: Float[Tensor, "... seq_len half_d_qk"] = self.buffer_cos[token_positions, :]

        rotation_matrix: Float[Tensor, "2 2 ... seq_len half_d_qk"] = torch.stack([
            torch.stack([sliced_cos, -sliced_sin], dim=0),
            torch.stack([sliced_sin, sliced_cos], dim=0)
        ], dim=0)
        print("Rotation matrix shape:", rotation_matrix.shape)

        reshaped_x = einops.rearrange(x, "... (half_d_qk two) -> two ... half_d_qk", two=2)
        print("Reshaped x shape:", reshaped_x.shape)

        reshaped_y = einsum(
            "a b ... seq_len half_d_qk, b ... seq_len half_d_qk-> a ... seq_len half_d_qk",
            rotation_matrix, reshaped_x
        )
        print("Reshaped y shape:", reshaped_y.shape)
        y = einops.rearrange(reshaped_y, "two ... half_d_qk -> ... (half_d_qk two)", two=2)
        return y



def softmax(x: Float[Tensor, "..."], dim: int=-1):
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    x_exp_sum = torch.sum(x_exp, dim=dim, keepdim=True)
    return x_exp / x_exp_sum


def scaled_dot_product_attention(
    queries: Float[Tensor, "batch_size ... seq_len d_qk"],
    keys: Float[Tensor, "batch_size ... seq_len d_qk"],
    values: Float[Tensor, "batch_size ... seq_len d_v"],
    attention_mask: Optional[Bool[Tensor, "seq_len seq_len"]],
) -> Float[Tensor, "batch_size ... d_v"]:

    d_qk = queries.shape[-1]
    attn_scores = einsum("batch_size ... seq_len_1 d_qk, batch_size ... seq_len_2 d_qk -> batch_size ... seq_len_1 seq_len_2", queries, keys) / np.sqrt(d_qk)

    if attention_mask is not None:
        attn_scores[..., ~attention_mask] = -torch.inf

    attn_pattern = softmax(attn_scores, dim=-1)
    attn_output = einsum("batch_size ... seq_len_1 seq_len_2, batch_size ... seq_len_2 d_v -> batch_size ... seq_len_1 d_v", attn_pattern, values)
    return attn_output



