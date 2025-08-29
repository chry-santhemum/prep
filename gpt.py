
from jaxtyping import Float, Int

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module, ModuleList
from modules import (
    Swish,
    RMSNorm,
    Embedding,
    Linear,
    MultiHeadSelfAttention,
    RotaryPositionalEmbedding,
    GatedLinearUnit,
)


class TransformerBlock(Module):
    def __init__(self, d_model, n_head, d_mlp, pos_emb: RotaryPositionalEmbedding, device=None, dtype=None):
        super().__init__()

        self.pos_emb = pos_emb
        self.attn = MultiHeadSelfAttention(n_head, d_model, pos_emb, device, dtype)
        self.mlp = GatedLinearUnit(Swish(), d_model, d_mlp, device, dtype)

        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, x: Float[Tensor, "... d_model"]):
        x_post_attn = x + self.attn(self.ln1(x))
        return x_post_attn + self.mlp(self.ln2(x_post_attn))


class Transformer(Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        d_mlp: int,
        context_length: int,
        vocab_size: int,
        n_layer: int,
        rope_theta: float,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.token_embeddings = Embedding(vocab_size, d_model, device, dtype)
        self.pos_emb = RotaryPositionalEmbedding(
            rope_theta, d_model//n_head, context_length, device
        )
        self.layers = ModuleList([
            TransformerBlock(d_model, n_head, d_mlp, self.pos_emb, device, dtype)
            for _ in range(n_layer)
        ])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device, dtype)

    
    def forward(self, x: Int[Tensor, "..."]):
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x





