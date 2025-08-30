
from typing import Callable, Optional, Iterable
from jaxtyping import Float, Int

import numpy as np
import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.nn import Parameter

def cross_entropy(logits: Float[Tensor, "... vocab_size"], labels: Int[Tensor, "..."]) -> float:
    logits_minus_max = logits - logits.max(dim=-1, keepdim=True).values
    logits_exp_sum = logits_minus_max.exp().sum(dim=-1)
    ce_denominator = logits_exp_sum.log().mean()
    ce_numerator = logits_minus_max.gather(dim=-1, index=labels.unsqueeze(-1)).mean()
    return ce_denominator - ce_numerator  # negative log



class SGD(Optimizer):
    def __init__(self, params, lr=1e-3):
        assert lr >= 0, "SGD optimizer: learning rate must be at least 0"
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    
    def step(self, closure: Optional[Callable]=None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]  # optmizer state
                t = state.get("t", 0)  # step

                p.data -= lr * grad
                state["t"] = t+1
        
        return loss


class AdamW(Optimizer):
    def __init__(self, params, lr, weight_decay, betas, eps=1e-8):
        assert lr >= 0, "AdamW: learning rate must be at least 0"
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "beta_1": betas[0],
            "beta_2": betas[1],
            "eps": eps
        }

        super().__init__(params, defaults)


    def step(self, closure: Optional[Callable]=None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            beta_1 = group["beta_1"]
            beta_2 = group["beta_2"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 1)
                
                lr_multiplier = np.sqrt(1 - beta_2 ** t) / (1 - beta_1 ** t)

                # weight decay
                p.data -= weight_decay * lr * p.data
                
                # Adam
                grad = p.grad.data
                m = state.get("first_moment", torch.zeros_like(p))
                v = state.get("second_moment", torch.zeros_like(p))

                m = beta_1 * m + (1 - beta_1) * grad
                v = beta_2 * v + (1 - beta_2) * (grad ** 2)   
                state["first_moment"] = m
                state["second_moment"] = v            
                p.data -= lr * lr_multiplier * (m / (torch.sqrt(v) + eps))
                state["t"] = t+1

        return loss


def cosine_anneal_scheduler(t: int, lr_max: float, lr_min: float, T_warmup: int, T_anneal: int):
    if t < T_warmup:
        return lr_max * t / T_warmup
    if t > T_anneal:
        return lr_min

    ratio = (t - T_warmup) / (T_anneal - T_warmup)
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(np.pi * ratio))


def gradient_clipping(params: Iterable[Parameter], max_l2_norm: float, eps: float=1e-6):
    all_norms = []
    for p in params:
        if p.requires_grad:
            all_norms.append(p.grad.norm())

    if torch.tensor(all_norms).norm() > max_l2_norm:
        scale = max_l2_norm / torch.tensor(all_norms).norm()

        for p in params:
            if p.requires_grad:
                p.grad.copy_(p.grad * scale)

