from typing import Tuple
from jaxtyping import Int

import numpy as np
from numpy.typing import NDArray
import torch

def data_loading(x: Int[NDArray, "..."], batch_size: int, context_len: int, device: torch.device) -> Tuple[Int[NDArray, "..."], Int[NDArray, "..."]]:
    starting_pos = np.random.choice(x.shape[0] - context_len, batch_size, replace=False)
    data_pos = starting_pos[:, None] + np.arange(context_len)[None, :]
    labels_pos = starting_pos[:, None] + np.arange(1, 1+context_len)[None, :]

    data = x[data_pos]
    labels = x[labels_pos]

    return torch.tensor(data, device=device), torch.tensor(labels, device=device)

