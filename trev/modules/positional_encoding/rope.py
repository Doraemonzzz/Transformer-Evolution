import math
import numpy as np
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from trev import utils
from trev.incremental_decoding_utils import with_incremental_state
from trev.modules.trev_dropout import TrevDropout
from trev.modules.quant_noise import quant_noise
from torch import Tensor, nn
from torch.nn import Parameter
from torch.nn import Dropout
import sys

# https://github.com/JunnYu/FLASHQuad_pytorch/blob/main/flash/gau.py
def rope(x, dim):
    """RoPE position embedding."""
    shape = x.shape
    if isinstance(dim, int):
        dim = [dim]
    spatial_shape = [shape[i] for i in dim]
    total_len = 1
    for i in spatial_shape:
        total_len *= i
    position = torch.reshape(
        torch.arange(total_len, dtype=x.dtype,
                     device=x.device), spatial_shape
    )
    for i in range(dim[-1] + 1, len(shape) - 1, 1):
        position = position.unsqueeze(-1)
    half_size = shape[-1] // 2
    freq_seq = -torch.arange(half_size, dtype=x.dtype, device=x.device) / float(
        half_size
    )
    inv_freq = 10000 ** freq_seq
    sinusoid = torch.einsum("...,d->...d", position, inv_freq)
    sin = sinusoid.sin()
    cos = sinusoid.cos()
    x1, x2 = torch.chunk(x, 2, dim=-1)

    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)