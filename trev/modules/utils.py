import torch.nn.functional as F
import torch.nn as nn
import torch
from trev.modules import SimpleRMSNorm
from trev.modules import GatedRMSNorm
from trev.modules import RMSNorm
from trev.modules import ScaleNorm

NEG_INFINITY = float('-inf')
POS_INFINITY = float('inf')

def get_activation_fn(activation):
    if activation == "gelu":
        return F.gelu
    elif activation == "relu":
        return F.relu
    elif activation == "elu":
        return F.elu
    elif activation == "sigmoid":
        return F.sigmoid
    elif activation == "exp":
        return torch.exp
    elif activation == "leak":
        return F.leaky_relu
    elif activation == "1+elu":
        def f(x):
            return 1 + F.elu(x)
        return f
    elif activation == "silu":
        return F.silu
    else:
        return lambda x: x

def get_norm(norm_type, embed_dim):
    if norm_type == "rmsnorm":
        return RMSNorm(embed_dim)
    elif norm_type == "gatedrmsnorm":
        return GatedRMSNorm(embed_dim)
    elif norm_type == "simplermsnorm":
        return SimpleRMSNorm(embed_dim)
    elif norm_type == "scalenorm":
        return ScaleNorm(embed_dim)
    else:
        return nn.LayerNorm(embed_dim)

def init_params(module, init_method="default"):
    if init_method == "default":
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.normal_(module.bias)