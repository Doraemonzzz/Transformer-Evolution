import torch.nn.functional as F
import torch.nn as nn
import torch
from taco.attention import *
from taco.ffn import *
from taco.norm import *

NEG_INFINITY = float('-inf')
POS_INFINITY = float('inf')

def get_attn(attn_type):
    if attn_type == "vanilla":
        return VanillaAttention
    else:
        return VanillaAttention

def get_ffn(ffn_type):
    if ffn_type == "vanilla":
        return VanillaFeedForward
    else:
        return VanillaFeedForward

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