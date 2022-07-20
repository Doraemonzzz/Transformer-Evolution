import torch
import torch.nn as nn
import logging

from torch import Tensor
from typing import Dict, List, Optional

from trev import utils
from trev.modules.trev_dropout import TrevDropout
from trev.modules.quant_noise import quant_noise
from trev.modules import VanillaFeedForward
from .random_feature_attention import RandomFeatureAttention
from ..utils import get_norm
from .vanilla_attention_layer import VanillaTransformerEncoderLayer, VanillaTransformerDecoderLayer

class RFAEncoderLayer(VanillaTransformerEncoderLayer):
    def __init__(self, args):
        super().__init__(args)

    def build_self_attention(self, embed_dim, args):
        return RandomFeatureAttention(
            embed_dim=embed_dim,
            num_heads=args.encoder_attention_heads,
            causal=False,
            dropout=args.attention_dropout,
            index=args.index,
            init_method=getattr(args, "init_method", "default"),
            proj_dim=getattr(args, "proj_dim", 64),
        )
        
class RFADecoderLayer(VanillaTransformerDecoderLayer):
    def __init__(
        self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__(args, no_encoder_attn, add_bias_kv, add_zero_attn)

    def build_self_attention(
        self, embed_dim, args,
    ):
        return RandomFeatureAttention(
            embed_dim=embed_dim,
            num_heads=args.decoder_attention_heads,
            causal=True,
            dropout=args.attention_dropout,
            index=args.index,
            init_method=getattr(args, "init_method", "default"),
            proj_dim=getattr(args, "proj_dim", 64),
        )