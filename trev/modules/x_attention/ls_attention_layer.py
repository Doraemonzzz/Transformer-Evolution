# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from trev import utils
from trev.modules import LayerNorm, TransformerEncoderLayer, TransformerDecoderLayer
from trev.modules.trev_dropout import TrevDropout
from trev.modules.quant_noise import quant_noise
from torch import Tensor
from .ls_non_causal_attention import LSNonCausalAttention

class LSAttentionEncoderLayer(TransformerEncoderLayer):
    def __init__(self, args):
        super().__init__(args)

    def build_self_attention(self, embed_dim, args):
        return LSNonCausalAttention(
            dim=embed_dim,
            num_heads=args.encoder_attention_heads,
            max_seq_len=getattr(args, "max_seq_len", 512),
            dropout=args.attention_dropout,
            num_landmarks=getattr(args, "num_landmarks", 32),
            window_size=getattr(args, "window_size", 8),
        )