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
from .cosformer_attention import CosformerAttention

class CosformerEncoderLayer(TransformerEncoderLayer):
    def __init__(self, args):
        super().__init__(args)

    def build_self_attention(self, embed_dim, args):
        return CosformerAttention(
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            # add
            has_out=getattr(args, "has_out", False),
            use_relu=getattr(args, "use_relu", False),
            use_elu=getattr(args, "use_elu", False),
            use_leak=getattr(args, "use_leak", False),
            index=args.index,
            max_l=getattr(args, "max_l", 1024),
            causal=getattr(args, "causal", False),
            resi=getattr(args, "resi", False),
        )

class CosformerDecoderLayer(TransformerDecoderLayer):
    def __init__(
        elf, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__(args, no_encoder_attn, add_bias_kv, add_zero_attn)

    def build_self_attention(
        self, embed_dim, args, add_bias_kv=False, add_zero_attn=False
    ):
        return CosformerAttention(
            embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not getattr(args, "cross_self_attention", False),
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            # add
            has_out=getattr(args, "has_out", False),
            use_relu=getattr(args, "use_relu", False),
            use_elu=getattr(args, "use_elu", False),
            use_leak=getattr(args, "use_leak", False),
            index=args.index,
            max_l=getattr(args, "max_l", 1024),
            causal=getattr(args, "causal", False),
            resi=getattr(args, "resi", False),
        )

    def build_encoder_attention(self, embed_dim, args):
        return CosformerAttention(
            embed_dim,
            args.decoder_attention_heads,
            kdim=getattr(args, "encoder_embed_dim", None),
            vdim=getattr(args, "encoder_embed_dim", None),
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            # add
            has_out=getattr(args, "has_out", False),
            use_relu=getattr(args, "use_relu", False),
            use_elu=getattr(args, "use_elu", False),
            use_leak=getattr(args, "use_leak", False),
            index=args.index,
            max_l=getattr(args, "max_l", 1024),
            causal=getattr(args, "causal", False),
            resi=getattr(args, "resi", False),
        )