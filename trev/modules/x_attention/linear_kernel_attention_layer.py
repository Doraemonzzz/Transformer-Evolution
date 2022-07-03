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
from .linear_kernel_attention import LinearKernelAttention

class LinearKernelAttentionEncoderLayer(TransformerEncoderLayer):
    def __init__(self, args):
        super().__init__(args)

    def build_self_attention(self, embed_dim, args):
        return LinearKernelAttention(
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            # add
            causal=getattr(args, "causal", False),
            use_urpe=getattr(args, "use_urpe", True),
            kernel_type=getattr(args, "kernel_type", "1+elu"),
            core_matrix=getattr(args, "core_matrix", 1),
            p_matrix=getattr(args, "p_matrix", 1),
            max_positions=getattr(args, "max_positions", 512),
            theta_type=getattr(args, "theta_type", "a"),
            theta_learned=getattr(args, "theta_learned", False), 
            householder_learned=getattr(args, "householder_learned", False),
            use_rope=getattr(args, "use_rope", False),
            use_spe=getattr(args, "use_spe", False),
            use_permutate=getattr(args, "use_permutate", False),
            max_seq_len=getattr(args, "max_seq_len", 512),
            # index
            index=args.index
        )

class LinearKernelAttentionDecoderLayer(TransformerDecoderLayer):
    def __init__(
        elf, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__(args, no_encoder_attn, add_bias_kv, add_zero_attn)

    def build_self_attention(
        self, embed_dim, args, add_bias_kv=False, add_zero_attn=False
    ):
        return LinearKernelAttention(
            embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not getattr(args, "cross_self_attention", False),
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            # add
            causal=getattr(args, "causal", False),
            use_urpe=getattr(args, "use_urpe", True),
            kernel_type=getattr(args, "kernel_type", "1+elu"),
            core_matrix=getattr(args, "core_matrix", 1),
            p_matrix=getattr(args, "p_matrix", 1),
            max_positions=getattr(args, "max_positions", 512),
            theta_type=getattr(args, "theta_type", "a"),
            theta_learned=getattr(args, "theta_learned", False), 
            householder_learned=getattr(args, "householder_learned", False),
            use_rope=getattr(args, "use_rope", False),
            use_spe=getattr(args, "use_spe", False),
            use_permutate=getattr(args, "use_permutate", False),
            max_seq_len=getattr(args, "max_seq_len", 512),
            # index
            index=args.index
        )

    def build_encoder_attention(self, embed_dim, args):
        return LinearKernelAttention(
            embed_dim,
            args.decoder_attention_heads,
            kdim=getattr(args, "encoder_embed_dim", None),
            vdim=getattr(args, "encoder_embed_dim", None),
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            # add
            causal=getattr(args, "causal", False),
            use_urpe=getattr(args, "use_urpe", True),
            kernel_type=getattr(args, "kernel_type", "1+elu"),
            core_matrix=getattr(args, "core_matrix", 1),
            p_matrix=getattr(args, "p_matrix", 1),
            max_positions=getattr(args, "max_positions", 512),
            theta_type=getattr(args, "theta_type", "a"),
            theta_learned=getattr(args, "theta_learned", False), 
            householder_learned=getattr(args, "householder_learned", False),
            use_rope=getattr(args, "use_rope", False),
            use_spe=getattr(args, "use_spe", False),
            use_permutate=getattr(args, "use_permutate", False),
            max_seq_len=getattr(args, "max_seq_len", 512),
            # index
            index=args.index
        )