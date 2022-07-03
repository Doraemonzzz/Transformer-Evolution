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
from .norm_mix_attention import NormMixAttention

class NormMixAttentionEncoderLayer(TransformerEncoderLayer):
    def __init__(self, args):
        super().__init__(args)

    def build_self_attention(self, embed_dim, args):
        return NormMixAttention(
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            # add
            index=args.index,
            use_relu=getattr(args, "use_relu", False),
            use_elu=getattr(args, "use_elu", False),
            use_leak=getattr(args, "use_leak", False),
            use_bound=getattr(args, "use_bound", False),
            max_l=getattr(args, "max_l", 1024),
            has_out=getattr(args, "has_out", False),
            weight_type=getattr(args, "weight_type", 1),
            c=getattr(args, "c", 1.0),
            v_act=getattr(args, "v_act", False),
            use_dropout=getattr(args, "use_dropout", False),
            p=getattr(args, "p", 0.5),
            use_layer_norm=getattr(args, "use_layer_norm", False),
            qk_layer_norm=getattr(args, "qk_layer_norm", False),
            seq_dropout=getattr(args, "seq_dropout", False),
            seq_p=getattr(args, "seq_p", 0.3),
            lambda_=getattr(args, "lambda_", 0.001),
            use_gelu=getattr(args, "use_gelu", False),
            mem_use_gelu=getattr(args, "mem_use_gelu", False),
            mem_use_grad=getattr(args, "mem_use_grad", True),
            mem_use_q=getattr(args, "mem_use_q", True),
            mem_use_k=getattr(args, "mem_use_k", False),
            attention_use_layer_norm=getattr(args, "attention_use_layer_norm", True),
            model_update_freq=getattr(args, "model_update_freq", 1),
            linear_act_fun=getattr(args, "linear_act_fun", "gelu"),
            local_act_fun=getattr(args, "local_act_fun", "relu"),
            out_use_act=getattr(args, "out_use_act", True),
            init_type=getattr(args, "init_type", "default"),
            norm_type=getattr(args, "norm_type", "layernorm"),
            use_rope=getattr(args, "use_rope", False),
            rope_type=getattr(args, "rope_type", "a"),
            use_v=getattr(args, "use_v", False),
            negative_slope=getattr(args, "negative_slope", 0.1),
            # add
            causal=getattr(args, "encoder_causal", False),
            use_urpe=getattr(args, "encoder_use_urpe", True),
            core_matrix=getattr(args, "encoder_core_matrix", 1),
            p_matrix=getattr(args, "encoder_p_matrix", 1),
            max_positions=getattr(args, "encoder_max_positions", 512),
            theta_type=getattr(args, "encoder_theta_type", "a"),
            theta_learned=getattr(args, "encoder_theta_learned", False), 
            householder_learned=getattr(args, "encoder_householder_learned", False),
            # add
            chunk_size=getattr(args, "chunk_size", 32),
            forward_type=getattr(args, "forward_type", 1),
        )

class NormMixAttentionDecoderLayer(TransformerDecoderLayer):
    def __init__(
        elf, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__(args, no_encoder_attn, add_bias_kv, add_zero_attn)

    def build_self_attention(
        self, embed_dim, args, add_bias_kv=False, add_zero_attn=False
    ):
        return NormMixAttention(
            embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not getattr(args, "cross_self_attention", False),
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            # add
            index=args.index,
            use_relu=getattr(args, "use_relu", False),
            use_elu=getattr(args, "use_elu", False),
            use_leak=getattr(args, "use_leak", False),
            use_bound=getattr(args, "use_bound", False),
            max_l=getattr(args, "max_l", 1024),
            has_out=getattr(args, "has_out", False),
            weight_type=getattr(args, "weight_type", 1),
            c=getattr(args, "c", 1.0),
            v_act=getattr(args, "v_act", False),
            use_dropout=getattr(args, "use_dropout", False),
            p=getattr(args, "p", 0.5),
            use_layer_norm=getattr(args, "use_layer_norm", False),
            qk_layer_norm=getattr(args, "qk_layer_norm", False),
            seq_dropout=getattr(args, "seq_dropout", False),
            seq_p=getattr(args, "seq_p", 0.3),
            lambda_=getattr(args, "lambda_", 0.001),
            use_gelu=getattr(args, "use_gelu", False),
            mem_use_gelu=getattr(args, "mem_use_gelu", False),
            mem_use_grad=getattr(args, "mem_use_grad", True),
            mem_use_q=getattr(args, "mem_use_q", True),
            mem_use_k=getattr(args, "mem_use_k", False),
            attention_use_layer_norm=getattr(args, "attention_use_layer_norm", True),
            model_update_freq=getattr(args, "model_update_freq", 1),
            linear_act_fun=getattr(args, "linear_act_fun", "gelu"),
            local_act_fun=getattr(args, "local_act_fun", "relu"),
            out_use_act=getattr(args, "out_use_act", True),
            init_type=getattr(args, "init_type", "default"),
            norm_type=getattr(args, "norm_type", "layernorm"),
            use_rope=getattr(args, "use_rope", False),
            rope_type=getattr(args, "rope_type", "a"),
            use_v=getattr(args, "use_v", False),
            negative_slope=getattr(args, "negative_slope", 0.1),
            # add
            causal=getattr(args, "decoder_causal", True),
            use_urpe=getattr(args, "decoder_use_urpe", True),
            core_matrix=getattr(args, "decoder_core_matrix", 1),
            p_matrix=getattr(args, "decoder_p_matrix", 1),
            max_positions=getattr(args, "decoder_max_positions", 512),
            theta_type=getattr(args, "decoder_theta_type", "a"),
            theta_learned=getattr(args, "decoder_theta_learned", False), 
            householder_learned=getattr(args, "decoder_householder_learned", False),
            # add
            chunk_size=getattr(args, "chunk_size", 32),
            forward_type=getattr(args, "forward_type", 1),
        )

    def build_encoder_attention(self, embed_dim, args):
        return NormMixAttention(
            embed_dim,
            args.decoder_attention_heads,
            kdim=getattr(args, "encoder_embed_dim", None),
            vdim=getattr(args, "encoder_embed_dim", None),
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            # add
            # add
            index=args.index,
            use_relu=getattr(args, "use_relu", False),
            use_elu=getattr(args, "use_elu", False),
            use_leak=getattr(args, "use_leak", False),
            use_bound=getattr(args, "use_bound", False),
            max_l=getattr(args, "max_l", 1024),
            has_out=getattr(args, "has_out", False),
            weight_type=getattr(args, "weight_type", 1),
            c=getattr(args, "c", 1.0),
            v_act=getattr(args, "v_act", False),
            use_dropout=getattr(args, "use_dropout", False),
            p=getattr(args, "p", 0.5),
            use_layer_norm=getattr(args, "use_layer_norm", False),
            qk_layer_norm=getattr(args, "qk_layer_norm", False),
            seq_dropout=getattr(args, "seq_dropout", False),
            seq_p=getattr(args, "seq_p", 0.3),
            lambda_=getattr(args, "lambda_", 0.001),
            use_gelu=getattr(args, "use_gelu", False),
            mem_use_gelu=getattr(args, "mem_use_gelu", False),
            mem_use_grad=getattr(args, "mem_use_grad", True),
            mem_use_q=getattr(args, "mem_use_q", True),
            mem_use_k=getattr(args, "mem_use_k", False),
            attention_use_layer_norm=getattr(args, "attention_use_layer_norm", True),
            model_update_freq=getattr(args, "model_update_freq", 1),
            linear_act_fun=getattr(args, "linear_act_fun", "gelu"),
            local_act_fun=getattr(args, "local_act_fun", "relu"),
            out_use_act=getattr(args, "out_use_act", True),
            init_type=getattr(args, "init_type", "default"),
            norm_type=getattr(args, "norm_type", "layernorm"),
            use_rope=getattr(args, "use_rope", False),
            rope_type=getattr(args, "rope_type", "a"),
            use_v=getattr(args, "use_v", False),
            negative_slope=getattr(args, "negative_slope", 0.1),
            # add
            causal=getattr(args, "encoder_causal", False),
            use_urpe=getattr(args, "encoder_use_urpe", True),
            core_matrix=getattr(args, "encoder_core_matrix", 1),
            p_matrix=getattr(args, "encoder_p_matrix", 1),
            max_positions=getattr(args, "encoder_max_positions", 512),
            theta_type=getattr(args, "encoder_theta_type", "a"),
            theta_learned=getattr(args, "encoder_theta_learned", False), 
            householder_learned=getattr(args, "encoder_householder_learned", False),
            # add
            chunk_size=getattr(args, "chunk_size", 32),
            forward_type=getattr(args, "forward_type", 1),
        )