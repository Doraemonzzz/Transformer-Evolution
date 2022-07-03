# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from trev import utils
from trev.modules import MultiheadAttention, LayerNorm, TransformerEncoderLayer, TransformerDecoderLayer
from trev.modules.trev_dropout import TrevDropout
from trev.modules.quant_noise import quant_noise
from torch import Tensor

from .norm_local_attention import NormLocalAttention
from .norm_linear_attention import NormLinearAttention
from .cosformer_attention import CosformerAttention
from .performer_attention import PerformerAttention
from .linear_kernel_attention import LinearKernelAttention
from ..ffn import GLU
from ..norm import SimpleRMSNorm, RMSNorm, GatedRMSNorm

class LinearCombinationEncoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embed_dim = args.encoder_embed_dim
        self.quant_noise = getattr(args, 'quant_noise_pq', 0)
        self.quant_noise_block_size = getattr(args, 'quant_noise_pq_block_size', 8) or 8
        self.self_attn = self.build_self_attention(self.embed_dim, args)
        attn_type = getattr(args, 'attn_type', 'layernorm')
        print(f"Encoder Norm Type: {attn_type}")
        if attn_type == "simplermsnorm":
            self.self_attn_layer_norm = SimpleRMSNorm(self.embed_dim)
        else:
            self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout_module = TrevDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, 'activation_fn', 'relu') or "relu"
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0) or 0
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0) or 0
        self.activation_dropout_module = TrevDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.encoder_normalize_before
        self.use_glu = getattr(args, "use_glu", False)
        self.glu_act = getattr(args, "glu_act", False)
        self.fina_act = getattr(args, "fina_act", "None")
        self.glu_dropout = getattr(args, "glu_dropout", 0.0)
        print("=============================")
        print("Encoder")
        print(f"self.use_glu {self.use_glu}")
        print(f"self.glu_act {self.glu_act}")
        print(f"self.fina_act {self.fina_act}")
        print(f"self.glu_dropout {self.glu_dropout}")
        print("=============================")

        if self.use_glu:
            d1 = self.embed_dim
            p = 8 / 3
            p = getattr(args, "multiple", p)
            d2 = int(p * d1)
            print(f"GLU multiple {p}")
            self.glu = GLU(d1, d2, self.glu_act, self.fina_act, self.glu_dropout)
        else:
            self.fc1 = self.build_fc1(
                self.embed_dim,
                args.encoder_ffn_embed_dim,
                self.quant_noise,
                self.quant_noise_block_size,
            )
            self.fc2 = self.build_fc2(
                args.encoder_ffn_embed_dim,
                self.embed_dim,
                self.quant_noise,
                self.quant_noise_block_size,
            )

        if attn_type == "simplermsnorm":
            self.final_layer_norm = SimpleRMSNorm(self.embed_dim)
        else:
            self.final_layer_norm = LayerNorm(self.embed_dim)

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_self_attention(self, embed_dim, args):
        if args.attention_type == 1:
            print("===========")
            print("cos")
            print("===========")
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
        elif args.attention_type == 2:
            print("===========")
            print("performer")
            print("===========")
            return PerformerAttention(
                embed_dim,
                args.encoder_attention_heads,
                dropout=args.attention_dropout,
                self_attention=True,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
                # add
                approx_attn_dim=getattr(args, 'approx_attn_dim', 64),
                causal=getattr(args, 'causal', False),
            )
        elif args.attention_type == 3:
            print("===========")
            print("1+elu")
            print("===========")
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
        elif args.attention_type == 4:
            print("======================")
            print("use norm_linear")
            print("======================")
            return NormLinearAttention(
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
                weight_type=getattr(args, "weight_type", -1),
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
                act_fun=getattr(args, "linear_act_fun", "gelu"),
                out_use_act=getattr(args, "out_use_act", True),
                init_type=getattr(args, "init_type", "default"),
                norm_type=getattr(args, "norm_type", "layernorm"),
                use_rope=getattr(args, "use_rope", False),
                rope_type=getattr(args, "rope_type", "a"),
                use_v=getattr(args, "use_v", False),
                negative_slope=getattr(args, "negative_slope", 0.1),
                # add
                causal=getattr(args, "encoder_causal", False),
                use_urpe=getattr(args, "encoder_use_urpe", False),
                core_matrix=getattr(args, "encoder_core_matrix", 1),
                p_matrix=getattr(args, "encoder_p_matrix", 1),
                max_positions=getattr(args, "encoder_max_positions", 512),
                theta_type=getattr(args, "encoder_theta_type", "a"),
                theta_learned=getattr(args, "encoder_theta_learned", False), 
                householder_learned=getattr(args, "encoder_householder_learned", False),
                kv_act=getattr(args, "encoder_kv_act", "identity"),
                # final dropout
                use_final_dropout=getattr(args, "use_final_dropout", False),
                final_dropout=getattr(args, "final_dropout", 0.0)
            )
        elif args.attention_type == 5:
            print("======================")
            print("use norm_local")
            print("======================")
            return NormLocalAttention(
                embed_dim,
                args.encoder_attention_heads,
                dropout=args.attention_dropout,
                self_attention=True,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
                # add
                act_fun=getattr(args, "local_act_fun", "relu"),
                negative_slope=getattr(args, "negative_slope", 0.1),
                # add
                causal=getattr(args, "encoder_causal", False),
                use_urpe=getattr(args, "encoder_use_urpe", False),
                core_matrix=getattr(args, "encoder_core_matrix", 1),
                p_matrix=getattr(args, "encoder_p_matrix", 1),
                max_positions=getattr(args, "encoder_max_positions", 512),
                theta_type=getattr(args, "encoder_theta_type", "a"),
                theta_learned=getattr(args, "encoder_theta_learned", False), 
                householder_learned=getattr(args, "encoder_householder_learned", False),
                # add
                chunk_size=getattr(args, "encoder_chunk_size", 32),
                left_window=getattr(args, "left_window", 1),
                right_window=getattr(args, "right_window", 1),
                group_type=getattr(args, "group_type", "chunk"),
                use_softmax=getattr(args, "use_softmax", False),
                norm_type=getattr(args, "local_norm_type", "gatedrmsnorm"),
                # weight
                weight_type=getattr(args, "weight_type", -1),
                # final dropout
                use_final_dropout=getattr(args, "use_final_dropout", False),
                final_dropout=getattr(args, "final_dropout", 0.0),
                index=args.index,
            )
        elif args.attention_type == -1:
            print("===========")
            print("vanilla")
            print("===========")
            return MultiheadAttention(
                embed_dim,
                args.encoder_attention_heads,
                dropout=args.attention_dropout,
                self_attention=True,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
                index=args.index,
            )

    def residual_connection(self, x, residual):
        return residual + x

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def forward(self, x, encoder_padding_mask: Optional[Tensor], attn_mask: Optional[Tensor] = None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)
        # print("encoder")
        # print("before")
        # print(x.shape)
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            need_weights=False,
            attn_mask=attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        if self.use_glu:
            x = self.glu(x)
        else:
            x = self.activation_fn(self.fc1(x))
            x = self.activation_dropout_module(x)
            x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        # print("after")
        # print(x.shape)
        return x


