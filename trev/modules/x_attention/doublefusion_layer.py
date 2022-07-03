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

from .doublefusion import DoubleFusion
from trev.modules import TokenShift

########################################################################
# DoubleFusion
class DoubleFusionEncoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.quant_noise = getattr(args, 'quant_noise_pq', 0)
        self.quant_noise_block_size = getattr(args, 'quant_noise_pq_block_size', 8) or 8
        self.embed_dim = args.encoder_embed_dim
        self.self_attn = self.build_self_attention(self.embed_dim, args)
        attn_type = getattr(args, 'attn_type', 'layernorm')
        print(f"Encoder Norm Type: {attn_type}")
        if attn_type == "simplermsnorm":
            self.self_attn_layer_norm = SimpleRMSNorm(self.embed_dim)
        else:
            self.self_attn_layer_norm = LayerNorm(self.embed_dim)

        ##### token_shift
        self.token_shift = getattr(args, 'token_shift', False)
        print(f"token_shift: {self.token_shift}")
        if self.token_shift:
            self.token_shift_layer = TokenShift()
        ##### token_shift

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_self_attention(self, embed_dim, args):
        return DoubleFusion(
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
            final_dropout=getattr(args, "final_dropout", 0.0),
            ##### transformer v2
            glu_act=getattr(args, "glu_act", "swish"),
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
        
        residual = x

        ##### token_shift
        x = self.token_shift_layer(x)
        ##### token_shift
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            need_weights=False,
            attn_mask=attn_mask,
        )

        x = self.residual_connection(x, residual)
        x = self.self_attn_layer_norm(x)

        return x

class DoubleFusionDecoderLayer(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__()
        self.use_layernorm = getattr(args, "use_layernorm", True)
        self.embed_dim = args.decoder_embed_dim
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            args,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )

        export = getattr(args, "char_inputs", False)
        self.normalize_before = args.decoder_normalize_before

        attn_type = getattr(args, 'attn_type', 'layernorm')
        print(f"Decoder Norm Type: {attn_type}")
        if attn_type == "simplermsnorm":
            self.self_attn_layer_norm = SimpleRMSNorm(self.embed_dim)
        else:
            self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        
        ##### token_shift
        self.token_shift = getattr(args, 'token_shift', False)
        print(f"token_shift: {self.token_shift}")
        if self.token_shift:
            self.token_shift_layer = TokenShift()
        ##### token_shift


        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = self.build_encoder_attention(self.embed_dim, args)
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.need_attn = True

        self.onnx_trace = False

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_self_attention(
        self, embed_dim, args, add_bias_kv=False, add_zero_attn=False
    ):
        return DoubleFusion(
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
            causal=getattr(args, "decoder_causal", True),
            use_urpe=getattr(args, "decoder_use_urpe", False),
            core_matrix=getattr(args, "decoder_core_matrix", 1),
            p_matrix=getattr(args, "decoder_p_matrix", 1),
            max_positions=getattr(args, "decoder_max_positions", 512),
            theta_type=getattr(args, "decoder_theta_type", "a"),
            theta_learned=getattr(args, "decoder_theta_learned", False), 
            householder_learned=getattr(args, "decoder_householder_learned", False),
            kv_act=getattr(args, "decoder_kv_act", "identity"),
            # final dropout
            use_final_dropout=getattr(args, "use_final_dropout", False),
            final_dropout=getattr(args, "final_dropout", 0.0),
            ##### transformer v2
            glu_act=getattr(args, "glu_act", "swish"),
        )

    def build_encoder_attention(self, embed_dim, args):
        return DoubleFusion(
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
            final_dropout=getattr(args, "final_dropout", 0.0),
            ##### transformer v2
            glu_act=getattr(args, "glu_act", "swish"),
        )

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def residual_connection(self, x, residual):
        return residual + x

    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        #print(x.shape)
        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)

            ##### token shift
            if self.token_shift:
                x = self.token_shift_layer(x)
                y = self.token_shift_layer(y)
            ##### token shift
        else:
            ##### token shift
            if self.token_shift:
                x = self.token_shift_layer(x)
            ##### token shift
            y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )

        x = self.residual_connection(x, residual)
        #print('residule connection', x.shape)

        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            ##### token shift
            if self.token_shift:
                x = self.token_shift_layer(x)
                encoder_out = self.token_shift_layer(encoder_out)
            ##### token shift
            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )

            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        return x, attn, None

    def make_generation_fast_(self, need_attn: bool = False, **kwargs):
        self.need_attn = need_attn