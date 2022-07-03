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
from ..positional_encoding import rope
from ..norm import ScaleNorm

# https://github.com/JunnYu/FLASHQuad_pytorch/blob/main/flash/gau.py
# Flash attention
@with_incremental_state
class FlashQuadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        encoder_decoder_attention=False,
        q_noise=0.0,
        qn_block_size=8,
        # add
        s=128,
        norm_type="layer_norm",
        eps=1e-5,
        max_position_embeddings=512,
        expansion_factor=2,
    ):
        super().__init__()
        self.s = s
        # self.e = int(embed_dim * expansion_factor)
        # self.uv = nn.Linear(expansion_factor, 2 * self.e + self.s)
        self.embed_dim = embed_dim
        self.e = int(self.embed_dim * expansion_factor)
        self.u_proj = nn.Linear(embed_dim, self.e)
        self.v_proj = nn.Linear(embed_dim, self.e)
        self.base_proj = nn.Linear(embed_dim, self.s)
        self.q_weight = nn.Parameter(torch.randn(1, self.s))
        self.q_bias = nn.Parameter(torch.zeros(1, self.s))
        self.k_weight = nn.Parameter(torch.randn(1, self.s))
        self.k_bias = nn.Parameter(torch.zeros(1, self.s))
        self.o = nn.Linear(self.e, self.embed_dim)
        self.norm = nn.LayerNorm(self.embed_dim, eps=eps) if norm_type == "layer_norm" else ScaleNorm(eps=eps)
        self.w = nn.Parameter(torch.randn(2 * max_position_embeddings - 1))
        self.a = nn.Parameter(torch.randn(1, self.s))
        self.b = nn.Parameter(torch.randn(1, self.s))
        self.act_fn = F.silu
        self.max_position_embeddings = max_position_embeddings

        nn.init.normal_(self.q_weight, std=0.02)
        nn.init.normal_(self.k_weight, std=0.02)
        nn.init.normal_(self.w, std=0.02)
        nn.init.normal_(self.a, std=0.02)
        nn.init.normal_(self.b, std=0.02)

        print("flash attention")
        print(f"s {self.s}")
        print(f"norm_type {norm_type}")
        print(f"eps {eps}")
        print(f"max_position_embeddings {max_position_embeddings}")
        print(f"expansion_factor {expansion_factor}")

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.v_proj.weight)

        if self.has_out:
            nn.init.xavier_uniform_(self.out_proj.weight)
            if self.out_proj.bias is not None:
                nn.init.constant_(self.out_proj.bias, 0.0)

    def rel_pos_bias(self, seq_len):
        """Relative position bias."""
        if seq_len <= 512:
            # Construct Toeplitz matrix directly when the sequence length is less than 512
            t = F.pad(self.w[: 2 * seq_len - 1], [0, seq_len]).repeat(seq_len)
            t = t[..., :-seq_len].reshape(-1, seq_len, 3 * seq_len - 2)
            r = (2 * seq_len - 1) // 2
            t = t[..., r:-r]
        else:
            # Construct Toeplitz matrix using RoPE when the sequence length is over 512.
            a = rope(self.a.repeat(seq_len, 1), dim=0)
            b = rope(self.b.repeat(seq_len, 1), dim=0)
            t = torch.einsum("mk,nk ->mn", a, b)

        return t

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        assert key is not None and value is not None

        '''
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        '''
        tgt_len, bsz, embed_dim = query.size()
        # bsz, tgt_len, embed_dim
        query = query.transpose(0, 1)

        shortcut, x = query, self.norm(query)
        # bsz, tgt_len, e
        u = self.act_fn(self.u_proj(x))
        # bsz, tgt_len, e
        v = self.act_fn(self.v_proj(x))
        # bsz, tgt_len, s
        base = self.act_fn(self.base_proj(x))
        # base = base * weight + bias
        q_base = base * self.q_weight + self.q_bias
        k_base = base * self.k_weight + self.k_bias
        # base = torch.einsum("...r,hr->...hr", base, self.weight) + self.bias
        q = rope(q_base, dim=1)
        k = rope(k_base, dim=1)
        # bsz, tgt_len, tgt_len
        qk = torch.bmm(q, k.transpose(1, 2))
        bias = self.rel_pos_bias(self.max_position_embeddings)[:, :tgt_len, :tgt_len]
        kernel = torch.square(torch.relu(qk / self.max_position_embeddings + bias))
        if attn_mask is not None:
            kernel = kernel.masked_fill(attn_mask==float("-inf"), 0)

        x = u * torch.bmm(kernel, v)
        x = self.o(x)

        # bsz, tgt_len, s
        output = x + shortcut
        # tgt_len, bsz, s
        output = output.contiguous().transpose(0, 1)

        return output, None

    @staticmethod
    def _append_prev_key_padding_mask(
        key_padding_mask: Optional[Tensor],
        prev_key_padding_mask: Optional[Tensor],
        batch_size: int,
        src_len: int,
        static_kv: bool,
    ) -> Optional[Tensor]:
        # saved key padding masks have shape (bsz, seq_len)
        if prev_key_padding_mask is not None and static_kv:
            new_key_padding_mask = prev_key_padding_mask
        elif prev_key_padding_mask is not None and key_padding_mask is not None:
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), key_padding_mask.float()], dim=1
            )
        # During incremental decoding, as the padding token enters and
        # leaves the frame, there will be a time when prev or current
        # is None
        elif prev_key_padding_mask is not None:
            filler = torch.zeros(
                (batch_size, src_len - prev_key_padding_mask.size(1)),
                device=prev_key_padding_mask.device,
            )
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), filler.float()], dim=1
            )
        elif key_padding_mask is not None:
            filler = torch.zeros(
                (batch_size, src_len - key_padding_mask.size(1)),
                device=key_padding_mask.device,
            )
            new_key_padding_mask = torch.cat(
                [filler.float(), key_padding_mask.float()], dim=1
            )
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask

    @torch.jit.export
    def reorder_incremental_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        new_order: Tensor,
    ):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None:
                    if self.encoder_decoder_attention and input_buffer_k.size(
                        0
                    ) == new_order.size(0):
                        break
                    input_buffer[k] = input_buffer_k.index_select(0, new_order)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def _get_input_buffer(
        self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "attn_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        buffer: Dict[str, Optional[Tensor]],
    ):
        return self.set_incremental_state(incremental_state, "attn_state", buffer)

    def apply_sparse_mask(self, attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""
        items_to_add = {}
        keys_to_remove = []
        for k in state_dict.keys():
            if k.endswith(prefix + "in_proj_weight"):
                # in_proj_weight used to be q + k + v with same dimensions
                dim = int(state_dict[k].shape[0] / 3)
                items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
                items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim : 2 * dim]
                items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim :]

                keys_to_remove.append(k)

                k_bias = prefix + "in_proj_bias"
                if k_bias in state_dict.keys():
                    dim = int(state_dict[k].shape[0] / 3)
                    items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
                    items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][
                        dim : 2 * dim
                    ]
                    items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim :]

                    keys_to_remove.append(prefix + "in_proj_bias")

        for k in keys_to_remove:
            del state_dict[k]

        for key, value in items_to_add.items():
            state_dict[key] = value

