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
from trev.modules import GatedRMSNorm
from trev.modules import SimpleRMSNorm
from trev.modules import RMSNorm
from trev.modules import Urpe
from trev.modules import UrpeV2

@with_incremental_state
class NormMixAttention(nn.Module):
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
        index=0,
        use_relu=True,
        use_elu=False,
        use_leak=False,
        use_bound=False,
        max_l=1024,
        has_out=False,
        causal=False,
        weight_type=1,
        c=1.0,
        v_act=False,
        use_dropout=False,
        p=0.5,
        use_layer_norm=False,
        qk_layer_norm=False,
        seq_dropout=False,
        seq_p=0.3,
        lambda_=0.001,
        use_gelu=False,
        mem_use_gelu=False,
        mem_use_grad=True,
        mem_use_q=True,
        mem_use_k=False,
        attention_use_layer_norm=True,
        model_update_freq=1,
        linear_act_fun="gelu",
        local_act_fun="relu",
        out_use_act=True,
        init_type="default",
        norm_type="layernorm",
        use_rope=False,
        rope_type="a",
        use_v=False,
        negative_slope=0.1,
        # urpe
        use_urpe=False,
        core_matrix=1, 
        p_matrix=1, 
        max_positions=512,
        theta_type="a",
        theta_learned=False, 
        householder_learned=False,
        # chunk_size
        chunk_size=32,
        # forward形式, 1为并联, 2为local + linear, 3为linear + local
        forward_type=1
    ):
        # add
        self.index = index

        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        # self.dropout_module = TrevDropout(
        #     dropout, module_name=self.__class__.__name__
        # )

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )

        d = embed_dim // 2
        
        self.k_proj_linear = quant_noise(
            nn.Linear(self.kdim, d, bias=bias), q_noise, qn_block_size
        )
        self.q_proj_linear = quant_noise(
            nn.Linear(embed_dim, d, bias=bias), q_noise, qn_block_size
        )
        self.v_proj_linear = quant_noise(
            nn.Linear(embed_dim, d, bias=bias), q_noise, qn_block_size
        )
        self.out_proj_linear = quant_noise(
            nn.Linear(d, embed_dim, bias=bias), q_noise, qn_block_size
        )

        self.k_proj_local = quant_noise(
            nn.Linear(self.kdim, d, bias=bias), q_noise, qn_block_size
        )
        self.q_proj_local = quant_noise(
            nn.Linear(embed_dim, d, bias=bias), q_noise, qn_block_size
        )
        self.v_proj_local = quant_noise(
            nn.Linear(embed_dim, d, bias=bias), q_noise, qn_block_size
        )
        self.out_proj_local = quant_noise(
            nn.Linear(d, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.dropout_module = TrevDropout(
            dropout, module_name=self.__class__.__name__
        )

        # self.gated_rms_norm = GatedRMSNorm(d)

        self.attention_use_layer_norm = attention_use_layer_norm
        self.norm_type = norm_type
        if self.attention_use_layer_norm:
            if self.norm_type == "rmsnorm":
                self.layer_norm = RMSNorm(d)
                self.gated_rms_norm = RMSNorm(d)
            elif self.norm_type == "gatedrmsnorm":
                print("here! gatedrmsnorm")
                self.layer_norm = GatedRMSNorm(d)
                self.gated_rms_norm = GatedRMSNorm(d)
            elif self.norm_type == "simplermsnorm":
                print("here! simple rmsnorm")
                self.layer_norm = SimpleRMSNorm(d)
                self.gated_rms_norm = SimpleRMSNorm(d)
            else:
                self.layer_norm = nn.LayerNorm(d)
                self.gated_rms_norm = nn.LayerNorm(d)

        self.i = 0
        self.model_update_freq = model_update_freq
        self.lambda_ = lambda_

        # add begin
        self.use_relu = use_relu
        self.use_elu = use_elu
        self.use_leak = use_leak
        self.use_bound = use_bound
        self.bound = embed_dim ** -0.5
        self.causal = causal
        self.use_gelu = use_gelu
        self.mem_use_gelu = mem_use_gelu
        self.has_out = has_out
        self.mem_use_q = mem_use_q
        self.mem_use_k = mem_use_k
        self.linear_act_fun = linear_act_fun
        self.local_act_fun = local_act_fun
        self.out_use_act = out_use_act
        self.init_type = init_type
        self.seq_dropout = seq_dropout
        self.seq_p = seq_p
        self.use_v = use_v
        self.negative_slope = negative_slope
        self.chunk_size = chunk_size

        # urpe
        self.core_matrix = core_matrix
        self.p_matrix = p_matrix
        self.max_positions = max_positions
        self.use_urpe = use_urpe
        self.theta_learned = theta_learned
        self.householder_learned = householder_learned
        if self.use_urpe:
            self.urpe1 = UrpeV2(self.core_matrix, self.p_matrix, embedding_dim=self.head_dim // 2, theta_type=theta_type, theta_learned=theta_learned, householder_learned=householder_learned)
            self.urpe2 = UrpeV2(self.core_matrix, self.p_matrix, embedding_dim=self.head_dim // 2, theta_type=theta_type, theta_learned=theta_learned, householder_learned=householder_learned)
            # self.urpe = Urpe(self.core_matrix, self.p_matrix, embedding_dim=self.head_dim, theta_type=theta_type, theta_learned=theta_learned, householder_learned=householder_learned)

        self.linear_act = self.get_act_fun(self.linear_act_fun)
        self.local_act = self.get_act_fun(self.local_act_fun)
        self.forward_type = forward_type

        print(f"causal {self.causal}")
        print(f"has_out {self.has_out}")
        print(f"attention_use_layer_norm {self.attention_use_layer_norm}")
        print(f"num_heads {self.num_heads}")
        print(f"linear_act_fun: {self.linear_act_fun}")
        print(f"local_act_fun: {self.local_act_fun}")
        print(f"norm_type {self.norm_type}")
        print(f"init_type {self.init_type}")
        print(f"use_urpe {self.use_urpe}")
        print(f"chunk_size {self.chunk_size}")
        print(f"self.forward_type {self.forward_type}")

        if self.init_type == "gelu":
            self.gelu_reset()
        elif self.init_type == "default":
            self.reset_parameters()

    def get_act_fun(self, act_fun):
        if act_fun == "gelu":
            return F.gelu
        elif act_fun == "relu":
            return F.relu
        elif act_fun == "elu":
            return F.elu
        elif act_fun == "sigmoid":
            return F.sigmoid
        elif act_fun == "exp":
            return torch.exp
        elif act_fun == "leak":
            def f(x):
                return F.leaky_relu(x, negative_slope=self.negative_slope)
            return f
        elif act_fun == "1+elu":
            def f(x):
                return 1 + F.elu(x)
            return f
        else:
            return None

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        print("normal init")
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj_linear.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj_linear.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj_linear.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.k_proj_local.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj_local.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj_local.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj_linear.weight)
            nn.init.xavier_uniform_(self.q_proj_linear.weight)
            nn.init.xavier_uniform_(self.v_proj_linear.weight)
            nn.init.xavier_uniform_(self.k_proj_local.weight)
            nn.init.xavier_uniform_(self.q_proj_local.weight)
            nn.init.xavier_uniform_(self.v_proj_local.weight)

        if self.has_out:
            nn.init.xavier_uniform_(self.out_proj_linear.weight)
            nn.init.xavier_uniform_(self.out_proj_local.weight)
            if self.out_proj_linear.bias is not None:
                nn.init.constant_(self.out_proj_linear.bias, 0.0)
            if self.out_proj_local.bias is not None:
                nn.init.constant_(self.out_proj_local.bias, 0.0)

            if self.out_proj_linear.bias is not None:
                nn.init.constant_(self.out_proj_linear.bias, 0.0)
            if self.out_proj_local.bias is not None:
                nn.init.constant_(self.out_proj_local.bias, 0.0)

    def gelu_reset(self):
        print("use gelu init")
        # std gelu
        c = 0.5874
        d1, d2 = self.k_proj.weight.shape
        nn.init.normal_(self.k_proj.weight, std=c * np.sqrt(2 / (d1 + d2)))
        d1, d2 = self.q_proj.weight.shape
        nn.init.normal_(self.q_proj.weight, std=c * np.sqrt(2 / (d1 + d2)))
        d1, d2 = self.out_proj.weight.shape
        nn.init.normal_(self.out_proj.weight, std=np.sqrt(2 / (d1 + d2)))

    def fft_coef(self, k):
        return (1 - ((-1) ** k) * np.exp(-1)) / (1 + (np.pi * k) ** 2)

    def get_weight(self, max_l):
        # cosformer
        if (self.weight_type == 1):
            a = np.pi / 2
            index = a * torch.arange(1, max_l + 1).reshape(1, -1, 1)

            return nn.Parameter(index, requires_grad=False)
        elif (self.weight_type == 2) or (self.weight_type == 3) or (self.weight_type == 4):
            # 1 - x^2
            index = torch.arange(1, max_l + 1).reshape(1, -1, 1)

            return nn.Parameter(index, requires_grad=False)

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
        # 并联
        if self.forward_type == 1:
            # print("a")
            o1, _ = self.forward_linear(query, key, value)
            o2, _ = self.forward_local(query, key, value)
            o = (o1 + o2) / 2
            return o, None
        elif self.forward_type == 2:
            # print("b")
            # 串联 linear + local
            o1, _ = self.forward_linear(query, key, value)
            o2, _ = self.forward_local(o1, key, value)
            return o2, _
        else:
            # print("c")
            # 串联 loal + linear
            o1, _ = self.forward_local(query, key, value)
            o2, _ = self.forward_linear(o1, key, value)
            return o2, _

        

    def forward_linear(
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
        num_heads = self.num_heads
        
        src_len = key.size(0)
        eps = 1e-4
        self.i += 1

        # L, N, E1
        q = self.q_proj_linear(query)
        # S, N, E1
        k = self.k_proj_linear(key)
        v = self.v_proj_linear(value)

        tgt_len, bsz, embed_dim = q.size()

        # N, L, H, E, batch, length, head, dim
        # N, L, e1
        head_dim = embed_dim // num_heads


        l = max(src_len, tgt_len)

        # (N * h, L, d)
        q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        q = self.linear_act(q)
        k = self.linear_act(k)

        if self.use_urpe:
            q = self.urpe1(q)
            k = self.urpe1(k)

        if self.causal:
            attn_mask = (torch.triu(torch.ones(tgt_len, tgt_len)) == 1).transpose(0, 1)
            attn_mask = attn_mask.float().masked_fill(attn_mask == 0, float('-inf')).to(q)
            
            weights = torch.bmm(q, k.transpose(1, 2))
            weights = weights.masked_fill(attn_mask==float("-inf"), 0)
            output = torch.bmm(weights, v)
        else:
            o1 = torch.matmul(k.transpose(1, 2), v)
            output = torch.bmm(q, o1)

        # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
        output = output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        # B, N, e2
        output = self.layer_norm(output)

        # L, N, e1
        output = self.out_proj_linear(output)

        return output, None

    def forward_local(
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
        # self.cnt += 1
        # if self.cnt == 10:
        #     sys.exit(0)
        num_heads = self.num_heads
        src_len = key.size(0)
        
        # L, N, E1
        q = self.q_proj_local(query)
        # S, N, E1
        k = self.k_proj_local(key)
        # S, N, E2
        v = self.v_proj_local(value)

        # scaling
        tgt_len, bsz, embed_dim = q.size()
        head_dim = embed_dim // self.num_heads
        scaling = float(head_dim) ** -0.5
        # scale
        q *= scaling


        # pad至chunk_size整数倍
        tgt_len_pad = (self.chunk_size - tgt_len % self.chunk_size) % self.chunk_size
        src_len_pad = (self.chunk_size - src_len % self.chunk_size) % self.chunk_size
        tgt_g = (tgt_len + tgt_len_pad) // self.chunk_size
        src_g = (src_len + src_len_pad) // self.chunk_size

        # 填充0
        q = F.pad(q, (0, 0, 0, 0, 0, tgt_len_pad))
        k = F.pad(k, (0, 0, 0, 0, 0, src_len_pad))
        v = F.pad(v, (0, 0, 0, 0, 0, src_len_pad))

        # N, L, H, E: batch, length, head, dim
        # N, L, E1 -> N * h, L, e1 -> N * h, g, l, e1
        q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1).contiguous().view(bsz * num_heads, -1, self.chunk_size, head_dim)
        # N, S, E1 -> N * h, S, e1 -> N * h, g, s, e1
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1).contiguous().view(bsz * num_heads, -1, self.chunk_size, head_dim)
        # N, S, E2 -> N * h, S, e2 -> N * h, g, s, e2
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1).contiguous().view(bsz * num_heads, -1, self.chunk_size, head_dim)

        if self.use_urpe:
            q = self.urpe2(q)
            k = self.urpe2(k)

        # (N * h, g, l, e1), (N * h, g, s, e1) -> (N * h, g, l, s)
        logits = torch.einsum("bgle,bgse->bgls", q, k)
        prob = self.local_act(logits)

        if self.causal:
            attn_mask = (torch.triu(torch.ones(self.chunk_size, self.chunk_size)) == 1).transpose(0, 1)
            attn_mask = attn_mask.float().masked_fill(attn_mask == 0, float('-inf')).to(q)
            prob = prob.masked_fill(attn_mask==float("-inf"), 0)
        weights = self.dropout_module(prob)

        # (N * h, g, l, s), (N * h, g, s, e2) -> (N * h, g, l, e2)
        output = torch.einsum("bgls,bgsd->bgld", weights, v)
        # (N * h, g, l, e2) -> (N * h, L, e2) -> (L, N * h, e2) -> (L, N, E2)
        output = output.contiguous().view(bsz * num_heads, tgt_len + tgt_len_pad, -1).transpose(0, 1).contiguous().view(tgt_len + tgt_len_pad, bsz, -1)[:tgt_len, ...]
        # perform RMSNorm to stabilize running
        output = self.gated_rms_norm(output)
        # outprojection
        output = self.out_proj_local(output)

        return output, prob

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

