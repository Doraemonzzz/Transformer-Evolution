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
from trev.modules import SimpleRMSNorm
from trev.modules import GatedRMSNorm
from trev.modules import RMSNorm
from trev.modules import Urpe
from trev.modules import UrpeV2
from trev.modules import Toeplizt
from einops import rearrange

@with_incremental_state
class NormLinearAttention(nn.Module):
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
        weight_type=-1,
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
        act_fun="gelu",
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
        kv_act='identity',
        # final dropout
        use_final_dropout=False,
        final_dropout=0.0,
        # Toeplizt
        use_toeplizt=False,
        type_num=-1
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
        
        self.k_proj = quant_noise(
            nn.Linear(self.kdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.q_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.v_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )

        self.attention_use_layer_norm = attention_use_layer_norm
        self.norm_type = norm_type
        if self.attention_use_layer_norm:
            if self.norm_type == "rmsnorm":
                print("here! rmsnorm")
                self.layer_norm = RMSNorm(embed_dim)
                if use_toeplizt:
                    self.toeplizt_norm = RMSNorm(embed_dim)
            elif self.norm_type == "gatedrmsnorm":
                print("here! gatedrmsnorm")
                self.layer_norm = GatedRMSNorm(embed_dim)
                if use_toeplizt:
                    self.toeplizt_norm = GatedRMSNorm(embed_dim)
            elif self.norm_type == "simplermsnorm":
                print("here! simple rmsnorm")
                self.layer_norm = SimpleRMSNorm(embed_dim)
                if use_toeplizt:
                    self.toeplizt_norm = SimpleRMSNorm(embed_dim)
            else:
                print("here! layer norm")
                self.layer_norm = nn.LayerNorm(embed_dim)
                if use_toeplizt:
                    self.toeplizt_norm = nn.LayerNorm(embed_dim)

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
        self.act_fun = act_fun
        self.out_use_act = out_use_act
        self.init_type = init_type
        self.seq_dropout = seq_dropout
        self.seq_p = seq_p
        self.use_v = use_v
        self.negative_slope = negative_slope
        self.use_dropout = use_dropout
        self.use_toeplizt = use_toeplizt
        self.type_num = type_num
        self.max_l = max_l

        if self.use_toeplizt:
            self.toeplizt = Toeplizt(self.max_l, self.type_num, self.causal)

        if self.use_dropout:
            self.dropout_module = TrevDropout(
                dropout, module_name=self.__class__.__name__
            )
        self.use_final_dropout = use_final_dropout
        if use_final_dropout:
            self.final_dropout_module = TrevDropout(
                final_dropout, module_name=self.__class__.__name__
            )

        # urpe
        self.core_matrix = core_matrix
        self.p_matrix = p_matrix
        self.max_positions = max_positions
        self.use_urpe = use_urpe
        self.theta_learned = theta_learned
        self.householder_learned = householder_learned
        if self.use_urpe:
            print("=====================================")
            self.urpe = UrpeV2(self.core_matrix, self.p_matrix, embedding_dim=self.head_dim, theta_type=theta_type, theta_learned=theta_learned, householder_learned=householder_learned)
            # self.urpe = Urpe(self.core_matrix, self.p_matrix, embedding_dim=self.head_dim, theta_type=theta_type, theta_learned=theta_learned, householder_learned=householder_learned)
            print("=====================================")

        print("=========================")
        print("qk_act")
        self.act = self.get_act_fun(self.act_fun)
        print("=========================")
        print("kv_act")
        self.kv_act = self.get_act_fun(kv_act)
        print("=========================")

        if self.has_out:
            self.out_proj = quant_noise(
                nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
            )
        self.weight_type = weight_type
        if self.weight_type == 1:
            a0 = 1 - np.exp(-1)
            a2 = 25 / 2 - 35 * np.exp(-1)
            self.b0 = 3 * a2 / 2
            self.b1 = a0 - a2 / 2
        elif self.weight_type == 2:
            # self.register_buffer("ratio", torch.sigmoid(torch.randn(1)))
            self.r = 0.5
            self.c = -2 * np.log(self.r)

        print(f"causal {self.causal}")
        print(f"has_out {self.has_out}")
        print(f"attention_use_layer_norm {self.attention_use_layer_norm}")
        print(f"num_heads {self.num_heads}")
        print(f"act_fun_type: {act_fun}")
        print(f"norm_type {self.norm_type}")
        print(f"init_type {self.init_type}")
        print(f"use_urpe {self.use_urpe}")
        print(f"use_dropout {self.use_dropout}")
        print(f"kv_act {kv_act}")
        print(f"self.weight_type {self.weight_type}")
        print(f"self.use_final_dropout {self.use_final_dropout}")
        print(f"self.final_dropout {final_dropout}")
        print(f"self.use_toeplizt {use_toeplizt}")
        print(f"self.type_num {self.type_num}")

        if self.init_type == "gelu":
            self.gelu_reset()
        elif self.init_type == "default":
            self.reset_parameters()

    def get_act_fun(self, act_fun):
        print(act_fun)
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
        elif act_fun == "silu":
            return F.silu
        elif self.act_fun == "relu2":
            def f(x):
                return torch.square(torch.relu(x))
            return f
        else:
            return lambda x: x

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        print("normal init")
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)

        if self.has_out:
            nn.init.xavier_uniform_(self.out_proj.weight)
            if self.out_proj.bias is not None:
                nn.init.constant_(self.out_proj.bias, 0.0)

            if self.out_proj.bias is not None:
                nn.init.constant_(self.out_proj.bias, 0.0)

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
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        eps = 1e-4
        self.i += 1

        # L, N, E1
        q = self.q_proj(query)
        # S, N, E1
        k = self.k_proj(key)
        v = self.v_proj(value)

        # N, L, H, E, batch, length, head, dim
        # N, L, e1
        head_dim = embed_dim // num_heads

        l = max(src_len, tgt_len)

        # (N * h, L, d)
        q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        if self.use_toeplizt:
            toeplizt_part = self.toeplizt(v)

        q = self.act(q)
        k = self.act(k)

        if self.use_urpe:
            q = self.urpe(q)
            k = self.urpe(k)

        #### for save
        # q1, k1, v1 = map(lambda t: rearrange(t, '(b h) n d -> b h n d', h=self.num_heads), [q, k, v])

        # # b h n d
        # attn_output_weights = torch.einsum('bhnd,bhmd->bhnm', q1, k1)
        # denorm = torch.sum(attn_output_weights, dim=-1, keepdim=True)
        # attn_output_weights = attn_output_weights / denorm
        # # print(attn_output_weights.shape)

        # data = attn_output_weights[0]
        # if tgt_len == 512:
        #     print(self.index)
        #     print(data.shape)
        #     print(torch.sum(attn_output_weights, dim=-1))
        #     np.save(f"./matrix/lg_softmax/l{self.index}.npy", attn_output_weights.cpu().detach().numpy())
        #### for save

        if self.weight_type == 1:
            # print("linear laplace")
            m = max(tgt_len, src_len)
            index = torch.arange(1, m + 1).reshape(1, -1, 1).to(q)
            q_index = index[:, :tgt_len, :] / m
            k_index = index[:, :src_len, :] / m
            q = torch.cat([(self.b1 + self.b0 * torch.square(q_index)) * q, - 2 * self.b0 * q_index * q, self.b0 * q], dim=-1)
            k = torch.cat([k, k_index * k, torch.square(k_index) * k], dim=-1)
        elif self.weight_type == 2:
            # print("linear guassian")
            m = max(tgt_len, src_len)
            index = torch.arange(m).reshape(1, -1, 1).to(q)
            q_index = index[:, :tgt_len, :] / m
            k_index = index[:, :src_len, :] / m
            q = (self.r ** (q_index ** 2)) * q
            k = (self.r ** (k_index ** 2)) * k
            q = torch.cat([q, self.c * q], dim=-1)
            k = torch.cat([k, k], dim=-1)

        if self.causal:
            if (attn_mask == None):
                attn_mask = (torch.triu(torch.ones(tgt_len, tgt_len)) == 1).transpose(0, 1)
                attn_mask = attn_mask.float().masked_fill(attn_mask == 0, float('-inf')).to(q)
            weights = torch.bmm(q, k.transpose(1, 2))
            weights = weights.masked_fill(attn_mask==float("-inf"), 0)
            output = torch.bmm(weights, v)
        else:
            o1 = torch.matmul(k.transpose(1, 2), v)
            output = torch.bmm(q, o1)

        if self.use_toeplizt:
            toeplizt_part = toeplizt_part.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)

        # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
        output = output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        # B, N, e2
        if self.attention_use_layer_norm:
            output = self.layer_norm(output)
            if self.use_toeplizt:
                output = output + self.toeplizt_norm(toeplizt_part)

        if self.use_dropout:
            output = self.dropout_module(output)

        # L, N, e1
        output = self.out_proj(output)
        if self.use_final_dropout:
            output = self.final_dropout_module(output)

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

