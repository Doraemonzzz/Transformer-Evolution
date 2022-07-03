# https://github.com/cpcp1998/PermuteFormer/blob/master/language_model/permute-p/__init__.py

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
from einops import rearrange

########## positional encoding
from trev.modules import Urpe, SineSPE, SPEFilter, T5RPE, RpeVanilla, rope
########## positional encoding

# memory attention
@with_incremental_state
class LinearKernelAttention(nn.Module):
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
        causal=False,
        use_urpe=True,
        kernel_type="1+elu",
        core_matrix=1, 
        p_matrix=1, 
        max_positions=512,
        theta_type="a",
        theta_learned=False, 
        householder_learned=False,
        # rope
        use_rope=False,
        use_spe=False,
        use_permutate=False,
        max_seq_len=512,
        # index
        index=0,
    ):
        super().__init__()
        self.index = index
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout_module = TrevDropout(
            dropout, module_name=self.__class__.__name__
        )

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )

        self.k_proj = quant_noise(
            nn.Linear(self.kdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.v_proj = quant_noise(
            nn.Linear(self.vdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.q_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )

        self.out_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

        self.onnx_trace = False

        # add
        self.kernel_type = kernel_type
        self.core_matrix = core_matrix
        self.p_matrix = p_matrix
        self.max_positions = max_positions
        self.causal = causal
        self.use_urpe = use_urpe
        self.theta_learned = theta_learned
        self.householder_learned = householder_learned
        self.use_rope = use_rope
        if self.use_urpe:
            self.urpe = Urpe(self.core_matrix, self.p_matrix, embedding_dim=self.head_dim, theta_type=theta_type, theta_learned=theta_learned, householder_learned=householder_learned)
        self.act = self.get_kernel_transform()
        self.use_spe = use_spe
        if self.use_spe:
            self.spe_encoder = SineSPE(num_heads=self.num_heads,          # Number of attention heads
                                       in_features=self.head_dim,       # Dimension of keys and queries
                                       num_realizations=self.head_dim,  # New dimension of keys and queries
                                       num_sines=1)          # Number of sinusoidal components
            self.spe_filter = SPEFilter(gated=True, code_shape=self.spe_encoder.code_shape)
        self.use_permutate = use_permutate
        if self.use_permutate:
            raw_permutation = self.generate_random_permutation(self.num_heads, self.kdim // self.num_heads, 0xdeadbeefdeadbeef)
            permutation = self.expand_permutation(max_seq_len, raw_permutation)
            self.register_buffer("permutation", permutation.unsqueeze(0))
            self.register_buffer("ratio", torch.sigmoid(torch.arange(self.num_heads) / self.num_heads * 3 + 2))

        print(f"kernel_type {kernel_type}")
        print(f"use_urpe {self.use_urpe}")
        print(f"use_rope {self.use_rope}")
        print(f"causal {self.causal}")
        print(f"use_spe {self.use_spe}")
        print(f"use_permutate {self.use_permutate}")

    # https://github.com/cpcp1998/PermuteFormer/blob/master/language_model/permute/__init__.py
    def generate_random_permutation(self, num_head, head_size, seed):
        rng = torch.Generator()
        rng.manual_seed(seed)
        permutate = torch.randperm(head_size, generator=rng)
        permutation = [permutate for _ in range(num_head)]
        # change to the same setting in urpe
        permutation = torch.stack(permutation, dim=0)
        return permutation

    # https://github.com/cpcp1998/PermuteFormer/blob/master/language_model/permute/__init__.py
    def expand_permutation(self, max_seq_length, permutation):
        num_head, head_size = permutation.shape
        expanded = [torch.arange(head_size).unsqueeze(0).expand(num_head, head_size)]
        for _ in range(max_seq_length - 1):
            previous = expanded[-1]
            current = previous.gather(-1, permutation)
            expanded.append(current)
        expanded = torch.stack(expanded, dim=1)
        return expanded

    def get_kernel_transform(self):
        if self.kernel_type == "1+elu":
            def f(x):
                return 1 + F.elu(x)
            return f
        elif self.kernel_type == "relu":
            return F.relu
        elif self.kernel_type == "elu":
            return F.elu

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        # add begin
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        # add end
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def get_index(self, seq_len):
        index = np.pi / 2 * torch.arange(1, seq_len + 1).reshape(1, -1, 1)

        return nn.Parameter(index, requires_grad=False)

    def get_matrix(self, l1, l2):
        i1 = torch.arange(1, l1 + 1).reshape(1, -1, 1)
        i2 = torch.arange(1, l2 + 1).reshape(1, 1, -1)
        weight = np.pi / 2 * (i1 - i2) / max(l1, l2)
        m = torch.cos(weight)

        return nn.Parameter(m, requires_grad=False)


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
        eps=1e-6,
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
        head_dim = embed_dim // num_heads

        tgt_len, bsz, embed_dim = query.size()
        # L, N, E
        q = self.q_proj(query)
        # S, N, E
        k = self.k_proj(key)
        # S, N, E
        v = self.v_proj(value)

        if self.use_spe:
            q = rearrange(q, 'l n (h d) -> l n h d', h=num_heads)
            k = rearrange(k, 'l n (h d) -> l n h d', h=num_heads)
            v = rearrange(v, 'l n (h d) -> l (n h) d', h=num_heads)
            v = rearrange(v, 'l n d -> n l d')
            q = self.act(q)
            k = self.act(k)
            pos_codes = self.spe_encoder(q.shape[:2])  # pos_codes is a tuple (qbar, kbar)
            q, k = self.spe_filter(q, k, pos_codes)
            q = rearrange(q, 'l n h d -> (n h) l d')
            k = rearrange(k, 'l n h d -> (n h) l d')
        elif self.use_permutate:
            q = rearrange(q, 'l n (h d) -> n l h d', h=num_heads)
            q = rearrange(q, 'n l h d -> n h l d')
            k = rearrange(k, 'l n (h d) -> n l h d', h=num_heads)
            k = rearrange(k, 'n l h d -> n h l d')
            q = q.gather(-1, self.permutation[:, :, :q.shape[2]].expand_as(q))
            k = k.gather(-1, self.permutation[:, :, :k.shape[2]].expand_as(k))
            # act
            q = self.act(q)
            k = self.act(k)
            q *= (self.ratio.unsqueeze(-1) ** torch.arange(q.shape[2], device=q.device).unsqueeze(0)).unsqueeze(-1)
            k *= ((1 / self.ratio).unsqueeze(-1) ** torch.arange(k.shape[2], device=k.device).unsqueeze(0)).unsqueeze(-1)
            # change shape
            q = rearrange(q, 'n h l d -> (n h) l d')
            k = rearrange(k, 'n h l d -> (n h) l d')
            v = rearrange(v, 'l n (h d) -> l (n h) d', h=num_heads)
            v = rearrange(v, 'l n d -> n l d')
        else:
            # ### for save
            # q1, k1, v1 = map(lambda t: rearrange(t, 'n b (h d) -> b h n d', h=self.num_heads), [q, k, v])
            # q1 = self.act(q1)
            # k1 = self.act(k1)

            # # b h n d
            # attn_output_weights = torch.einsum('bhnd,bhmd->bhnm', q1, k1)
            # print(torch.mean((attn_output_weights > 1.0).double()))
            # denorm = torch.sum(attn_output_weights, dim=-1, keepdim=True)
            # attn_output_weights = attn_output_weights / denorm

            # data = attn_output_weights[0]
            # if tgt_len == 512:
            #     print(self.index)
            #     print(data.shape)
            #     np.save(f"./matrix/linear/l{self.index}.npy", attn_output_weights.cpu().detach().numpy())
            # ### for save

            # N * h, L, d
            q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
            # N * h, S, d
            k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
            # N * h, S, d
            v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

            # act
            q = self.act(q)
            k = self.act(k)

        # urpe
        if self.use_urpe:
            q = self.urpe(q)
            k = self.urpe(k)
        if self.use_rope:
            q = rope(q, dim=1)
            k = rope(k, dim=1)

        if self.use_urpe and self.urpe.core_matrix == 4:
            q = torch.cat([q.real, q.imag], dim=-1)
            k = torch.cat([k.real, k.imag], dim=-1)

        if self.causal:
            if (attn_mask == None):
                attn_mask = (torch.triu(torch.ones(tgt_len, tgt_len)) == 1).transpose(0, 1)
                attn_mask = attn_mask.float().masked_fill(attn_mask == 0, float('-inf')).to(q)
            
            weights = torch.bmm(q, k.transpose(1, 2))
            if self.causal:
                weights = weights.masked_fill(attn_mask==float("-inf"), 0)
            # (N * h, L, S) -> (N * h, L, S)
            denom = torch.clamp_min(weights.sum(dim=-1, keepdim=True), eps)
            # (N * h, L, S) (N * h, L, S) -> (N * h, L, S)
            attn_weights = weights / denom
            # (N * h, L, S) (N * h, S, d) -> (N * h, L, d)
            attn_output = torch.bmm(attn_weights, v)
            # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
            attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        else:
            # (N * h, L, 2 * d) (N * h, L, d) -> (N * h, 2 * d, d)
            kv = torch.einsum('nld,nlm->ndm', k, v)
            # (N * h, L, 2 * d) (N * h, 2 * d) -> (N * h, L)
            z = 1 / torch.clamp_min(torch.einsum('nld,nd->nl', q, torch.sum(k, axis=1)), eps)
            # (N * h, L, 2 * d) (N * h, d, 2 * d) (N * h, L) -> (N * h, L, d)
            attn_output = torch.einsum('nld,ndm,nl->nlm', q, kv, z)
            # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
            attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)

        # L, N, E
        attn_output = self.out_proj(attn_output)

        return attn_output, None

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