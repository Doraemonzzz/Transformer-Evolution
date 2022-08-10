import torch
import torch.nn as nn
import logging

from torch import Tensor
from typing import Dict, List, Optional

from trev import utils
from trev.modules.trev_dropout import TrevDropout
from trev.modules.quant_noise import quant_noise
from trev.modules import VanillaFeedForward
from .vanilla_attention import VanillaAttention
from ..utils import get_norm

class VanillaTransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embed_dim = args.encoder_embed_dim
        self.attn = self.build_self_attention(self.embed_dim, args)
        self.norm_type = getattr(args, "norm_type", "layernorm")
        self.attn_norm = get_norm(self.norm_type, self.embed_dim)
        self.ffn = VanillaFeedForward(
            embed_dim=self.embed_dim,
            hidden_dim=args.encoder_ffn_embed_dim,
            act_dropout=getattr(args, "activation_dropout", 0) or 0,
            final_dropout=args.dropout,
            activation=getattr(args, 'activation_fn', 'relu') or "relu",
        )
        self.ffn_norm = get_norm(self.norm_type, self.embed_dim)

        self.pre_norm = args.encoder_normalize_before
        if self.pre_norm:
            self.forward = self.forward_pre_norm
        else:
            self.forward = self.forward_post_norm

        logging.info(f"norm_type {self.norm_type}")

    def build_self_attention(self, embed_dim, args):
        kwargs = getattr(args, "kwargs", {})
        return VanillaAttention(
            embed_dim=embed_dim,
            num_heads=args.encoder_attention_heads,
            causal=False,
            dropout=args.attention_dropout,
            index=args.index,
            init_method=getattr(args, "init_method", "default"),
            **kwargs,
        )

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...attn_norm.weight` and `...layer_norms.1.weight` to
        `...ffn_norm.weight`
        """
        layer_norm_map = {"0": "attn_norm", "1": "ffn_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def forward_pre_norm(self, x, mask=None):
        # attention
        x = self.attn_norm(x)
        x = x + self.attn(x, x, mask)
        # ffn
        x = self.ffn_norm(x)
        x = x + self.ffn(x)

        return x

    def forward_post_norm(self, x, mask=None):
        # attention
        x = x + self.attn(x, x, mask)
        x = self.attn_norm(x)
        # ffn
        x = x + self.ffn(x)
        x = self.ffn_norm(x)

        return x

class VanillaTransformerDecoderLayer(nn.Module):
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
        self.embed_dim = args.decoder_embed_dim
        self.attn = self.build_self_attention(self.embed_dim, args)
        self.norm_type = getattr(args, "norm_type", "layernorm")
        self.attn_norm = get_norm(self.norm_type, self.embed_dim)
        self.ffn = VanillaFeedForward(
            embed_dim=self.embed_dim,
            hidden_dim=args.decoder_ffn_embed_dim,
            act_dropout=getattr(args, "activation_dropout", 0) or 0,
            final_dropout=args.dropout,
            activation=getattr(args, 'activation_fn', 'relu') or "relu",
        )
        self.ffn_norm = get_norm(self.norm_type, self.embed_dim)

        self.pre_norm = args.decoder_normalize_before
        if self.pre_norm:
            self.forward = self.forward_pre_norm
        else:
            self.forward = self.forward_post_norm

        logging.info(f"norm_type {self.norm_type}")

    def build_self_attention(
        self, embed_dim, args,
    ):
        kwargs = getattr(args, "kwargs", {})
        return VanillaAttention(
            embed_dim=embed_dim,
            num_heads=args.decoder_attention_heads,
            causal=True,
            dropout=args.attention_dropout,
            index=args.index,
            init_method=getattr(args, "init_method", "default"),
            **kwargs,
        )

    def forward_pre_norm(self, x, mask=None):
        # attention
        x = self.attn_norm(x)
        x = x + self.attn(x, x, mask)
        # ffn
        x = self.ffn_norm(x)
        x = x + self.ffn(x)

        return x

    def forward_post_norm(self, x, mask=None):
        # attention
        x = x + self.attn(x, x, mask)
        x = self.attn_norm(x)
        # ffn
        x = x + self.ffn(x)
        x = self.ffn_norm(x)

        return x

    def make_generation_fast_(self, need_attn: bool = False, **kwargs):
        self.need_attn = need_attn
