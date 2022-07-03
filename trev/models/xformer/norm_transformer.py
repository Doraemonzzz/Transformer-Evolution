# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from trev import utils
from trev.distributed import fsdp_wrap
from trev.models import (
    TrevEncoder,
    TrevEncoderDecoderModel,
    TrevIncrementalDecoder,
    register_model,
    register_model_architecture,
)

from trev.modules import AdaptiveInput, CharacterTokenEmbedder
from omegaconf import II
from typing import Dict, List, Optional
import torch
from trev.modules.checkpoint_activations import checkpoint_wrapper
from trev.modules.quant_noise import quant_noise as apply_quant_noise_
from torch import Tensor

from trev.models.transformer import (
    TransformerDecoder, 
    TransformerEncoder, 
    TransformerModel, 
    base_architecture,
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS,
    DEFAULT_MIN_PARAMS_TO_WRAP,
)

from trev.modules import NormAttentionEncoderLayer, NormAttentionDecoderLayer

class NormAttentionEncoder(TransformerEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~trev.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)

    def build_encoder_layer(self, args):
        layer = NormAttentionEncoderLayer(args)
        checkpoint = getattr(args, "checkpoint_activations", False)
        if checkpoint:
            offload_to_cpu = getattr(args, "offload_activations", False)
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = (
            getattr(args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP)
            if not checkpoint else 0
        )
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

class NormAttentionDecoder(TransformerDecoder):
    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
    ):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn, output_projection)

    def build_decoder_layer(self, args, no_encoder_attn=False):
        layer = NormAttentionDecoderLayer(args, no_encoder_attn)
        checkpoint = getattr(args, "checkpoint_activations", False)
        if checkpoint:
            offload_to_cpu = getattr(args, "offload_activations", False)
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = (
            getattr(args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP)
            if not checkpoint else 0
        )
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    def build_encoder_layer(self, args):
        layer = NormAttentionEncoderLayer(args)
        checkpoint = getattr(args, "checkpoint_activations", False)
        if checkpoint:
            offload_to_cpu = getattr(args, "offload_activations", False)
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = (
            getattr(args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP)
            if not checkpoint else 0
        )
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

@register_model("normattention")
class TransformerNormModel(TransformerModel):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: trev.models.transformer_parser
        :prog:
    """

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return NormAttentionEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return NormAttentionDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )

@register_model("normattention_only_encoder")
class TransformerNormOnlyEncoderModel(TransformerModel):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: trev.models.transformer_parser
        :prog:
    """

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return NormAttentionEncoder(args, src_dict, embed_tokens)

@register_model_architecture("normattention", "vanilla_wmt_en_de_norm_glu_de_linear")
def transformer_vanilla_wmt_en_de_norm_glu_de_linear(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_use_urpe = False
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.decoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    args.decoder_attention_types = [1 for _ in range(args.decoder_layers)]
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "layernorm"
    args.norm_type = "layernorm"

@register_model_architecture("normattention", "vanilla_wmt_en_de_norm_glu_de_local")
def transformer_vanilla_wmt_en_de_norm_glu_de_local(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_use_urpe = False
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.decoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers)]
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "layernorm"
    args.norm_type = "layernorm"

@register_model_architecture("normattention", "vanilla_wmt_en_de_norm_glu")
def transformer_vanilla_wmt_en_de_norm_glu(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_use_urpe = False
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.decoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "layernorm"
    args.norm_type = "layernorm"

@register_model_architecture("normattention", "vanilla_wmt_en_de_norm_glu_small")
def transformer_vanilla_wmt_en_de_norm_glu_small(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_use_urpe = False
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.decoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "layernorm"
    args.norm_type = "layernorm"
    args.multiple = 2

@register_model_architecture("normattention", "vanilla_wmt_en_de_norm_ffn")
def transformer_vanilla_wmt_en_de_norm_ffn(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_use_urpe = False
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.decoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ##### glu
    args.local_norm_type = "layernorm"
    args.norm_type = "layernorm"

########## only encoder norm
@register_model_architecture("normattention_only_encoder", "vanilla_wmt_en_de_norm_glu_small_only_encoder")
def transformer_vanilla_wmt_en_de_norm_glu_small_only_encoder(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_use_urpe = False
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.decoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "layernorm"
    args.norm_type = "layernorm"
    args.multiple = 2