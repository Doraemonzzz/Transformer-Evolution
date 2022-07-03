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

from trev.modules import TransformerEncoderLayerPlus, TransformerDecoderLayerPlus

class TransformerDecoderPlus(TransformerDecoder):
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
        layer = TransformerDecoderLayerPlus(args, no_encoder_attn)
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

class TransformerEncoderPlus(TransformerEncoder):
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
        layer = TransformerEncoderLayerPlus(args)
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

@register_model("encoder_transformer_plus")
class TransfomerPlusModel(TransformerModel):
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
        return TransformerEncoderPlus(args, src_dict, embed_tokens)

@register_model_architecture("encoder_transformer_plus", "vanilla_wmt_en_de")
def transformer_vanilla_wmt_en_de(args):
    base_architecture(args)

##### Identity
@register_model_architecture("encoder_transformer_plus", "vanilla_wmt_en_de_1_1")
def transformer_vanilla_wmt_en_de_1_1(args):
    base_architecture(args)
    ##### add
    args.weight_type = -1
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 1

@register_model_architecture("encoder_transformer_plus", "vanilla_wmt_en_de_1d_1")
def transformer_vanilla_wmt_en_de_1d_1(args):
    base_architecture(args)
    ##### add
    args.weight_type = -1
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 1
    args.theta_learned = True

@register_model_architecture("encoder_transformer_plus", "vanilla_wmt_en_de_2_1")
def transformer_vanilla_wmt_en_de_2_1(args):
    base_architecture(args)
    ##### add
    args.weight_type = -1
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 2
    args.p_matrix = 1

@register_model_architecture("encoder_transformer_plus", "vanilla_wmt_en_de_3_1")
def transformer_vanilla_wmt_en_de_3_1(args):
    base_architecture(args)
    ##### add
    args.weight_type = -1
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 3
    args.p_matrix = 1
##### Identity

##### Householder
@register_model_architecture("encoder_transformer_plus", "vanilla_wmt_en_de_1_3")
def transformer_vanilla_wmt_en_de_1_3(args):
    base_architecture(args)
    ##### add
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 1
    args.p_matrix = 3

@register_model_architecture("encoder_transformer_plus", "vanilla_wmt_en_de_1d_3")
def transformer_vanilla_wmt_en_de_1d_3(args):
    base_architecture(args)
    ##### add
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 1
    args.p_matrix = 3
    args.theta_learned = True

@register_model_architecture("encoder_transformer_plus", "vanilla_wmt_en_de_2_3")
def transformer_vanilla_wmt_en_de_2_3(args):
    base_architecture(args)
    ##### add
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 2
    args.p_matrix = 3

@register_model_architecture("encoder_transformer_plus", "vanilla_wmt_en_de_3_3")
def transformer_vanilla_wmt_en_de_3_3(args):
    base_architecture(args)
    ##### add
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 3
    args.p_matrix = 3

@register_model_architecture("encoder_transformer_plus", "vanilla_wmt_en_de_1d_3a")
def transformer_vanilla_wmt_en_de_1d_3a(args):
    base_architecture(args)
    ##### add
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 1
    args.theta_learned = True
    args.p_matrix = 3
    args.householder_learned = True
##### Householder

##### Odd Even
@register_model_architecture("encoder_transformer_plus", "vanilla_wmt_en_de_1_5")
def transformer_vanilla_wmt_en_de_1_5(args):
    base_architecture(args)
    ##### add
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 1
    args.p_matrix = 5

@register_model_architecture("encoder_transformer_plus", "vanilla_wmt_en_de_1d_5")
def transformer_vanilla_wmt_en_de_1d_5(args):
    base_architecture(args)
    ##### add
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 1
    args.p_matrix = 5
    args.theta_learned = True

@register_model_architecture("encoder_transformer_plus", "vanilla_wmt_en_de_2_5")
def transformer_vanilla_wmt_en_de_2_5(args):
    base_architecture(args)
    ##### add
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 2
    args.p_matrix = 5

@register_model_architecture("encoder_transformer_plus", "vanilla_wmt_en_de_3_5")
def transformer_vanilla_wmt_en_de_3_5(args):
    base_architecture(args)
    ##### add
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 3
    args.p_matrix = 5
##### Odd Even

##### Fourier
@register_model_architecture("encoder_transformer_plus", "vanilla_wmt_en_de_4_4")
def transformer_vanilla_wmt_en_de_4_4(args):
    base_architecture(args)
    ##### add
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 4
    args.p_matrix = 4

@register_model_architecture("encoder_transformer_plus", "vanilla_wmt_en_de_4d_4")
def transformer_vanilla_wmt_en_de_4d_4(args):
    base_architecture(args)
    ##### add
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 4
    args.p_matrix = 4
    args.theta_learned = True
##### Fourier

##### abl
@register_model_architecture("encoder_transformer_plus", "vanilla_wmt_en_de_spe")
def transformer_vanilla_wmt_en_de_spe(args):
    base_architecture(args)
    ##### add
    args.weight_type = -1
    args.use_urpe = False
    args.use_spe = True

@register_model_architecture("encoder_transformer_plus", "vanilla_wmt_en_de_per")
def transformer_vanilla_wmt_en_de_per(args):
    base_architecture(args)
    ##### add
    args.weight_type = -1
    args.use_urpe = False
    args.use_spe = False
    args.use_permutate = True

@register_model_architecture("encoder_transformer_plus", "vanilla_wmt_en_de_t5")
def transformer_vanilla_wmt_en_de_t5(args):
    base_architecture(args)
    ##### add
    args.weight_type = -1
    args.use_urpe = False
    args.use_spe = False
    args.use_permutate = False
    args.causal = False
    args.use_t5 = True

@register_model_architecture("encoder_transformer_plus", "vanilla_wmt_en_de_rpe_vanilla")
def transformer_vanilla_wmt_en_de_rpe_vanilla(args):
    base_architecture(args)
    ##### add
    args.weight_type = -1
    args.use_urpe = False
    args.use_spe = False
    args.use_permutate = False
    args.causal = False
    args.use_t5 = False
    args.use_rpe_vanilla = True
##### abl

##### only rel
@register_model_architecture("encoder_transformer_plus", "vanilla_wmt_en_de_1d_3_no_abs")
def transformer_vanilla_wmt_en_de_1d_3_no_abs(args):
    base_architecture(args)
    ##### add
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 1
    args.p_matrix = 3
    args.theta_learned = True
    ##### add
    args.no_encoder_token_positional_embeddings = True

@register_model_architecture("encoder_transformer_plus", "vanilla_wmt_en_de_1_1_no_abs")
def transformer_vanilla_wmt_en_de_1_1_no_abs(args):
    base_architecture(args)
    ##### add
    args.weight_type = -1
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 1
    ##### add
    args.no_encoder_token_positional_embeddings = True

@register_model_architecture("encoder_transformer_plus", "vanilla_wmt_en_de_1d_5_no_abs")
def transformer_vanilla_wmt_en_de_1d_5_no_abs(args):
    base_architecture(args)
    ##### add
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 1
    args.p_matrix = 5
    args.theta_learned = True
    ##### add
    args.no_encoder_token_positional_embeddings = True
##### only rel