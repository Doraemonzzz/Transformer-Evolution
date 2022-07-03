# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import logging
from dataclasses import dataclass, field
from typing import Optional

from trev import options, utils
from trev.dataclass import ChoiceEnum, TrevDataclass
from trev.models import (
    TrevIncrementalDecoder,
    TrevLanguageModel,
    register_model,
    register_model_architecture,
)
logger = logging.getLogger(__name__)
from trev.models.transformer import (
    DEFAULT_MIN_PARAMS_TO_WRAP, Embedding, TransformerDecoder
)

from trev.modules import AdaptiveInput, CharacterTokenEmbedder
from omegaconf import II
from typing import Dict, List, Optional
import torch

from trev.models.transformer_lm import (
    DEFAULT_MAX_TARGET_POSITIONS, 
    TransformerLanguageModel,
    TransformerLanguageModelConfig,
    base_lm_architecture,
    transformer_lm_big,
)

from ..xformer import DoubleFusionV3Decoder

@register_model("double_fusion_v3_lm", dataclass=TransformerLanguageModelConfig)
class DoubleFusionV3LanguageModel(TransformerLanguageModel):
    def __init__(self, decoder):
        super(DoubleFusionV3LanguageModel, self).__init__(decoder)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = getattr(
                args, "tokens_per_sample", DEFAULT_MAX_TARGET_POSITIONS
            )

        if args.character_embeddings:
            embed_tokens = CharacterTokenEmbedder(
                task.source_dictionary,
                eval(args.character_filters),
                args.character_embedding_dim,
                args.decoder_embed_dim,
                args.char_embedder_highway_layers,
            )
        elif args.adaptive_input:
            embed_tokens = AdaptiveInput(
                len(task.source_dictionary),
                task.source_dictionary.pad(),
                args.decoder_input_dim,
                args.adaptive_input_factor,
                args.decoder_embed_dim,
                options.eval_str_list(args.adaptive_input_cutoff, type=int),
                args.quant_noise_pq,
                args.quant_noise_pq_block_size,
            )
        else:
            embed_tokens = cls.build_embedding(
                args, task.source_dictionary, args.decoder_input_dim
            )

        if args.tie_adaptive_weights:
            assert args.adaptive_input
            assert args.adaptive_input_factor == args.adaptive_softmax_factor
            assert (
                args.adaptive_softmax_cutoff == args.adaptive_input_cutoff
            ), "{} != {}".format(
                args.adaptive_softmax_cutoff, args.adaptive_input_cutoff
            )
            assert args.decoder_input_dim == args.decoder_output_dim

        decoder = DoubleFusionV3Decoder(
            args, task.target_dictionary, embed_tokens, no_encoder_attn=True
        )
        return cls(decoder)

##### base
@register_model_architecture("double_fusion_v3_lm", "double_fusion_v3_lm_no_norm_gelu")
def double_fusion_v3_lm_no_norm_1_elu(args):
    base_lm_architecture(args)
    args.has_out = True
    args.decoder_layers = 12
    args.attention_use_layer_norm = False
    args.linear_act_fun = "gelu"
    args.causal = True
##### base

##### norm test
@register_model_architecture("double_fusion_v3_lm", "double_fusion_v3_lm_layernorm_gelu")
def double_fusion_v3_lm_layernorm_1_elu(args):
    base_lm_architecture(args)
    args.has_out = True
    args.decoder_layers = 12
    args.attention_use_layer_norm = True
    args.norm_type = "layernorm"
    args.linear_act_fun = "gelu"
    args.causal = True

@register_model_architecture("double_fusion_v3_lm", "double_fusion_v3_lm_rmsnorm_gelu")
def double_fusion_v3_lm_rmsnorm_1_elu(args):
    base_lm_architecture(args)
    args.has_out = True
    args.decoder_layers = 12
    args.attention_use_layer_norm = True
    args.norm_type = "rmsnorm"
    args.linear_act_fun = "gelu"
    args.causal = True

@register_model_architecture("double_fusion_v3_lm", "double_fusion_v3_lm_gatedrmsnorm_gelu")
def double_fusion_v3_lm_gatedrmsnorm_1_elu(args):
    base_lm_architecture(args)
    args.has_out = True
    args.decoder_layers = 12
    args.attention_use_layer_norm = True
    args.norm_type = "gatedrmsnorm"
    args.linear_act_fun = "gelu"
    args.causal = True

@register_model_architecture("double_fusion_v3_lm", "double_fusion_v3_lm_simplermsnorm_gelu")
def double_fusion_v3_lm_simplermsnorm_1_elu(args):
    base_lm_architecture(args)
    args.has_out = True
    args.decoder_layers = 12
    args.attention_use_layer_norm = True
    args.norm_type = "simplermsnorm"
    args.linear_act_fun = "gelu"
    args.causal = True

@register_model_architecture("double_fusion_v3_lm", "double_fusion_v3_lm_scalenorm_gelu")
def double_fusion_v3_lm_scalenorm_1_elu(args):
    base_lm_architecture(args)
    args.has_out = True
    args.decoder_layers = 12
    args.attention_use_layer_norm = True
    args.norm_type = "scalenorm"
    args.linear_act_fun = "gelu"
    args.causal = True
##### norm test

##### act test
@register_model_architecture("double_fusion_v3_lm", "double_fusion_v3_lm_simplermsnorm_elu")
def double_fusion_v3_lm_simplermsnorm_elu(args):
    base_lm_architecture(args)
    args.has_out = True
    args.decoder_layers = 12
    args.attention_use_layer_norm = True
    args.norm_type = "simplermsnorm"
    args.linear_act_fun = "elu"
    args.causal = True

@register_model_architecture("double_fusion_v3_lm", "double_fusion_v3_lm_simplermsnorm_relu")
def double_fusion_v3_lm_simplermsnorm_relu(args):
    base_lm_architecture(args)
    args.has_out = True
    args.decoder_layers = 12
    args.attention_use_layer_norm = True
    args.norm_type = "simplermsnorm"
    args.linear_act_fun = "relu"
    args.causal = True

@register_model_architecture("double_fusion_v3_lm", "double_fusion_v3_lm_simplermsnorm_relu2")
def double_fusion_v3_lm_simplermsnorm_relu2(args):
    base_lm_architecture(args)
    args.has_out = True
    args.decoder_layers = 12
    args.attention_use_layer_norm = True
    args.norm_type = "simplermsnorm"
    args.linear_act_fun = "relu2"
    args.causal = True

@register_model_architecture("double_fusion_v3_lm", "double_fusion_v3_lm_simplermsnorm_sigmoid")
def double_fusion_v3_lm_simplermsnorm_sigmoid(args):
    base_lm_architecture(args)
    args.has_out = True
    args.decoder_layers = 12
    args.attention_use_layer_norm = True
    args.norm_type = "simplermsnorm"
    args.linear_act_fun = "sigmoid"
    args.causal = True
##### act test

##### urpe test
@register_model_architecture("double_fusion_v3_lm", "double_fusion_v3_lm_simplermsnorm_gelu_urpe_1d3")
def double_fusion_v3_lm_simplermsnorm_1_elu_urpe_1d3(args):
    base_lm_architecture(args)
    args.has_out = True
    args.decoder_layers = 12
    args.attention_use_layer_norm = True
    args.norm_type = "simplermsnorm"
    args.linear_act_fun = "gelu"
    args.causal = True
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True

@register_model_architecture("double_fusion_v3_lm", "double_fusion_v3_lm_simplermsnorm_gelu_urpe_1")
def double_fusion_v3_lm_simplermsnorm_1_elu_urpe_1(args):
    base_lm_architecture(args)
    args.has_out = True
    args.decoder_layers = 12
    args.attention_use_layer_norm = True
    args.norm_type = "simplermsnorm"
    args.linear_act_fun = "gelu"
    args.causal = True
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
##### urpe test

##### head test
@register_model_architecture("double_fusion_v3_lm", "double_fusion_v3_lm_simplermsnorm_gelu_single_head")
def double_fusion_v3_lm_simplermsnorm_1_elu_single_head(args):
    base_lm_architecture(args)
    args.has_out = True
    args.decoder_layers = 12
    args.attention_use_layer_norm = True
    args.norm_type = "simplermsnorm"
    args.linear_act_fun = "gelu"
    args.causal = True
    args.decoder_attention_heads = 1
##### head test

##### token shift
@register_model_architecture("double_fusion_v3_lm", "double_fusion_v3_lm_simplermsnorm_gelu_token_shift")
def double_fusion_v3_lm_simplermsnorm_1_elu_token_shift(args):
    base_lm_architecture(args)
    args.has_out = True
    args.decoder_layers = 12
    args.attention_use_layer_norm = True
    args.norm_type = "simplermsnorm"
    args.linear_act_fun = "gelu"
    args.causal = True
    args.token_shift = True
##### token shift

##### postnorm
@register_model_architecture("double_fusion_v3_lm", "double_fusion_v3_lm_simplermsnorm_gelu_postnorm")
def double_fusion_v3_lm_simplermsnorm_1_elu_postnorm(args):
    base_lm_architecture(args)
    args.has_out = True
    args.decoder_layers = 12
    args.attention_use_layer_norm = True
    args.norm_type = "simplermsnorm"
    args.linear_act_fun = "gelu"
    args.causal = True
    args.decoder_normalize_before = False
##### postnorm