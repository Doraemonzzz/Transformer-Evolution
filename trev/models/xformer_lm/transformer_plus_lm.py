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

from ..xformer import TransformerDecoderPlus

@register_model("transformer_plus_lm", dataclass=TransformerLanguageModelConfig)
class TransformerPlusLanguageModel(TransformerLanguageModel):
    def __init__(self, decoder):
        super(TransformerPlusLanguageModel, self).__init__(decoder)

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

        decoder = TransformerDecoderPlus(
            args, task.target_dictionary, embed_tokens, no_encoder_attn=True
        )
        return cls(decoder)

########## large model
##### base model
@register_model_architecture("transformer_plus_lm", "transformer_lm_cos")
def transformer_lm_cos(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)

@register_model_architecture("transformer_plus_lm", "transformer_lm_cos_type2")
def transformer_lm_cos_type2(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    args.weight_type = 2

@register_model_architecture("transformer_plus_lm", "transformer_lm_rope")
def transformer_lm_rope(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    args.weight_type = -1
    args.use_rope = True

##### urpe
##### 单位阵
@register_model_architecture("transformer_plus_lm", "transformer_lm_urpe_1_1")
def transformer_lm_urpe_1_1(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 1
    args.p_matrix = 1

@register_model_architecture("transformer_plus_lm", "transformer_lm_urpe_1b_1")
def transformer_lm_urpe_1b_1(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 1
    args.p_matrix = 1
    args.theta_type = "b"

@register_model_architecture("transformer_plus_lm", "transformer_lm_urpe_1d_1")
def transformer_lm_urpe_1d_1(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 1
    args.p_matrix = 1
    args.theta_learned = True

@register_model_architecture("transformer_plus_lm", "transformer_lm_urpe_2_1")
def transformer_lm_urpe_2_1(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 2
    args.p_matrix = 1

@register_model_architecture("transformer_plus_lm", "transformer_lm_urpe_3_1")
def transformer_lm_urpe_3_1(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 3
    args.p_matrix = 1
##### 单位阵

##### Odd Even
@register_model_architecture("transformer_plus_lm", "transformer_lm_urpe_1_5")
def transformer_lm_urpe_1_5(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 1
    args.p_matrix = 5
##### Odd Even

##### DCT
@register_model_architecture("transformer_plus_lm", "transformer_lm_urpe_1_2")
def transformer_lm_urpe_1_2(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 1
    args.p_matrix = 2

@register_model_architecture("transformer_plus_lm", "transformer_lm_urpe_1b_2")
def transformer_lm_urpe_1b_2(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 1
    args.p_matrix = 2
    args.theta_type = "b"

@register_model_architecture("transformer_plus_lm", "transformer_lm_urpe_2_2")
def transformer_lm_urpe_2_2(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 2
    args.p_matrix = 2

@register_model_architecture("transformer_plus_lm", "transformer_lm_urpe_3_2")
def transformer_lm_urpe_3_2(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 3
    args.p_matrix = 2
##### DCT

##### Householder
@register_model_architecture("transformer_plus_lm", "transformer_lm_urpe_1_3")
def transformer_lm_urpe_1_3(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 1
    args.p_matrix = 3

@register_model_architecture("transformer_plus_lm", "transformer_lm_urpe_2_3")
def transformer_lm_urpe_2_3(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 2
    args.p_matrix = 3

@register_model_architecture("transformer_plus_lm", "transformer_lm_urpe_3_3")
def transformer_lm_urpe_3_3(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 3
    args.p_matrix = 3
##### Householder

##### Householder learned
@register_model_architecture("transformer_plus_lm", "transformer_lm_urpe_1_3a")
def transformer_lm_urpe_1_3a(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 1
    args.p_matrix = 3
    args.householder_learned = True
##### Householder learned
########## large model

########## small model
##### 单位阵
@register_model_architecture("transformer_plus_lm", "transformer_lm_urpe_1_1_base")
def transformer_lm_urpe_1_1_base(args):
    base_lm_architecture(args)
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 1
    args.p_matrix = 1

@register_model_architecture("transformer_plus_lm", "transformer_lm_urpe_1b_1_base")
def transformer_lm_urpe_1b_1_base(args):
    base_lm_architecture(args)
    ##### add
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 1
    args.p_matrix = 1
    args.theta_type = "b"

@register_model_architecture("transformer_plus_lm", "transformer_lm_urpe_1c_1_base")
def transformer_lm_urpe_1c_1_base(args):
    base_lm_architecture(args)
    ##### add
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 1
    args.p_matrix = 1
    args.theta_type = "c"

@register_model_architecture("transformer_plus_lm", "transformer_lm_urpe_1d_1_base")
def transformer_lm_urpe_1d_1_base(args):
    base_lm_architecture(args)
    ##### add
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 1
    args.p_matrix = 1
    args.theta_learned = True

@register_model_architecture("transformer_plus_lm", "transformer_lm_urpe_2_1_base")
def transformer_lm_urpe_2_1_base(args):
    base_lm_architecture(args)
    ##### add
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 2
    args.p_matrix = 1

@register_model_architecture("transformer_plus_lm", "transformer_lm_urpe_3_1_base")
def transformer_lm_urpe_3_1_base(args):
    base_lm_architecture(args)
    ##### add
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 3
    args.p_matrix = 1
##### 单位阵

##### Householder
@register_model_architecture("transformer_plus_lm", "transformer_lm_urpe_1_3_base")
def transformer_lm_urpe_1_3_base(args):
    base_lm_architecture(args)
    ##### add
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 1
    args.p_matrix = 3

@register_model_architecture("transformer_plus_lm", "transformer_lm_urpe_1d_3_base")
def transformer_lm_urpe_1d_3_base(args):
    base_lm_architecture(args)
    ##### add
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 1
    args.p_matrix = 3
    args.theta_learned = True

@register_model_architecture("transformer_plus_lm", "transformer_lm_urpe_2_3_base")
def transformer_lm_urpe_2_3_base(args):
    base_lm_architecture(args)
    ##### add
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 2
    args.p_matrix = 3

@register_model_architecture("transformer_plus_lm", "transformer_lm_urpe_3_3_base")
def transformer_lm_urpe_3_3_base(args):
    base_lm_architecture(args)
    ##### add
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 3
    args.p_matrix = 3

@register_model_architecture("transformer_plus_lm", "transformer_lm_urpe_1d_3a_base")
def transformer_lm_urpe_1d_3a_base(args):
    base_lm_architecture(args)
    ##### add
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 1
    args.theta_learned = True
    args.p_matrix = 3
    args.householder_learned = True
##### Householder

###### Fourier
@register_model_architecture("transformer_plus_lm", "transformer_lm_urpe_4_4_base")
def transformer_lm_urpe_4_4_base(args):
    base_lm_architecture(args)
    ##### add
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 4
    args.p_matrix = 4

@register_model_architecture("transformer_plus_lm", "transformer_lm_urpe_4d_4_base")
def transformer_lm_urpe_4d_4_base(args):
    base_lm_architecture(args)
    ##### add
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 4
    args.p_matrix = 4
    args.theta_learned = True
###### Fourier

##### Odd Even
@register_model_architecture("transformer_plus_lm", "transformer_lm_urpe_1_5_base")
def transformer_lm_urpe_1_5_base(args):
    base_lm_architecture(args)
    ##### add
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 1
    args.p_matrix = 5

@register_model_architecture("transformer_plus_lm", "transformer_lm_urpe_1d_5_base")
def transformer_lm_urpe_1d_5_base(args):
    base_lm_architecture(args)
    ##### add
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 1
    args.p_matrix = 5
    args.theta_learned = True

@register_model_architecture("transformer_plus_lm", "transformer_lm_urpe_2_5_base")
def transformer_lm_urpe_2_5_base(args):
    base_lm_architecture(args)
    ##### add
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 2
    args.p_matrix = 5

@register_model_architecture("transformer_plus_lm", "transformer_lm_urpe_3_5_base")
def transformer_lm_urpe_3_5_base(args):
    base_lm_architecture(args)
    ##### add
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 3
    args.p_matrix = 5
##### Odd Even

##### abl
@register_model_architecture("transformer_plus_lm", "transformer_lm_spe_base")
def transformer_lm_spe_base(args):
    base_lm_architecture(args)
    ##### add
    args.weight_type = -1
    args.use_urpe = False
    args.use_spe = True

@register_model_architecture("transformer_plus_lm", "transformer_lm_per_base")
def transformer_lm_per_base(args):
    base_lm_architecture(args)
    ##### add
    args.weight_type = -1
    args.use_permutate = True
    args.use_urpe = False
    args.use_spe = False

@register_model_architecture("transformer_plus_lm", "transformer_lm_t5_base")
def transformer_lm_t5_base(args):
    base_lm_architecture(args)
    ##### add
    args.weight_type = -1
    args.use_urpe = False
    args.use_spe = False
    args.causal = True
    args.use_t5 = True

@register_model_architecture("transformer_plus_lm", "transformer_lm_rpe_vanilla_base")
def transformer_lm_rpe_vanilla_base(args):
    base_lm_architecture(args)
    ##### add
    args.weight_type = -1
    args.use_urpe = False
    args.use_spe = False
    args.causal = True
    args.use_t5 = False
    args.use_rpe_vanilla = True
##### abl

@register_model_architecture("transformer_plus_lm", "transformer_lm_urpe_1d_3_base_no_abs")
def transformer_lm_urpe_1d_3_base_no_abs(args):
    base_lm_architecture(args)
    ##### add
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 1
    args.p_matrix = 3
    args.theta_learned = True
    # add
    args.no_token_positional_embeddings = True

@register_model_architecture("transformer_plus_lm", "transformer_lm_urpe_1_1_base_no_abs")
def transformer_lm_urpe_1_1_base_no_abs(args):
    base_lm_architecture(args)
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 1
    args.p_matrix = 1
    # add
    args.no_token_positional_embeddings = True
########## small model