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

from ..xformer import LinearKernelAttentionDecoder

@register_model("linear_urpe_lm", dataclass=TransformerLanguageModelConfig)
class LKOrLanguageModel(TransformerLanguageModel):
    def __init__(self, decoder):
        super(LKOrLanguageModel, self).__init__(decoder)

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

        decoder = LinearKernelAttentionDecoder(
            args, task.target_dictionary, embed_tokens, no_encoder_attn=True
        )
        return cls(decoder)

########## linear urpe
########## large model
@register_model_architecture("linear_urpe_lm", "1+elu_wiki")
def transformer_1_elu_wiki(args):
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
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = True
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_normalize_before = True
    args.use_gelu = True
    args.decoder_attention_heads = 1
    ##### add
    args.causal = True
    args.use_urpe = False
    args.kernel_type = "1+elu"

##### 单位阵
@register_model_architecture("linear_urpe_lm", "1+elu_1_1_wiki")
def transformer_1_elu_1_1_wiki(args):
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
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = True
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_normalize_before = True
    args.use_gelu = True
    args.decoder_attention_heads = 1
    ##### add
    args.causal = True
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 1

@register_model_architecture("linear_urpe_lm", "1+elu_1b_1_wiki")
def transformer_1_elu_1b_1_wiki(args):
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
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = True
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_normalize_before = True
    args.use_gelu = True
    args.decoder_attention_heads = 1
    ##### add
    args.causal = True
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 1
    args.theta_type = "b"

@register_model_architecture("linear_urpe_lm", "1+elu_1d_1_wiki")
def transformer_1_elu_1d_1_wiki(args):
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
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = True
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_normalize_before = True
    args.use_gelu = True
    args.decoder_attention_heads = 1
    ##### add
    args.causal = True
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 1
    args.theta_learned = True

@register_model_architecture("linear_urpe_lm", "1+elu_2_1_wiki")
def transformer_1_elu_2_1_wiki(args):
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
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = True
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_normalize_before = True
    args.use_gelu = True
    args.decoder_attention_heads = 1
    ##### add
    args.causal = True
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 2
    args.p_matrix = 1

@register_model_architecture("linear_urpe_lm", "1+elu_3_1_wiki")
def transformer_1_elu_3_1_wiki(args):
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
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = True
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_normalize_before = True
    args.use_gelu = True
    args.decoder_attention_heads = 1
    ##### add
    args.causal = True
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 3
    args.p_matrix = 1
##### 单位阵

##### Rope
@register_model_architecture("linear_urpe_lm", "1+elu_rope_wiki")
def transformer_1_elu_rope_wiki(args):
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
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = True
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_normalize_before = True
    args.use_gelu = True
    args.decoder_attention_heads = 1
    ##### add
    args.causal = True
    args.use_urpe = False
    args.use_rope = True
    args.kernel_type = "1+elu"

@register_model_architecture("linear_urpe_lm", "relu_rope_wiki")
def transformer_relu_rope_wiki(args):
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
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = True
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_normalize_before = True
    args.use_gelu = True
    args.decoder_attention_heads = 1
    ##### add
    args.causal = True
    args.use_urpe = False
    args.use_rope = True
    args.kernel_type = "relu"
##### Rope

##### Odd Even
@register_model_architecture("linear_urpe_lm", "1+elu_1_5_wiki")
def transformer_1_elu_1_5_wiki(args):
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
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = True
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_normalize_before = True
    args.use_gelu = True
    args.decoder_attention_heads = 1
    ##### add
    args.causal = True
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 5

@register_model_architecture("linear_urpe_lm", "relu_1_5_wiki")
def transformer_relu_1_5_wiki(args):
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
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = True
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_normalize_before = True
    args.use_gelu = True
    args.decoder_attention_heads = 1
    ##### add
    args.causal = True
    args.use_urpe = True
    args.kernel_type = "relu"
    args.core_matrix = 1
    args.p_matrix = 5
##### Odd Even

##### DCT
@register_model_architecture("linear_urpe_lm", "1+elu_1_2_wiki")
def transformer_1_elu_1_2_wiki(args):
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
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = True
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_normalize_before = True
    args.use_gelu = True
    args.decoder_attention_heads = 1
    ##### add
    args.causal = True
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 2

@register_model_architecture("linear_urpe_lm", "1+elu_1b_2_wiki")
def transformer_1_elu_1b_2_wiki(args):
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
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = True
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_normalize_before = True
    args.use_gelu = True
    args.decoder_attention_heads = 1
    ##### add
    args.causal = True
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 2
    args.theta_type = "b"

@register_model_architecture("linear_urpe_lm", "1+elu_2_2_wiki")
def transformer_1_elu_2_2_wiki(args):
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
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = True
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_normalize_before = True
    args.use_gelu = True
    args.decoder_attention_heads = 1
    ##### add
    args.causal = True
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 2
    args.p_matrix = 2

@register_model_architecture("linear_urpe_lm", "1+elu_3_2_wiki")
def transformer_1_elu_3_2_wiki(args):
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
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = True
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_normalize_before = True
    args.use_gelu = True
    args.decoder_attention_heads = 1
    ##### add
    args.causal = True
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 3
    args.p_matrix = 2
##### DCT

##### Householder
@register_model_architecture("linear_urpe_lm", "1+elu_1_3_wiki")
def transformer_1_elu_1_3_wiki(args):
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
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = True
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_normalize_before = True
    args.use_gelu = True
    args.decoder_attention_heads = 1
    ##### add
    args.causal = True
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 3

@register_model_architecture("linear_urpe_lm", "1+elu_2_3_wiki")
def transformer_1_elu_2_3_wiki(args):
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
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = True
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_normalize_before = True
    args.use_gelu = True
    args.decoder_attention_heads = 1
    ##### add
    args.causal = True
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 2
    args.p_matrix = 3

@register_model_architecture("linear_urpe_lm", "1+elu_3_3_wiki")
def transformer_1_elu_3_3_wiki(args):
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
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = True
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_normalize_before = True
    args.use_gelu = True
    args.decoder_attention_heads = 1
    ##### add
    args.causal = True
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 3
    args.p_matrix = 3
##### Householder

##### Householder learned
@register_model_architecture("linear_urpe_lm", "1+elu_1_3a_wiki")
def transformer_1_elu_1_3a_wiki(args):
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
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = True
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_normalize_before = True
    args.use_gelu = True
    args.decoder_attention_heads = 1
    ##### add
    args.causal = True
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 3
    args.householder_learned = True
##### Householder learned
########## large model

########## small model
@register_model_architecture("linear_urpe_lm", "1+elu_wiki_base")
def transformer_1_elu_wiki_base(args):
    base_lm_architecture(args)
    ##### add
    args.causal = True
    args.use_urpe = False
    args.kernel_type = "1+elu"

######## Identity
@register_model_architecture("linear_urpe_lm", "1+elu_1_1_wiki_base")
def transformer_1_elu_1_1_wiki_base(args):
    base_lm_architecture(args)
    args.causal = True
    ##### add
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 1

@register_model_architecture("linear_urpe_lm", "1+elu_1b_1_wiki_base")
def transformer_1_elu_1b_1_wiki_base(args):
    base_lm_architecture(args)
    args.causal = True
    ##### add
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 1
    args.theta_type = "b"

@register_model_architecture("linear_urpe_lm", "1+elu_1c_1_wiki_base")
def transformer_1_elu_1c_1_wiki_base(args):
    base_lm_architecture(args)
    args.causal = True
    ##### add
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 1
    args.theta_type = "c"

@register_model_architecture("linear_urpe_lm", "1+elu_1d_1_wiki_base")
def transformer_1_elu_1d_1_wiki_base(args):
    base_lm_architecture(args)
    args.causal = True
    ##### add
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 1
    args.theta_learned = True

@register_model_architecture("linear_urpe_lm", "1+elu_2_1_wiki_base")
def transformer_1_elu_2_1_wiki_base(args):
    base_lm_architecture(args)
    args.causal = True
    ##### add
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 2
    args.p_matrix = 1

@register_model_architecture("linear_urpe_lm", "1+elu_3_1_wiki_base")
def transformer_1_elu_3_1_wiki_base(args):
    base_lm_architecture(args)
    args.causal = True
    ##### add
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 3
    args.p_matrix = 1

######## Identity


######## DCT
@register_model_architecture("linear_urpe_lm", "1+elu_1_2_wiki_base")
def transformer_1_elu_1_2_wiki_base(args):
    base_lm_architecture(args)
    args.causal = True
    ##### add
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 2

@register_model_architecture("linear_urpe_lm", "1+elu_1d_2_wiki_base")
def transformer_1_elu_1d_2_wiki_base(args):
    base_lm_architecture(args)
    args.causal = True
    ##### add
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 2
    args.theta_learned = True

@register_model_architecture("linear_urpe_lm", "1+elu_2_2_wiki_base")
def transformer_1_elu_2_2_wiki_base(args):
    base_lm_architecture(args)
    args.causal = True
    ##### add
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 2
    args.p_matrix = 2

@register_model_architecture("linear_urpe_lm", "1+elu_3_2_wiki_base")
def transformer_1_elu_3_2_wiki_base(args):
    base_lm_architecture(args)
    args.causal = True
    ##### add
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 3
    args.p_matrix = 2

######## DCT


######## Householder
@register_model_architecture("linear_urpe_lm", "1+elu_1_3_wiki_base")
def transformer_1_elu_1_3_wiki_base(args):
    base_lm_architecture(args)
    args.causal = True
    ##### add
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 3

@register_model_architecture("linear_urpe_lm", "1+elu_1d_3_wiki_base")
def transformer_1_elu_1d_3_wiki_base(args):
    base_lm_architecture(args)
    args.causal = True
    ##### add
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 3
    args.theta_learned = True

@register_model_architecture("linear_urpe_lm", "1+elu_2_3_wiki_base")
def transformer_1_elu_2_3_wiki_base(args):
    base_lm_architecture(args)
    args.causal = True
    ##### add
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 2
    args.p_matrix = 3

@register_model_architecture("linear_urpe_lm", "1+elu_3_3_wiki_base")
def transformer_1_elu_3_3_wiki_base(args):
    base_lm_architecture(args)
    args.causal = True
    ##### add
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 3
    args.p_matrix = 3

@register_model_architecture("linear_urpe_lm", "1+elu_1d_3a_wiki_base")
def transformer_1_elu_1d_3a_wiki_base(args):
    base_lm_architecture(args)
    args.causal = True
    ##### add
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.theta_learned = True
    args.p_matrix = 3
    args.householder_learned = True
######## Householder

######## Fourier
@register_model_architecture("linear_urpe_lm", "1+elu_4_4_wiki_base")
def transformer_1_elu_4_4_wiki_base(args):
    base_lm_architecture(args)
    args.causal = True
    ##### add
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 4
    args.p_matrix = 4

@register_model_architecture("linear_urpe_lm", "1+elu_4d_4_wiki_base")
def transformer_1_elu_4d_4_wiki_base(args):
    base_lm_architecture(args)
    args.causal = True
    ##### add
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 4
    args.p_matrix = 4
    args.theta_learned = True
######## Fourier

######## Odd Even
@register_model_architecture("linear_urpe_lm", "1+elu_1_5_wiki_base")
def transformer_1_elu_1_5_wiki_base(args):
    base_lm_architecture(args)
    args.causal = True
    ##### add
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 5

@register_model_architecture("linear_urpe_lm", "1+elu_1d_5_wiki_base")
def transformer_1_elu_1d_5_wiki_base(args):
    base_lm_architecture(args)
    args.causal = True
    ##### add
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 5
    args.theta_learned = True

@register_model_architecture("linear_urpe_lm", "1+elu_2_5_wiki_base")
def transformer_1_elu_2_5_wiki_base(args):
    base_lm_architecture(args)
    args.causal = True
    ##### add
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 2
    args.p_matrix = 5

@register_model_architecture("linear_urpe_lm", "1+elu_3_5_wiki_base")
def transformer_1_elu_3_5_wiki_base(args):
    base_lm_architecture(args)
    args.causal = True
    ##### add
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 3
    args.p_matrix = 5

######## Odd Even

######## abl
@register_model_architecture("linear_urpe_lm", "1+elu_spe_wiki_base")
def transformer_1_elu_spe_wiki_base(args):
    base_lm_architecture(args)
    args.causal = True
    ##### add
    args.use_urpe = False
    args.use_spe = True

@register_model_architecture("linear_urpe_lm", "1+elu_per_wiki_base")
def transformer_1_elu_per_wiki_base(args):
    base_lm_architecture(args)
    args.causal = True
    ##### add
    args.use_urpe = False
    args.use_spe = False
    args.use_permutate = True
######## abl

######## only rel
@register_model_architecture("linear_urpe_lm", "1+elu_1d_3_wiki_base_no_abs")
def transformer_1_elu_1d_3_wiki_base_no_abs(args):
    base_lm_architecture(args)
    args.causal = True
    ##### add
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 3
    args.theta_learned = True
    ##### add
    args.no_token_positional_embeddings = True

@register_model_architecture("linear_urpe_lm", "1+elu_1_1_wiki_base_no_abs")
def transformer_1_elu_1_1_wiki_base_no_abs(args):
    base_lm_architecture(args)
    args.causal = True
    ##### add
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 1
    ##### add
    args.no_token_positional_embeddings = True
########## small model