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
from trev.models.transformer.transformer import (
    DEFAULT_MIN_PARAMS_TO_WRAP, Embedding, TransformerDecoder
)

from trev.modules import AdaptiveInput, CharacterTokenEmbedder
from omegaconf import II
from typing import Dict, List, Optional
import torch

from trev.models.transformer_lm.transformer_lm import (
    DEFAULT_MAX_TARGET_POSITIONS, 
    TransformerLanguageModel,
    TransformerLanguageModelConfig,
    base_lm_architecture,
)

from ..xformer import LinearTransformerDecoder

@register_model("linear_transformer", dataclass=TransformerLanguageModelConfig)
class LinearTransformerLanguageModel(TransformerLanguageModel):
    def __init__(self, decoder):
        super().__init__(decoder)

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

        decoder = LinearTransformerDecoder(
            args, task.target_dictionary, embed_tokens, no_encoder_attn=True
        )
        return cls(decoder)

##### act test
@register_model_architecture("linear_transformer", "linear_transformer_lm_relu")
def linear_transformer_lm_relu(args):
    base_lm_architecture(args)
    args.act_fun = "relu"

@register_model_architecture("linear_transformer", "linear_transformer_lm_relu2")
def linear_transformer_lm_relu2(args):
    base_lm_architecture(args)
    args.act_fun = "relu2"

@register_model_architecture("linear_transformer", "linear_transformer_lm_sigmoid")
def linear_transformer_lm_sigmoid(args):
    base_lm_architecture(args)
    args.act_fun = "sigmoid"
    
@register_model_architecture("linear_transformer", "linear_transformer_lm_silu")
def linear_transformer_lm_silu(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    
@register_model_architecture("linear_transformer", "linear_transformer_lm_1+elu")
def linear_transformer_lm_1_elu(args):
    base_lm_architecture(args)
    args.act_fun = "1+elu"
##### act test