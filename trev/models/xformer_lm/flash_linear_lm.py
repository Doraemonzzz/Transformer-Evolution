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

from ..xformer import FlashLinearDecoder

@register_model("flash_linear_lm", dataclass=TransformerLanguageModelConfig)
class FlashLinearLanguageModel(TransformerLanguageModel):
    def __init__(self, decoder):
        super(FlashLinearLanguageModel, self).__init__(decoder)

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

        decoder = FlashLinearDecoder(
            args, task.target_dictionary, embed_tokens, no_encoder_attn=True
        )
        return cls(decoder)

@register_model_architecture("flash_linear_lm", "flash_linear_wiki_ada_v1")
def flash_linear_wiki_ada_v1(args):
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
    args.decoder_layers = 32
    args.s = 128
    args.norm_type = "scale_norm"
    args.eps = 1e-5
    args.max_position_embeddings = 512
    args.expansion_factor = 2
    args.chunk_size = 64

@register_model_architecture("flash_linear_lm", "flash_linear_wiki_one_head")
def flash_linear_wiki_one_head(args):
    transformer_lm_big(args)
    args.decoder_layers = 10
    args.s = 128
    # args.s = 512
    args.norm_type = "scale_norm"
    args.eps = 1e-5
    args.max_position_embeddings = 512
    args.expansion_factor = 2
    args.chunk_size = 64
    args.decoder_attention_types = []
    args.decoder_attention_heads = 1

@register_model_architecture("flash_linear_lm", "flash_linear_wiki")
def flash_linear_wiki(args):
    transformer_lm_big(args)
    args.decoder_layers = 10
    args.s = 128
    # args.s = 512
    args.norm_type = "scale_norm"
    args.eps = 1e-5
    args.max_position_embeddings = 512
    args.expansion_factor = 2
    args.chunk_size = 64
    args.decoder_attention_types = []
