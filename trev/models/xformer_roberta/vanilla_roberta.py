# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
RoBERTa: A Robustly Optimized BERT Pretraining Approach.
"""

import logging
from numpy import False_

import torch
import torch.nn as nn
import torch.nn.functional as F
from trev import utils
from trev.models import (
    TrevEncoder,
    TrevEncoderModel,
    register_model,
    register_model_architecture,
)
from trev.models.transformer.transformer import DEFAULT_MIN_PARAMS_TO_WRAP, TransformerEncoder
from trev.modules.transformer_sentence_encoder import init_bert_params
from trev.models.roberta import RobertaEncoder, RobertaModel, base_architecture, roberta_tiny_architecture

from trev.models.xformer import VanillaTransformerEncoder

class RobertaEncoderVanilla(RobertaEncoder):
    """RoBERTa encoder."""

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)

    def build_encoder(self, args, dictionary, embed_tokens):
        encoder = VanillaTransformerEncoder(args, dictionary, embed_tokens)
        encoder.apply(init_bert_params)
        return encoder

@register_model("roberta_vanilla")
class RobertaModelVanilla(RobertaModel):
    def __init__(self, args, encoder):
        super().__init__(args, encoder)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample

        encoder = RobertaEncoderVanilla(args, task.source_dictionary)
        return cls(args, encoder)

@register_model_architecture("roberta_vanilla", "roberta_vanilla_tiny")
def roberta_vanilla_tiny(args):
    roberta_tiny_architecture(args)
    args.causal = False

##### norm test
@register_model_architecture("roberta_vanilla", "roberta_vanilla_tiny_simplermsnorm")
def roberta_vanilla_tiny_simplermsnorm(args):
    roberta_tiny_architecture(args)
    args.causal = False
    args.norm_type = "simplermsnorm"

@register_model_architecture("roberta_vanilla", "roberta_vanilla_tiny_rmsnorm")
def roberta_vanilla_tiny_rmsnorm(args):
    roberta_tiny_architecture(args)
    args.causal = False
    args.norm_type = "rmsnorm"

@register_model_architecture("roberta_vanilla", "roberta_vanilla_tiny_gatedrmsnorm")
def roberta_vanilla_tiny_gatedrmsnorm(args):
    roberta_tiny_architecture(args)
    args.causal = False
    args.norm_type = "gatedrmsnorm"

@register_model_architecture("roberta_vanilla", "roberta_vanilla_tiny_scalenorm")
def roberta_vanilla_tiny_scalenorm(args):
    roberta_tiny_architecture(args)
    args.causal = False
    args.norm_type = "scalenorm"
##### norm test

##### head test
@register_model_architecture("roberta_vanilla", "roberta_vanilla_tiny_one_head")
def roberta_vanilla_tiny_one_head(args):
    roberta_tiny_architecture(args)
    args.causal = False
    args.encoder_attention_heads = 1
##### head test