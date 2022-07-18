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

from trev.models.xformer import LinearTransformerEncoder

class RobertaEncoderLinear(RobertaEncoder):
    """RoBERTa encoder."""

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)

    def build_encoder(self, args, dictionary, embed_tokens):
        encoder = LinearTransformerEncoder(args, dictionary, embed_tokens)
        encoder.apply(init_bert_params)
        return encoder

@register_model("roberta_linear")
class RobertaModelLinear(RobertaModel):
    def __init__(self, args, encoder):
        super().__init__(args, encoder)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample

        encoder = RobertaEncoderLinear(args, task.source_dictionary)
        return cls(args, encoder)
    
##### act test
@register_model_architecture("roberta_linear", "roberta_linear_tiny_relu")
def roberta_linear_tiny_relu(args):
    roberta_tiny_architecture(args)
    args.act_fun = "relu"
    
@register_model_architecture("roberta_linear", "roberta_linear_tiny_relu2")
def roberta_linear_tiny_relu2(args):
    roberta_tiny_architecture(args)
    args.act_fun = "relu2"
    
@register_model_architecture("roberta_linear", "roberta_linear_tiny_sigmoid")
def roberta_linear_tiny_sigmoid(args):
    roberta_tiny_architecture(args)
    args.act_fun = "sigmoid"
    
@register_model_architecture("roberta_linear", "roberta_linear_tiny_silu")
def roberta_linear_tiny_silu(args):
    roberta_tiny_architecture(args)
    args.act_fun = "silu"
    
@register_model_architecture("roberta_linear", "roberta_linear_tiny_1+elu")
def roberta_linear_tiny_1_elu(args):
    roberta_tiny_architecture(args)
    args.act_fun = "1+elu"
##### act test