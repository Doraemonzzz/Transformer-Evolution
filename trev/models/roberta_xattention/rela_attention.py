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
from trev.models.transformer import DEFAULT_MIN_PARAMS_TO_WRAP, TransformerEncoder
from trev.modules import LayerNorm
from trev.modules.quant_noise import quant_noise as apply_quant_noise_
from trev.modules.transformer_sentence_encoder import init_bert_params
from trev.models.roberta import RobertaEncoder, RobertaModel, base_architecture

from trev.models.xformer import ReLAEncoder

class RobertaReLAEncoder(RobertaEncoder):
    """RoBERTa encoder."""

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)

    def build_encoder(self, args, dictionary, embed_tokens):
        encoder = ReLAEncoder(args, dictionary, embed_tokens)
        encoder.apply(init_bert_params)
        return encoder

@register_model("roberta_rela")
class RobertaReLAModel(RobertaModel):
    def __init__(self, args, encoder):
        super().__init__(args, encoder)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample

        encoder = RobertaReLAEncoder(args, task.source_dictionary)
        return cls(args, encoder)

@register_model_architecture("roberta_rela", "roberta_rela_v1")
def roberta_rela_architecture_rela_v1(args):
    base_architecture(args)

@register_model_architecture("roberta_rela", "roberta_rela_relu2")
def roberta_rela_architecture_rela_relu2(args):
    base_architecture(args)
    args.act_fun = "relu2"

@register_model_architecture("roberta_rela", "roberta_rela_1+elu")
def roberta_rela_architecture_rela_1_elu(args):
    base_architecture(args)
    args.act_fun = "1+elu"

@register_model_architecture("roberta_rela", "roberta_rela_leak")
def roberta_rela_architecture_rela_leak(args):
    base_architecture(args)
    args.act_fun = "leak"

@register_model_architecture("roberta_rela", "roberta_rela_1+relu")
def roberta_rela_architecture_rela_1_relu(args):
    base_architecture(args)
    args.act_fun = "1+relu"

@register_model_architecture("roberta_rela", "roberta_rela_2+elu")
def roberta_rela_architecture_rela_2_elu(args):
    base_architecture(args)
    args.act_fun = "2+elu"

@register_model_architecture("roberta_rela", "roberta_rela_elu")
def roberta_rela_architecture_rela_elu(args):
    base_architecture(args)
    args.act_fun = "elu"

@register_model_architecture("roberta_rela", "roberta_rela_noact")
def roberta_rela_architecture_rela_noact(args):
    base_architecture(args)
    args.act_fun = "noact"