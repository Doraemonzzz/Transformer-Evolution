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

from trev.models.xformer import DoubleFusionEncoder

class RobertaDoubleFusionEncoder(RobertaEncoder):
    """RoBERTa encoder."""

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)

    def build_encoder(self, args, dictionary, embed_tokens):
        encoder = DoubleFusionEncoder(args, dictionary, embed_tokens)
        encoder.apply(init_bert_params)
        return encoder

@register_model("roberta_double_fusion")
class RobertaDoubleFusionModel(RobertaModel):
    def __init__(self, args, encoder):
        super().__init__(args, encoder)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample

        encoder = RobertaDoubleFusionEncoder(args, task.source_dictionary)
        return cls(args, encoder)

##### base
@register_model_architecture("roberta_double_fusion", "roberta_double_fusion_denorm_1+elu")
def roberta_double_fusion_denorm_1_elu(args):
    base_architecture(args)
    args.has_out = True
    args.encoder_layers = 24
    args.attention_use_layer_norm = False
    args.linear_act_fun = "1+elu"
##### base

##### 1+elu
@register_model_architecture("roberta_double_fusion", "roberta_double_fusion_layernorm_1+elu")
def roberta_double_fusion_layernorm_1_elu(args):
    base_architecture(args)
    args.has_out = True
    args.encoder_layers = 24
    args.attention_use_layer_norm = True
    args.norm_type = "layernorm"
    args.linear_act_fun = "1+elu"

@register_model_architecture("roberta_double_fusion", "roberta_double_fusion_rmsnorm_1+elu")
def roberta_double_fusion_rmsnorm_1_elu(args):
    base_architecture(args)
    args.has_out = True
    args.encoder_layers = 24
    args.attention_use_layer_norm = True
    args.norm_type = "rmsnorm"
    args.linear_act_fun = "1+elu"

@register_model_architecture("roberta_double_fusion", "roberta_double_fusion_gatedrmsnorm_1+elu")
def roberta_double_fusion_gatedrmsnorm_1_elu(args):
    base_architecture(args)
    args.has_out = True
    args.encoder_layers = 24
    args.attention_use_layer_norm = True
    args.norm_type = "gatedrmsnorm"
    args.linear_act_fun = "1+elu"

@register_model_architecture("roberta_double_fusion", "roberta_double_fusion_simplermsnorm_1+elu")
def roberta_double_fusion_simplermsnorm_1_elu(args):
    base_architecture(args)
    args.has_out = True
    args.encoder_layers = 24
    args.attention_use_layer_norm = True
    args.norm_type = "simplermsnorm"
    args.linear_act_fun = "1+elu"
##### 1+elu