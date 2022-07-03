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

from trev.models.xformer import NormMixAttentionEncoder

class RobertaNormMixEncoder(RobertaEncoder):
    """RoBERTa encoder."""

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)

    def build_encoder(self, args, dictionary, embed_tokens):
        encoder = NormMixAttentionEncoder(args, dictionary, embed_tokens)
        encoder.apply(init_bert_params)
        return encoder

@register_model("roberta_norm_mix_attention")
class RobertaNormMixModel(RobertaModel):
    def __init__(self, args, encoder):
        super().__init__(args, encoder)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample

        encoder = RobertaNormMixEncoder(args, task.source_dictionary)
        return cls(args, encoder)

@register_model_architecture("roberta_norm_mix_attention", "roberta_norm_mix_type_1")
def roberta_base_architecture_norm_mix_type_1(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_use_urpe = False
    args.encoder_chunk_size = 32

@register_model_architecture("roberta_norm_mix_attention", "roberta_norm_mix_type_2")
def roberta_base_architecture_norm_mix_type_2(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.encoder_chunk_size = 64
