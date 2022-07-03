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

from trev.models.xformer import LinearCombinationEncoder

class RobertaLinearCombinationEncoder(RobertaEncoder):
    """RoBERTa encoder."""

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)

    def build_encoder(self, args, dictionary, embed_tokens):
        encoder = LinearCombinationEncoder(args, dictionary, embed_tokens)
        encoder.apply(init_bert_params)
        return encoder

@register_model("roberta_linear_combination")
class RobertaLinearCombinationModel(RobertaModel):
    def __init__(self, args, encoder):
        super().__init__(args, encoder)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample

        encoder = RobertaLinearCombinationEncoder(args, task.source_dictionary)
        return cls(args, encoder)

##### linear + vanilla
@register_model_architecture("roberta_linear_combination", "roberta_performer_vanilla")
def roberta_base_architecture_performer_vanilla(args):
    base_architecture(args)
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [-1 for _ in range(args.encoder_layers // 2)]
    args.approx_attn_dim = 64
    args.causal = False

@register_model_architecture("roberta_linear_combination", "roberta_vanilla_performer")
def roberta_base_architecture_vanilla_performer(args):
    base_architecture(args)
    args.encoder_attention_types = [-1 for _ in range(args.encoder_layers // 2)] + [2 for _ in range(args.encoder_layers // 2)] 
    args.approx_attn_dim = 64
    args.causal = False

@register_model_architecture("roberta_linear_combination", "roberta_1+elu_vanilla")
def roberta_base_architecture_1elu_vanilla(args):
    base_architecture(args)
    args.encoder_attention_types = [3 for _ in range(args.encoder_layers // 2)] + [-1 for _ in range(args.encoder_layers // 2)]
    args.kernel_type = "1+elu"

@register_model_architecture("roberta_linear_combination", "roberta_vanilla_1+elu")
def roberta_base_architecture_vanilla_1elu(args):
    base_architecture(args)
    args.encoder_attention_types = [-1 for _ in range(args.encoder_layers // 2)] + [3 for _ in range(args.encoder_layers // 2)]
    args.kernel_type = "1+elu"

@register_model_architecture("roberta_linear_combination", "roberta_1+elu_vanilla_no_urpe")
def roberta_base_architecture_1elu_vanilla(args):
    base_architecture(args)
    args.encoder_attention_types = [3 for _ in range(args.encoder_layers // 2)] + [-1 for _ in range(args.encoder_layers // 2)]
    args.kernel_type = "1+elu"
    args.use_urpe = False

@register_model_architecture("roberta_linear_combination", "roberta_vanilla_1+elu_no_urpe")
def roberta_base_architecture_vanilla_1elu(args):
    base_architecture(args)
    args.encoder_attention_types = [-1 for _ in range(args.encoder_layers // 2)] + [3 for _ in range(args.encoder_layers // 2)]
    args.kernel_type = "1+elu"
    args.use_urpe = False
##### linear + vanilla

##### local softmax + other linear
@register_model_architecture("roberta_linear_combination", "roberta_local_softmax_cosformer")
def roberta_base_architecture_local_cos(args):
    base_architecture(args)
    args.encoder_attention_types = [5 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    ##### add
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    ##### softmax
    args.use_softmax = True
    ##### cosformer
    args.use_relu = True

@register_model_architecture("roberta_linear_combination", "roberta_local_softmax_1+elu")
def roberta_base_architecture_local_1elu(args):
    base_architecture(args)
    args.encoder_attention_types = [5 for _ in range(args.encoder_layers // 2)] + [3 for _ in range(args.encoder_layers // 2)]
    args.kernel_type = "1+elu"
    ##### add
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    ##### softmax
    args.use_softmax = True

@register_model_architecture("roberta_linear_combination", "roberta_local_softmax_performer")
def roberta_base_architecture_local_1elu(args):
    base_architecture(args)
    args.encoder_attention_types = [5 for _ in range(args.encoder_layers // 2)] + [2 for _ in range(args.encoder_layers // 2)]
    args.kernel_type = "1+elu"
    ##### add
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    ##### softmax
    args.use_softmax = True
##### local softmax + other linear

##### local relu + other linear
@register_model_architecture("roberta_linear_combination", "roberta_local_relu_cosformer")
def roberta_base_architecture_local_cos(args):
    base_architecture(args)
    args.encoder_attention_types = [5 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    ##### add
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    ##### cosformer
    args.use_relu = True

@register_model_architecture("roberta_linear_combination", "roberta_local_relu_1+elu")
def roberta_base_architecture_local_1elu(args):
    base_architecture(args)
    args.encoder_attention_types = [5 for _ in range(args.encoder_layers // 2)] + [3 for _ in range(args.encoder_layers // 2)]
    args.kernel_type = "1+elu"
    ##### add
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"

@register_model_architecture("roberta_linear_combination", "roberta_local_relu_performer")
def roberta_base_architecture_local_1elu(args):
    base_architecture(args)
    args.encoder_attention_types = [5 for _ in range(args.encoder_layers // 2)] + [2 for _ in range(args.encoder_layers // 2)]
    args.kernel_type = "1+elu"
    ##### add
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
##### local relu + other linear