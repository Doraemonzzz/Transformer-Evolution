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

from trev.models.xformer import TransformerEncoderPlus

class RobertaEncoderPlus(RobertaEncoder):
    """RoBERTa encoder."""

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)

    def build_encoder(self, args, dictionary, embed_tokens):
        encoder = TransformerEncoderPlus(args, dictionary, embed_tokens)
        encoder.apply(init_bert_params)
        return encoder

@register_model("roberta_plus")
class RobertaPlusModel(RobertaModel):
    def __init__(self, args, encoder):
        super().__init__(args, encoder)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample

        encoder = RobertaEncoderPlus(args, task.source_dictionary)
        return cls(args, encoder)

@register_model_architecture("roberta_plus", "roberta_plus_base")
def roberta_plus_architecture(args):
    base_architecture(args)

##### base model
@register_model_architecture("roberta_plus", "roberta_rope")
def roberta_base_architecture_rope(args):
    base_architecture(args)
    args.weight_type = -1
    args.use_rope = True

########## urpe
##### 单位阵
@register_model_architecture("roberta_plus", "roberta_urpe_1_1")
def roberta_base_architecture_urpe_1_1(args):
    base_architecture(args)
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 1
    args.p_matrix = 1

@register_model_architecture("roberta_plus", "roberta_urpe_1b_1")
def roberta_base_architecture_urpe_1b_1(args):
    base_architecture(args)
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 1
    args.p_matrix = 1
    args.theta_type = "b"

@register_model_architecture("roberta_plus", "roberta_urpe_1c_1")
def roberta_base_architecture_urpe_1c_1(args):
    base_architecture(args)
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 1
    args.p_matrix = 1
    args.theta_type = "c"

@register_model_architecture("roberta_plus", "roberta_urpe_1d_1")
def roberta_base_architecture_urpe_1d_1(args):
    base_architecture(args)
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 1
    args.p_matrix = 1
    args.theta_learned = True
    
@register_model_architecture("roberta_plus", "roberta_urpe_2_1")
def roberta_base_architecture_urpe_2_1(args):
    base_architecture(args)
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 2
    args.p_matrix = 1

@register_model_architecture("roberta_plus", "roberta_urpe_3_1")
def roberta_base_architecture_urpe_3_1(args):
    base_architecture(args)
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 3
    args.p_matrix = 1
##### 单位阵

##### Odd_Even
@register_model_architecture("roberta_plus", "roberta_urpe_1_5")
def roberta_base_architecture_urpe_1_5(args):
    base_architecture(args)
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 1
    args.p_matrix = 5

@register_model_architecture("roberta_plus", "roberta_urpe_1d_5")
def roberta_base_architecture_urpe_1d_5(args):
    base_architecture(args)
    ##### add
    args.weight_type = -1
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 5
    args.theta_learned = True

@register_model_architecture("roberta_plus", "roberta_urpe_2_5")
def roberta_base_architecture_urpe_2_5(args):
    base_architecture(args)
    ##### add
    args.weight_type = -1
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 2
    args.p_matrix = 5

@register_model_architecture("roberta_plus", "roberta_urpe_3_5")
def roberta_base_architecture_urpe_3_5(args):
    base_architecture(args)
    ##### add
    args.weight_type = -1
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 3
    args.p_matrix = 5
##### Odd_Even

##### DCT
@register_model_architecture("roberta_plus", "roberta_urpe_1_2")
def roberta_base_architecture_urpe_1_2(args):
    base_architecture(args)
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 1
    args.p_matrix = 2

@register_model_architecture("roberta_plus", "roberta_urpe_1b_2")
def roberta_base_architecture_urpe_1b_2(args):
    base_architecture(args)
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 1
    args.p_matrix = 2
    args.theta_type = "b"

@register_model_architecture("roberta_plus", "roberta_urpe_1c_2")
def roberta_base_architecture_urpe_1c_2(args):
    base_architecture(args)
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 1
    args.p_matrix = 2
    args.theta_type = "c"

@register_model_architecture("roberta_plus", "roberta_urpe_1d_2")
def roberta_base_architecture_urpe_1d_2(args):
    base_architecture(args)
    ##### add
    args.weight_type = -1
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 2
    args.theta_learned = True

@register_model_architecture("roberta_plus", "roberta_urpe_2_2")
def roberta_base_architecture_urpe_2_2(args):
    base_architecture(args)
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 2
    args.p_matrix = 2

@register_model_architecture("roberta_plus", "roberta_urpe_3_2")
def roberta_base_architecture_urpe_3_2(args):
    base_architecture(args)
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 3
    args.p_matrix = 2
##### DCT

##### Householder
@register_model_architecture("roberta_plus", "roberta_urpe_1_3")
def roberta_base_architecture_urpe_1_3(args):
    base_architecture(args)
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 1
    args.p_matrix = 3

@register_model_architecture("roberta_plus", "roberta_urpe_1d_3")
def roberta_base_architecture_urpe_1d_3(args):
    base_architecture(args)
    ##### add
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 1
    args.p_matrix = 3
    args.theta_learned = True

@register_model_architecture("roberta_plus", "roberta_urpe_2_3")
def roberta_base_architecture_urpe_2_3(args):
    base_architecture(args)
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 2
    args.p_matrix = 3

@register_model_architecture("roberta_plus", "roberta_urpe_3_3")
def roberta_base_architecture_urpe_3_3(args):
    base_architecture(args)
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 3
    args.p_matrix = 3
##### Householder

##### Householder learned
@register_model_architecture("roberta_plus", "roberta_urpe_1_3a")
def roberta_base_architecture_urpe_1_3a(args):
    base_architecture(args)
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 1
    args.p_matrix = 3
    args.householder_learned = True

@register_model_architecture("roberta_plus", "roberta_urpe_1d_3a")
def roberta_base_architecture_urpe_1d_3a(args):
    base_architecture(args)
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 1
    args.theta_learned = True
    args.p_matrix = 3
    args.householder_learned = True
##### Householder learned

##### Fourier
@register_model_architecture("roberta_plus", "roberta_urpe_4_4")
def roberta_base_architecture_urpe_4_4(args):
    base_architecture(args)
    ##### add
    args.weight_type = -1
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 4
    args.p_matrix = 4

@register_model_architecture("roberta_plus", "roberta_urpe_4d_4")
def roberta_base_architecture_urpe_4d_4(args):
    base_architecture(args)
    ##### add
    args.weight_type = -1
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 4
    args.p_matrix = 4
    args.theta_learned = True
##### Fourier
########## urpe

##### abl
@register_model_architecture("roberta_plus", "roberta_spe")
def roberta_base_architecture_spe(args):
    base_architecture(args)
    ##### add
    args.weight_type = -1
    args.use_urpe = False
    args.use_spe = True

@register_model_architecture("roberta_plus", "roberta_per")
def roberta_base_architecture_per(args):
    base_architecture(args)
    ##### add
    args.weight_type = -1
    args.use_urpe = False
    args.use_spe = False
    args.use_permutate = True

@register_model_architecture("roberta_plus", "roberta_t5")
def roberta_base_architecture_t5(args):
    base_architecture(args)
    ##### add
    args.weight_type = -1
    args.use_urpe = False
    args.use_spe = False
    args.causal = False
    args.use_t5 = True

@register_model_architecture("roberta_plus", "roberta_rpe_vanilla")
def roberta_base_architecture_rpe_vanilla(args):
    base_architecture(args)
    ##### add
    args.weight_type = -1
    args.use_urpe = False
    args.use_spe = False
    args.causal = False
    args.use_t5 = False
    args.use_rpe_vanilla = True
##### abl

##### noabs
@register_model_architecture("roberta_plus", "roberta_urpe_1_1_no_abs")
def roberta_base_architecture_urpe_1_1_no_abs(args):
    base_architecture(args)
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 1
    args.p_matrix = 1
    ##### add
    args.no_token_positional_embeddings = True

@register_model_architecture("roberta_plus", "roberta_urpe_1d_3_no_abs")
def roberta_base_architecture_urpe_1d_3_no_abs(args):
    base_architecture(args)
    ##### add
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 1
    args.p_matrix = 3
    args.theta_learned = True
    ##### add
    args.no_token_positional_embeddings = True
##### noabs