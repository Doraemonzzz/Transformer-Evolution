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

########## v2
##### base
@register_model_architecture("gau_lm", "gau_lm_v2_softmax")
def gau_lm_v2_softmax(args):
    base_lm_architecture(args)
    args.has_out = True
    args.decoder_layers = 2 * args.decoder_layers
    args.attention_use_layer_norm = False
    args.act_fun = "silu"
    args.causal = True
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers)]

@register_model_architecture("gau_lm", "gau_lm_v2_softmax_single_head")
def gau_lm_v2_softmax_single_head(args):
    base_lm_architecture(args)
    args.has_out = True
    args.decoder_layers = 2 * args.decoder_layers
    args.attention_use_layer_norm = False
    args.act_fun = "silu"
    args.causal = True
    args.decoder_attention_heads = 1
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers)]
##### base

##### norm test
@register_model_architecture("gau_lm", "gau_lm_v2_layernorm_1+elu")
def gau_lm_v2_layernorm_1_elu(args):
    base_lm_architecture(args)
    args.has_out = True
    args.decoder_layers = 2 * args.decoder_layers
    args.attention_use_layer_norm = True
    args.norm_type = "layernorm"
    args.act_fun = "silu"
    args.causal = True
    args.norm_act = "1+elu"
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers)]

@register_model_architecture("gau_lm", "gau_lm_v2_rmsnorm_1+elu")
def gau_lm_v2_rmsnorm_1_elu(args):
    base_lm_architecture(args)
    args.has_out = True
    args.decoder_layers = 2 * args.decoder_layers
    args.attention_use_layer_norm = True
    args.norm_type = "rmsnorm"
    args.act_fun = "silu"
    args.causal = True
    args.norm_act = "1+elu"
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers)]

@register_model_architecture("gau_lm", "gau_lm_v2_gatedrmsnorm_1+elu")
def gau_lm_v2_gatedrmsnorm_1_elu(args):
    base_lm_architecture(args)
    args.has_out = True
    args.decoder_layers = 2 * args.decoder_layers
    args.attention_use_layer_norm = True
    args.norm_type = "gatedrmsnorm"
    args.act_fun = "silu"
    args.causal = True
    args.norm_act = "1+elu"
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers)]

@register_model_architecture("gau_lm", "gau_lm_v2_simplermsnorm_1+elu")
def gau_lm_v2_simplermsnorm_1_elu(args):
    base_lm_architecture(args)
    args.has_out = True
    args.decoder_layers = 2 * args.decoder_layers
    args.attention_use_layer_norm = True
    args.norm_type = "simplermsnorm"
    args.act_fun = "silu"
    args.causal = True
    args.norm_act = "1+elu"
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers)]

@register_model_architecture("gau_lm", "gau_lm_v2_scalenorm_1+elu")
def gau_lm_v2_scalenorm_1_elu(args):
    base_lm_architecture(args)
    args.has_out = True
    args.decoder_layers = 2 * args.decoder_layers
    args.attention_use_layer_norm = True
    args.norm_type = "scalenorm"
    args.act_fun = "silu"
    args.causal = True
    args.norm_act = "1+elu"
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers)]
##### norm test

##### act test
@register_model_architecture("gau_lm", "gau_lm_v2_simplermsnorm_elu")
def gau_lm_v2_simplermsnorm_elu(args):
    base_lm_architecture(args)
    args.has_out = True
    args.decoder_layers = 2 * args.decoder_layers
    args.attention_use_layer_norm = True
    args.norm_type = "simplermsnorm"
    args.act_fun = "silu"
    args.norm_act = "elu"
    args.causal = True
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers)]

@register_model_architecture("gau_lm", "gau_lm_v2_simplermsnorm_relu")
def gau_lm_v2_simplermsnorm_relu(args):
    base_lm_architecture(args)
    args.has_out = True
    args.decoder_layers = 2 * args.decoder_layers
    args.attention_use_layer_norm = True
    args.norm_type = "simplermsnorm"
    args.act_fun = "silu"
    args.norm_act = "relu"
    args.causal = True
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers)]

@register_model_architecture("gau_lm", "gau_lm_v2_simplermsnorm_relu2")
def gau_lm_v2_simplermsnorm_relu2(args):
    base_lm_architecture(args)
    args.has_out = True
    args.decoder_layers = 2 * args.decoder_layers
    args.attention_use_layer_norm = True
    args.norm_type = "simplermsnorm"
    args.act_fun = "silu"
    args.norm_act = "relu2"
    args.causal = True
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers)]

@register_model_architecture("gau_lm", "gau_lm_v2_simplermsnorm_sigmoid")
def gau_lm_v2_simplermsnorm_sigmoid(args):
    base_lm_architecture(args)
    args.has_out = True
    args.decoder_layers = 2 * args.decoder_layers
    args.attention_use_layer_norm = True
    args.norm_type = "simplermsnorm"
    args.act_fun = "silu"
    args.norm_act = "sigmoid"
    args.causal = True
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers)]
##### act test

##### urpe test
@register_model_architecture("gau_lm", "gau_lm_v2_simplermsnorm_softmax_urpe_1d3")
def gau_lm_v2_simplermsnorm_softmax_urpe_1d3(args):
    base_lm_architecture(args)
    args.has_out = True
    args.decoder_layers = 2 * args.decoder_layers
    args.attention_use_layer_norm = False
    args.norm_type = "simplermsnorm"
    args.act_fun = "silu"
    args.causal = True
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers)]
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True

@register_model_architecture("gau_lm", "gau_lm_v2_simplermsnorm_softmax_urpe_1")
def gau_lm_v2_simplermsnorm_softmax_urpe_1(args):
    base_lm_architecture(args)
    args.has_out = True
    args.decoder_layers = 2 * args.decoder_layers
    args.attention_use_layer_norm = False
    args.norm_type = "simplermsnorm"
    args.act_fun = "silu"
    args.causal = True
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers)]
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1

@register_model_architecture("gau_lm", "gau_lm_v2_simplermsnorm_softmax_urpe_1d3_one_head")
def gau_lm_v2_simplermsnorm_softmax_urpe_1d3_one_head(args):
    base_lm_architecture(args)
    args.has_out = True
    args.decoder_layers = 2 * args.decoder_layers
    args.attention_use_layer_norm = False
    args.norm_type = "simplermsnorm"
    args.act_fun = "silu"
    args.causal = True
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers)]
    args.decoder_attention_heads = 1
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True

@register_model_architecture("gau_lm", "gau_lm_v2_simplermsnorm_softmax_urpe_1_one_head")
def gau_lm_v2_simplermsnorm_softmax_urpe_1_one_head(args):
    base_lm_architecture(args)
    args.has_out = True
    args.decoder_layers = 2 * args.decoder_layers
    args.attention_use_layer_norm = False
    args.norm_type = "simplermsnorm"
    args.act_fun = "silu"
    args.causal = True
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers)]
    args.decoder_attention_heads = 1
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
##### urpe test
########## v2