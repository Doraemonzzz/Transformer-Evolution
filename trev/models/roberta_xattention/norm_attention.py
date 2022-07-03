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

from trev.models.xformer import NormAttentionEncoder

############# NormAttentionEncoder
class RobertaNormEncoder(RobertaEncoder):
    """RoBERTa encoder."""

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)

    def build_encoder(self, args, dictionary, embed_tokens):
        encoder = NormAttentionEncoder(args, dictionary, embed_tokens)
        init_method = getattr(args, 'init_method', "default")
        encoder.apply(init_bert_params)
        return encoder
    
    def _init_weights(self, module):
        print("small init")
        if isinstance(module, (nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=0.02)        

        if isinstance(module, (nn.Embedding)):
            print("here")
            print(torch.norm(module.weight.data))
            module.weight.data.normal_(mean=0.0, std=1e-5)
            print(torch.norm(module.weight.data))

        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

@register_model("roberta_norm_attention")
class RobertaNormUrpeModel(RobertaModel):
    def __init__(self, args, encoder):
        super().__init__(args, encoder)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample

        encoder = RobertaNormEncoder(args, task.source_dictionary)
        return cls(args, encoder)

# linear: attention_type = 1
# local: attention_type = 2
# local, ... , local, linear, ... ,linear
# 统一格式_2_1_w32_h1
# 1, 2, 3: 1表示linear, 2表示chunk, 3表示window, c表示window size, h表示头数
##### for test
@register_model_architecture("roberta_norm_attention", "roberta_norm_type_1")
def roberta_base_architecture_norm_type_1(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_use_urpe = False
    args.encoder_chunk_size = 32
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_2")
def roberta_base_architecture_norm_type_2(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_use_urpe = False
    args.encoder_chunk_size = 32
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers)] 
##### for test

##### pure window
@register_model_architecture("roberta_norm_attention", "roberta_norm_type_3_3_w32_h1")
def roberta_base_architecture_norm_type_3_3_w32_h1(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_use_urpe = False
    args.group_type = "window"
    args.encoder_chunk_size = 32
    args.left_window = 1
    args.right_window = 1
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers)] 

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_3_3_w64_h1")
def roberta_base_architecture_norm_type_3_3_w64_h1(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_use_urpe = False
    args.group_type = "window"
    args.encoder_chunk_size = 64
    args.left_window = 1
    args.right_window = 1
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers)] 

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_3_3_w32_h8")
def roberta_base_architecture_norm_type_3_3_w32_h8(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 8
    args.encoder_use_urpe = False
    args.group_type = "window"
    args.encoder_chunk_size = 64
    args.left_window = 1
    args.right_window = 1
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers)] 

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_3_3_w64_h8")
def roberta_base_architecture_norm_type_3_3_w64_h8(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 8
    args.encoder_use_urpe = False
    args.group_type = "window"
    args.encoder_chunk_size = 64
    args.left_window = 1
    args.right_window = 1
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers)] 
##### pure window

##### pure chunk
@register_model_architecture("roberta_norm_attention", "roberta_norm_type_2_2_w32_h1")
def roberta_base_architecture_norm_type_2_2_w32_h1(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 32
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers)] 

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_2_2_w64_h1")
def roberta_base_architecture_norm_type_2_2_w64_h1(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers)]

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_2_2_w32_h8")
def roberta_base_architecture_norm_type_2_2_w32_h8(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 8
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers)] 

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_2_2_w64_h8")
def roberta_base_architecture_norm_type_2_2_w64_h8(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 8
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers)] 
##### pure chunk

##### mix
@register_model_architecture("roberta_norm_attention", "roberta_norm_type_2_1_w64_h1")
def roberta_base_architecture_norm_type_2_1_w64_h1(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_2_1_w64_h8")
def roberta_base_architecture_norm_type_2_1_w64_h8(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 8
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_3_1_w64_h1")
def roberta_base_architecture_norm_type_3_1_w64_h1(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_use_urpe = False
    args.group_type = "window"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_3_1_w64_h8")
def roberta_base_architecture_norm_type_3_1_w64_h8(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 8
    args.encoder_use_urpe = False
    args.group_type = "window"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_2_1_w32_h1")
def roberta_base_architecture_norm_type_2_1_w32_h1(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 32
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_2_1_w32_h8")
def roberta_base_architecture_norm_type_2_1_w32_h8(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 8
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 32
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_3_1_w32_h1")
def roberta_base_architecture_norm_type_3_1_w32_h1(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_use_urpe = False
    args.group_type = "window"
    args.encoder_chunk_size = 32
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_3_1_w32_h8")
def roberta_base_architecture_norm_type_3_1_w32_h8(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 8
    args.encoder_use_urpe = False
    args.group_type = "window"
    args.encoder_chunk_size = 32
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
##### mix

##### percent test
@register_model_architecture("roberta_norm_attention", "roberta_norm_type_2_1_w64_h12_p0.25")
def roberta_base_architecture_norm_type_2_1_w64_h12_p0_25(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 64
    l = int(args.encoder_layers * 0.25)
    args.encoder_attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_2_1_w64_h12_p0.5")
def roberta_base_architecture_norm_type_2_1_w64_h12_p0_5(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 64
    l = int(args.encoder_layers * 0.5)
    args.encoder_attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_2_1_w64_h12_p0.75")
def roberta_base_architecture_norm_type_2_1_w64_h12_p0_75(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 64
    l = int(args.encoder_layers * 0.75)
    args.encoder_attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_2_1_w32_h12_p0.25")
def roberta_base_architecture_norm_type_2_1_w32_h12_p0_25(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 32
    l = int(args.encoder_layers * 0.25)
    args.encoder_attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_2_1_w32_h12_p0.5")
def roberta_base_architecture_norm_type_2_1_w32_h12_p0_5(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 32
    l = int(args.encoder_layers * 0.5)
    args.encoder_attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_2_1_w32_h12_p0.75")
def roberta_base_architecture_norm_type_2_1_w32_h12_p0_75(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 32
    l = int(args.encoder_layers * 0.75)
    args.encoder_attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]
##### percent test

##### chunk size
@register_model_architecture("roberta_norm_attention", "roberta_norm_type_2_1_w128_h12_p0.5")
def roberta_base_architecture_norm_type_2_1_w128_h12_p0_5(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 128
    l = int(args.encoder_layers * 0.5)
    args.encoder_attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_2_1_w256_h12_p0.5")
def roberta_base_architecture_norm_type_2_1_w256_h12_p0_5(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 256
    l = int(args.encoder_layers * 0.5)
    args.encoder_attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]
##### chunk size

##### Urpe
@register_model_architecture("roberta_norm_attention", "roberta_norm_type_w32_h12_13")
def roberta_base_architecture_norm_type_w32_h12_13(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 32
    l = int(args.encoder_layers * 0.5)
    args.encoder_attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]
    ##### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_w64_h12_13")
def roberta_base_architecture_norm_type_w64_h12_13(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 64
    l = int(args.encoder_layers * 0.5)
    args.encoder_attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]
    ##### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_w128_h12_13")
def roberta_base_architecture_norm_type_w128_h12_13(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 128
    l = int(args.encoder_layers * 0.5)
    args.encoder_attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]
    ##### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_w256_h12_13")
def roberta_base_architecture_norm_type_w256_h12_13(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 256
    l = int(args.encoder_layers * 0.5)
    args.encoder_attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]
    ##### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
##### Urpe

##### Pure Linear
@register_model_architecture("roberta_norm_attention", "roberta_norm_type_11")
def roberta_base_architecture_norm_type_11(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_attention_types = [1 for _ in range(args.encoder_layers)]
##### Pure Linear

##### Linear + Norm 分比测试
@register_model_architecture("roberta_norm_attention", "roberta_norm_type_1_2_w64_h12_p0.25")
def roberta_base_architecture_norm_type_1_2_w64_h12_p0_25(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 64
    l = int(args.encoder_layers * 0.25)
    args.encoder_attention_types = [1 for _ in range(l)] + [2 for _ in range(args.encoder_layers - l)]

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_1_2_w64_h12_p0.5")
def roberta_base_architecture_norm_type_1_2_w64_h12_p0_5(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 64
    l = int(args.encoder_layers * 0.5)
    args.encoder_attention_types = [1 for _ in range(l)] + [2 for _ in range(args.encoder_layers - l)]

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_1_2_w64_h12_p0.75")
def roberta_base_architecture_norm_type_1_2_w64_h12_p0_75(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 64
    l = int(args.encoder_layers * 0.75)
    args.encoder_attention_types = [1 for _ in range(l)] + [2 for _ in range(args.encoder_layers - l)]
##### Linear + Norm 分比测试

##### act测试
##### 0: elu, 1: relu, 2: silu
@register_model_architecture("roberta_norm_attention", "roberta_norm_type_stand_act21")
def roberta_base_architecture_norm_type_stand_act21(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'silu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 64
    l = int(args.encoder_layers * 0.5)
    args.encoder_attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_stand_act22")
def roberta_base_architecture_norm_type_stand_act22(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = 'silu'
    args.local_act_fun = 'silu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 64
    l = int(args.encoder_layers * 0.5)
    args.encoder_attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_stand_softmax")
def roberta_base_architecture_norm_type_stand_softmax(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.use_softmax = True
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 64
    l = int(args.encoder_layers * 0.5)
    args.encoder_attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]
##### act测试

##### dropout
@register_model_architecture("roberta_norm_attention", "roberta_norm_type_stand_drop")
def roberta_base_architecture_norm_type_stand_drop(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 64
    args.use_dropout = True
    l = int(args.encoder_layers * 0.5)
    args.encoder_attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]
##### dropout

##### norm type
##### 0: default
##### 1: simple rms norm
@register_model_architecture("roberta_norm_attention", "roberta_norm_type_stand_norm_10")
def roberta_base_architecture_norm_type_stand_norm_10(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 8
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_stand_norm_01")
def roberta_base_architecture_norm_type_stand_norm_01(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 8
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.norm_type = "simplermsnorm"

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_stand_norm_11")
def roberta_base_architecture_norm_type_stand_norm_11(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 8
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_stand_norm_11_h12")
def roberta_base_architecture_norm_type_stand_norm_11_h12(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_stand_norm_33_h12")
def roberta_base_architecture_norm_type_stand_norm_33_h12(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "layernorm"
    args.norm_type = "layernorm"
##### norm type

##### chunk从大变小
@register_model_architecture("roberta_norm_attention", "roberta_norm_type_stand_chunk_stl")
def roberta_base_architecture_norm_type_stand_chunk_stl(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = [16, 16, 32, 32, 64, 64] + [64] * 6
    l = int(args.encoder_layers * 0.5)
    args.encoder_attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]
##### chunk从大变小

##### Linear + kv act
@register_model_architecture("roberta_norm_attention", "roberta_linear_standard_11")
def roberta_base_architecture_linear_standard_11(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_attention_types = [1 for _ in range(args.encoder_layers)]
    ##### add
    args.encoder_kv_act = "sigmoid"

@register_model_architecture("roberta_norm_attention", "roberta_linear_standard_12")
def roberta_base_architecture_linear_standard_12(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_attention_types = [1 for _ in range(args.encoder_layers)]
    ##### add
    args.encoder_kv_act = "relu"

@register_model_architecture("roberta_norm_attention", "roberta_linear_standard_01")
def roberta_base_architecture_linear_standard_01(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = 'identity'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_attention_types = [1 for _ in range(args.encoder_layers)]
    ##### add
    args.encoder_kv_act = "sigmoid"

@register_model_architecture("roberta_norm_attention", "roberta_linear_standard_02")
def roberta_base_architecture_linear_standard_02(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = 'identity'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_attention_types = [1 for _ in range(args.encoder_layers)]
    ##### add
    args.encoder_kv_act = "relu"
##### Linear + kv act

##### GLU
@register_model_architecture("roberta_norm_attention", "roberta_normtype_21_glu_1")
def roberta_base_architecture_normtype_21_glu_1(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 64
    l = int(args.encoder_layers * 0.5)
    args.encoder_attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"

@register_model_architecture("roberta_norm_attention", "roberta_normtype_11_glu_1")
def roberta_base_architecture_normtype_11_glu_1(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [1 for _ in range(args.encoder_layers)]
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"

@register_model_architecture("roberta_norm_attention", "roberta_normtype_22_glu_1")
def roberta_base_architecture_normtype_22_glu_1(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers)]
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"

@register_model_architecture("roberta_norm_attention", "roberta_glu_rms_layer")
def roberta_base_architecture_glu_rms_layer(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_layernorm")
def roberta_base_architecture_glu_all_layernorm(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "layernorm"
    args.norm_type = "layernorm"

@register_model_architecture("roberta_norm_attention", "roberta_glu_dropout_rms_layer")
def roberta_base_architecture_glu_dropout_rms_layer(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.use_dropout = True
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"

@register_model_architecture("roberta_norm_attention", "roberta_glu_moreheads_rms_layer")
def roberta_base_architecture_glu_moreheads_rms_layer(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    l = int(args.encoder_layers * 0.5)
    args.encoder_attention_heads_list = [12 for _ in range(l)] + [24 for _ in range(args.encoder_layers - l)]
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer")
def roberta_base_architecture_glu_all_rms_layer(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
##### GLU

##### speed test
@register_model_architecture("roberta_norm_attention", "roberta_rms_layer_standard")
def roberta_base_architecture_rms_layer_standard(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
##### speed test

##### change glu multiple
@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_small")
def roberta_base_architecture_glu_all_rms_layer_small(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.multiple = 2

@register_model_architecture("roberta_norm_attention", "roberta_glu_rms_layer_small")
def roberta_base_architecture_glu_rms_layer_small(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.multiple = 2

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_layernorm_small")
def roberta_base_architecture_glu_all_layernorm_small(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "layernorm"
    args.norm_type = "layernorm"
    args.multiple = 2
##### change glu multiple

##### layer norm rms
@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_ln_rms")
def roberta_base_architecture_glu_all_rms_layer_ln_rms(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_small_ln_rms")
def roberta_base_architecture_glu_all_rms_layer_small_ln_rms(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.multiple = 2
    args.attn_type = "simplermsnorm"
##### layer norm rms

##### GLU + URPE
@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_ln_rms_no_urpe")
def roberta_base_architecture_glu_all_rms_layer_ln_rms_no_urpe(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.encoder_use_urpe = False

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_ln_rms_urpe_1d3")
def roberta_base_architecture_glu_all_rms_layer_ln_rms_urpe_1d3(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_small_ln_rms_urpe_1d3")
def roberta_base_architecture_glu_all_rms_layer_small_ln_rms_urpe_1d3(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.multiple = 2
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_layernorm_urpe_1d3")
def roberta_base_architecture_glu_all_layernorm_urpe_1d3(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "layernorm"
    args.norm_type = "layernorm"
    ##### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_layernorm_small_urpe_1d3")
def roberta_base_architecture_glu_all_layernorm_small_urpe_1d3(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "layernorm"
    args.norm_type = "layernorm"
    args.multiple = 2
    ##### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
##### GLU + URPE

##### GLU + FINAL ACT
@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_elu")
def roberta_base_architecture_glu_all_rms_layer_elu(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.fina_act = "elu"

##### change multiple
@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_small_elu")
def roberta_base_architecture_glu_all_rms_layer_small_elu(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.multiple = 2
    args.fina_act = "elu"

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_layernorm_elu")
def roberta_base_architecture_glu_all_layernorm_elu(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "layernorm"
    args.norm_type = "layernorm"
    args.fina_act = "elu"

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_layernorm_small_elu")
def roberta_base_architecture_glu_all_layernorm_small_elu(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "layernorm"
    args.norm_type = "layernorm"
    args.multiple = 2
    args.fina_act = "elu"

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_swish")
def roberta_base_architecture_glu_all_rms_layer_swish(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.fina_act = "swish"

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_small_swish")
def roberta_base_architecture_glu_all_rms_layer_small_swish(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.multiple = 2
    args.fina_act = "swish"

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_layernorm_swish")
def roberta_base_architecture_glu_all_layernorm_swish(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "layernorm"
    args.norm_type = "layernorm"
    args.fina_act = "swish"

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_layernorm_small_swish")
def roberta_base_architecture_glu_all_layernorm_small_swish(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "layernorm"
    args.norm_type = "layernorm"
    args.multiple = 2
    args.fina_act = "swish"
##### GLU + FINAL ACT

##### GLU + URPE + DROPOUT
@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_ln_rms_urpe_1d3_dropout02")
def roberta_base_architecture_glu_all_rms_layer_ln_rms_urpe_1d3_dropout02(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    args.glu_dropout = 0.2

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_small_ln_rms_urpe_1d3_dropout02")
def roberta_base_architecture_glu_all_rms_layer_small_ln_rms_urpe_1d3_dropout02(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.multiple = 2
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    args.glu_dropout = 0.2

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_layernorm_urpe_1d3_dropout02")
def roberta_base_architecture_glu_all_layernorm_urpe_1d3_dropout02(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "layernorm"
    args.norm_type = "layernorm"
    ##### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    args.glu_dropout = 0.2

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_layernorm_small_urpe_1d3_dropout02")
def roberta_base_architecture_glu_all_layernorm_small_urpe_1d3_dropout02(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "layernorm"
    args.norm_type = "layernorm"
    args.multiple = 2
    ##### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    args.glu_dropout = 0.2
##### GLU + URPE + DROPOUT

##### GLU + URPE + NO ABS
@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_ln_rms_urpe_1d3_no_abs")
def roberta_base_architecture_glu_all_rms_layer_ln_rms_urpe_1d3_no_abs(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    ######## no abs
    args.no_token_positional_embeddings = True

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_small_ln_rms_urpe_1d3_no_abs")
def roberta_base_architecture_glu_all_rms_layer_small_ln_rms_urpe_1d3_no_abs(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.multiple = 2
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    ######## no abs
    args.no_token_positional_embeddings = True

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_layernorm_urpe_1d3_no_abs")
def roberta_base_architecture_glu_all_layernorm_urpe_1d3_no_abs(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "layernorm"
    args.norm_type = "layernorm"
    ##### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    ######## no abs
    args.no_token_positional_embeddings = True

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_layernorm_small_urpe_1d3_no_abs")
def roberta_base_architecture_glu_all_layernorm_small_urpe_1d3_no_abs(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "layernorm"
    args.norm_type = "layernorm"
    args.multiple = 2
    ##### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    ######## no abs
    args.no_token_positional_embeddings = True
##### GLU + URPE + NO ABS

##### Heads
@register_model_architecture("roberta_norm_attention", "roberta_normtype_21_head_12_1")
def roberta_base_architecture_normtype_21_head_12_1(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 64
    l = int(args.encoder_layers * 0.5)
    args.encoder_attention_heads_list = [12 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]
    args.encoder_attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]

@register_model_architecture("roberta_norm_attention", "roberta_normtype_11_head_1_1")
def roberta_base_architecture_normtype_11_head_1_1(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_use_urpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [1 for _ in range(args.encoder_layers)]

@register_model_architecture("roberta_norm_attention", "roberta_normtype_22_head_1_1")
def roberta_base_architecture_normtype_22_head_1_1(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_use_urpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers)]

@register_model_architecture("roberta_norm_attention", "roberta_normtype_21_head_12_24")
def roberta_base_architecture_normtype_21_head_12_24(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 64
    l = int(args.encoder_layers * 0.5)
    args.encoder_attention_heads_list = [12 for _ in range(l)] + [24 for _ in range(args.encoder_layers - l)]
    args.encoder_attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]

@register_model_architecture("roberta_norm_attention", "roberta_normtype_21_head_24_24")
def roberta_base_architecture_normtype_21_head_24_24(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 64
    l = int(args.encoder_layers * 0.5)
    args.encoder_attention_heads_list = [12 for _ in range(l)] + [24 for _ in range(args.encoder_layers - l)]
    args.encoder_attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]
##### Heads

##### small init + pure rms norm + urpe
@register_model_architecture("roberta_norm_attention", "roberta_glu_pure_rms_urpe_1d3_small_init")
def roberta_base_architecture_glu_pure_rms_urpe_1d3_small_init(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    ##### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"
    args.init_method = "small_embdding"

@register_model_architecture("roberta_norm_attention", "roberta_glu_small_pure_rms_urpe_1d3_small_init")
def roberta_base_architecture_glu_small_pure_rms_urpe_1d3_small_init(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.multiple = 2
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    ##### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"
    args.init_method = "small_embdding"

@register_model_architecture("roberta_norm_attention", "roberta_glu_pure_rms_urpe_1d3_small_init_no_abs")
def roberta_base_architecture_glu_pure_rms_urpe_1d3_small_init_no_abs(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    ##### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"
    args.init_method = "small_embdding"
    # add
    args.no_token_positional_embeddings = True

@register_model_architecture("roberta_norm_attention", "roberta_glu_small_pure_rms_urpe_1d3_small_init_no_abs")
def roberta_base_architecture_glu_small_pure_rms_urpe_1d3_small_init_no_abs(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.multiple = 2
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    ##### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"
    args.init_method = "small_embdding"
    # add
    args.no_token_positional_embeddings = True
##### small init + pure rms norm + urpe

##### pure rms norm + urpe + GEGLU
@register_model_architecture("roberta_norm_attention", "roberta_glu_pure_rms_urpe_1d3_geglu")
def roberta_base_architecture_glu_pure_rms_urpe_1d3_geglu(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ##### glu
    args.use_glu = True
    args.glu_act = "gelu"
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    ##### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"

@register_model_architecture("roberta_norm_attention", "roberta_glu_small_pure_rms_urpe_1d3_geglu")
def roberta_base_architecture_glu_small_pure_rms_urpe_1d3_geglu(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ##### glu
    args.use_glu = True
    args.glu_act = "gelu"
    args.multiple = 2
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    ##### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"

@register_model_architecture("roberta_norm_attention", "roberta_glu_pure_rms_urpe_1d3_geglu_small_init")
def roberta_base_architecture_glu_pure_rms_urpe_1d3_geglu_small_init(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ##### glu
    args.use_glu = True
    args.glu_act = "gelu"
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    ##### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"
    args.init_method = "small_embdding"

@register_model_architecture("roberta_norm_attention", "roberta_glu_small_pure_rms_urpe_1d3_geglu_small_init")
def roberta_base_architecture_glu_small_pure_rms_urpe_1d3_geglu_small_init(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ##### glu
    args.use_glu = True
    args.glu_act = "gelu"
    args.multiple = 2
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    ##### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"
    args.init_method = "small_embdding"
##### pure rms norm + urpe + GEGLU

##### pure rms norm + urpe + weight
@register_model_architecture("roberta_norm_attention", "roberta_glu_pure_rms_urpe_1d3")
def roberta_base_architecture_glu_pure_rms_urpe_1d3(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    ##### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"

@register_model_architecture("roberta_norm_attention", "roberta_glu_small_pure_rms_urpe_1d3")
def roberta_base_architecture_glu_small_pure_rms_urpe_1d3(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.multiple = 2
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    ##### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"

@register_model_architecture("roberta_norm_attention", "roberta_glu_pure_rms_urpe_1d3_laplace")
def roberta_base_architecture_glu_pure_rms_urpe_1d3_laplace(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    ##### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"
    ##### weight
    args.weight_type = 1

@register_model_architecture("roberta_norm_attention", "roberta_glu_small_pure_rms_urpe_1d3_laplace")
def roberta_base_architecture_glu_small_pure_rms_urpe_1d3_laplace(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.multiple = 2
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    ##### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"
    ##### weight
    args.weight_type = 1

@register_model_architecture("roberta_norm_attention", "roberta_glu_pure_rms_urpe_1d3_gaussian")
def roberta_base_architecture_glu_pure_rms_urpe_1d3_gaussian(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    ##### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"
    ##### weight
    args.weight_type = 2

@register_model_architecture("roberta_norm_attention", "roberta_glu_small_pure_rms_urpe_1d3_gaussian")
def roberta_base_architecture_glu_small_pure_rms_urpe_1d3_gaussian(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.multiple = 2
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    ##### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"
    ##### weight
    args.weight_type = 2
##### pure rms norm + urpe + weight

##### final dropout
@register_model_architecture("roberta_norm_attention", "roberta_glu_pure_rms_urpe_1d3_final_dropout")
def roberta_base_architecture_glu_pure_rms_urpe_1d3_final_dropout(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    ##### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"
    ##### final dropout
    args.use_final_dropout = True
    args.final_dropout = 0.1

@register_model_architecture("roberta_norm_attention", "roberta_glu_small_pure_rms_urpe_1d3_final_dropout")
def roberta_base_architecture_glu_small_pure_rms_urpe_1d3_final_dropout(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.multiple = 2
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    ##### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"
    ##### final dropout
    args.use_final_dropout = True
    args.final_dropout = 0.1
##### final dropout

##### relu2
@register_model_architecture("roberta_norm_attention", "roberta_glu_pure_rms_urpe_1d3_relu2")
def roberta_base_architecture_glu_pure_rms_urpe_1d3_relu2(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "relu2"
    args.local_act_fun = "relu2"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    ##### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"

@register_model_architecture("roberta_norm_attention", "roberta_glu_small_pure_rms_urpe_1d3_relu2")
def roberta_base_architecture_glu_small_pure_rms_urpe_1d3_relu2(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "relu2"
    args.local_act_fun = "relu2"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.multiple = 2
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    ##### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"
##### relu2

##### linear chunk
@register_model_architecture("roberta_norm_attention", "roberta_glu_pure_rms_urpe_1d3_linear_chunk")
def roberta_base_architecture_glu_pure_rms_urpe_1d3_linear_chunk(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "linear_chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    ##### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"

@register_model_architecture("roberta_norm_attention", "roberta_glu_small_pure_rms_urpe_1d3_linear_chunk")
def roberta_base_architecture_glu_small_pure_rms_urpe_1d3_linear_chunk(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "linear_chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.multiple = 2
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    ##### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"

@register_model_architecture("roberta_norm_attention", "roberta_glu_pure_rms_urpe_1d3_linear_chunk_32")
def roberta_base_architecture_glu_pure_rms_urpe_1d3_linear_chunk_32(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "linear_chunk"
    args.encoder_chunk_size = 32
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    ##### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"

@register_model_architecture("roberta_norm_attention", "roberta_glu_small_pure_rms_urpe_1d3_linear_chunk_32")
def roberta_base_architecture_glu_small_pure_rms_urpe_1d3_linear_chunk_32(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "linear_chunk"
    args.encoder_chunk_size = 32
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.multiple = 2
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    ##### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"

@register_model_architecture("roberta_norm_attention", "roberta_glu_pure_rms_urpe_1d3_linear_chunk_16")
def roberta_base_architecture_glu_pure_rms_urpe_1d3_linear_chunk_16(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "linear_chunk"
    args.encoder_chunk_size = 16
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    ##### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"

@register_model_architecture("roberta_norm_attention", "roberta_glu_small_pure_rms_urpe_1d3_linear_chunk_16")
def roberta_base_architecture_glu_small_pure_rms_urpe_1d3_linear_chunk_16(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "linear_chunk"
    args.encoder_chunk_size = 16
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.multiple = 2
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    ##### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"
##### linear chunk

##### abl
@register_model_architecture("roberta_norm_attention", "roberta_ffn_all_rms_layer_ln_rms_urpe_1d3")
def roberta_base_architecture_ffn_all_rms_layer_ln_rms_urpe_1d3(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ##### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_ln_rms_urpe_1d3_pure_chunk")
def roberta_base_architecture_glu_all_rms_layer_ln_rms_urpe_1d3_pure_chunk(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_ln_rms_urpe_1d3_pure_linear")
def roberta_base_architecture_glu_all_rms_layer_ln_rms_urpe_1d3_pure_linear(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [1 for _ in range(args.encoder_layers)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_ln_rms_urpe_1d3_linear_chunk")
def roberta_base_architecture_glu_all_rms_layer_ln_rms_urpe_1d3_linear_chunk(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [1 for _ in range(args.encoder_layers // 2)] + [2 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_ln_rms_urpe_1d3_softmax")
def roberta_base_architecture_glu_all_rms_layer_ln_rms_urpe_1d3_softmax(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    ##### softmax
    args.use_softmax = True
##### abl

##### mix 并联
@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_ln_rms_urpe_1d3_parallel")
def roberta_base_architecture_glu_all_rms_layer_ln_rms_urpe_1d3_parallel(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
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
    ##### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    ##### type
    args.encoder_attention_types = [3 for _ in range(args.encoder_layers)]
    args.forward_type = 1

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_ln_rms_urpe_1d3_linear_local")
def roberta_base_architecture_glu_all_rms_layer_ln_rms_urpe_1d3_linear_local(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
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
    ##### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    ##### type
    args.encoder_attention_types = [3 for _ in range(args.encoder_layers)]
    args.forward_type = 2

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_ln_rms_urpe_1d3_local_linear")
def roberta_base_architecture_glu_all_rms_layer_ln_rms_urpe_1d3_local_linear(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
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
    ##### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    ##### type
    args.encoder_attention_types = [3 for _ in range(args.encoder_layers)]
    args.forward_type = 3

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_ln_rms_urpe_1d3_25_75")
def roberta_base_architecture_glu_all_rms_layer_ln_rms_urpe_1d3_25_75(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    l = int(args.encoder_layers * 0.25)
    args.encoder_attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_ln_rms_urpe_1d3_75_25")
def roberta_base_architecture_glu_all_rms_layer_ln_rms_urpe_1d3_75_25(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    l = int(args.encoder_layers * 0.75)
    args.encoder_attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
##### mix 并联

##### chunk size
@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_ln_rms_urpe_1d3_chunk32")
def roberta_base_architecture_glu_all_rms_layer_ln_rms_urpe_1d3_chunk32(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 32
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_ln_rms_urpe_1d3_chunk128")
def roberta_base_architecture_glu_all_rms_layer_ln_rms_urpe_1d3_chunk128(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 128
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
##### chunk size

##### softmax + 1 + elu
@register_model_architecture("roberta_norm_attention", "roberta_ffn_all_rms_layer_ln_rms_urpe_1d3_softmax_1+elu")
def roberta_base_architecture_ffn_all_rms_layer_ln_rms_urpe_1d3_softmax_1_elu(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "1+elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ##### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    ##### softmax
    args.use_softmax = True


@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_ln_rms_urpe_1d3_softmax_1+elu_pure_chunk")
def roberta_base_architecture_glu_all_rms_layer_ln_rms_urpe_1d3_softmax_1_elu_pure_chunk(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "1+elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    ##### softmax
    args.use_softmax = True

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_ln_rms_urpe_1d3_softmax_1+elu_pure_linear")
def roberta_base_architecture_glu_all_rms_layer_ln_rms_urpe_1d3_softmax_1_elu_pure_linear(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "1+elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [1 for _ in range(args.encoder_layers)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    ##### softmax
    args.use_softmax = True

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_ln_rms_urpe_1d3_softmax_1+elu_linear_chunk")
def roberta_base_architecture_glu_all_rms_layer_ln_rms_urpe_1d3_softmax_1_elu_linear_chunk(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "1+elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [1 for _ in range(args.encoder_layers // 2)] + [2 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    ##### softmax
    args.use_softmax = True

##### mix 并联
@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_ln_rms_urpe_1d3_softmax_1+elu_parallel")
def roberta_base_architecture_glu_all_rms_layer_ln_rms_urpe_1d3_softmax_1_elu_parallel(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "1+elu"
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
    ##### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    ##### type
    args.encoder_attention_types = [3 for _ in range(args.encoder_layers)]
    args.forward_type = 1
    ##### softmax
    args.use_softmax = True

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_ln_rms_urpe_1d3_softmax_1+elu_linear_local")
def roberta_base_architecture_glu_all_rms_layer_ln_rms_urpe_1d3_softmax_1_elu_linear_local(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "1+elu"
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
    ##### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    ##### type
    args.encoder_attention_types = [3 for _ in range(args.encoder_layers)]
    args.forward_type = 2
    ##### softmax
    args.use_softmax = True

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_ln_rms_urpe_1d3_softmax_1+elu_local_linear")
def roberta_base_architecture_glu_all_rms_layer_ln_rms_urpe_1d3_softmax_1_elu_local_linear(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "1+elu"
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
    ##### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    ##### type
    args.encoder_attention_types = [3 for _ in range(args.encoder_layers)]
    args.forward_type = 3
    ##### softmax
    args.use_softmax = True

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_ln_rms_urpe_1d3_softmax_1+elu_25_75")
def roberta_base_architecture_glu_all_rms_layer_ln_rms_urpe_1d3_softmax_1_elu_25_75(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "1+elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    l = int(args.encoder_layers * 0.25)
    args.encoder_attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    ##### softmax
    args.use_softmax = True

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_ln_rms_urpe_1d3_softmax_1+elu_75_25")
def roberta_base_architecture_glu_all_rms_layer_ln_rms_urpe_1d3_softmax_1_elu_75_25(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "1+elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    l = int(args.encoder_layers * 0.75)
    args.encoder_attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    ##### softmax
    args.use_softmax = True
##### mix 并联

##### local global new version
@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_ln_rms_no_urpe_softmax_1+elu")
def roberta_base_architecture_glu_all_rms_layer_ln_rms_no_urpe_softmax_1_elu(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "1+elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.encoder_use_urpe = False
    ##### softmax
    args.use_softmax = True

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_ln_rms_urpe_1d3_softmax_1+elu")
def roberta_base_architecture_glu_all_rms_layer_ln_rms_urpe_1d3_softmax_1_elu(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "1+elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    ##### softmax
    args.use_softmax = True

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_ln_rms_urpe_1d3_softmax_1+elu_small")
def roberta_base_architecture_glu_all_rms_layer_ln_rms_urpe_1d3_softmax_1_elu_small(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "1+elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    ##### softmax
    args.use_softmax = True
    args.multiple = 2

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_ln_rms_urpe_1d3_softmax_1+elu_chunk32")
def roberta_base_architecture_glu_all_rms_layer_ln_rms_urpe_1d3_softmax_1_elu_chunk32(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "1+elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 32
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    ##### softmax
    args.use_softmax = True

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_ln_rms_urpe_1d3_softmax_1+elu_chunk128")
def roberta_base_architecture_glu_all_rms_layer_ln_rms_urpe_1d3_softmax_1_elu_chunk128(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "1+elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 128
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    ##### softmax
    args.use_softmax = True

@register_model_architecture("roberta_norm_attention", "roberta_window_softmax_1+elu")
def roberta_base_architecture_window_norm_1elu(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "1+elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "window"
    args.encoder_chunk_size = 64
    args.left_window = 1
    args.right_window = 1
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    ##### softmax
    args.use_softmax = True
##### softmax + 1 + elu

@register_model_architecture("roberta_norm_attention", "roberta_window_relu_elu")
def roberta_base_architecture_window_relu_elu(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "window"
    args.encoder_chunk_size = 64
    args.left_window = 1
    args.right_window = 1
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"

##### norm(1 + elu)
@register_model_architecture("roberta_norm_attention", "roberta_norm_1+elu")
def roberta_base_architecture_norm_1_elu(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "1+elu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.norm_type = "simplermsnorm"
    args.encoder_attention_types = [1 for _ in range(args.encoder_layers)]

@register_model_architecture("roberta_norm_attention", "roberta_layernorm_1+elu")
def roberta_base_architecture_roberta_layernorm_1_elu(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "1+elu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.norm_type = "layernorm"
    args.encoder_attention_types = [1 for _ in range(args.encoder_layers)]

@register_model_architecture("roberta_norm_attention", "roberta_gatedrms_1+elu")
def roberta_base_architecture_roberta_gatedrms_1_elu(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "1+elu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.norm_type = "gatedrmsnorm"
    args.encoder_attention_types = [1 for _ in range(args.encoder_layers)]

@register_model_architecture("roberta_norm_attention", "roberta_norm_elu")
def roberta_base_elu_architecture(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.norm_type = "simplermsnorm"
    args.encoder_attention_types = [1 for _ in range(args.encoder_layers)]

@register_model_architecture("roberta_norm_attention", "roberta_no_norm_1+elu")
def roberta_no_norm_architecture(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "1+elu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.norm_type = "simplermsnorm"
    args.attention_use_layer_norm = False
    args.encoder_attention_types = [1 for _ in range(args.encoder_layers)]
##### norm(1 + elu)
