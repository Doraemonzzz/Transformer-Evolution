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

from ..xformer import NormAttentionDecoder

def small_init_weights(module):
    if isinstance(module, (nn.Linear)):
        module.weight.data.normal_(mean=0.0, std=0.02)        

    if isinstance(module, (nn.Embedding)):
        # nn.init.uniform_(module.weight, a=-1e-4, b=1e-4) # SmallInit(Emb)
        print("Embdding norm before")
        print(torch.norm(module.weight.data))
        module.weight.data.normal_(mean=0.0, std=1e-5)
        print("Embdding norm after")
        print(torch.norm(module.weight.data))

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

@register_model("norm_attention_lm", dataclass=TransformerLanguageModelConfig)
class NormAttentionLanguageModel(TransformerLanguageModel):
    def __init__(self, decoder):
        super(NormAttentionLanguageModel, self).__init__(decoder)

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

        init_method = getattr(args, 'init_method', "default")
        print(f"init_method {init_method}")
        if init_method != "default":
            print("small init")
            embed_tokens.apply(small_init_weights)

        decoder = NormAttentionDecoder(
            args, task.target_dictionary, embed_tokens, no_encoder_attn=True
        )
        return cls(decoder)

# norm attention
# linear: attention_type = 1
# local: attention_type = 2
# local, ... , local, linear, ... ,linear
@register_model_architecture("norm_attention_lm", "norm_attention_lm_type1")
def transformer_norm_attention_lm_type1(args):
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
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_attention_heads = 1
    args.decoder_use_urpe = False
    args.decoder_chunk_size = 32
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]


########## norm attention(local + linear)
@register_model_architecture("norm_attention_lm", "norm_ln_glu_lm_base")
def transformer_norm_ln_glu_lm_base(args):
    base_lm_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "layernorm"
    args.norm_type = "layernorm"

@register_model_architecture("norm_attention_lm", "norm_ln_glu_small_lm_base")
def transformer_norm_ln_glu_small_lm_base(args):
    base_lm_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "layernorm"
    args.norm_type = "layernorm"
    args.multiple = 2

@register_model_architecture("norm_attention_lm", "norm_ln_ffn_lm_base")
def transformer_norm_ln_ffn_lm_base(args):
    base_lm_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ##### glu
    args.local_norm_type = "layernorm"
    args.norm_type = "layernorm"

########## norm attention + glu act
@register_model_architecture("norm_attention_lm", "norm_all_rms_glu_lm_base_elu")
def transformer_norm_all_rms_glu_lm_base_elu(args):
    base_lm_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.fina_act = "elu"
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"

@register_model_architecture("norm_attention_lm", "norm_all_rms_glu_small_lm_base_elu")
def transformer_norm_all_rms_glu_small_lm_base_elu(args):
    base_lm_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.fina_act = "elu"
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    args.multiple = 2
########## norm attention + glu act

########## norm attention + urpe
@register_model_architecture("norm_attention_lm", "norm_all_rms_glu_lm_base_urpe_1d3")
def transformer_norm_all_rms_glu_lm_base_urpe_1d3(args):
    base_lm_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True

@register_model_architecture("norm_attention_lm", "norm_all_rms_glu_small_lm_base_urpe_1d3")
def transformer_norm_all_rms_glu_small_lm_base_urpe_1d3(args):
    base_lm_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    args.multiple = 2
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True

@register_model_architecture("norm_attention_lm", "norm_all_rms_glu_lm_base_ln_rms_urpe_1d3")
def transformer_norm_all_rms_glu_lm_base_ln_rms_urpe_1d3(args):
    base_lm_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True

@register_model_architecture("norm_attention_lm", "norm_all_rms_glu_small_lm_base_ln_rms_urpe_1d3")
def transformer_norm_all_rms_glu_small_lm_base_ln_rms_urpe_1d3(args):
    base_lm_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    args.multiple = 2
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True

@register_model_architecture("norm_attention_lm", "norm_all_layernorm_glu_lm_base_urpe_1d3")
def transformer_norm_all_layernorm_glu_lm_base_urpe_1d3(args):
    base_lm_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "layernorm"
    args.norm_type = "layernorm"
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True

@register_model_architecture("norm_attention_lm", "norm_all_layernorm_glu_small_lm_base_urpe_1d3")
def transformer_norm_all_layernorm_glu_small_lm_base_urpe_1d3(args):
    base_lm_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "layernorm"
    args.norm_type = "layernorm"
    args.multiple = 2
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
########## norm attention + urpe

########## norm attention + urpe + dropout
@register_model_architecture("norm_attention_lm", "norm_all_rms_glu_lm_base_urpe_1d3_dropout02")
def transformer_norm_all_rms_glu_lm_base_urpe_1d3_dropout02(args):
    base_lm_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    args.glu_dropout = 0.2

@register_model_architecture("norm_attention_lm", "norm_all_rms_glu_small_lm_base_urpe_1d3_dropout02")
def transformer_norm_all_rms_glu_small_lm_base_urpe_1d3_dropout02(args):
    base_lm_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    args.multiple = 2
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    args.glu_dropout = 0.2
########## norm attention + urpe + dropout

########## norm attention + urpe + no_abs
@register_model_architecture("norm_attention_lm", "norm_all_rms_glu_lm_base_ln_rms_urpe_1d3_no_abs")
def transformer_norm_all_rms_glu_lm_base_ln_rms_urpe_1d3_no_abs(args):
    base_lm_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    ##### no abs
    args.no_token_positional_embeddings = True

@register_model_architecture("norm_attention_lm", "norm_all_rms_glu_small_lm_base_ln_rms_urpe_1d3_no_abs")
def transformer_norm_all_rms_glu_small_lm_base_ln_rms_urpe_1d3_no_abs(args):
    base_lm_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    args.multiple = 2
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    ##### no abs
    args.no_token_positional_embeddings = True

@register_model_architecture("norm_attention_lm", "norm_all_layernorm_glu_lm_base_urpe_1d3_no_abs")
def transformer_norm_all_layernorm_glu_lm_base_urpe_1d3_no_abs(args):
    base_lm_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "layernorm"
    args.norm_type = "layernorm"
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    ##### no abs
    args.no_token_positional_embeddings = True

@register_model_architecture("norm_attention_lm", "norm_all_layernorm_glu_small_lm_base_urpe_1d3_no_abs")
def transformer_norm_all_layernorm_glu_small_lm_base_urpe_1d3_no_abs(args):
    base_lm_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "layernorm"
    args.norm_type = "layernorm"
    args.multiple = 2
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    ##### no abs
    args.no_token_positional_embeddings = True
########## norm attention + urpe + no_abs

########## norm attention + urpe + pure rms norm
@register_model_architecture("norm_attention_lm", "norm_glu_lm_base_pure_rms_urpe_1d3_small_init")
def transformer_norm_glu_lm_base_pure_rms_urpe_1d3_small_init(args):
    base_lm_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    ##### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"
    args.init_method = "small_embdding"

@register_model_architecture("norm_attention_lm", "norm_small_glu_lm_base_pure_rms_urpe_1d3_small_init")
def transformer_norm_small_glu_lm_base_pure_rms_urpe_1d3_small_init(args):
    base_lm_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    args.multiple = 2
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    ##### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"
    args.init_method = "small_embdding"

@register_model_architecture("norm_attention_lm", "norm_glu_lm_base_pure_rms_urpe_1d3_no_abs_small_init")
def transformer_norm_glu_lm_base_pure_rms_urpe_1d3_no_abs_small_init(args):
    base_lm_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    ##### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"
    args.init_method = "small_embdding"
    ##### no abs
    args.no_token_positional_embeddings = True

@register_model_architecture("norm_attention_lm", "norm_small_glu_lm_base_pure_rms_urpe_1d3_no_abs_small_init")
def transformer_norm_small_glu_lm_base_pure_rms_urpe_1d3_no_abs_small_init(args):
    base_lm_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    args.multiple = 2
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    ##### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"
    args.init_method = "small_embdding"
    ##### no abs
    args.no_token_positional_embeddings = True
########## norm attention + urpe + pure rms norm

########## norm attention + urpe + pure rms norm + geglu
@register_model_architecture("norm_attention_lm", "norm_glu_lm_base_pure_rms_urpe_1d3_geglu")
def transformer_norm_glu_lm_base_pure_rms_urpe_1d3_geglu(args):
    base_lm_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ##### glu
    args.use_glu = True
    args.glu_act = "gelu"
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    ##### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"

@register_model_architecture("norm_attention_lm", "norm_small_glu_lm_base_pure_rms_urpe_1d3_geglu")
def transformer_norm_small_glu_lm_base_pure_rms_urpe_1d3_geglu(args):
    base_lm_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ##### glu
    args.use_glu = True
    args.glu_act = "gelu"
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    args.multiple = 2
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    ##### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"

@register_model_architecture("norm_attention_lm", "norm_glu_lm_base_pure_rms_urpe_1d3_small_init_geglu")
def transformer_norm_glu_lm_base_pure_rms_urpe_1d3_small_init_geglu(args):
    base_lm_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ##### glu
    args.use_glu = True
    args.glu_act = "gelu"
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    ##### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"
    args.init_method = "small_embdding"

@register_model_architecture("norm_attention_lm", "norm_small_glu_lm_base_pure_rms_urpe_1d3_small_init_geglu")
def transformer_norm_small_glu_lm_base_pure_rms_urpe_1d3_small_init_geglu(args):
    base_lm_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ##### glu
    args.use_glu = True
    args.glu_act = "gelu"
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    args.multiple = 2
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    ##### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"
    args.init_method = "small_embdding"
########## norm attention + urpe + pure rms norm + geglu

########## pure rms norm + urpe + weight
@register_model_architecture("norm_attention_lm", "norm_glu_lm_base_pure_rms_urpe_1d3")
def transformer_norm_glu_lm_base_pure_rms_urpe_1d3(args):
    base_lm_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    ##### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"

@register_model_architecture("norm_attention_lm", "norm_small_glu_lm_base_pure_rms_urpe_1d3")
def transformer_norm_small_glu_lm_base_pure_rms_urpe_1d3(args):
    base_lm_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    args.multiple = 2
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    ##### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"

@register_model_architecture("norm_attention_lm", "norm_glu_lm_base_pure_rms_urpe_1d3_laplace")
def transformer_norm_glu_lm_base_pure_rms_urpe_1d3_laplace(args):
    base_lm_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    ##### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"
    #### weight
    args.weight_type = 1

@register_model_architecture("norm_attention_lm", "norm_small_glu_lm_base_pure_rms_urpe_1d3_laplace")
def transformer_norm_small_glu_lm_base_pure_rms_urpe_1d3_laplace(args):
    base_lm_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    args.multiple = 2
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    ##### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"
    #### weight
    args.weight_type = 1

@register_model_architecture("norm_attention_lm", "norm_glu_lm_base_pure_rms_urpe_1d3_gaussian")
def transformer_norm_glu_lm_base_pure_rms_urpe_1d3_gaussian(args):
    base_lm_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    ##### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"
    #### weight
    args.weight_type = 2

@register_model_architecture("norm_attention_lm", "norm_small_glu_lm_base_pure_rms_urpe_1d3_gaussian")
def transformer_norm_small_glu_lm_base_pure_rms_urpe_1d3_gaussian(args):
    base_lm_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    args.multiple = 2
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    ##### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"
    #### weight
    args.weight_type = 2

@register_model_architecture("norm_attention_lm", "norm_glu_lm_base_pure_rms_urpe_1d3_final_dropout")
def transformer_norm_glu_lm_base_pure_rms_urpe_1d3_final_dropout(args):
    base_lm_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    ##### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"
    ##### final dropout
    args.use_final_dropout = True
    args.final_dropout = 0.1

@register_model_architecture("norm_attention_lm", "norm_small_glu_lm_base_pure_rms_urpe_1d3_final_dropout")
def transformer_norm_small_glu_lm_base_pure_rms_urpe_1d3_final_dropout(args):
    base_lm_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    args.multiple = 2
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    ##### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"
    ##### final dropout
    args.use_final_dropout = True
    args.final_dropout = 0.1
########## pure rms norm + urpe + weight

##### relu2
@register_model_architecture("norm_attention_lm", "norm_glu_lm_base_pure_rms_urpe_1d3_relu2")
def transformer_norm_glu_lm_base_pure_rms_urpe_1d3_relu2(args):
    base_lm_architecture(args)
    ##### add
    args.linear_act_fun = "relu2"
    args.local_act_fun = "relu2"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    ##### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"

@register_model_architecture("norm_attention_lm", "norm_small_glu_lm_base_pure_rms_urpe_1d3_relu2")
def transformer_norm_small_glu_lm_base_pure_rms_urpe_1d3_relu2(args):
    base_lm_architecture(args)
    ##### add
    args.linear_act_fun = "relu2"
    args.local_act_fun = "relu2"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    args.multiple = 2
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    ##### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"
##### relu2

##### linear_chunk
@register_model_architecture("norm_attention_lm", "norm_glu_lm_base_pure_rms_urpe_1d3_linear_chunk")
def transformer_norm_glu_lm_base_pure_rms_urpe_1d3_linear_chunk(args):
    base_lm_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "linear_chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    ##### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"

@register_model_architecture("norm_attention_lm", "norm_small_glu_lm_base_pure_rms_urpe_1d3_linear_chunk")
def transformer_norm_small_glu_lm_base_pure_rms_urpe_1d3_linear_chunk(args):
    base_lm_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "linear_chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    args.multiple = 2
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    ##### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"

@register_model_architecture("norm_attention_lm", "norm_glu_lm_base_pure_rms_urpe_1d3_linear_chunk_32")
def transformer_norm_glu_lm_base_pure_rms_urpe_1d3_linear_chunk_32(args):
    base_lm_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "linear_chunk"
    args.decoder_chunk_size = 32
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    ##### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"

@register_model_architecture("norm_attention_lm", "norm_small_glu_lm_base_pure_rms_urpe_1d3_linear_chunk_32")
def transformer_norm_small_glu_lm_base_pure_rms_urpe_1d3_linear_chunk_32(args):
    base_lm_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "linear_chunk"
    args.decoder_chunk_size = 32
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    args.multiple = 2
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    ##### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"

@register_model_architecture("norm_attention_lm", "norm_glu_lm_base_pure_rms_urpe_1d3_linear_chunk_16")
def transformer_norm_glu_lm_base_pure_rms_urpe_1d3_linear_chunk_16(args):
    base_lm_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "linear_chunk"
    args.decoder_chunk_size = 16
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    ##### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"

@register_model_architecture("norm_attention_lm", "norm_small_glu_lm_base_pure_rms_urpe_1d3_linear_chunk_16")
def transformer_norm_small_glu_lm_base_pure_rms_urpe_1d3_linear_chunk_16(args):
    base_lm_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "linear_chunk"
    args.decoder_chunk_size = 16
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    args.multiple = 2
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    ##### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"
##### linear_chunk

##### speed test
##### 删除mask, 不影响速度
@register_model_architecture("norm_attention_lm", "norm_glu_lm_base_abl")
def transformer_norm_glu_lm_base_abl(args):
    base_lm_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ##### glu
    args.decoder_causal = False
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "layernorm"
    args.norm_type = "layernorm"

@register_model_architecture("norm_attention_lm", "norm_glu_lm_base_pure_chunk")
def transformer_norm_glu_lm_base_pure_chunk(args):
    base_lm_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers)]
    ##### glu
    args.use_glu = False
    args.glu_act = "swish"
    args.local_norm_type = "layernorm"
    args.norm_type = "layernorm"

@register_model_architecture("norm_attention_lm", "norm_glu_lm_base_pure_linear")
def transformer_norm_glu_lm_base_pure_linear(args):
    base_lm_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [1 for _ in range(args.decoder_layers)]
    ##### glu
    args.use_glu = False
    args.glu_act = "swish"
    args.local_norm_type = "layernorm"
    args.norm_type = "layernorm"
##### speed test

########## softmax + 1 + elu
@register_model_architecture("norm_attention_lm", "norm_glu_lm_base_pure_rms_urpe_1d3_softmax_1+elu")
def transformer_norm_glu_lm_base_pure_rms_urpe_1d3_softmax_1_elu(args):
    base_lm_architecture(args)
    ##### add
    args.linear_act_fun = "1+elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ##### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    args.attn_type = "simplermsnorm"
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    ##### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"
    ###### softmax
    args.use_softmax = True
########## softmax + 1 + elu

########## norm linear + toeplizt
@register_model_architecture("norm_attention_lm", "norm_lm_pure_linear_1+elu")
def transofrmer_norm_lm_pure_linear_1_elu(args):
    base_lm_architecture(args)
    ##### add
    args.linear_act_fun = "elu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_attention_types = [1 for _ in range(args.decoder_layers)]
    args.norm_type = "simplermsnorm"

@register_model_architecture("norm_attention_lm", "norm_lm_pure_linear_1+elu_toep_learn")
def transofrmer_norm_lm_pure_linear_1_elu_toep_learn(args):
    base_lm_architecture(args)
    ##### add
    args.linear_act_fun = "1+elu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_attention_types = [1 for _ in range(args.decoder_layers)]
    args.norm_type = "simplermsnorm"
    ##### topelitz
    args.use_toeplizt = True
    args.type_num = -1

@register_model_architecture("norm_attention_lm", "norm_lm_pure_linear_1+elu_toep_exp")
def transofrmer_norm_lm_pure_linear_1_elu_toep_exp(args):
    base_lm_architecture(args)
    ##### add
    args.linear_act_fun = "1+elu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_attention_types = [1 for _ in range(args.decoder_layers)]
    args.norm_type = "simplermsnorm"
    ##### topelitz
    args.use_toeplizt = True
    args.type_num = 1
########## norm linear + toeplizt