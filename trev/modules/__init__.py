# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""isort:skip_file"""

from .adaptive_input import AdaptiveInput
from .adaptive_softmax import AdaptiveSoftmax
from .base_layer import BaseLayer
from .beamable_mm import BeamableMM
from .character_token_embedder import CharacterTokenEmbedder
from .conv_tbc import ConvTBC
from .cross_entropy import cross_entropy
from .downsampled_multihead_attention import DownsampledMultiHeadAttention
from .dynamic_convolution import DynamicConv, DynamicConv1dTBC
from .dynamic_crf_layer import DynamicCRF
from .trev_dropout import TrevDropout
from .fp32_group_norm import Fp32GroupNorm
from .gelu import gelu, gelu_accurate
from .grad_multiply import GradMultiply
from .gumbel_vector_quantizer import GumbelVectorQuantizer
from .kmeans_vector_quantizer import KmeansVectorQuantizer
from .layer_drop import LayerDropModuleList
from .layer_norm import Fp32LayerNorm, LayerNorm
from .learned_positional_embedding import LearnedPositionalEmbedding
from .lightweight_convolution import LightweightConv, LightweightConv1dTBC
from .linearized_convolution import LinearizedConvolution
from .multihead_attention import MultiheadAttention
from .positional_embedding import PositionalEmbedding
from .same_pad import SamePad
from .scalar_bias import ScalarBias
from .sinusoidal_positional_embedding import SinusoidalPositionalEmbedding
from .transformer_sentence_encoder_layer import TransformerSentenceEncoderLayer
from .transformer_sentence_encoder import TransformerSentenceEncoder
from .transpose_last import TransposeLast
from .unfold import unfold1d
from .transformer_layer import TransformerDecoderLayer, TransformerEncoderLayer
from .vggblock import VGGBlock

########## positional encoding
from .positional_encoding import rope
from .positional_encoding import RpeVanilla
from .positional_encoding import SineSPE, ConvSPE, SPEFilter
from .positional_encoding import T5RPE
from .positional_encoding import Urpe
from .positional_encoding import UrpeV2
from .positional_encoding import Toeplizt
from .positional_encoding import ToepliztV2
########## positional encoding

########## norm
from .norm import SimpleRMSNorm, RMSNorm, GatedRMSNorm, ScaleNorm, OffsetScale
########## norm

########## token mix
from .token_mix import TokenShift
from .token_mix import ConvMix
########## token mix

########## ffn
from .ffn import GLU
########## ffn

########## xattention
##### linearkernel
from .x_attention import LinearKernelAttention
from .x_attention import LinearKernelAttentionEncoderLayer, LinearKernelAttentionDecoderLayer
##### performer
from .x_attention import PerformerAttention
from .x_attention import PerformerEncoderLayer, PerformerDecoderLayer
##### flash
from .x_attention import FlashQuadAttention
from .x_attention import FlashQuadEncoderLayer, FlashQuadDecoderLayer
from .x_attention import FlashLinearAttention
from .x_attention import FlashLinearEncoderLayer, FlashLinearDecoderLayer
##### mha plus
from .x_attention import MultiheadAttentionPlus
from .x_attention import TransformerEncoderLayerPlus, TransformerDecoderLayerPlus
##### long short attention
from .x_attention import LSCausalAttention
from .x_attention import LSNonCausalAttention
from .x_attention import LSAttentionEncoderLayer
from .x_attention import TransformerLSModel
##### ReLA
from .x_attention import ReLAttention
from .x_attention import ReLAEncoderLayer, ReLADecoderLayer
##### cosformer
from .x_attention import CosformerAttention
from .x_attention import CosformerEncoderLayer, CosformerDecoderLayer
##### norm mix attention
from .x_attention import NormMixAttention
from .x_attention import NormMixAttentionDecoderLayer, NormMixAttentionEncoderLayer
##### norm local/linear attention
from .x_attention import NormLinearAttention
from .x_attention import NormLocalAttention
from .x_attention import NormAttentionDecoderLayer, NormAttentionEncoderLayer
##### linear combination
from .x_attention import LinearCombinationEncoderLayer
##### sparse attetntioin
from .x_attention import SparseMultiheadAttention
from .x_attention import SparseTransformerEncoderLayer, SparseTransformerDecoderLayer
##### sparse attetntioin
##### double fusion
from .x_attention import DoubleFusion
from .x_attention import DoubleFusionEncoderLayer, DoubleFusionDecoderLayer
from .x_attention import DoubleFusionV2
from .x_attention import DoubleFusionV2EncoderLayer, DoubleFusionV2DecoderLayer
from .x_attention import DoubleFusionV3
from .x_attention import DoubleFusionV3EncoderLayer, DoubleFusionV3DecoderLayer
from .x_attention import DoubleFusionQuad
from .x_attention import DoubleFusionQuadEncoderLayer, DoubleFusionQuadDecoderLayer
##### double fusion
##### toeplitz attention
from .x_attention import ToeplitzAttention
from .x_attention import ToeplitzAttentionEncoderLayer, ToeplitzAttentionDecoderLayer
##### toeplitz attention
##### gau
from .x_attention import GauQuad
from .x_attention import GauQuadV2
from .x_attention import GauQuadV3
from .x_attention import GauEncoderLayer, GauDecoderLayer
##### gau
########## xattention

########## sparse
# from .sparse_multihead_attention import SparseMultiheadAttention
# from .sparse_transformer_layer import SparseTransformerEncoderLayer, SparseTransformerDecoderLayer
########## sparse relu


__all__ = [
    "AdaptiveInput",
    "AdaptiveSoftmax",
    "BaseLayer",
    "BeamableMM",
    "CharacterTokenEmbedder",
    "ConvTBC",
    "cross_entropy",
    "DownsampledMultiHeadAttention",
    "DynamicConv1dTBC",
    "DynamicConv",
    "DynamicCRF",
    "TrevDropout",
    "Fp32GroupNorm",
    "Fp32LayerNorm",
    "gelu",
    "gelu_accurate",
    "GradMultiply",
    "GumbelVectorQuantizer",
    "KmeansVectorQuantizer",
    "LayerDropModuleList",
    "LayerNorm",
    "LearnedPositionalEmbedding",
    "LightweightConv1dTBC",
    "LightweightConv",
    "LinearizedConvolution",
    "MultiheadAttention",
    "PositionalEmbedding",
    "SamePad",
    "ScalarBias",
    "SinusoidalPositionalEmbedding",
    "TransformerSentenceEncoderLayer",
    "TransformerSentenceEncoder",
    "TransformerDecoderLayer",
    "TransformerEncoderLayer",
    "TransposeLast",
    "VGGBlock",
    "unfold1d",
]
