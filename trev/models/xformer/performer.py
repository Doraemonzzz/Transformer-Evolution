import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from trev import utils
from trev.distributed import fsdp_wrap
from trev.models import (
    TrevEncoder,
    TrevEncoderDecoderModel,
    TrevIncrementalDecoder,
    register_model,
    register_model_architecture,
)
from trev.modules import (
    AdaptiveSoftmax,
    BaseLayer,
    TrevDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
)

from trev.modules.checkpoint_activations import checkpoint_wrapper
from trev.modules.quant_noise import quant_noise as apply_quant_noise_
from torch import Tensor

from trev.modules import PerformerEncoderLayer, PerformerDecoderLayer
from .vanilla_transformer import VanillaTransformerEncoder, VanillaTransformerDecoder
from trev.modules.utils import get_norm
import logging

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024

DEFAULT_MIN_PARAMS_TO_WRAP = int(1e8)


class PerformerEncoder(VanillaTransformerEncoder):
    """
    Performer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`PerformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~trev.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)

    def build_encoder_layer(self, args):
        layer = PerformerEncoderLayer(args)
        checkpoint = getattr(args, "checkpoint_activations", False)
        if checkpoint:
            offload_to_cpu = getattr(args, "offload_activations", False)
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = (
            getattr(args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP)
            if not checkpoint else 0
        )
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

class PerformerDecoder(VanillaTransformerDecoder):
    """
    Performer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`PerformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~trev.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
    ):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn, output_projection)

    def build_decoder_layer(self, args, no_encoder_attn=False):
        layer = PerformerDecoderLayer(args, no_encoder_attn)
        checkpoint = getattr(args, "checkpoint_activations", False)
        if checkpoint:
            offload_to_cpu = getattr(args, "offload_activations", False)
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = (
            getattr(args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP)
            if not checkpoint else 0
        )
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer