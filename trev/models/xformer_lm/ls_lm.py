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

from trev.modules import TransformerLSModel

@dataclass
class TransformerLSConfig(TrevDataclass):
    # defaults come from https://github.com/facebookresearch/adaptive-span/blob/master/experiments/enwik8_small.sh
    vocab_size: int = 50
    d_model: int = 256
    n_head: int = 4
    d_inner: int = 1024
    n_layer: int = 8
    dropout: float = 0.0
    emb_dropout: float = 0.0
    chunk_rank: int = 1
    chunk_size: int = 32
    mem_len: int = 4096
    window_len: int = 256
    grad_chk: bool = False
    pre_ln: bool = False
    use_gelu: bool = False
    use_bias: bool = False
    clamp_len: int = -1
    cpos_clamp_len: int = -1
    probing: bool = False

@register_model("transformer-ls", dataclass=TransformerLanguageModelConfig)
class TransformerLS(TrevLanguageModel):
    @classmethod
    def build_model(cls, args, task):
        print(args)
        return cls(TransformerLSDecoder(args, task))

    def get_aux_loss(self):
        return self.decoder.get_aux_loss()

    def get_current_max_span(self):
        return self.decoder.get_current_max_span()

    def get_current_avg_span(self):
        return self.decoder.get_current_avg_span()

class TransformerLSDecoder(TrevIncrementalDecoder):
    def __init__(self, args, task):

        super().__init__(task.target_dictionary)

        config = TransformerLSConfig(
            vocab_size=len(task.target_dictionary),
            d_model=args.d_model,
            n_head=args.n_head,
            d_inner=args.d_inner,
            n_layer=args.n_layer,
            dropout=args.dropout,
            emb_dropout=args.emb_dropout,
            mem_len=args.mem_len,
            chunk_rank=args.chunk_rank,
            chunk_size=args.chunk_size,
            window_len=args.window_len,
            grad_chk=args.grad_chk,
            pre_ln=args.pre_ln,
            use_gelu=args.use_gelu,
            use_bias=args.use_bias,
            clamp_len=args.clamp_len,
            cpos_clamp_len=args.cpos_clamp_len,
            probing=args.probing,
        )
        self.config = config
        logger.info(config)
        del config.__dict__['_name']
        self.model = TransformerLSModel(**config.__dict__)
        self.cache_size = args.mem_len

        self._mems = None

    def forward(
        self,
        src_tokens,
        incremental_state: Optional[Dict[str, List[torch.Tensor]]] = None,
        encoder_out=None,
        src_lengths=None,
    ):
        bsz = src_tokens.size(0)
        if incremental_state is not None:  # used during inference
            mems = self.get_incremental_state("mems")
            src_tokens = src_tokens[:, -1:]  # only keep the most recent token
        else:
            mems = self._mems

        if mems is None:
            # first time init
            mems = self.init_hid_cache(bsz)
        output = self.model(x=src_tokens, h_cache=mems,)
        if incremental_state is not None:
            self.set_incremental_state(incremental_state, "mems", output[1])
        else:
            self._mems = output[1]
        return (output[0],)

    def init_hid_cache(self, batch_sz):
        hid = []
        for layer in self.model.layers:
            param = next(self.model.parameters())
            h = torch.zeros(
                batch_sz,
                self.cache_size,
                self.config.d_model,
                dtype=param.dtype,
                device=param.device,
            )
            hid.append(h)
        return hid

    def get_aux_loss(self):
        return self.model.get_aux_loss()

    def get_current_max_span(self):
        return self.model.get_current_max_span()

    def get_current_avg_span(self):
        return self.model.get_current_avg_span()

    def reorder_incremental_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[torch.Tensor]]],
        new_order: torch.Tensor,
    ):
        """Reorder incremental state.
        This will be called when the order of the input has changed from the
        previous time step. A typical use case is beam search, where the input
        order changes between time steps based on the selection of beams.
        """
        raise NotImplementedError("This is required for generation/beam search")
        # mems = self.get_incremental_state(incremental_state, "mems")
        # if mems is not None:
        #     new_mems = [mems_i.index_select(1, new_order) for mems_i in mems]
        #     self.set_incremental_state(incremental_state, "mems", new_mems)

@register_model_architecture("transformer-ls", "ls_attention_lm")
def transformer_ls_attention_lm(args):
    base_lm_architecture(args)
    args.d_model = args.decoder_embed_dim
    args.n_head = args.decoder_attention_heads
    args.d_inner = args.decoder_ffn_embed_dim
    args.n_layer = args.decoder_layers
    args.dropout = args.dropout
    args.emb_dropout = 0.0
    args.chunk_rank = 1
    args.chunk_size = 32
    args.mem_len = 512
    args.window_len = 64
    args.grad_chk = False
    args.pre_ln = False
    args.use_gelu = False
    args.use_bias = True
    args.clamp_len = -1
    args.cpos_clamp_len = -1
    args.probing = False
