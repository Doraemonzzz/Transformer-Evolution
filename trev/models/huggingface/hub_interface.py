import torch
import re
import torch.nn as nn
from trev.dataclass.utils import convert_namespace_to_omegaconf
from trev import options

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from trev import utils
from trev.data import encoders

def from_pretrained(pretrained_model_path, task=None, model=None, cfg=None, arch='hf_gpt2'):
    from trev import tasks
    if task is None:
        args = options.get_args('data-bin/wikitext-103/', 'lm_gpt2', arch)
        cfg = convert_namespace_to_omegaconf(args)
        task = tasks.setup_task(cfg.task)
    if model is None:
        model = task.build_model(cfg.model)
    state_dict = torch.load(pretrained_model_path, map_location=torch.device("cpu"))
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()

    missing_keys = []
    unexpected_keys = []
    error_msgs = []

    if metadata is not None:
        state_dict._metadata = metadata
    def load(module: nn.Module, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        args = (state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        module._load_from_state_dict(*args)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")
    start_prefix = ""
    model_decoder = getattr(model, 'decoder')
    model_decoder_model = getattr(model_decoder, 'model')
    has_prefix_module = any(s.startswith('transformer') for s in state_dict.keys())
    if not hasattr(model_decoder_model, 'transformer') and has_prefix_module:
        start_prefix = 'transformer' + "."
    if hasattr(model_decoder_model, 'transformer') and not has_prefix_module:
        model_to_load = getattr(model_decoder_model, 'transformer')
    load(model_to_load, prefix=start_prefix)
    if model_decoder_model.__class__.__name__ != model_to_load.__class__.__name__:
        base_model_state_dict = model_to_load.state_dict().keys()
        head_model_state_dict_without_base_prefix = [
            key.split('transformer' + ".")[-1] for key in model_decoder_model.state_dict().keys()
        ]
        missing_keys.extend(head_model_state_dict_without_base_prefix - base_model_state_dict)
    if model_decoder_model._keys_to_ignore_on_load_missing is not None:
        for pat in model_decoder_model._keys_to_ignore_on_load_missing:
            missing_keys = [k for k in missing_keys if re.search(pat, k) is None]

    if model_decoder_model._keys_to_ignore_on_load_unexpected is not None:
        for pat in model_decoder_model._keys_to_ignore_on_load_unexpected:
            unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]
    model_decoder_model.tie_weights()
    return {
        "args": cfg,
        "task": task,
        "models": [model],
    }


class GPT2HubInterface(nn.Module):

    def __init__(self, cfg, task, model):
        super().__init__()
        self.cfg = cfg
        self.task = task
        self.model = model

        #self.bpe = encoders.build_bpe('gpt2')

        # this is useful for determining the device
        self.register_buffer("_float_tensor", torch.tensor([0], dtype=torch.float))

    @property
    def device(self):
        return self._float_tensor.device

    def encode(
        self, sentence: str, *addl_sentences, no_separator=False
    ) -> torch.LongTensor:
        tokens = []
        tokens.append(self.task.source_dictionary.encode_line(sentence))
        for s in addl_sentences:
            tokens.append(self.task.source_dictionary.encode_line(s))
        tokens = torch.cat(tokens,dim=0)
        return tokens.long()

    def decode(self, tokens: torch.LongTensor):
        sentences = tokens[:]
        sentences = self.task.source_dictionary.bpe.decode([int(s) for s in sentences])
        return sentences

    def extract_features(
        self, tokens: torch.LongTensor
    ) -> torch.Tensor:
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        if tokens.size(-1) > self.model.max_positions():
            raise ValueError(
                "tokens exceeds maximum length: {} > {}".format(
                    tokens.size(-1), self.model.max_positions()
                )
            )
        logits, transformer_outputs = self.model(
            tokens.to(device=self.device)
        )
        return logits,transformer_outputs  # just the last layer's features

    def predict(self, tokens: torch.LongTensor, return_logits: bool = False):
        logits,features = self.extract_features(tokens.to(device=self.device))
        pooled_logits = logits[range(1), len(tokens)-1]
        if return_logits:
            return pooled_logits
        return F.log_softmax(pooled_logits, dim=-1)



