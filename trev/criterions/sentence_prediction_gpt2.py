# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F
from trev import metrics, utils
from trev.criterions import TrevCriterion, register_criterion


@register_criterion("sentence_prediction_gpt2")
class SentencePredictionCriterionGPT2(TrevCriterion):
    def __init__(self, task, classification_head_name, regression_target):
        super().__init__(task)
        self.classification_head_name = classification_head_name
        self.regression_target = regression_target

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--classification-head-name',
                            default='sentence_classification_head',
                            help='name of the classification head to use')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        logits, _ = model(
            **sample["net_input"]
        )
        targets = model.get_targets(sample, [logits]).view(-1)
        sample_size = targets.numel()
        input_ids = sample['net_input']['src_tokens']
        if input_ids is not None:
            sequence_lengths = torch.ne(input_ids, model.decoder.pad_idx).sum(-1) - 1
        pooled_logits = logits[range(sample_size), sequence_lengths]
        if not self.regression_target:
            pooled_logits = F.log_softmax(pooled_logits, dim=-1, dtype=torch.float32)
            targets = torch.where(targets<0,0,targets)
            loss = F.nll_loss(pooled_logits, targets, reduction="sum")
        else:
            pooled_logits = pooled_logits.view(-1).float()
            targets = targets.float()
            loss = F.mse_loss(pooled_logits, targets, reduction="sum")

        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample_size,
            "sample_size": sample_size,
        }
        if not self.regression_target:
            preds = pooled_logits.argmax(dim=1)
            logging_output["ncorrect"] = (preds == targets).sum()

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )

        if len(logging_outputs) > 0 and "ncorrect" in logging_outputs[0]:
            ncorrect = sum(log.get("ncorrect", 0) for log in logging_outputs)
            metrics.log_scalar(
                "accuracy", 100.0 * ncorrect / nsentences, nsentences, round=1
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
