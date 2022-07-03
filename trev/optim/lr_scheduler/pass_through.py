# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from trev.dataclass import TrevDataclass
from trev.optim.lr_scheduler import TrevLRScheduler, register_lr_scheduler


@dataclass
class PassThroughScheduleConfig(TrevDataclass):
    pass


@register_lr_scheduler("pass_through", dataclass=PassThroughScheduleConfig)
class PassThroughScheduleSchedule(TrevLRScheduler):
    """Delegate lr scheduling to the optimizer."""

    def __init__(self, cfg: PassThroughScheduleConfig, optimizer):
        super().__init__(cfg, optimizer)
        assert (
            hasattr(optimizer, "lr_scheduler") and optimizer.lr_scheduler is not None
        ), "Pass-through schedule can only be used with optimizers with their own schedulers"

    def state_dict(self):
        return self.optimizer.lr_scheduler.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.lr_scheduler.load_state_dict(state_dict)

    def step_begin_epoch(self, epoch):
        """Update the learning rate at the beginning of the given epoch."""
        return self.optimizer.lr_scheduler.step_begin_epoch(epoch)

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        return self.optimizer.lr_scheduler.step_update(num_updates)
