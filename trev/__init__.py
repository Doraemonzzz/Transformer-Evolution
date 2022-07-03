# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""isort:skip_file"""

import os
import sys

try:
    from .version import __version__  # noqa
except ImportError:
    version_txt = os.path.join(os.path.dirname(__file__), "version.txt")
    with open(version_txt) as f:
        __version__ = f.read().strip()

__all__ = ["pdb"]

# backwards compatibility to support `from trev.X import Y`
from trev.distributed import utils as distributed_utils
from trev.logging import meters, metrics, progress_bar  # noqa

sys.modules["trev.distributed_utils"] = distributed_utils
sys.modules["trev.meters"] = meters
sys.modules["trev.metrics"] = metrics
sys.modules["trev.progress_bar"] = progress_bar

# initialize hydra
from trev.dataclass.initialize import hydra_init
hydra_init()

import trev.criterions  # noqa
import trev.distributed  # noqa
import trev.models  # noqa
import trev.modules  # noqa
import trev.optim  # noqa
import trev.optim.lr_scheduler  # noqa
import trev.pdb  # noqa
import trev.scoring  # noqa
import trev.tasks  # noqa
import trev.token_generation_constraints  # noqa

import trev.benchmark  # noqa
import trev.model_parallel  # noqa
