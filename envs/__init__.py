# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os, sys
VAE_PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    "../task_embed/clutr_RVAE/")
)
sys.path.append(VAE_PROJECT_ROOT)

from .multigrid.adversarial import *

