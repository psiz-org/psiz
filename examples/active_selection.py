# -*- coding: utf-8 -*-
# Copyright 2018 The PsiZ Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Example that demonstrates the advantage of using active selection.

This example using simulated behavior to illustrate the theoretical
advantage of using active selection over random trial selection. The
entire simulation takes a long time.

"""

import copy
import itertools

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from psiz.trials import UnjudgedTrials, stack
from psiz.models import Exponential
from psiz.generator import ActiveGenerator


def main():
    """Sample from posterior of pre-defined embedding model."""
    # Settings.
    np.random.seed(126)
    n_sample = 2000


if __name__ == "__main__":
    main()
