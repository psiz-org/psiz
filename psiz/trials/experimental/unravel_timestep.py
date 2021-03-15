# -*- coding: utf-8 -*-
# Copyright 2020 The PsiZ Authors. All Rights Reserved.
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
# ============================================================================
"""Module for trials.

Functions:
    unravel_timestep: Utility function for unraveling timestep-based
        data structures.

"""

import numpy as np


def unravel_timestep(x):
    """Unravel sequence and timestep axis into a single axis.

    Argumetns:
        x: A time-step based data structure.
            shape=(n_sequence, max_timestep, [m, n, ...])

    Returns:
        New data structure.
            shape=(n_sequence*max_timestep, [m, n, ...])

    """
    x_shape = x.shape
    n_sequence = x_shape[0]
    max_timestep = x_shape[1]
    new_shape = np.hstack(
        ([n_sequence * max_timestep], x_shape[2:])
    ).astype(dtype=int)
    return np.reshape(x, new_shape)
