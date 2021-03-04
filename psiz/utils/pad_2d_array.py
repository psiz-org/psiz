
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
"""Module of utility functions.

Functions:
    pad_2d_array: Pad a 2D array with a value.

"""

import numpy as np


def pad_2d_array(arr, n_column, value=-1):
    """Pad 2D array with columns composed of -1.

    Argument:
        arr: A 2D array denoting the stimulus set.
        n_column: The total number of columns that the array should
            have.
        value (optional): The value to use to pad the array.

    Returns:
        Padded array.

    """
    n_trial = arr.shape[0]
    n_pad = n_column - arr.shape[1]
    if n_pad > 0:
        pad_mat = value * np.ones((n_trial, n_pad), dtype=np.int32)
        arr = np.hstack((arr, pad_mat))
    return arr
