# -*- coding: utf-8 -*-
# Copyright 2023 The PsiZ Authors. All Rights Reserved.
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

"""Utility module.

Functions:
    sort_based_mask

"""


import copy

import numpy as np


def sort_based_mask(a, n_retain, **kwargs):
    """Mask 1D or 2D array of values based on ascending sorted order.

    For each row, mask elements based on ascending sorted order. If
    elements should be sorted in descending order, pass in `-a`.

    Args:
        a: A 1D or 2D matrix.
        n_retain: The number of elements to retain (keep unmasked) for
            each row.
        kwargs: Key-word arguments to pass to `np.argsort`.

    Returns
        mask: A Boolean mask.

    """
    a_copy = copy.copy(a)

    if a_copy.ndim == 1:
        a_copy = np.expand_dims(a_copy, axis=0)
        is_1d = True
    else:
        is_1d = False

    # Identify indices to keep unmasked.
    idx_sorted = np.argsort(a_copy, **kwargs)
    idx_retain = idx_sorted[:, 0:n_retain]

    # Create Boolean mask
    mask = np.zeros_like(a_copy, dtype=bool)
    n_row = a_copy.shape[0]
    for i_row in range(n_row):
        mask[i_row, idx_retain[i_row]] = True

    # Undo added axis before returning.
    if is_1d:
        mask = mask[0]

    return mask
