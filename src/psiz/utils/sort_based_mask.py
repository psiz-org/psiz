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
    mask_matrix

"""


import copy

import numpy as np


def sort_based_mask(a, n_retain, mask_value=0, direction="ascending"):
    """Mask 1D or 2D array of values based on ascending sorted order.

    For each row, mask elements based on ascending sorted order.

    Args:
        a: A 1D or 2D matrix.
        n_retain: The number of elements to retain (keep unmasked) for
            each row.
        mask_value (optional): The value to use for the mask.
        direction (optional): Sort in "ascending" or "decending" order.

    Returns
        a_masked: The matrix with mask applied.

    """
    a_masked = copy.copy(a)

    if a_masked.ndim == 1:
        a_masked = np.expand_dims(a_masked, axis=0)
        is_1d = True
    else:
        is_1d = False

    n_row = a_masked.shape[0]
    for i_row in range(n_row):
        if direction == "ascending":
            idx_sorted = np.argsort(a_masked[i_row])
        elif direction == "decending":
            idx_sorted = np.argsort(-a_masked[i_row])
        else:
            raise NotImplementedError(
                "The argument `direction` must be eight 'acending' or 'decending'."
            )
        # Retain necessary indices.
        idx_mask = idx_sorted[n_retain:]
        # Mask lowest values.
        a_masked[i_row, idx_mask] = mask_value

    # Undo added axis before returning.
    if is_1d:
        a_masked = a_masked[0]

    return a_masked
