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
    matrix_comparison: Compute correlation between two matrices.

"""

import numpy as np
from scipy.stats import pearsonr


def matrix_comparison(mat_a, mat_b, score='r2', elements='upper'):
    """Return a comparison score between two square matrices.

    Arguments:
        mat_a: A square matrix.
        mat_b: A square matrix the same size as mat_a
        score (optional): The type of comparison to use. Can be 'r2',
            'pearson', or 'mse'.
        elements (optional): Which elements to use in the computation.
            The options are upper triangular elements (upper), lower
            triangular elements (lower), off-diagonal elements (off),
            or all elements (all).

    Returns:
        The comparison score.

    """
    n_row = mat_a.shape[0]
    if elements == 'upper':
        idx = np.triu_indices(n_row, 1)
    elif elements == 'lower':
        idx = np.tril_indices(n_row, -1)
    elif elements == 'off':
        idx_upper = np.triu_indices(n_row, 1)
        idx_lower = np.tril_indices(n_row, -1)
        idx = (
            np.hstack((idx_upper[0], idx_lower[0])),
            np.hstack((idx_upper[1], idx_lower[1])),
        )
    elif elements == 'all':
        idx_mesh = np.meshgrid(np.arange(n_row), np.arange(n_row))
        idx = (idx_mesh[0].flatten(), idx_mesh[1].flatten())
    else:
        raise ValueError(
            'The argument to `elements` must be "upper", "lower", or "off".')

    if score == 'r2':
        rho, _ = pearsonr(mat_a[idx], mat_b[idx])
        score = rho**2
    elif score == 'pearson':
        score, _ = pearsonr(mat_a[idx], mat_b[idx])
    elif score == 'mse':
        score = np.mean((mat_a[idx] - mat_b[idx])**2)
    else:
        raise ValueError(
            'The provided `score` argument is not valid.')
    return score
