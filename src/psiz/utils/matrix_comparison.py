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
        score (optional): The type of comparison to use.
        elements (optional): Which elements to use in the computation.
            The options are upper triangular elements (upper), lower
            triangular elements (lower), or off-diagonal elements
            (off).

    Returns:
        The comparison score.

    Notes:
        'r2'
        When computing R^2 values of two similarity matrices, it is
        assumed, by definition, that the corresponding diagonal
        elements are the same between the two matrices being compared.
        Therefore, the diagonal elements are not included in the R^2
        computation to prevent inflating the R^2 value. On a similar
        note, including both the upper and lower triangular portion
        does not artificially inflate R^2 values for symmetric
        matrices.

    """
    n_row = mat_a.shape[0]
    idx_upper = np.triu_indices(n_row, 1)
    idx_lower = np.triu_indices(n_row, 1)
    if elements == 'upper':
        idx = idx_upper
    elif elements == 'lower':
        idx = idx_lower
    elif elements == 'off':
        idx = (
            np.hstack((idx_upper[0], idx_lower[0])),
            np.hstack((idx_upper[1], idx_lower[1])),
        )
    else:
        raise ValueError(
            'The argument to `elements` must be "upper", "lower", or "off".')

    if score == 'pearson':
        score, _ = pearsonr(mat_a[idx], mat_b[idx])
    elif score == 'r2':
        rho, _ = pearsonr(mat_a[idx], mat_b[idx])
        score = rho**2
        # ALT: score = sklearn.metrics.r2_score(mat_a[idx], mat_b[idx])
    elif score == 'mse':
        score = np.mean((mat_a[idx] - mat_b[idx])**2)
    else:
        raise ValueError(
            'The provided `score` argument is not valid.')
    return score
