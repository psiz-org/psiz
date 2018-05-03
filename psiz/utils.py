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

"""Module of helpful utility functions."""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from itertools import permutations


def matrix_correlation(mat_a, mat_b):
    """Return the R^2 score between two square matrices.

    Args:
        mat_a: A square matrix.
        mat_b: A square matrix the same size as mat_a

    Returns:
        The R^2 score between the two matrices.

    Notes:
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
    idx = (
        np.hstack((idx_upper[0], idx_lower[0])),
        np.hstack((idx_upper[1], idx_lower[1])),
    )
    # Explained variance score.
    return r2_score(mat_a[idx], mat_b[idx])


def possible_outcomes(trial_configuration):
    """Return the possible outcomes of a trial configuration.

    Args:
        trial_configuration: A trial configuration Pandas Series.

    Returns:
        An 2D array indicating all possible outcomes where the values
            indicate indices of the reference stimuli. Each row
            corresponds to one outcome. Note the indices refer to
            references only and does not include an index for the
            query.

    """
    n_reference = trial_configuration['n_reference']
    n_selected = int(trial_configuration['n_selected'])

    reference_list = range(n_reference)

    # Get all permutations of length n_selected.
    perm = permutations(reference_list, n_selected)

    selection = list(perm)
    n_outcome = len(selection)

    outcomes = np.empty((n_outcome, n_reference), dtype=np.int32)
    for i_outcome in range(n_outcome):
        # Fill in selections.
        outcomes[i_outcome, 0:n_selected] = selection[i_outcome]
        # Fill in unselected.
        dummy_idx = np.arange(n_reference)
        for i_selected in range(n_selected):
            loc = dummy_idx != outcomes[i_outcome, i_selected]
            dummy_idx = dummy_idx[loc]

        outcomes[i_outcome, n_selected:] = dummy_idx

    return outcomes
