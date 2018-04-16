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


def matrix_correlation(mat_A, mat_B):
    """Return the R^2 score between two square matrices.

    Args:
        mat_A: A square matrix.
        mat_B: A square matrix the same size as mat_A
        is_sym: Boolean indicating if the matrices are symmetric.

    Returns:
        The R^2 score between the two matrices.

    """
    n_row = mat_A.shape[0]
    iu1 = np.triu_indices(n_row, 1)

    # Explained variance score.
    return r2_score(mat_A[iu1], mat_B[iu1])


def possible_outcomes(trial_configuration):
    """Return the possible outcomes of a trial configuration.

    Args:
        trial_configuration: A trial configuration Pandas Series.

    Returns:
        An array indicating all possible outcomes where the values
            indicate indices of the reference stimuli.

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
