# -*- coding: utf-8 -*-
# Copyright 2022 The PsiZ Authors. All Rights Reserved.
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

from itertools import permutations

import numpy as np


def m_prefer_n(m_option, n_select):
    """Return the possible outcomes of an m-prefer-n event.

    Given `m` options, select the `n` options that are most preferred
    and rank the selected options.

    Args:
        m_option: Integer indicating the number of options.
        n_select: Integer indicating the number of most preferred
            options. Must be less than or equal to `m_options`.

    Returns:
        A 2D array indicating all possible outcomes where the values
        indicate indices of the options. Each row corresponds to one
        outcome. Note the indices refer to the options only and
        does not include an index for a query or prompt. Also note that
        the unpermuted index is returned first.

    """
    # Cast if necessary.
    m_option = int(m_option)
    n_select = int(n_select)

    option_list = range(m_option)

    # Get all permutations of length n_select.
    perm = permutations(option_list, n_select)

    selection = list(perm)
    n_outcome = len(selection)

    outcomes = np.empty((n_outcome, m_option), dtype=np.int32)
    for i_outcome in range(n_outcome):
        # Fill in selections.
        outcomes[i_outcome, 0:n_select] = selection[i_outcome]
        # Fill in unselected.
        dummy_idx = np.arange(m_option)
        for i_selected in range(n_select):
            loc = dummy_idx != outcomes[i_outcome, i_selected]
            dummy_idx = dummy_idx[loc]

        outcomes[i_outcome, n_select:] = dummy_idx

    return outcomes
