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
    generate_group_matrix: Generate group membership matrix.

"""

import numpy as np


def generate_group_matrix(n_row, group_idx=[]):
    """Generate group ID data structure.

    The first column is reserved and is composed of all zeros.
    Additional columns are optional and determined by the user.

    Arguments:
        n_row: The number of rows.
        group_idx (optional): Array-like integers indicating
            hierarchical group index information. For example, `[4,3]`
            indicates that the first hierarchical level has `group_id`
            4 and the second hierarchical level has `group_id` 3.

    Returns:
        group_matrix: The completed group matrix where each column
            corresponds to a group level and each row a datum.
            shape=(n_row,1+len(group_id))

    """
    group_matrix = np.hstack(([0], group_idx))
    group_matrix = np.expand_dims(group_matrix, axis=0)
    group_matrix = np.repeat(group_matrix, n_row, axis=0)
    return group_matrix

