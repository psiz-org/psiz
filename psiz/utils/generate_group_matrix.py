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


def generate_group_matrix(n_row, groups=None):
    """Generate group ID data structure.

    Arguments:
        n_row: The number of rows.
        groups (optional): Array-like integers indicating group
            membership information. For example, `[4, 3]` indicates
            that the first column has the value 4 and the second
            column has the value 3.

    Returns:
        group_matrix: The completed group matrix where each column
            corresponds to a different distinction and each row
            corresponds to something like number of trials.
            shape=(n_row, len(groups))

    """
    if groups is None:
        groups = [0]

    group_matrix = np.asarray(groups, dtype=np.int32)
    group_matrix = np.expand_dims(group_matrix, axis=0)
    group_matrix = np.repeat(group_matrix, n_row, axis=0)
    return group_matrix
