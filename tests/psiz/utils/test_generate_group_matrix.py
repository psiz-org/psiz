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
"""Module for testing utils.py."""

import numpy as np

from psiz.utils import generate_group_matrix


def test_generate_group_matrix():
    """Test generate_group_matrix."""
    n_row = 3

    # Test default.
    group_matrix = generate_group_matrix(n_row)
    desired_group_matrix = np.array([
        [0],
        [0],
        [0]
    ])
    np.testing.assert_array_equal(group_matrix, desired_group_matrix)

    # Test one-level hierarchy.
    groups = [0, 0]
    group_matrix = generate_group_matrix(n_row, groups=groups)
    desired_group_matrix = np.array([
        [0, 0],
        [0, 0],
        [0, 0]
    ])
    np.testing.assert_array_equal(group_matrix, desired_group_matrix)

    # Test three-level hierarchy.
    groups = [0, 6, 7, 3]
    group_matrix = generate_group_matrix(n_row, groups=groups)
    desired_group_matrix = np.array([
        [0, 6, 7, 3],
        [0, 6, 7, 3],
        [0, 6, 7, 3]
    ])
    np.testing.assert_array_equal(group_matrix, desired_group_matrix)
