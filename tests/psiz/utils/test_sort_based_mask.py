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

"""Test sort_based_mask.

NOTE: When `mask_zero=True`, the element IDs start at 1, not 0.

"""


import numpy as np

from psiz.utils.sort_based_mask import sort_based_mask


def test_1d():
    w = np.array([1.0, 0.8, 0.6, 0.4])

    w_mask = sort_based_mask(w, 2)

    w_masked_desired = np.array([False, False, True, True])
    np.testing.assert_almost_equal(w_mask, w_masked_desired)

    w_mask = sort_based_mask(-w, 2)
    w_masked_desired = np.array([True, True, False, False])
    np.testing.assert_almost_equal(w_mask, w_masked_desired)


def test_2d():
    w = np.array(
        [
            [1.0, 0.8, 0.6, 0.4],
            [0.7, 0.0, 0.2, 0.1],
            [0.1, 0.2, 0.6, 0.6],
            [0.5, 0.5, 0.7, 0.1],
        ]
    )

    w_mask = sort_based_mask(w, 2)

    w_masked_desired = np.array(
        [
            [False, False, True, True],
            [False, True, False, True],
            [True, True, False, False],
            [True, False, False, True],
        ]
    )
    np.testing.assert_equal(w_mask, w_masked_desired)

    w_mask = sort_based_mask(-w, 2)
    w_masked_desired = np.array(
        [
            [True, True, False, False],
            [True, False, True, False],
            [False, False, True, True],
            [True, False, True, False],
        ]
    )
    np.testing.assert_equal(w_mask, w_masked_desired)
