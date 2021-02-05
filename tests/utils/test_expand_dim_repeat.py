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
import pytest
import tensorflow as tf

from psiz.utils import expand_dim_repeat


def test_expand_dim_repeat_empty():
    x = tf.constant(
        np.array([
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [9, 10, 11],
            [12, 13, 14]
        ])
    )

    n_sample = ()
    with pytest.raises(Exception) as e_info:
        output = expand_dim_repeat(x, n_sample, axis=1)
    assert str(e_info.value) == (
        'Dimensions 1 and 0 are not compatible'
    )


def test_expand_dim_repeat_1():
    x = tf.constant(
        np.array([
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [9, 10, 11],
            [12, 13, 14]
        ])
    )

    n_sample = 1
    output = expand_dim_repeat(x, n_sample, axis=1)

    desired_output = np.array([
        [[0, 1, 2]],
        [[3, 4, 5]],
        [[6, 7, 8]],
        [[9, 10, 11]],
        [[12, 13, 14]]
    ])
    np.testing.assert_array_equal(
        desired_output, output
    )


def test_expand_dim_repeat_2():
    x = tf.constant(
        np.array([
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [9, 10, 11],
            [12, 13, 14]
        ])
    )

    n_sample = 2
    output = expand_dim_repeat(x, n_sample, axis=1)

    desired_output = np.array([
        [[0, 1, 2], [0, 1, 2]],
        [[3, 4, 5], [3, 4, 5]],
        [[6, 7, 8], [6, 7, 8]],
        [[9, 10, 11], [9, 10, 11]],
        [[12, 13, 14], [12, 13, 14]]
    ])
    np.testing.assert_array_equal(
        desired_output, output
    )
