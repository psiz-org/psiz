# -*- coding: utf-8 -*-
# Copyright 2021 The PsiZ Authors. All Rights Reserved.
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

from psiz.utils import matrix_comparison


@pytest.fixture
def matrix_a():
    """A synthetic pair-wise matrix."""
    a = np.array((
        (1.0, .50, .90, .13),
        (.50, 1.0, .10, .80),
        (.90, .10, 1.0, .12),
        (.13, .80, .12, 1.0)
    ))
    return a


@pytest.fixture
def matrix_b():
    """A synthetic pair-wise matrix."""
    b = np.array((
        (1.0, .45, .90, .11),
        (.45, 1.0, .20, .82),
        (.90, .20, 1.0, .02),
        (.11, .82, .02, 1.0)
    ))
    return b


@pytest.mark.parametrize(
    "subtest",
    [
        ("upper", 0.96723696), ("lower", 0.96723696), ("off", 0.96723696),
        ("all", 0.98109966)
    ]
)
def test_r2(matrix_a, matrix_b, subtest):
    """Test matrix correlation."""
    elements = subtest[0]
    desired_score = subtest[1]

    actual_score = matrix_comparison(
        matrix_a, matrix_b, score='r2', elements=elements
    )
    np.testing.assert_almost_equal(actual_score, desired_score)


@pytest.mark.parametrize(
    "subtest",
    [
        ("upper", 0.98348206), ("lower", 0.98348206), ("off", 0.98348206),
        ("all", 0.99050475)
    ]
)
def test_pearson(matrix_a, matrix_b, subtest):
    """Test matrix correlation."""
    elements = subtest[0]
    desired_score = subtest[1]

    actual_score = matrix_comparison(
        matrix_a, matrix_b, score='pearson', elements=elements
    )
    np.testing.assert_almost_equal(actual_score, desired_score)


@pytest.mark.parametrize(
    "subtest",
    [
        ("upper", 0.00388333), ("lower", 0.00388333), ("off", 0.00388333),
        ("all", 0.00291249)
    ]
)
def test_mse(matrix_a, matrix_b, subtest):
    """Test matrix correlation."""
    elements = subtest[0]
    desired_score = subtest[1]

    actual_score = matrix_comparison(
        matrix_a, matrix_b, score='mse', elements=elements
    )
    np.testing.assert_almost_equal(actual_score, desired_score)


def test_invalid_0(matrix_a, matrix_b):
    """Test invalid inputs."""
    with pytest.raises(Exception) as e_info:
        matrix_comparison(matrix_a, matrix_b, score='garbage')
    assert e_info.type == ValueError

    with pytest.raises(Exception) as e_info:
        matrix_comparison(matrix_a, matrix_b, elements='garbage')
    assert e_info.type == ValueError
