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
"""Test utils module."""
import numpy as np

from psiz.utils.m_prefer_n import m_prefer_n


def test_2prefer1():
    """Test outcomes 2 choose 1 ranked trial."""
    m_option = 2
    n_select = 1
    outcomes = m_prefer_n(m_option, n_select)

    desired_outcomes = np.array(((0, 1), (1, 0)))
    np.testing.assert_array_equal(outcomes, desired_outcomes)


def test_3prefer2():
    """Test outcomes 3 choose 2 ranked trial."""
    m_option = 3
    n_select = 2
    outcomes = m_prefer_n(m_option, n_select)

    desired_outcomes = np.array((
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0)
    ))
    np.testing.assert_array_equal(outcomes, desired_outcomes)


def test_4prefer2():
    """Test outcomes 4 choose 2 ranked trial."""
    m_option = 4
    n_select = 2
    outcomes = m_prefer_n(m_option, n_select)

    desired_outcomes = np.array((
        (0, 1, 2, 3),
        (0, 2, 1, 3),
        (0, 3, 1, 2),
        (1, 0, 2, 3),
        (1, 2, 0, 3),
        (1, 3, 0, 2),
        (2, 0, 1, 3),
        (2, 1, 0, 3),
        (2, 3, 0, 1),
        (3, 0, 1, 2),
        (3, 1, 0, 2),
        (3, 2, 0, 1)
    ))
    np.testing.assert_array_equal(outcomes, desired_outcomes)


def test_8prefer1():
    """Test outcomes 8 choose 1 ranked trial."""
    m_option = 8
    n_select = 1
    outcomes = m_prefer_n(m_option, n_select)

    desired_outcomes = np.array((
        (0, 1, 2, 3, 4, 5, 6, 7),
        (1, 0, 2, 3, 4, 5, 6, 7),
        (2, 0, 1, 3, 4, 5, 6, 7),
        (3, 0, 1, 2, 4, 5, 6, 7),
        (4, 0, 1, 2, 3, 5, 6, 7),
        (5, 0, 1, 2, 3, 4, 6, 7),
        (6, 0, 1, 2, 3, 4, 5, 7),
        (7, 0, 1, 2, 3, 4, 5, 6)
    ))
    np.testing.assert_array_equal(outcomes, desired_outcomes)
