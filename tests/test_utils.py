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

"""Module for testing utils.py."""


import numpy as np

from psiz.utils import possible_outcomes, matrix_correlation
from psiz.trials import UnjudgedTrials


def test_possible_outcomes_2c1():
    """Test outcomes 2 choose 1 ranked trial."""
    stimulus_set = np.array(((0, 1, 2), (9, 12, 7)))
    n_selected = 1 * np.ones((2))
    tasks = UnjudgedTrials(stimulus_set, n_selected=n_selected)

    po = possible_outcomes(tasks.config_list.iloc[0])

    correct = np.array(((0, 1), (1, 0)))
    np.testing.assert_array_equal(po, correct)


def test_possible_outcomes_3c2():
    """Test outcomes 3 choose 2 ranked trial."""
    stimulus_set = np.array(((0, 1, 2, 3), (33, 9, 12, 7)))
    n_selected = 2 * np.ones((2))
    tasks = UnjudgedTrials(stimulus_set, n_selected=n_selected)

    po = possible_outcomes(tasks.config_list.iloc[0])

    correct = np.array((
        (0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0),
        (2, 0, 1), (2, 1, 0)))
    np.testing.assert_array_equal(po, correct)


def test_possible_outcomes_4c2():
    """Test outcomes 4 choose 2 ranked trial."""
    stimulus_set = np.array(((0, 1, 2, 3, 4), (45, 33, 9, 12, 7)))
    n_selected = 2 * np.ones((2))
    tasks = UnjudgedTrials(stimulus_set, n_selected=n_selected)

    po = possible_outcomes(tasks.config_list.iloc[0])

    correct = np.array((
        (0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2),
        (1, 0, 2, 3), (1, 2, 0, 3), (1, 3, 0, 2),
        (2, 0, 1, 3), (2, 1, 0, 3), (2, 3, 0, 1),
        (3, 0, 1, 2), (3, 1, 0, 2), (3, 2, 0, 1)))
    np.testing.assert_array_equal(po, correct)


def test_possible_outcomes_8c1():
    """Test outcomes 8 choose 1 ranked trial."""
    stimulus_set = np.array((
        (0, 1, 2, 3, 4, 5, 6, 7, 8),
        (45, 33, 9, 12, 7, 2, 5, 4, 3)))
    n_selected = 1 * np.ones((2))
    tasks = UnjudgedTrials(stimulus_set, n_selected=n_selected)

    po = possible_outcomes(tasks.config_list.iloc[0])

    correct = np.array((
        (0, 1, 2, 3, 4, 5, 6, 7),
        (1, 0, 2, 3, 4, 5, 6, 7),
        (2, 0, 1, 3, 4, 5, 6, 7),
        (3, 0, 1, 2, 4, 5, 6, 7),
        (4, 0, 1, 2, 3, 5, 6, 7),
        (5, 0, 1, 2, 3, 4, 6, 7),
        (6, 0, 1, 2, 3, 4, 5, 7),
        (7, 0, 1, 2, 3, 4, 5, 6)))
    np.testing.assert_array_equal(po, correct)


def test_matrix_correlation():
    """Test matrix correlation."""
    a = np.array((
        (1.0, .50, .90, .13),
        (.50, 1.0, .10, .80),
        (.90, .10, 1.0, .12),
        (.13, .80, .12, 1.0)
    ))

    b = np.array((
        (1.0, .45, .90, .11),
        (.45, 1.0, .20, .82),
        (.90, .20, 1.0, .02),
        (.11, .82, .02, 1.0)
    ))

    r2_score_1 = matrix_correlation(a, b)
    np.testing.assert_almost_equal(r2_score_1, 0.96456543)
