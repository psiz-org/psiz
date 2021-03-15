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
"""Test trials module."""

import numpy as np
import pytest
import tensorflow as tf

from psiz.trials.experimental.contents.rank_similarity import RankSimilarity
from psiz.trials.experimental.contents.rate_similarity import RateSimilarity
from psiz.trials.experimental.outcomes.continuous import Continuous
from psiz.trials.experimental.outcomes.sparse_categorical import SparseCategorical


@pytest.fixture(scope="module")
def rank_sim_0():
    """Content RankedSimilarity with minimal rank arguments."""
    stimulus_set = np.array(
        [
            [3, 1, 2, 0, 0, 0, 0, 0, 0, 0],
            [9, 12, 7, 0, 0, 0, 0, 0, 0, 0],
            [3, 4, 5, 6, 7, 0, 0, 0, 0, 0],
            [3, 4, 5, 6, 13, 14, 15, 16, 17, 0]
        ], dtype=np.int32
    )
    n_select = np.array([1, 1, 1, 2], dtype=np.int32)

    return RankSimilarity(stimulus_set, n_select=n_select)


@pytest.fixture(scope="module")
def rank_sim_1():
    """Content RankedSimilarity with true rank arguments.

    Notes:
    The input arguments are full rank, but singleton on the
        timestep axis.
    There is intentionally an extra reference placeholder that should
        be trimmed during initalization.

    """
    stimulus_set = np.array(
        [
            [[3, 1, 2, 0, 0, 0, 0, 0, 0, 0]],
            [[9, 12, 7, 0, 0, 0, 0, 0, 0, 0]],
            [[3, 4, 5, 6, 7, 0, 0, 0, 0, 0]],
            [[3, 4, 5, 6, 13, 14, 15, 16, 17, 0]]
        ], dtype=np.int32
    )
    n_select = np.array([[1], [1], [1], [2]], dtype=np.int32)

    return RankSimilarity(stimulus_set, n_select=n_select)


@pytest.fixture(scope="module")
def rank_sim_2():
    """Content RankedSimilarity with true rank arguments.

    Notes:
    There is intentionally an extra reference placeholder that should
        be trimmed during initalization.

    """
    stimulus_set = np.array(
        [
            [
                [3, 1, 2, 0, 0, 0, 0, 0, 0, 0],
                [3, 1, 2, 0, 0, 0, 0, 0, 0, 0]
            ],
            [
                [9, 12, 7, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ],
            [
                [3, 4, 5, 6, 7, 0, 0, 0, 0, 0],
                [3, 4, 5, 0, 0, 0, 0, 0, 0, 0]
            ],
            [
                [3, 4, 5, 6, 13, 14, 15, 16, 17, 0],
                [3, 4, 5, 6, 13, 14, 15, 16, 17, 0]
            ]
        ], dtype=np.int32
    )
    n_select = np.array(
        [[1, 1], [1, 0], [1, 1], [2, 1]], dtype=np.int32
    )

    return RankSimilarity(stimulus_set, n_select=n_select)


@pytest.fixture(scope="module")
def rank_sim_3():
    """Content RankedSimilarity with true rank arguments.

    This instance also has an extra timestep and reference that should
    be trimmed at initialization.

    """
    stimulus_set = np.array(
        [
            [
                [3, 1, 2, 0, 0, 0, 0, 0, 0, 0],
                [3, 1, 2, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ],
            [
                [9, 12, 7, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ],
            [
                [3, 4, 5, 6, 7, 0, 0, 0, 0, 0],
                [3, 4, 5, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ],
            [
                [3, 4, 5, 6, 13, 14, 15, 16, 17, 0],
                [3, 4, 5, 6, 13, 14, 15, 16, 17, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ]
        ], dtype=np.int32
    )
    n_select = np.array(
        [[1, 1, 0], [1, 0, 0], [1, 1, 0], [2, 1, 0]], dtype=np.int32
    )

    return RankSimilarity(stimulus_set, n_select=n_select)


@pytest.fixture(scope="module")
def rank_sim_4():
    """Content RankedSimilarity with true rank arguments.

    A set of trials with relatively simple outcomes.

    """
    stimulus_set = np.array(
        [
            [
                [1, 2, 3, 0],
                [4, 5, 6, 0],
            ],
            [
                [7, 8, 9, 0],
                [0, 0, 0, 0],
            ],
            [
                [10, 11, 12, 13],
                [14, 15, 16, 0],
            ]
        ], dtype=np.int32
    )
    n_select = np.array(
        [[1, 1], [1, 0], [1, 1]], dtype=np.int32
    )

    return RankSimilarity(stimulus_set, n_select=n_select)


@pytest.fixture(scope="module")
def rank_sim_5():
    """Content RankedSimilarity with true rank arguments.

    A set of trials with relatively simple outcomes.

    """
    stimulus_set = np.array(
        [
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ],
            [
                [10, 11, 12],
                [13, 14, 15],
                [16, 17, 18],
            ],
        ], dtype=np.int32
    )
    n_select = np.array(
        [[1, 1, 1], [1, 1, 1]], dtype=np.int32
    )

    return RankSimilarity(stimulus_set, n_select=n_select)


@pytest.fixture(scope="module")
def rate_sim_0():
    """Content RateSimilarity with minimal rank arguments."""
    stimulus_set = np.array(
        [
            [3, 1],
            [9, 12],
            [3, 4],
            [3, 4]
        ], dtype=np.int32
    )

    return RateSimilarity(stimulus_set)


@pytest.fixture(scope="module")
def rate_sim_1():
    """Content RateSimilarity with true rank arguments.

    Notes:
    The input arguments are full rank, but singleton on the
        timestep axis.
    There is intentionally an extra reference placeholder that should
        be trimmed during initalization.

    """
    stimulus_set = np.array(
        [
            [[3, 1]],
            [[9, 12]],
            [[3, 4]],
            [[3, 4]]
        ], dtype=np.int32
    )

    return RateSimilarity(stimulus_set)


@pytest.fixture(scope="module")
def rate_sim_2():
    """Content RateSimilarity with true rank arguments.

    Notes:
    There is intentionally an extra reference placeholder that should
        be trimmed during initalization.

    """
    stimulus_set = np.array(
        [
            [
                [3, 1],
                [3, 1]
            ],
            [
                [9, 12],
                [0, 0]
            ],
            [
                [3, 4],
                [3, 4]
            ],
            [
                [3, 4],
                [3, 4]
            ]
        ], dtype=np.int32
    )

    return RateSimilarity(stimulus_set)


@pytest.fixture(scope="module")
def rate_sim_3():
    """Content RateSimilarity with true rank arguments.

    This instance also has an extra timestep and reference that should
    be trimmed at initialization.

    """
    stimulus_set = np.array(
        [
            [
                [3, 1],
                [3, 1],
                [0, 0]
            ],
            [
                [9, 12],
                [0, 0],
                [0, 0]
            ],
            [
                [3, 4],
                [3, 4],
                [0, 0]
            ],
            [
                [3, 4],
                [3, 4],
                [0, 0]
            ]
        ], dtype=np.int32
    )

    return RateSimilarity(stimulus_set)


@pytest.fixture(scope="module")
def rate_sim_4():
    """Content RateSimilarity with true rank arguments.

    Used in test_stack.

    """
    stimulus_set = np.array(
        [
            [
                [5, 6],
                [7, 8],
                [9, 10],
            ],
            [
                [1, 2],
                [3, 4],
                [0, 0]
            ],
        ], dtype=np.int32
    )

    return RateSimilarity(stimulus_set)


@pytest.fixture(scope="module")
def continuous_0():
    """Outcome Continuous with minimal rank arguments."""
    outcome = np.array(
        [[0.0], [2.0], [-0.1], [1.3]], dtype=np.float32
    )
    return Continuous(outcome)


@pytest.fixture(scope="module")
def continuous_1():
    """Outcome Continuous with rull rank arguments (singleton)."""
    outcome = np.array(
        [[[0.0]], [[2.0]], [[-0.1]], [[1.3]]], dtype=np.float32
    )
    return Continuous(outcome)


@pytest.fixture(scope="module")
def continuous_2():
    """Outcome Continuous with rull rank arguments."""
    outcome = np.array(
        [
            [[0.0], [0.0], [0.0]],
            [[2.0], [0.0], [0.0]],
            [[-0.1], [-1.0], [0.3]],
            [[1.0], [1.0], [1.0]],
        ], dtype=np.float32
    )
    return Continuous(outcome)


@pytest.fixture(scope="module")
def continuous_3():
    """Outcome Continuous with rull rank arguments."""
    outcome = np.array(
        [
            [[0.0, 0.1], [0.0, 0.2], [0.0, 0.3]],
            [[2.0, 0.4], [0.0, 0.5], [0.0, 0.6]],
            [[-0.1, 0.7], [-1.0, 0.8], [0.3, 0.9]],
            [[1.0, 1.1], [1.0, 1.2], [1.0, 1.3]],
        ], dtype=np.float32
    )
    return Continuous(outcome)


@pytest.fixture(scope="module")
def continuous_4():
    """Outcome Continuous with rull rank arguments."""
    outcome = np.array(
        [
            [[2.0, 2.1], [2.0, 2.2], [2.0, 2.3]],
            [[3.0, 3.4], [3.0, 3.5], [3.0, 3.6]],
        ], dtype=np.float32
    )
    return Continuous(outcome)


@pytest.fixture(scope="module")
def sparse_cat_0():
    """Outcome SparseCategorical with minimal rank arguments."""
    outcome_idx = np.array(
        [0, 2, 0, 1], dtype=np.int32
    )
    return SparseCategorical(outcome_idx, depth=3)


@pytest.fixture(scope="module")
def sparse_cat_1():
    """Outcome SparseCategorical with rull rank arguments (singleton)."""
    outcome_idx = np.array(
        [[0], [2], [0], [1]], dtype=np.int32
    )
    return SparseCategorical(outcome_idx, depth=5)


@pytest.fixture(scope="module")
def sparse_cat_2():
    """Outcome SparseCategorical with rull rank arguments."""
    outcome_idx = np.array(
        [
            [0, 0, 0],
            [2, 0, 0],
            [0, 1, 0],
            [1, 1, 1],
        ], dtype=np.int32
    )
    return SparseCategorical(outcome_idx, depth=3)
