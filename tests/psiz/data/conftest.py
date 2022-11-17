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
"""Test data module.

DatasetComponent pytest fixtures follow the naming convention:
x_y_z_mxn
x: trial component type (c: content, o: outcome, g: group)
y: helpful name (rank, rate, rt, continous, catsparse)
z: variant identifier
mxn: n_sample x sequence_length

"""

import numpy as np
import pytest
from tensorflow.keras.utils import to_categorical

from psiz.data.contents.categorize import Categorize
from psiz.data.contents.rank import Rank
from psiz.data.contents.rate import Rate
from psiz.data.groups.group import Group
from psiz.data.outcomes.continuous import Continuous
from psiz.data.outcomes.sparse_categorical import SparseCategorical


@pytest.fixture(scope="module")
def c_2rank1_a_4x1():
    """Content RankedSimilarity with rank-2 arguments."""
    stimulus_set = np.array(
        [
            [3, 1, 2],
            [9, 12, 7],
            [5, 6, 7],
            [13, 14, 15]
        ], dtype=np.int32
    )
    n_select = 1
    return Rank(stimulus_set, n_select=n_select)


@pytest.fixture(scope="module")
def c_2rank1_aa_4x1():
    """Content Rank with rank-3 arguments.

    Notes:
    The input arguments are full rank, but singleton on the
        timestep axis.
    There is intentionally an extra reference placeholder that should
        be trimmed during initalization.

    """
    stimulus_set = np.array(
        [
            [[3, 1, 2]],
            [[9, 12, 7]],
            [[5, 6, 7]],
            [[13, 14, 15]]
        ], dtype=np.int32
    )
    n_select = 1
    return Rank(stimulus_set, n_select=n_select)


@pytest.fixture(scope="module")
def o_2rank1_aa_4x1():
    outcome_idx = np.zeros([4, 1], dtype=np.int32)
    sample_weight = .9 * np.ones([4, 1])
    n_outcome = 2
    rank_outcome = SparseCategorical(
        outcome_idx,
        depth=n_outcome,
        sample_weight=sample_weight,
        name='rank_prob'
    )
    return rank_outcome


@pytest.fixture(scope="module")
def c_4rank2_b_4x2():
    """Content RankedSimilarity with rank-3 arguments.

    Notes:
    There is intentionally a placeholder trial in one sequence.

    """
    stimulus_set = np.array(
        [
            [
                [1, 2, 3, 4, 5],
                [5, 4, 3, 2, 1]
            ],
            [
                [9, 12, 7, 13, 14],
                [0, 0, 0, 0, 0]
            ],
            [
                [3, 4, 5, 6, 7],
                [7, 8, 9, 10, 11]
            ],
            [
                [13, 14, 15, 16, 17],
                [14, 15, 16, 17, 18]
            ]
        ], dtype=np.int32
    )
    n_select = 2
    return Rank(stimulus_set, n_select=n_select)


@pytest.fixture(scope="module")
def c_4rank2_c_4x3():
    """Content RankedSimilarity with rank-3 arguments.

    * An extra sequence that should not be trimmed.
    * An extra reference that should be trimmed at initialization.

    """
    stimulus_set = np.array(
        [
            [
                [3, 1, 2, 4, 5],
                [3, 1, 2, 6, 7],
                [0, 0, 0, 0, 0]
            ],
            [
                [9, 12, 7, 14, 5],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]
            ],
            [
                [3, 4, 5, 6, 7],
                [8, 9, 10, 11, 12],
                [0, 0, 0, 0, 0]
            ],
            [
                [1, 3, 5, 7, 9],
                [11, 9, 7, 5, 3],
                [0, 0, 0, 0, 0]
            ]
        ], dtype=np.int32
    )
    n_select = 2
    return Rank(stimulus_set, n_select=n_select)


@pytest.fixture(scope="module")
def o_4rank2_c_4x3():
    outcome_idx = np.zeros([4, 3], dtype=np.int32)
    sample_weight = .9 * np.ones([4, 3])
    n_outcome = 12
    rank_outcome = SparseCategorical(
        outcome_idx,
        depth=n_outcome,
        sample_weight=sample_weight,
        name='rank_prob'
    )
    return rank_outcome


@pytest.fixture(scope="module")
def c_2rank1_d_3x2():
    """Content RankedSimilarity with rank-3 arguments.

    A set of trials with relatively simple outcomes.

    """
    stimulus_set = np.array(
        [
            [
                [1, 2, 3],
                [4, 5, 6],
            ],
            [
                [7, 8, 9],
                [0, 0, 0],
            ],
            [
                [10, 11, 12],
                [14, 15, 16],
            ]
        ], dtype=np.int32
    )
    n_select = 1
    return Rank(stimulus_set, n_select=n_select)


@pytest.fixture(scope="module")
def o_2rank1_d_3x2():
    outcome_idx = np.zeros([3, 2], dtype=np.int32)
    sample_weight = .9 * np.ones([3, 2])
    n_outcome = 2
    rank_outcome = SparseCategorical(
        outcome_idx,
        depth=n_outcome,
        sample_weight=sample_weight,
        name='rank_prob'
    )
    return rank_outcome


@pytest.fixture(scope="module")
def o_rt_a_3x2():
    rt = np.array(
        [
            [[4.1], [4.2]],
            [[5.1], [5.2]],
            [[6.1], [6.2]],
        ]
    )
    sample_weight = .8 * np.ones([3, 2])
    return Continuous(
        rt, sample_weight=sample_weight, name='rt'
    )


@pytest.fixture(scope="module")
def o_rt_a_3x2_noname():
    rt = np.array(
        [
            [[4.1], [4.2]],
            [[5.1], [5.2]],
            [[6.1], [6.2]],
        ]
    )
    sample_weight = .8 * np.ones([3, 2])
    return Continuous(rt, sample_weight=sample_weight)


@pytest.fixture(scope="module")
def c_2rank1_e_3x2():
    """Content RankedSimilarity with rank-3 arguments.

    A set of trials with relatively simple outcomes.

    """
    stimulus_set = np.array(
        [
            [
                [1, 2, 3],
                [4, 5, 6],
            ],
            [
                [7, 8, 9],
                [10, 11, 12],
            ],
            [
                [13, 14, 15],
                [16, 17, 18],
            ],
        ], dtype=np.int32
    )
    n_select = 1
    return Rank(stimulus_set, n_select=n_select)


# TODO use or delete
@pytest.fixture(scope="module")
def c_2rank1_f_2x4():
    """Content RankedSimilarity with rank-3 arguments.

    A set of trials with relatively simple outcomes.

    """
    stimulus_set = np.array(
        [
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
                [7, 8, 9],
            ],
            [
                [10, 11, 12],
                [13, 14, 15],
                [16, 17, 18],
                [16, 17, 18],
            ],
        ], dtype=np.int32
    )
    n_select = 1
    return Rank(stimulus_set, n_select=n_select)


@pytest.fixture(scope="module")
def c_rate2_a_4x1():
    """Content Rate with minimal rank arguments."""
    stimulus_set = np.array(
        [
            [3, 1],
            [9, 12],
            [3, 4],
            [3, 4]
        ], dtype=np.int32
    )

    return Rate(stimulus_set)


@pytest.fixture(scope="module")
def c_rate2_aa_4x1():
    """Content Rate with rank-3 arguments.

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

    return Rate(stimulus_set)


@pytest.fixture(scope="module")
def c_rate2_b_4x2():
    """Content Rate with rank-3 arguments.

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

    return Rate(stimulus_set)


@pytest.fixture(scope="module")
def c_rate2_c_4x3():
    """Content Rate with rank-3 arguments.

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

    return Rate(stimulus_set)


@pytest.fixture(scope="module")
def c_rate2_d_2x3():
    """Content Rate with rank-3 arguments.

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

    return Rate(stimulus_set)


@pytest.fixture(scope="module")
def c_rate2_e_3x2():
    """Content Rate with rank-3 arguments."""
    stimulus_set = np.array(
        [
            [
                [5, 6],
                [7, 8],
            ],
            [
                [1, 9],
                [8, 7],
            ],
            [
                [1, 2],
                [3, 4]
            ]
        ], dtype=np.int32
    )

    return Rate(stimulus_set)


@pytest.fixture(scope="module")
def o_rate2_a_3x2():
    """Content Rate with rank-3 arguments."""
    ratings = np.array(
        [
            [
                [0.5, 0.6],
                [0.7, 0.8],
            ],
            [
                [1.0, 0.9],
                [0.8, 0.7],
            ],
            [
                [0.1, 0.2],
                [0.3, 0.4]
            ]
        ], dtype=np.float32
    )

    return Continuous(ratings, name="rate_a")


@pytest.fixture(scope="module")
def c_categorize_a_4x10():
    """Content.

    n_sample = 4
    sequence_length = 10
    n_stimuli = 20
    n_output = 3

    """
    # NOTE: Last sequence has 2 placeholder trials.
    stimulus_set = np.array(
        [
            [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]],
            [[11], [12], [13], [14], [15], [16], [17], [18], [19], [20]],
            [[1], [3], [5], [7], [9], [11], [13], [15], [17], [19]],
            [[2], [4], [6], [8], [10], [12], [14], [16], [0], [0]],
        ], dtype=np.int32
    )
    objective_query_label = np.array(
        [
            [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0]],
            [[1], [1], [1], [1], [1], [2], [2], [2], [2], [2]],
            [[0], [0], [0], [0], [0], [1], [1], [1], [2], [2]],
            [[0], [0], [0], [0], [0], [1], [1], [2], [0], [0]],
        ], dtype=np.int32
    )
    objective_query_label = to_categorical(
        objective_query_label, num_classes=3
    )
    content = Categorize(
        stimulus_set=stimulus_set, objective_query_label=objective_query_label
    )
    return content


@pytest.fixture(scope="module")
def c_categorize_b_4x3():
    """Content.

    n_sample = 4
    sequence_length = 3
    n_stimuli = 20
    n_output = 3

    """
    stimulus_set = np.array(
        [
            [[1], [2], [3]],
            [[11], [12], [13]],
            [[1], [3], [5]],
            [[2], [4], [6]],
        ], dtype=np.int32
    )
    objective_query_label = np.array(
        [
            [[0], [0], [0]],
            [[1], [1], [2]],
            [[0], [1], [2]],
            [[2], [2], [0]],
        ], dtype=np.int32
    )
    objective_query_label = to_categorical(
        objective_query_label, num_classes=3
    )
    content = Categorize(
        stimulus_set=stimulus_set, objective_query_label=objective_query_label
    )
    return content


@pytest.fixture(scope="module")
def o_continuous_a_4x1():
    """Outcome Continuous with minimal rank arguments."""
    outcome = np.array(
        [[0.0], [2.0], [-0.1], [1.3]], dtype=np.float32
    )
    return Continuous(outcome, name='continuous_a')


@pytest.fixture(scope="module")
def o_continuous_aa_4x1():
    """Outcome Continuous with full rank arguments (singleton)."""
    outcome = np.array(
        [[[0.0]], [[2.0]], [[-0.1]], [[1.3]]], dtype=np.float32
    )
    return Continuous(outcome, name='continuous_aa')


@pytest.fixture(scope="module")
def o_continuous_b_4x3():
    """Outcome Continuous with full rank arguments."""
    outcome = np.array(
        [
            [[0.0], [0.0], [0.0]],
            [[2.0], [0.0], [0.0]],
            [[-0.1], [-1.0], [0.3]],
            [[1.0], [1.0], [1.0]],
        ], dtype=np.float32
    )
    return Continuous(outcome, name='continuous_b')


@pytest.fixture(scope="module")
def o_continuous_c_4x3():
    """Outcome Continuous with full rank arguments."""
    outcome = np.array(
        [
            [[0.0, 0.1], [0.0, 0.2], [0.0, 0.3]],
            [[2.0, 0.4], [0.0, 0.5], [0.0, 0.6]],
            [[-0.1, 0.7], [-1.0, 0.8], [0.3, 0.9]],
            [[1.0, 1.1], [1.0, 1.2], [1.0, 1.3]],
        ], dtype=np.float32
    )
    return Continuous(outcome, name='continuous_c')


@pytest.fixture(scope="module")
def o_continuous_d_2x3():
    """Outcome Continuous with full rank arguments."""
    outcome = np.array(
        [
            [[2.0, 2.1], [2.0, 2.2], [2.0, 2.3]],
            [[3.0, 3.4], [3.0, 3.5], [3.0, 3.6]],
        ], dtype=np.float32
    )
    return Continuous(outcome, name='continuous_d')


@pytest.fixture(scope="module")
def o_continuous_e_4x3():
    """Outcome Continuous with full rank arguments.

    * with sample weights

    """
    outcome = np.array(
        [
            [[0.0], [0.0], [0.0]],
            [[2.0], [0.0], [0.0]],
            [[-0.1], [-1.0], [0.3]],
            [[1.0], [1.0], [1.0]],
        ], dtype=np.float32
    )
    sample_weight = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
            [1.0, 1.0, 0.0],
        ], dtype=np.float32
    )
    return Continuous(
        outcome, name='continuous_e', sample_weight=sample_weight
    )


@pytest.fixture(scope="module")
def o_sparsecat_a_4x1():
    """Outcome SparseCategorical with minimal rank arguments."""
    outcome_idx = np.array(
        [0, 2, 0, 1], dtype=np.int32
    )
    return SparseCategorical(outcome_idx, depth=3, name='sparsecat_a')


@pytest.fixture(scope="module")
def o_sparsecat_aa_4x1():
    """Outcome SparseCategorical with full rank arguments (singleton)."""
    outcome_idx = np.array(
        [[0], [2], [0], [1]], dtype=np.int32
    )
    return SparseCategorical(outcome_idx, depth=5, name='sparsecat_aa')


@pytest.fixture(scope="module")
def o_sparsecat_b_4x3():
    """Outcome SparseCategorical with full rank arguments."""
    outcome_idx = np.array(
        [
            [0, 0, 0],
            [2, 0, 0],
            [0, 1, 0],
            [1, 1, 1],
        ], dtype=np.int32
    )
    return SparseCategorical(outcome_idx, depth=3, name='sparsecat_b')


@pytest.fixture(scope="module")
def o_sparsecat_c_2x3():
    """Outcome SparseCategorical with full rank arguments."""
    outcome_idx = np.array(
        [
            [0, 2, 1],
            [1, 2, 2],
        ], dtype=np.int32
    )
    return SparseCategorical(outcome_idx, depth=3, name='sparsecat_c')


@pytest.fixture(scope="module")
def o_sparsecat_d_4x3():
    """Outcome SparseCategorical with full rank arguments."""
    outcome_idx = np.array(
        [
            [0, 0, 0],
            [2, 0, 0],
            [0, 1, 0],
            [1, 1, 1],
        ], dtype=np.int32
    )
    sample_weight = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
            [1.0, 1.0, 0.0],
        ], dtype=np.float32
    )
    return SparseCategorical(
        outcome_idx, depth=3, sample_weight=sample_weight, name='sparsecat_d'
    )


@pytest.fixture(scope="module")
def g_condition_idx_4x1():
    value = np.array(
        [
            [[0]],
            [[1]],
            [[0]],
            [[0]],
        ]
    )
    return Group(
        value, name='condition_idx'
    )


@pytest.fixture(scope="module")
def g_condition_label_4x1():
    value = np.array(
        [
            [['block']],
            [['interleave']],
            [['block']],
            [['block']],
        ]
    )
    return Group(
        value, name='condition_label'
    )


@pytest.fixture(scope="module")
def g_condition_idx_4x3():
    value = np.array(
        [
            [[0], [0], [0]],
            [[1], [1], [1]],
            [[0], [0], [0]],
            [[0], [0], [0]],
        ]
    )
    return Group(
        value, name='condition_idx'
    )


@pytest.fixture(scope="module")
def g_condition_idx_3x2():
    value = np.array(
        [
            [[0], [0]],
            [[1], [1]],
            [[0], [0]]
        ], dtype=np.int32
    )
    return Group(
        value, name='condition_idx'
    )


@pytest.fixture(scope="module")
def g_mix2_4x3():
    value = np.array(
        [
            [[0.5, 0.5], [0.6, 0.4], [0.7, 0.3]],
            [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            [[0.8, 0.2], [0.8, 0.2], [0.8, 0.2]],
            [[0.2, 0.8], [0.2, 0.8], [0.2, 0.8]],
        ], dtype=np.float32
    )
    return Group(
        value, name='mix2'
    )
