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
"""Test data module."""

import numpy as np
import pytest
import tensorflow as tf

from psiz.data.contents.rank import Rank


def test_init_0(c_2rank1_a_4x1):
    """Test initialization with minimal rank arguments."""
    desired_n_sequence = 4
    desired_sequence_length = 1
    desired_stimulus_set = np.array([
        [[3, 1, 2]],
        [[9, 12, 7]],
        [[5, 6, 7]],
        [[13, 14, 15]]
    ], dtype=np.int32)
    desired_n_reference = 2
    desired_n_select = 1
    desired_max_outcome = 2

    assert c_2rank1_a_4x1.n_sequence == desired_n_sequence
    assert c_2rank1_a_4x1.sequence_length == desired_sequence_length
    np.testing.assert_array_equal(
        desired_stimulus_set, c_2rank1_a_4x1.stimulus_set
    )
    assert desired_n_reference == c_2rank1_a_4x1.n_reference
    assert desired_n_select == c_2rank1_a_4x1.n_select
    assert desired_max_outcome == c_2rank1_a_4x1.max_outcome


def test_init_1(c_2rank1_aa_4x1):
    """Test initialization with true rank arguments."""
    desired_n_sequence = 4
    desired_sequence_length = 1
    desired_stimulus_set = np.array([
        [[3, 1, 2]],
        [[9, 12, 7]],
        [[5, 6, 7]],
        [[13, 14, 15]]
    ], dtype=np.int32)
    desired_n_reference = 2
    desired_n_select = 1
    desired_max_outcome = 2

    assert c_2rank1_aa_4x1.n_sequence == desired_n_sequence
    assert c_2rank1_aa_4x1.sequence_length == desired_sequence_length
    np.testing.assert_array_equal(
        desired_stimulus_set, c_2rank1_aa_4x1.stimulus_set
    )
    assert desired_n_reference == c_2rank1_aa_4x1.n_reference
    assert desired_n_select == c_2rank1_aa_4x1.n_select
    assert desired_max_outcome == c_2rank1_aa_4x1.max_outcome


def test_init_2(c_4rank2_b_4x2):
    """Test initialization with true rank arguments."""
    desired_n_sequence = 4
    desired_sequence_length = 2
    desired_stimulus_set = np.array(
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
    desired_n_reference = 4
    desired_n_select = 2
    desired_max_outcome = 12

    assert c_4rank2_b_4x2.n_sequence == desired_n_sequence
    assert c_4rank2_b_4x2.sequence_length == desired_sequence_length
    np.testing.assert_array_equal(
        desired_stimulus_set, c_4rank2_b_4x2.stimulus_set
    )
    assert desired_n_reference == c_4rank2_b_4x2.n_reference
    assert desired_n_select == c_4rank2_b_4x2.n_select
    assert desired_max_outcome == c_4rank2_b_4x2.max_outcome


def test_init_3(c_4rank2_c_4x3):
    """Test initialization.

    Extra placeholder reference gets auto-trimmed.

    """
    desired_n_sequence = 4
    desired_sequence_length = 3
    desired_stimulus_set = np.array(
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
    desired_n_reference = 4
    desired_n_select = 2
    desired_max_outcome = 12

    assert c_4rank2_c_4x3.n_sequence == desired_n_sequence
    assert c_4rank2_c_4x3.sequence_length == desired_sequence_length
    np.testing.assert_array_equal(
        desired_stimulus_set, c_4rank2_c_4x3.stimulus_set
    )
    assert desired_n_reference == c_4rank2_c_4x3.n_reference
    assert desired_n_select == c_4rank2_c_4x3.n_select
    assert desired_max_outcome == c_4rank2_c_4x3.max_outcome


def test_init_4():
    """Test initialization without `n_select` argument.

    Note that you cannot simply create an array of ones as a default because
    some trials may be placeholders.

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
    content = Rank(stimulus_set)

    n_select_desired = 1
    n_select_arr_desired = np.array(
        [
            [1, 1],
            [1, 0],
            [1, 1],
        ], dtype=np.int32
    )
    assert content.n_select == n_select_desired
    np.testing.assert_array_equal(
        content._n_select, n_select_arr_desired
    )


def test_invalid_stimulus_set():
    """Test handling of invalid `stimulus_set` argument."""
    # Non-integer input.
    stimulus_set = np.array(
        (
            (3., 1, 2),
            (9, 12, 7),
            (3, 4, 5),
            (3, 4, 5)
        )
    )
    with pytest.raises(Exception) as e_info:
        Rank(stimulus_set)
    assert e_info.type == ValueError

    # Contains negative integers.
    stimulus_set = np.array(
        (
            (3, 1, -1),
            (9, 12, 7),
            (3, 4, 5),
            (3, 4, 5)
        )
    )
    with pytest.raises(Exception) as e_info:
        Rank(stimulus_set)
    assert e_info.type == ValueError

    # Does not contain enough references for each trial.
    stimulus_set = np.array(
        (
            (3, 1, 2),
            (9, 12, 7),
            (3, 4, 0),
            (3, 4, 5)
        )
    )
    with pytest.raises(Exception) as e_info:
        Rank(stimulus_set)
    assert e_info.type == ValueError
    assert str(e_info.value) == (
        "The argument `stimulus_set` must contain at least three positive int"
        "egers per a trial, i.e. one query and at least two reference stimuli."
    )

    # Is too large a rank.
    stimulus_set = np.array(
        [
            [
                [
                    [3, 1, 2, 4],
                    [9, 12, 7, 5]
                ],
                [
                    [3, 1, 2, 3],
                    [9, 12, 7, 9]
                ]
            ],
            [
                [
                    [13, 14, 15, 2],
                    [16, 17, 4, 5]
                ],
                [
                    [3, 4, 5, 6],
                    [3, 4, 5, 6]
                ]
            ]
        ]
    )
    with pytest.raises(Exception) as e_info:
        Rank(stimulus_set)
    assert e_info.type == ValueError

    # TODO not sure whether to test this.
    # # Integer is too large.
    # ii32 = np.iinfo(np.int32)
    # too_large = ii32.max + 1
    # stimulus_set = np.array(
    #     [
    #         [
    #             [3, 1, 2, 0],
    #             [9, too_large, 7, 0],
    #             [3, 1, 2, 0],
    #             [9, 12, 7, 0]
    #         ],
    #         [
    #             [13, 14, 15, 2],
    #             [16, 17, 0, 0],
    #             [3, 4, 5, 6],
    #             [3, 4, 5, 6]
    #         ]
    #     ]
    # )
    # with pytest.raises(Exception) as e_info:
    #     Rank(stimulus_set)
    # assert e_info.type == ValueError


def test_invalid_n_select():
    """Test handling of invalid 'n_select' argument."""
    stimulus_set = np.array(
        (
            (3, 1, 2, 4, 5),
            (9, 12, 7, 8, 9),
            (3, 4, 5, 6, 7),
            (13, 14, 15, 16, 17)
        )
    )

    # Below support.
    n_select = 0
    with pytest.raises(Exception) as e_info:
        Rank(stimulus_set, n_select=n_select)
    assert e_info.type == ValueError

    # Above support.
    n_select = 5
    with pytest.raises(Exception) as e_info:
        Rank(stimulus_set, n_select=n_select)
    assert e_info.type == ValueError

    # Not an integer
    n_select = np.array([
        [[2], [1]], [[1], [2]]
    ])
    with pytest.raises(Exception) as e_info:
        Rank(stimulus_set, n_select=n_select)
    assert e_info.type == ValueError
    assert str(e_info.value) == (
        "The argument `n_select` must be an integer."
    )


def test_is_actual(c_4rank2_b_4x2):
    """Test is_actual method."""
    desired_is_actual = np.array(
        [
            [1, 1],
            [1, 0],
            [1, 1],
            [1, 1]
        ], dtype=bool
    )
    np.testing.assert_array_equal(desired_is_actual, c_4rank2_b_4x2.is_actual)


def test_config_attrs():
    """Test _config_attrs()"""
    desired_list = ['_n_reference', '_n_select']
    assert desired_list == Rank._config_attrs()


def test_possible_outcomes_2c1():
    """Test outcomes 2 choose 1 ranked trial."""
    n_reference = 2
    n_select = 1
    outcomes = Rank.possible_outcomes(
        n_reference, n_select
    )

    desired_outcomes = np.array(((0, 1), (1, 0)))
    np.testing.assert_array_equal(outcomes, desired_outcomes)


def test_possible_outcomes_3c2():
    """Test outcomes 3 choose 2 ranked trial."""
    n_reference = 3
    n_select = 2
    outcomes = Rank.possible_outcomes(n_reference, n_select)

    desired_outcomes = np.array((
        (0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0),
        (2, 0, 1), (2, 1, 0)))
    np.testing.assert_array_equal(outcomes, desired_outcomes)


def test_possible_outcomes_4c2():
    """Test outcomes 4 choose 2 ranked trial."""
    n_reference = 4
    n_select = 2
    outcomes = Rank.possible_outcomes(n_reference, n_select)

    desired_outcomes = np.array((
        (0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2),
        (1, 0, 2, 3), (1, 2, 0, 3), (1, 3, 0, 2),
        (2, 0, 1, 3), (2, 1, 0, 3), (2, 3, 0, 1),
        (3, 0, 1, 2), (3, 1, 0, 2), (3, 2, 0, 1)))
    np.testing.assert_array_equal(outcomes, desired_outcomes)


def test_possible_outcomes_8c1():
    """Test outcomes 8 choose 1 ranked trial."""
    n_reference = 8
    n_select = 1
    outcomes = Rank.possible_outcomes(n_reference, n_select)

    correct = np.array((
        (0, 1, 2, 3, 4, 5, 6, 7),
        (1, 0, 2, 3, 4, 5, 6, 7),
        (2, 0, 1, 3, 4, 5, 6, 7),
        (3, 0, 1, 2, 4, 5, 6, 7),
        (4, 0, 1, 2, 3, 5, 6, 7),
        (5, 0, 1, 2, 3, 4, 6, 7),
        (6, 0, 1, 2, 3, 4, 5, 7),
        (7, 0, 1, 2, 3, 4, 5, 6)))
    np.testing.assert_array_equal(outcomes, correct)


# TODO delete or move elsewhere
# def test_stimulus_set_with_outcomes(c_2rank1_d_3x2):
#     """Test _stimulus_set_with_outcomes."""
#     # NOTE: To read the 4D matrix, read down the visual "columns" for a
#     # single outcome.
#     stimulus_set = c_2rank1_d_3x2._stimulus_set_with_outcomes()
#     desired_stimulus_set = np.array([
#         [
#             [
#                 [1, 1, 0],
#                 [2, 3, 0],
#                 [3, 2, 0],
#                 [0, 0, 0]
#             ],
#             [
#                 [4, 4, 0],
#                 [5, 6, 0],
#                 [6, 5, 0],
#                 [0, 0, 0],
#             ],
#         ],
#         [
#             [
#                 [7, 7, 0],
#                 [8, 9, 0],
#                 [9, 8, 0],
#                 [0, 0, 0],
#             ],
#             [
#                 [0, 0, 0],
#                 [0, 0, 0],
#                 [0, 0, 0],
#                 [0, 0, 0]
#             ]
#         ],
#         [
#             [
#                 [10, 10, 10],
#                 [11, 12, 13],
#                 [12, 11, 11],
#                 [13, 13, 12]
#             ],
#             [
#                 [14, 14, 0],
#                 [15, 16, 0],
#                 [16, 15, 0],
#                 [0, 0, 0]
#             ]
#         ]
#     ])
#     np.testing.assert_array_equal(desired_stimulus_set, stimulus_set)


def test_is_select_0(c_4rank2_b_4x2):
    """Test _is_select."""
    is_select = c_4rank2_b_4x2._is_select()
    desired_is_select = np.array(
        [
            [
                [0, 1, 1, 0, 0],
                [0, 1, 1, 0, 0]
            ],
            [
                [0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0]
            ],
            [
                [0, 1, 1, 0, 0],
                [0, 1, 1, 0, 0]
            ],
            [
                [0, 1, 1, 0, 0],
                [0, 1, 1, 0, 0]
            ]
        ], dtype=bool
    )
    np.testing.assert_array_equal(desired_is_select, is_select)


def test_is_select_compress_0(c_4rank2_b_4x2):
    """Test _is_select."""
    is_select = c_4rank2_b_4x2._is_select(compress=True)
    desired_is_select = np.array(
        [
            [
                [1, 1],
                [1, 1]
            ],
            [
                [1, 1],
                [0, 0]
            ],
            [
                [1, 1],
                [1, 1]
            ],
            [
                [1, 1],
                [1, 1]
            ]
        ], dtype=bool
    )
    np.testing.assert_array_equal(desired_is_select, is_select)


def test_is_select_1(c_2rank1_d_3x2):
    """Test _is_select."""
    is_select = c_2rank1_d_3x2._is_select()
    desired_is_select = np.array(
        [
            [
                [0, 1, 0],
                [0, 1, 0],
            ],
            [
                [0, 1, 0],
                [0, 0, 0],
            ],
            [
                [0, 1, 0],
                [0, 1, 0],
            ]
        ], dtype=bool
    )
    np.testing.assert_array_equal(desired_is_select, is_select)


def test_is_select_compress_1(c_2rank1_d_3x2):
    """Test _is_select."""
    is_select = c_2rank1_d_3x2._is_select(compress=True)
    desired_is_select = np.array(
        [
            [
                [1],
                [1],
            ],
            [
                [1],
                [0],
            ],
            [
                [1],
                [1],
            ]
        ], dtype=bool
    )
    np.testing.assert_array_equal(desired_is_select, is_select)


def test_export_0(c_2rank1_d_3x2):
    """Test export."""
    x = c_2rank1_d_3x2.export()
    desired_stimulus_set = tf.constant(
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
        ], dtype=tf.int32
    )
    desired_is_select = tf.constant(
        [
            [
                [False, True, False],
                [False, True, False],
            ],
            [
                [False, True, False],
                [False, False, False],
            ],
            [
                [False, True, False],
                [False, True, False],
            ]
        ]
    )
    tf.debugging.assert_equal(
            desired_stimulus_set, x['2rank1/stimulus_set']
    )
    tf.debugging.assert_equal(
        desired_is_select, x['2rank1/is_select']
    )


def test_export_1(c_2rank1_d_3x2):
    """Test export.

    Use with_timestep_axis=False.

    """
    x = c_2rank1_d_3x2.export(with_timestep_axis=False)
    desired_stimulus_set = tf.constant(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [0, 0, 0],
            [10, 11, 12],
            [14, 15, 16],
        ], dtype=tf.int32
    )
    desired_is_select = tf.constant(
        [
            [
                [False, True, False],
                [False, True, False],
                [False, True, False],
                [False, False, False],
                [False, True, False],
                [False, True, False],
            ]
        ]
    )
    tf.debugging.assert_equal(
        desired_stimulus_set, x['2rank1/stimulus_set']
    )
    tf.debugging.assert_equal(
        desired_is_select, x['2rank1/is_select']
    )


def test_export_wrong(c_2rank1_d_3x2):
    """Test export.

    Using incorrect `export_format`.

    """
    with pytest.raises(Exception) as e_info:
        c_2rank1_d_3x2.export(export_format='garbage')
    assert e_info.type == ValueError
    assert (
        str(e_info.value) == "Unrecognized `export_format` 'garbage'."
    )