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

import h5py
import numpy as np
import pytest
import tensorflow as tf

from psiz.trials.experimental.contents.rank_similarity import RankSimilarity
from psiz.trials import stack


def test_init_0(rank_sim_0):
    """Test initialization with minimal rank arguments."""
    desired_n_sequence = 4
    desired_max_timestep = 1
    desired_stimulus_set = np.array([
        [[3, 1, 2, 0, 0, 0, 0, 0, 0]],
        [[9, 12, 7, 0, 0, 0, 0, 0, 0]],
        [[3, 4, 5, 6, 7, 0, 0, 0, 0]],
        [[3, 4, 5, 6, 13, 14, 15, 16, 17]]
    ], dtype=np.int32)
    desired_n_reference = np.array([[2], [2], [4], [8]], dtype=np.int32)
    desired_n_select = np.array([[1], [1], [1], [2]], dtype=np.int32)
    desired_n_timestep = np.array([1, 1, 1, 1], dtype=np.int32)
    desired_max_outcome = 56

    assert rank_sim_0.n_sequence == desired_n_sequence
    assert rank_sim_0.max_timestep == desired_max_timestep
    np.testing.assert_array_equal(
        desired_stimulus_set, rank_sim_0.stimulus_set
    )
    np.testing.assert_array_equal(
        desired_n_reference, rank_sim_0.n_reference
    )
    np.testing.assert_array_equal(
        desired_n_select, rank_sim_0.n_select
    )
    np.testing.assert_array_equal(
        desired_n_timestep, rank_sim_0.n_timestep
    )
    assert desired_max_outcome == rank_sim_0.max_outcome


def test_init_1(rank_sim_1):
    """Test initialization with true rank arguments."""
    desired_n_sequence = 4
    desired_max_timestep = 1
    desired_stimulus_set = np.array([
        [[3, 1, 2, 0, 0, 0, 0, 0, 0]],
        [[9, 12, 7, 0, 0, 0, 0, 0, 0]],
        [[3, 4, 5, 6, 7, 0, 0, 0, 0]],
        [[3, 4, 5, 6, 13, 14, 15, 16, 17]]
    ], dtype=np.int32)
    desired_n_reference = np.array([[2], [2], [4], [8]], dtype=np.int32)
    desired_n_select = np.array([[1], [1], [1], [2]], dtype=np.int32)
    desired_n_timestep = np.array([1, 1, 1, 1], dtype=np.int32)
    desired_max_outcome = 56

    assert rank_sim_1.n_sequence == desired_n_sequence
    assert rank_sim_1.max_timestep == desired_max_timestep
    np.testing.assert_array_equal(
        desired_stimulus_set, rank_sim_1.stimulus_set
    )
    np.testing.assert_array_equal(
        desired_n_reference, rank_sim_1.n_reference
    )
    np.testing.assert_array_equal(
        desired_n_select, rank_sim_1.n_select
    )
    np.testing.assert_array_equal(
        desired_n_timestep, rank_sim_1.n_timestep
    )
    assert desired_max_outcome == rank_sim_1.max_outcome


def test_init_2(rank_sim_2):
    """Test initialization with true rank arguments."""
    desired_n_sequence = 4
    desired_max_timestep = 2
    desired_stimulus_set = np.array(
        [
            [
                [3, 1, 2, 0, 0, 0, 0, 0, 0],
                [3, 1, 2, 0, 0, 0, 0, 0, 0]
            ],
            [
                [9, 12, 7, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0]
            ],
            [
                [3, 4, 5, 6, 7, 0, 0, 0, 0],
                [3, 4, 5, 0, 0, 0, 0, 0, 0]
            ],
            [
                [3, 4, 5, 6, 13, 14, 15, 16, 17],
                [3, 4, 5, 6, 13, 14, 15, 16, 17]
            ]
        ], dtype=np.int32
    )
    desired_n_reference = np.array(
        [[2, 2], [2, 0], [4, 2], [8, 8]], dtype=np.int32
    )
    desired_n_select = np.array(
        [[1, 1], [1, 0], [1, 1], [2, 1]], dtype=np.int32
    )
    desired_n_timestep = np.array([2, 1, 2, 2], dtype=np.int32)
    desired_max_outcome = 56

    assert rank_sim_2.n_sequence == desired_n_sequence
    assert rank_sim_2.max_timestep == desired_max_timestep
    np.testing.assert_array_equal(
        desired_stimulus_set, rank_sim_2.stimulus_set
    )
    np.testing.assert_array_equal(
        desired_n_reference, rank_sim_2.n_reference
    )
    np.testing.assert_array_equal(
        desired_n_select, rank_sim_2.n_select
    )
    np.testing.assert_array_equal(
        desired_n_timestep, rank_sim_2.n_timestep
    )
    assert desired_max_outcome == rank_sim_2.max_outcome


def test_init_3(rank_sim_3):
    """Test initialization with true rank arguments."""
    desired_n_sequence = 4
    desired_max_timestep = 2
    desired_stimulus_set = np.array(
        [
            [
                [3, 1, 2, 0, 0, 0, 0, 0, 0],
                [3, 1, 2, 0, 0, 0, 0, 0, 0]
            ],
            [
                [9, 12, 7, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0]
            ],
            [
                [3, 4, 5, 6, 7, 0, 0, 0, 0],
                [3, 4, 5, 0, 0, 0, 0, 0, 0]
            ],
            [
                [3, 4, 5, 6, 13, 14, 15, 16, 17],
                [3, 4, 5, 6, 13, 14, 15, 16, 17]
            ]
        ], dtype=np.int32
    )
    desired_n_reference = np.array(
        [[2, 2], [2, 0], [4, 2], [8, 8]], dtype=np.int32
    )
    desired_n_select = np.array(
        [[1, 1], [1, 0], [1, 1], [2, 1]], dtype=np.int32
    )
    desired_n_timestep = np.array([2, 1, 2, 2], dtype=np.int32)
    desired_max_outcome = 56

    assert rank_sim_3.n_sequence == desired_n_sequence
    assert rank_sim_3.max_timestep == desired_max_timestep
    np.testing.assert_array_equal(
        desired_stimulus_set, rank_sim_3.stimulus_set
    )
    np.testing.assert_array_equal(
        desired_n_reference, rank_sim_3.n_reference
    )
    np.testing.assert_array_equal(
        desired_n_select, rank_sim_3.n_select
    )
    np.testing.assert_array_equal(
        desired_n_timestep, rank_sim_3.n_timestep
    )
    assert desired_max_outcome == rank_sim_3.max_outcome


def test_invalid_stimulus_set():
    """Test handling of invalid `stimulus_set` argument."""
    # Non-integer input.
    stimulus_set = np.array((
        (3., 1, 2, 0, 0, 0, 0, 0, 0),
        (9, 12, 7, 0, 0, 0, 0, 0, 0),
        (3, 4, 5, 6, 7, 0, 0, 0, 0),
        (3, 4, 5, 6, 13, 14, 15, 16, 17)))
    with pytest.raises(Exception) as e_info:
        obs = RankSimilarity(stimulus_set)

    # Contains negative integers.
    stimulus_set = np.array((
        (3, 1, -1, 0, 0, 0, 0, 0, 0),
        (9, 12, 7, 0, 0, 0, 0, 0, 0),
        (3, 4, 5, 6, 7, 0, 0, 0, 0),
        (3, 4, 5, 6, 13, 14, 15, 16, 17)))
    with pytest.raises(Exception) as e_info:
        obs = RankSimilarity(stimulus_set)

    # Does not contain enough references for each trial.
    stimulus_set = np.array((
        (3, 1, 2, 0, 0, 0, 0, 0, 0),
        (9, 12, 7, 0, 0, 0, 0, 0, 0),
        (3, 4, 0, 0, 0, 0, 0, 0, 0),
        (3, 4, 5, 6, 13, 14, 15, 16, 17)))
    with pytest.raises(Exception) as e_info:
        obs = RankSimilarity(stimulus_set)


def test_invalid_n_select():
    """Test handling of invalid 'n_select' argument."""
    stimulus_set = np.array((
        (3, 1, 2, 0, 0, 0, 0, 0, 0),
        (9, 12, 7, 0, 0, 0, 0, 0, 0),
        (3, 4, 5, 6, 7, 0, 0, 0, 0),
        (3, 4, 5, 6, 13, 14, 15, 16, 17)))

    # Mismatch in number of trials
    n_select = np.array((1, 1, 2))
    with pytest.raises(Exception) as e_info:
        content = RankSimilarity(stimulus_set, n_select=n_select)

    # Below support.
    n_select = np.array((1, 0, 1, 0))
    with pytest.raises(Exception) as e_info:
        content = RankSimilarity(stimulus_set, n_select=n_select)

    # Above support.
    n_select = np.array((2, 1, 1, 2))
    with pytest.raises(Exception) as e_info:
        content = RankSimilarity(stimulus_set, n_select=n_select)


def test_is_actual(rank_sim_2):
    """Test is_actual method."""
    desired_is_actual = np.array(
        [
            [1, 1],
            [1, 0],
            [1, 1],
            [1, 1]
        ], dtype=bool
    )
    np.testing.assert_array_equal(
        desired_is_actual,
        rank_sim_2.is_actual()
    )


def test_config_attrs():
    """Test _config_attrs()"""
    desired_list = ['n_reference', 'n_select']
    assert desired_list == RankSimilarity._config_attrs()


def test_possible_outcomes_2c1():
    """Test outcomes 2 choose 1 ranked trial."""
    n_reference = 2
    n_select = 1
    outcomes = RankSimilarity._possible_outcomes(
        n_reference, n_select
    )

    desired_outcomes = np.array(((0, 1), (1, 0)))
    np.testing.assert_array_equal(outcomes, desired_outcomes)


def test_possible_outcomes_3c2():
    """Test outcomes 3 choose 2 ranked trial."""
    n_reference = 3
    n_select = 2
    outcomes = RankSimilarity._possible_outcomes(n_reference, n_select)

    desired_outcomes = np.array((
        (0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0),
        (2, 0, 1), (2, 1, 0)))
    np.testing.assert_array_equal(outcomes, desired_outcomes)


def test_possible_outcomes_4c2():
    """Test outcomes 4 choose 2 ranked trial."""
    n_reference = 4
    n_select = 2
    outcomes = RankSimilarity._possible_outcomes(n_reference, n_select)

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
    outcomes = RankSimilarity._possible_outcomes(n_reference, n_select)

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


def test_stimulus_set_with_outcomes(rank_sim_4):
    """Test _stimulus_set_with_outcomes."""
    # NOTE: To read the 4D matrix, read down the visual "columns" for a
    # single outcome.
    stimulus_set = rank_sim_4._stimulus_set_with_outcomes()
    desired_stimulus_set = np.array([
        [
            [
                [1, 1, 0],
                [2, 3, 0],
                [3, 2, 0],
                [0, 0, 0]
            ],
            [
                [4, 4, 0],
                [5, 6, 0],
                [6, 5, 0],
                [0, 0, 0],
            ],
        ],
        [
            [
                [7, 7, 0],
                [8, 9, 0],
                [9, 8, 0],
                [0, 0, 0],
            ],
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]
            ]
        ],
        [
            [
                [10, 10, 10],
                [11, 12, 13],
                [12, 11, 11],
                [13, 13, 12]
            ],
            [
                [14, 14, 0],
                [15, 16, 0],
                [16, 15, 0],
                [0, 0, 0]
            ]
        ]
    ])
    np.testing.assert_array_equal(desired_stimulus_set, stimulus_set)


def test_is_select_0(rank_sim_2):
    """Test _is_select."""
    is_select = rank_sim_2._is_select()
    desired_is_select = np.array(
        [
            [
                [0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0]
            ],
            [
                [0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0]
            ],
            [
                [0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0]
            ],
            [
                [0, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0]
            ]
        ], dtype=bool
    )
    np.testing.assert_array_equal(desired_is_select, is_select)


def test_is_select_compress_0(rank_sim_2):
    """Test _is_select."""
    is_select = rank_sim_2._is_select(compress=True)
    desired_is_select = np.array(
        [
            [
                [1, 0],
                [1, 0]
            ],
            [
                [1, 0],
                [0, 0]
            ],
            [
                [1, 0],
                [1, 0]
            ],
            [
                [1, 1],
                [1, 0]
            ]
        ], dtype=bool
    )
    np.testing.assert_array_equal(desired_is_select, is_select)


def test_is_select_1(rank_sim_4):
    """Test _is_select."""
    is_select = rank_sim_4._is_select()
    desired_is_select = np.array(
        [
            [
                [0, 1, 0, 0],
                [0, 1, 0, 0],
            ],
            [
                [0, 1, 0, 0],
                [0, 0, 0, 0],
            ],
            [
                [0, 1, 0, 0],
                [0, 1, 0, 0],
            ]
        ], dtype=bool
    )
    np.testing.assert_array_equal(desired_is_select, is_select)


def test_is_select_compress_1(rank_sim_4):
    """Test _is_select."""
    is_select = rank_sim_4._is_select(compress=True)
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


def test_for_dataset_0(rank_sim_4):
    """Test _for_dataset."""
    x = rank_sim_4._for_dataset()
    desired_stimulus_set = tf.constant(
        np.array([
            [
                [
                    [1, 1, 0],
                    [2, 3, 0],
                    [3, 2, 0],
                    [0, 0, 0]
                ],
                [
                    [4, 4, 0],
                    [5, 6, 0],
                    [6, 5, 0],
                    [0, 0, 0],
                ],
            ],
            [
                [
                    [7, 7, 0],
                    [8, 9, 0],
                    [9, 8, 0],
                    [0, 0, 0],
                ],
                [
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]
                ]
            ],
            [
                [
                    [10, 10, 10],
                    [11, 12, 13],
                    [12, 11, 11],
                    [13, 13, 12]
                ],
                [
                    [14, 14, 0],
                    [15, 16, 0],
                    [16, 15, 0],
                    [0, 0, 0]
                ]
            ]
        ]), dtype=tf.int32
    )
    desired_is_select = tf.constant(
        np.expand_dims(
            np.array(
                [
                    [
                        [0, 1, 0, 0],
                        [0, 1, 0, 0],
                    ],
                    [
                        [0, 1, 0, 0],
                        [0, 0, 0, 0],
                    ],
                    [
                        [0, 1, 0, 0],
                        [0, 1, 0, 0],
                    ]
                ], dtype=bool
            ), axis=-1
        ),
        dtype=tf.bool
    )
    tf.debugging.assert_equal(desired_stimulus_set, x['stimulus_set'])
    tf.debugging.assert_equal(desired_is_select, x['is_select'])


def test_for_dataset_1(rank_sim_4):
    """Test _for_dataset.

    Use timestep=False.

    """
    x = rank_sim_4._for_dataset(timestep=False)
    desired_stimulus_set = tf.constant(
        np.array([
            [
                [
                    [1, 1, 0],
                    [2, 3, 0],
                    [3, 2, 0],
                    [0, 0, 0]
                ],
                [
                    [4, 4, 0],
                    [5, 6, 0],
                    [6, 5, 0],
                    [0, 0, 0],
                ],
                [
                    [7, 7, 0],
                    [8, 9, 0],
                    [9, 8, 0],
                    [0, 0, 0],
                ],
                [
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]
                ],
                [
                    [10, 10, 10],
                    [11, 12, 13],
                    [12, 11, 11],
                    [13, 13, 12]
                ],
                [
                    [14, 14, 0],
                    [15, 16, 0],
                    [16, 15, 0],
                    [0, 0, 0]
                ]
            ]
        ]), dtype=tf.int32
    )
    desired_is_select = tf.constant(
        np.expand_dims(
            np.array(
                [
                    [
                        [0, 1, 0, 0],
                        [0, 1, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 1, 0, 0],
                    ]
                ], dtype=bool
            ), axis=-1
        ),
        dtype=tf.bool
    )
    tf.debugging.assert_equal(desired_stimulus_set, x['stimulus_set'])
    tf.debugging.assert_equal(desired_is_select, x['is_select'])


def test_persistence(rank_sim_4, tmpdir):
    """Test _save and _load."""
    group_name = "content"

    original = rank_sim_4
    fn = tmpdir.join('content_test.hdf5')

    # Save group.
    f = h5py.File(fn, "w")
    grp_stimulus = f.create_group(group_name)
    original._save(grp_stimulus)
    f.close()

    # Load group.
    f = h5py.File(fn, "r")
    grp = f[group_name]
    # Encoding/read rules changed in h5py 3.0, requiring asstr() call.
    try:
        class_name = grp["class_name"].asstr()[()]
    except AttributeError:
        class_name = grp["class_name"][()]
    reconstructed = RankSimilarity._load(grp)
    f.close()

    # Check for equivalency.
    assert class_name == "RankSimilarity"
    assert original.n_sequence == reconstructed.n_sequence
    assert original.max_timestep == reconstructed.max_timestep
    np.testing.assert_array_equal(
        original.stimulus_set, reconstructed.stimulus_set
    )
    np.testing.assert_array_equal(
        original.n_select, reconstructed.n_select
    )
    np.testing.assert_array_equal(
        original.n_reference, reconstructed.n_reference
    )
    np.testing.assert_array_equal(
        original.n_timestep, reconstructed.n_timestep
    )


def test_subset_0(rank_sim_1):
    """Test subset."""
    desired_n_sequence = 2
    desired_max_timestep = 1
    desired_stimulus_set = np.array([
        [[9, 12, 7, 0, 0]],
        [[3, 4, 5, 6, 7]],
    ], dtype=np.int32)
    desired_n_reference = np.array([[2], [4]], dtype=np.int32)
    desired_n_select = np.array([[1], [1]], dtype=np.int32)
    desired_n_timestep = np.array([1, 1], dtype=np.int32)
    desired_max_outcome = 4

    sub = rank_sim_1.subset(np.array([1, 2]))
    assert sub.n_sequence == desired_n_sequence
    assert sub.max_timestep == desired_max_timestep
    np.testing.assert_array_equal(
        desired_stimulus_set, sub.stimulus_set
    )
    np.testing.assert_array_equal(
        desired_n_reference, sub.n_reference
    )
    np.testing.assert_array_equal(
        desired_n_select, sub.n_select
    )
    np.testing.assert_array_equal(
        desired_n_timestep, sub.n_timestep
    )
    assert desired_max_outcome == sub.max_outcome


def test_subset_1(rank_sim_4):
    """Test subset."""
    desired_n_sequence = 2
    desired_max_timestep = 2
    desired_stimulus_set = np.array([
        [
            [7, 8, 9, 0],
            [0, 0, 0, 0],
        ],
        [
            [10, 11, 12, 13],
            [14, 15, 16, 0],
        ]
    ], dtype=np.int32)
    desired_n_reference = np.array([[2, 0], [3, 2]], dtype=np.int32)
    desired_n_select = np.array([[1, 0], [1, 1]], dtype=np.int32)
    desired_n_timestep = np.array([1, 2], dtype=np.int32)
    desired_max_outcome = 3

    sub = rank_sim_4.subset(np.array([1, 2]))
    assert sub.n_sequence == desired_n_sequence
    assert sub.max_timestep == desired_max_timestep
    np.testing.assert_array_equal(
        desired_stimulus_set, sub.stimulus_set
    )
    np.testing.assert_array_equal(
        desired_n_reference, sub.n_reference
    )
    np.testing.assert_array_equal(
        desired_n_select, sub.n_select
    )
    np.testing.assert_array_equal(
        desired_n_timestep, sub.n_timestep
    )
    assert desired_max_outcome == sub.max_outcome


def test_stack(rank_sim_4, rank_sim_5):
    """Test stack."""
    desired_n_sequence = 5
    desired_max_timestep = 3
    desired_stimulus_set = np.array(
        [
            [
                [1, 2, 3, 0],
                [4, 5, 6, 0],
                [0, 0, 0, 0]
            ],
            [
                [7, 8, 9, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0]
            ],
            [
                [10, 11, 12, 13],
                [14, 15, 16, 0],
                [0, 0, 0, 0]
            ],
            [
                [1, 2, 3, 0],
                [4, 5, 6, 0],
                [7, 8, 9, 0]
            ],
            [
                [10, 11, 12, 0],
                [13, 14, 15, 0],
                [16, 17, 18, 0]
            ]
        ], dtype=np.int32
    )
    desired_n_select = np.array(
        [
            [1, 1, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 1, 1]
        ], dtype=np.int32
    )
    desired_max_n_referece = 3
    stacked = stack((rank_sim_4, rank_sim_5))

    assert desired_n_sequence == stacked.n_sequence
    assert desired_max_timestep == stacked.max_timestep
    assert desired_max_n_referece == stacked.max_n_reference
    np.testing.assert_array_equal(
        desired_stimulus_set, stacked.stimulus_set
    )
    np.testing.assert_array_equal(
        desired_n_select, stacked.n_select
    )
