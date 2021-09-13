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
from importlib.metadata import version
import numpy as np
import pytest
import tensorflow as tf

from psiz.trials import load_trials
from psiz.trials.experimental.outcomes.sparse_categorical import SparseCategorical
from psiz.trials.experimental.trial_dataset import TrialDataset
from psiz.trials import stack


def test_init_0(rank_sim_1):
    """Test initialization.

    Bare minimum arguments.

    """
    content = rank_sim_1
    td = TrialDataset(content)

    assert td.n_sequence == content.n_sequence
    assert td.max_timestep == content.max_timestep


def test_init_1(rank_sim_1):
    """Test initialization.

    Include outcome argument.

    """
    content = rank_sim_1

    outcome_idx = np.zeros(
        [content.n_sequence, content.max_timestep], dtype=np.int32
    )
    outcome = SparseCategorical(outcome_idx, depth=content.max_outcome)
    TrialDataset(content, outcome=outcome)


def test_init_2(rank_sim_1):
    """Test initialization.

    Include outcome, group, and weight argument.

    """
    content = rank_sim_1
    outcome_idx = np.zeros(
        [content.n_sequence, content.max_timestep], dtype=np.int32
    )
    outcome = SparseCategorical(outcome_idx, depth=content.max_outcome)
    groups = np.ones([content.n_sequence, content.max_timestep, 1], dtype=int)
    weight = .9 * np.ones([content.n_sequence, content.max_timestep])
    TrialDataset(content, outcome=outcome, groups=groups, weight=weight)


def test_init_3(rank_sim_1):
    """Test initialization.

    Include outcome, group, and weight argument.
    Pass in 2D group instead of 3D to force internal handling.

    """
    content = rank_sim_1
    outcome_idx = np.zeros(
        [content.n_sequence, content.max_timestep], dtype=np.int32
    )
    outcome = SparseCategorical(outcome_idx, depth=content.max_outcome)
    groups = np.ones([content.n_sequence, content.max_timestep], dtype=int)
    weight = .9 * np.ones([content.n_sequence, content.max_timestep])
    TrialDataset(content, outcome=outcome, groups=groups, weight=weight)


def test_invalid_init_0(rank_sim_1):
    """Test invalid initialization."""
    content = rank_sim_1
    outcome_idx = np.zeros(
        [content.n_sequence, content.max_timestep], dtype=np.int32
    )
    outcome = SparseCategorical(outcome_idx, depth=content.max_outcome)
    groups = np.ones([content.n_sequence, content.max_timestep, 1], dtype=int)
    weight = .9 * np.ones([content.n_sequence, content.max_timestep, 2])
    with pytest.raises(Exception) as e_info:
        TrialDataset(content, outcome=outcome, groups=groups, weight=weight)
    assert e_info.type == ValueError

    weight = .9 * np.ones([content.n_sequence + 1, content.max_timestep])
    with pytest.raises(Exception) as e_info:
        TrialDataset(content, outcome=outcome, groups=groups, weight=weight)
    assert e_info.type == ValueError
    assert str(e_info.value) == (
        "The argument 'weight' must have "
        "shape=(n_squence, max_timestep) as determined by `content`."
    )

    weight = .9 * np.ones([content.n_sequence, content.max_timestep + 1])
    with pytest.raises(Exception) as e_info:
        TrialDataset(content, outcome=outcome, groups=groups, weight=weight)
    assert e_info.type == ValueError
    assert str(e_info.value) == (
        "The argument 'weight' must have "
        "shape=(n_squence, max_timestep) as determined by `content`."
    )


def test_invalid_init_1(rank_sim_1):
    """Test invalid initialization.

    Bad group shapes:
    * invalid rank 4 groups.
    * mismatch in n_sequence
    * mismatch in max_timestep

    """
    content = rank_sim_1
    outcome_idx = np.zeros(
        [content.n_sequence, content.max_timestep], dtype=np.int32
    )
    outcome = SparseCategorical(outcome_idx, depth=content.max_outcome)
    groups = np.ones(
        [content.n_sequence, content.max_timestep, 1, 1], dtype=int
    )
    weight = .9 * np.ones([content.n_sequence, content.max_timestep, 2])
    with pytest.raises(Exception) as e_info:
        TrialDataset(content, outcome=outcome, groups=groups, weight=weight)
    assert e_info.type == ValueError
    assert str(e_info.value) == (
        "The argument 'groups' must be a rank 3 ND array."
    )

    groups = np.ones(
        [content.n_sequence + 1, content.max_timestep, 1], dtype=int
    )
    with pytest.raises(Exception) as e_info:
        TrialDataset(content, outcome=outcome, groups=groups, weight=weight)
    assert e_info.type == ValueError
    assert str(e_info.value) == (
        "The argument 'groups' must have "
        "shape=(n_squence, max_timestep) as determined by `content`."
    )

    groups = np.ones(
        [content.n_sequence, content.max_timestep + 1, 1], dtype=int
    )
    with pytest.raises(Exception) as e_info:
        TrialDataset(content, outcome=outcome, groups=groups, weight=weight)
    assert e_info.type == ValueError
    assert str(e_info.value) == (
        "The argument 'groups' must have "
        "shape=(n_squence, max_timestep) as determined by `content`."
    )

    groups = np.array(
        [
            [[0]], [[-2]], [[1]], [[-1]]
        ]
    )
    with pytest.raises(Exception) as e_info:
        TrialDataset(content, outcome=outcome, groups=groups, weight=weight)
    assert e_info.type == ValueError
    assert str(e_info.value) == (
        "The parameter 'groups' contains integers less than 0. "
        "Found 2 bad trial(s)."
    )


def test_invalid_init_2(rank_sim_1):
    """Test invalid initialization.

    Invalid outcome.
    * Number of sequences disagrees.

    """
    content = rank_sim_1
    outcome_idx = np.zeros(
        [content.n_sequence + 1, content.max_timestep], dtype=np.int32
    )
    outcome = SparseCategorical(outcome_idx, depth=content.max_outcome)
    groups = np.ones([content.n_sequence, content.max_timestep], dtype=int)
    weight = .9 * np.ones([content.n_sequence, content.max_timestep])
    with pytest.raises(Exception) as e_info:
        TrialDataset(content, outcome=outcome, groups=groups, weight=weight)
    assert e_info.type == ValueError
    assert str(e_info.value) == (
        "The user-provided 'outcome' object must agree with the "
        "`n_sequence` attribute of the user-provided"
        "`content` object."
    )

    outcome_idx = np.zeros(
        [content.n_sequence, content.max_timestep + 1], dtype=np.int32
    )
    outcome = SparseCategorical(outcome_idx, depth=content.max_outcome)
    with pytest.raises(Exception) as e_info:
        TrialDataset(content, outcome=outcome, groups=groups, weight=weight)
    assert e_info.type == ValueError
    assert str(e_info.value) == (
        "The user-provided 'outcome' object must agree with the "
        "`max_timestep` attribute of the user-provided "
        "`content` object."
    )


def test_is_actual(rank_sim_2):
    """Test is_actual."""
    outcome_idx = np.zeros(
        [rank_sim_2.n_sequence, rank_sim_2.max_timestep], dtype=np.int32
    )
    outcome = SparseCategorical(outcome_idx, depth=rank_sim_2.max_outcome)
    trials_4 = TrialDataset(rank_sim_2, outcome=outcome)

    is_actual = trials_4.is_actual
    is_actual_desired = np.array(
        [
            [1, 1],
            [1, 0],
            [1, 1],
            [1, 1],
        ], dtype=bool
    )
    np.testing.assert_array_equal(is_actual, is_actual_desired)


def test_export_0(rank_sim_4):
    """Test as_dataset.

    Return inputs, outputs, and weights.

    """
    content = rank_sim_4
    outcome_idx = np.zeros(
        [content.n_sequence, content.max_timestep], dtype=np.int32
    )
    outcome = SparseCategorical(outcome_idx, depth=content.max_outcome)
    groups = np.ones([content.n_sequence, content.max_timestep, 1], dtype=int)
    weight = .9 * np.ones([content.n_sequence, content.max_timestep])
    trials = TrialDataset(
        content, outcome=outcome, groups=groups, weight=weight
    )

    ds = trials.export().batch(4, drop_remainder=False)
    ds_list = list(ds)
    x = ds_list[0][0]
    y = ds_list[0][1]
    w = ds_list[0][2]

    desired_x_stimulus_set = tf.constant(
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
    desired_x_is_select = tf.constant(
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
    desired_y = tf.constant(
        np.array(
            [
                [
                    [1., 0., 0.],
                    [1., 0., 0.],
                ],
                [
                    [1., 0., 0.],
                    [1., 0., 0.],
                ],
                [
                    [1., 0., 0.],
                    [1., 0., 0.],
                ]
            ], dtype=np.float32
        )
    )
    desired_w = tf.constant(
        np.array(
            [
                [0.9, 0.9],
                [0.9, 0.9],
                [0.9, 0.9],
            ], dtype=np.float32
        )
    )
    tf.debugging.assert_equal(desired_x_stimulus_set, x['stimulus_set'])
    tf.debugging.assert_equal(desired_x_is_select, x['is_select'])
    tf.debugging.assert_equal(desired_y, y)
    tf.debugging.assert_equal(desired_w, w)


def test_export_1(rank_sim_4):
    """Test export.

    Inputs only.

    """
    content = rank_sim_4
    outcome_idx = np.zeros(
        [content.n_sequence, content.max_timestep], dtype=np.int32
    )
    outcome = SparseCategorical(outcome_idx, depth=content.max_outcome)
    groups = np.ones([content.n_sequence, content.max_timestep, 1], dtype=int)
    weight = .9 * np.ones([content.n_sequence, content.max_timestep])
    trials = TrialDataset(
        content, outcome=outcome, groups=groups, weight=weight
    )

    ds = trials.export(input_only=True).batch(4, drop_remainder=False)
    ds_list = list(ds)
    x = ds_list[0]

    desired_x_stimulus_set = tf.constant(
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
    desired_x_is_select = tf.constant(
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
    tf.debugging.assert_equal(desired_x_stimulus_set, x['stimulus_set'])
    tf.debugging.assert_equal(desired_x_is_select, x['is_select'])


def test_export_2(rank_sim_4):
    """Test export.

    Return dataset using `timestep=False`.

    """
    content = rank_sim_4
    outcome_idx = np.zeros(
        [content.n_sequence, content.max_timestep], dtype=np.int32
    )
    outcome = SparseCategorical(outcome_idx, depth=content.max_outcome)
    groups = np.ones([content.n_sequence, content.max_timestep, 1], dtype=int)
    weight = .9 * np.ones([content.n_sequence, content.max_timestep])
    trials = TrialDataset(
        content, outcome=outcome, groups=groups, weight=weight
    )

    ds = trials.export(timestep=False).batch(6, drop_remainder=False)
    ds_list = list(ds)
    x = ds_list[0][0]
    y = ds_list[0][1]
    w = ds_list[0][2]

    desired_x_stimulus_set = tf.constant(
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
    desired_x_is_select = tf.constant(
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
    desired_y = tf.constant(
        np.array(
            [
                [
                    [1., 0., 0.],
                    [1., 0., 0.],
                    [1., 0., 0.],
                    [1., 0., 0.],
                    [1., 0., 0.],
                    [1., 0., 0.],
                ]
            ], dtype=np.float32
        )
    )
    desired_w = tf.constant(
        np.array(
            [0.9, 0.9, 0.9, 0.9, 0.9, 0.9], dtype=np.float32
        )
    )
    tf.debugging.assert_equal(desired_x_stimulus_set, x['stimulus_set'])
    tf.debugging.assert_equal(desired_x_is_select, x['is_select'])
    tf.debugging.assert_equal(desired_y, y)
    tf.debugging.assert_equal(desired_w, w)


def test_invalid_export_0(rank_sim_4):
    """Test invalid export."""
    content = rank_sim_4
    trials = TrialDataset(content)

    with pytest.raises(Exception) as e_info:
        trials.export()
    assert e_info.type == ValueError
    assert (
        str(e_info.value) == "No outcome has been specified."
    )


def test_invalid_export_1(rank_sim_4):
    """Test export.

    Using incorrect `export_format`.

    """
    content = rank_sim_4
    outcome_idx = np.zeros(
        [content.n_sequence, content.max_timestep], dtype=np.int32
    )
    outcome = SparseCategorical(outcome_idx, depth=content.max_outcome)
    groups = np.ones([content.n_sequence, content.max_timestep, 1], dtype=int)
    weight = .9 * np.ones([content.n_sequence, content.max_timestep])
    trials = TrialDataset(
        content, outcome=outcome, groups=groups, weight=weight
    )

    with pytest.raises(Exception) as e_info:
        trials.export(export_format='garbage')
    assert e_info.type == ValueError
    assert (
        str(e_info.value) == "Unrecognized `export_format` 'garbage'."
    )


def test_persistence_0(rank_sim_4, tmpdir):
    """Test persistence.

    Minimal arguments.

    """
    fn = tmpdir.join('persistence_test.hdf5')

    content = rank_sim_4
    trials = TrialDataset(content)

    trials.save(fn)

    reconstructed = load_trials(fn)

    assert trials.n_sequence == reconstructed.n_sequence
    assert trials.max_timestep == reconstructed.max_timestep
    np.testing.assert_array_equal(
        trials.content.stimulus_set, reconstructed.content.stimulus_set
    )
    np.testing.assert_array_equal(trials.groups, reconstructed.groups)
    assert trials.outcome == reconstructed.outcome
    np.testing.assert_array_equal(trials.weight, reconstructed.weight)

    ver = version("psiz")
    ver = '.'.join(ver.split('.')[:3])
    f = h5py.File(fn, "r")
    # pylint: disable=no-member
    reconstructed_version = f["psiz_version"].asstr()[()]
    assert ver == reconstructed_version


def test_persistence_1(rank_sim_4, tmpdir):
    """Test persistence."""
    fn = tmpdir.join('persistence_test.hdf5')

    content = rank_sim_4
    outcome_idx = np.zeros(
        [content.n_sequence, content.max_timestep], dtype=np.int32
    )
    outcome = SparseCategorical(outcome_idx, depth=content.max_outcome)
    groups = np.ones([content.n_sequence, content.max_timestep, 1], dtype=int)
    weight = .9 * np.ones([content.n_sequence, content.max_timestep])
    trials = TrialDataset(
        content, outcome=outcome, groups=groups, weight=weight
    )

    trials.save(fn)

    reconstructed = load_trials(fn)

    assert trials.n_sequence == reconstructed.n_sequence
    assert trials.max_timestep == reconstructed.max_timestep
    np.testing.assert_array_equal(
        trials.content.stimulus_set, reconstructed.content.stimulus_set
    )
    np.testing.assert_array_equal(trials.groups, reconstructed.groups)
    np.testing.assert_array_equal(
        trials.outcome.index, reconstructed.outcome.index
    )
    np.testing.assert_array_equal(trials.weight, reconstructed.weight)


def test_subset_0(rank_sim_4):
    """Test subset

    With non-singleton timestep and outcome.

    """
    content = rank_sim_4
    outcome_idx = np.zeros(
        [content.n_sequence, content.max_timestep], dtype=np.int32
    )
    outcome = SparseCategorical(outcome_idx, depth=content.max_outcome)
    groups = np.ones([content.n_sequence, content.max_timestep, 1], dtype=int)
    weight = .9 * np.ones([content.n_sequence, content.max_timestep])
    trials = TrialDataset(
        content, outcome=outcome, groups=groups, weight=weight
    )

    trials_sub = trials.subset(np.array([1, 2]))

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
    desired_index = np.array(
        [[0, 0], [0, 0]], dtype=np.int32
    )
    desired_depth = 3

    assert trials_sub.n_sequence == desired_n_sequence
    assert trials_sub.max_timestep == desired_max_timestep
    np.testing.assert_array_equal(
        desired_stimulus_set, trials_sub.content.stimulus_set
    )
    np.testing.assert_array_equal(
        desired_n_reference, trials_sub.content.n_reference
    )
    np.testing.assert_array_equal(
        desired_n_select, trials_sub.content.n_select
    )
    np.testing.assert_array_equal(
        desired_n_timestep, trials_sub.content.n_timestep
    )
    assert desired_max_outcome == trials_sub.content.max_outcome

    assert desired_n_sequence == trials_sub.outcome.n_sequence
    assert desired_max_timestep == trials_sub.outcome.max_timestep
    np.testing.assert_array_equal(
        desired_index, trials_sub.outcome.index
    )
    assert desired_depth == trials_sub.outcome.depth


def test_subset_1(rank_sim_4):
    """Test subset.

    With non-singleton timestepand no outcome.

    """
    content = rank_sim_4
    groups = np.ones([content.n_sequence, content.max_timestep, 1], dtype=int)
    weight = .9 * np.ones([content.n_sequence, content.max_timestep])
    trials = TrialDataset(
        content, groups=groups, weight=weight
    )

    trials_sub = trials.subset(np.array([1, 2]))

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

    assert trials_sub.n_sequence == desired_n_sequence
    assert trials_sub.max_timestep == desired_max_timestep
    np.testing.assert_array_equal(
        desired_stimulus_set, trials_sub.content.stimulus_set
    )
    np.testing.assert_array_equal(
        desired_n_reference, trials_sub.content.n_reference
    )
    np.testing.assert_array_equal(
        desired_n_select, trials_sub.content.n_select
    )
    np.testing.assert_array_equal(
        desired_n_timestep, trials_sub.content.n_timestep
    )
    assert desired_max_outcome == trials_sub.content.max_outcome

    assert desired_n_sequence == trials_sub.content.n_sequence
    assert desired_max_timestep == trials_sub.content.max_timestep


def test_stack_0(rank_sim_4, rank_sim_5):
    """Test stack."""
    # Creat two trial datasets.
    outcome_idx = np.zeros(
        [rank_sim_4.n_sequence, rank_sim_4.max_timestep], dtype=np.int32
    )
    outcome = SparseCategorical(outcome_idx, depth=rank_sim_4.max_outcome)
    groups = 4 * np.ones(
        [rank_sim_4.n_sequence, rank_sim_4.max_timestep, 1], dtype=int
    )
    weight = .4 * np.ones([rank_sim_4.n_sequence, rank_sim_4.max_timestep])
    trials_4 = TrialDataset(
        rank_sim_4, outcome=outcome, groups=groups, weight=weight
    )

    outcome_idx = np.zeros(
        [rank_sim_5.n_sequence, rank_sim_5.max_timestep], dtype=np.int32
    )
    outcome = SparseCategorical(outcome_idx, depth=rank_sim_5.max_outcome)
    groups = 5 * np.ones(
        [rank_sim_5.n_sequence, rank_sim_5.max_timestep, 1], dtype=int
    )
    weight = .5 * np.ones([rank_sim_5.n_sequence, rank_sim_5.max_timestep])
    trials_5 = TrialDataset(
        rank_sim_5, outcome=outcome, groups=groups, weight=weight
    )

    stacked = stack((trials_4, trials_5, trials_4))

    desired_n_sequence = 8
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
            ],
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
        ], dtype=np.int32
    )
    desired_n_select = np.array(
        [
            [1, 1, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 1, 1], [1, 1, 0],
            [1, 0, 0], [1, 1, 0]
        ], dtype=np.int32
    )
    desired_max_n_referece = 3
    desired_weight = np.array(
        [
            [0.4, 0.4, 0.0],
            [0.4, 0.4, 0.0],
            [0.4, 0.4, 0.0],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [0.4, 0.4, 0.0],
            [0.4, 0.4, 0.0],
            [0.4, 0.4, 0.0],
        ]
    )
    desired_groups = np.array(
        [
            [
                [4], [4], [0]
            ],
            [
                [4], [4], [0]
            ],
            [
                [4], [4], [0]
            ],
            [
                [5], [5], [5]
            ],
            [
                [5], [5], [5]
            ],
            [
                [4], [4], [0]
            ],
            [
                [4], [4], [0]
            ],
            [
                [4], [4], [0]
            ],
        ], dtype=np.int32
    )

    assert desired_n_sequence == stacked.n_sequence
    assert desired_max_timestep == stacked.max_timestep
    assert desired_max_n_referece == stacked.content.max_n_reference
    np.testing.assert_array_equal(
        desired_stimulus_set, stacked.content.stimulus_set
    )
    np.testing.assert_array_equal(
        desired_n_select, stacked.content.n_select
    )
    np.testing.assert_array_equal(
        desired_weight, stacked.weight
    )
    np.testing.assert_array_equal(
        desired_groups, stacked.groups
    )


def test_invalid_stack_0(rank_sim_4, rank_sim_5):
    """Test invalid stack.

    The `groups` axis=2 does not agree.

    """
    # Creat two trial datasets.
    outcome_idx = np.zeros(
        [rank_sim_4.n_sequence, rank_sim_4.max_timestep], dtype=np.int32
    )
    outcome = SparseCategorical(outcome_idx, depth=rank_sim_4.max_outcome)
    groups = 4 * np.ones(
        [rank_sim_4.n_sequence, rank_sim_4.max_timestep, 1], dtype=int
    )
    weight = .4 * np.ones([rank_sim_4.n_sequence, rank_sim_4.max_timestep])
    trials_4 = TrialDataset(
        rank_sim_4, outcome=outcome, groups=groups, weight=weight
    )

    outcome_idx = np.zeros(
        [rank_sim_5.n_sequence, rank_sim_5.max_timestep], dtype=np.int32
    )
    outcome = SparseCategorical(outcome_idx, depth=rank_sim_5.max_outcome)
    groups = 5 * np.ones(
        [rank_sim_5.n_sequence, rank_sim_5.max_timestep, 2], dtype=int
    )
    weight = .5 * np.ones([rank_sim_5.n_sequence, rank_sim_5.max_timestep])
    trials_5 = TrialDataset(
        rank_sim_5, outcome=outcome, groups=groups, weight=weight
    )

    with pytest.raises(Exception) as e_info:
        stack((trials_4, trials_5))
    assert e_info.type == ValueError
    assert str(e_info.value) == (
        'The shape of `groups` for the different TrialDatasets '
        'must be identical on axis=2.'
    )
