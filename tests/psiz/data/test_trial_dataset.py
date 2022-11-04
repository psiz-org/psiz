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

import h5py
from importlib.metadata import version
import numpy as np
import pytest
import tensorflow as tf

from psiz.trials import load_trials
from psiz.data.groups.group import Group
from psiz.data.outcomes.sparse_categorical import SparseCategorical
from psiz.data.trial_dataset import TrialDataset
# from psiz.trials import stack  TODO finish or delete


def test_init_0(c_rank_aa_4x1):
    """Test initialization.

    Bare minimum arguments.

    """
    td = TrialDataset([c_rank_aa_4x1])

    assert td.n_sequence == c_rank_aa_4x1.n_sequence
    assert td.sequence_length == c_rank_aa_4x1.sequence_length
    assert len(td.content_list) == 1
    assert len(td.group_list) == 0
    assert len(td.outcome_list) == 0


def test_init_1(c_rank_aa_4x1):
    """Test initialization.

    With outcome, no sample weights.

    """
    outcome_idx = np.zeros(
        [c_rank_aa_4x1.n_sequence, c_rank_aa_4x1.sequence_length],
        dtype=np.int32
    )
    rank_outcome = SparseCategorical(
        outcome_idx, depth=c_rank_aa_4x1.max_outcome, name='rank_outcome'
    )

    td = TrialDataset([c_rank_aa_4x1, rank_outcome])

    assert td.n_sequence == c_rank_aa_4x1.n_sequence
    assert td.sequence_length == c_rank_aa_4x1.sequence_length
    assert len(td.content_list) == 1
    assert len(td.group_list) == 0
    assert len(td.outcome_list) == 1


def test_init_2(c_rank_aa_4x1, o_rank_aa_4x1):
    """Test initialization.

    With outcome, including sample_weight.
    With group, mixture format.

    """
    group_weights = np.array(
        [
            [[.1, .9]],
            [[.5, .5]],
            [[1., 0.]],
            [[.9, .1]],
        ]
    )
    group_0 = Group(
        group_weights=group_weights, name='group_id'
    )

    td = TrialDataset([c_rank_aa_4x1, group_0, o_rank_aa_4x1])

    assert td.n_sequence == c_rank_aa_4x1.n_sequence
    assert td.sequence_length == c_rank_aa_4x1.sequence_length
    assert len(td.content_list) == 1
    assert len(td.group_list) == 1
    assert len(td.outcome_list) == 1


def test_init_3(c_rank_aa_4x1):
    """Test initialization.

    With outcome, including sample_weight argument.
    With group, pass in sparse format.

    """
    # Create rank outcome.
    outcome_idx = np.zeros(
        [c_rank_aa_4x1.n_sequence, c_rank_aa_4x1.sequence_length],
        dtype=np.int32
    )
    sample_weight = .9 * np.ones(
        [c_rank_aa_4x1.n_sequence, c_rank_aa_4x1.sequence_length]
    )
    rank_outcome = SparseCategorical(
        outcome_idx,
        depth=c_rank_aa_4x1.max_outcome,
        sample_weight=sample_weight,
        name='rank_outcome'
    )

    group_weights = np.array(
        [
            [[0]],
            [[1]],
            [[0]],
            [[0]],
        ]
    )
    group_0 = Group(
        group_weights=group_weights, name='condition_id'
    )

    td = TrialDataset([c_rank_aa_4x1, group_0, rank_outcome])

    assert td.n_sequence == c_rank_aa_4x1.n_sequence
    assert td.sequence_length == c_rank_aa_4x1.sequence_length
    assert len(td.content_list) == 1
    assert len(td.group_list) == 1
    assert len(td.outcome_list) == 1


def test_init_4(c_rank_d_3x2, o_rank_d_3x2, o_rt_a_3x2):
    """Test initialization.

    One content, two outcomes.

    """
    td = TrialDataset([c_rank_d_3x2, o_rank_d_3x2, o_rt_a_3x2])

    assert td.n_sequence == 3
    assert td.sequence_length == 2
    assert len(td.content_list) == 1
    assert len(td.group_list) == 0
    assert len(td.outcome_list) == 2


def test_init_5(c_rank_d_3x2, o_rank_d_3x2, c_rate_e_3x2, o_rate_a_3x2):
    """Test initialization.

    * two contents
    * two outcomes

    """
    td = TrialDataset([c_rank_d_3x2, o_rank_d_3x2, c_rate_e_3x2, o_rate_a_3x2])

    assert td.n_sequence == 3
    assert td.sequence_length == 2
    assert len(td.content_list) == 2
    assert len(td.group_list) == 0
    assert len(td.outcome_list) == 2


def test_invalid_init_0(c_rank_aa_4x1, o_rank_d_3x2, o_rank_c_4x3):
    """Test invalid initialization.

    * Number of sequences disagrees.
    * Sequence length disagrees.

    """
    with pytest.raises(Exception) as e_info:
        TrialDataset([c_rank_aa_4x1, o_rank_d_3x2])
    assert e_info.type == ValueError
    assert str(e_info.value) == (
        "All user-provided 'TrialComponent' objects must have the same "
        "`n_sequence`. The 'TrialComponent' in position 1 does not match "
        "the previous components."
    )

    with pytest.raises(Exception) as e_info:
        TrialDataset([c_rank_aa_4x1, o_rank_c_4x3])
    assert e_info.type == ValueError
    assert str(e_info.value) == (
        "All user-provided 'TrialComponent' objects must have the same "
        "`sequence_length`. The 'TrialComponent' in position 1 does not "
        "match the previous components."
    )


def test_export_0(c_rank_d_3x2, g_condition_id_3x2):
    """Test export.

    * Include content and group only.

    """
    td = TrialDataset([c_rank_d_3x2, g_condition_id_3x2])

    desired_x_stimulus_set = tf.constant(
        [
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
        ], dtype=tf.int32
    )
    desired_x_is_select = tf.constant(
        [
            [
                [False, True, False, False],
                [False, True, False, False],
            ],
            [
                [False, True, False, False],
                [False, False, False, False],
            ],
            [
                [False, True, False, False],
                [False, True, False, False],
            ]
        ]
    )
    desired_x_is_select = tf.expand_dims(desired_x_is_select, axis=-1)
    desired_condition_id = tf.constant(
        [
            [[0], [0]],
            [[1], [1]],
            [[0], [0]]
        ], dtype=tf.int32
    )

    ds = td.export().batch(4, drop_remainder=False)
    ds_list = list(ds)
    x = ds_list[0]

    assert len(ds_list) == 1
    tf.debugging.assert_equal(
        desired_x_stimulus_set, x['rank_similarity_stimulus_set']
    )
    tf.debugging.assert_equal(
        desired_x_is_select, x['rank_similarity_is_select']
    )
    tf.debugging.assert_equal(
        desired_condition_id, x['condition_id']
    )


def test_export_1a(c_rank_d_3x2, g_condition_id_3x2, o_rank_d_3x2):
    """Test as_dataset.

    * Include content, group, and outcome.
    * A single output model, therefore drop dictionary keys of `y` and
        `w`.

    """
    td = TrialDataset([c_rank_d_3x2, g_condition_id_3x2, o_rank_d_3x2])

    desired_x_stimulus_set = tf.constant(
        [
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
        ], dtype=tf.int32
    )
    desired_x_is_select = tf.expand_dims(
        tf.constant(
            [
                [
                    [False, True, False, False],
                    [False, True, False, False],
                ],
                [
                    [False, True, False, False],
                    [False, False, False, False],
                ],
                [
                    [False, True, False, False],
                    [False, True, False, False],
                ]
            ]
        ), axis=-1
    )
    desired_condition_id = tf.constant(
        [
            [[0], [0]],
            [[1], [1]],
            [[0], [0]]
        ], dtype=tf.int32
    )
    desired_y = tf.constant(
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
        ], dtype=tf.float32
    )
    desired_w = tf.constant(
        [
            [0.9, 0.9],
            [0.9, 0.9],
            [0.9, 0.9],
        ], dtype=tf.float32
    )

    ds = td.export().batch(4, drop_remainder=False)
    ds_list = list(ds)
    x = ds_list[0][0]
    y = ds_list[0][1]
    w = ds_list[0][2]

    assert len(ds_list[0]) == 3
    tf.debugging.assert_equal(
        desired_x_stimulus_set, x['rank_similarity_stimulus_set']
    )
    tf.debugging.assert_equal(
        desired_x_is_select, x['rank_similarity_is_select']
    )
    tf.debugging.assert_equal(
        desired_condition_id, x['condition_id']
    )
    tf.debugging.assert_equal(desired_y, y)
    tf.debugging.assert_equal(desired_w, w)
    tf.debugging.assert_equal(desired_condition_id, x['condition_id'])

    # Export inputs only.
    ds = td.export(inputs_only=True).batch(4, drop_remainder=False)
    ds_list = list(ds)
    x = ds_list[0]
    assert len(ds_list) == 1
    tf.debugging.assert_equal(
        desired_x_stimulus_set, x['rank_similarity_stimulus_set']
    )
    tf.debugging.assert_equal(
        desired_x_is_select, x['rank_similarity_is_select']
    )
    tf.debugging.assert_equal(
        desired_condition_id, x['condition_id']
    )


def test_export_1b(c_rank_d_3x2, g_condition_id_3x2, o_rank_d_3x2):
    """Test export.

    Return dataset using `with_timestep_axis=False`.

    """
    td = TrialDataset([c_rank_d_3x2, g_condition_id_3x2, o_rank_d_3x2])

    desired_x_stimulus_set = tf.constant(
        [
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
        ], dtype=tf.int32
    )
    desired_x_is_select = tf.constant(
        [
            [
                [False, True, False, False],
                [False, True, False, False],
                [False, True, False, False],
                [False, False, False, False],
                [False, True, False, False],
                [False, True, False, False],
            ]
        ], dtype=tf.bool
    )
    desired_x_is_select = tf.expand_dims(desired_x_is_select, axis=-1)
    desired_condition_id = tf.constant(
        [
            [0], [0], [1], [1], [0], [0]
        ], dtype=tf.int32
    )
    desired_y = tf.constant(
        [
            [
                [1., 0., 0.],
                [1., 0., 0.],
                [1., 0., 0.],
                [1., 0., 0.],
                [1., 0., 0.],
                [1., 0., 0.],
            ]
        ], dtype=tf.float32
    )
    desired_w = tf.constant(
        [0.9, 0.9, 0.9, 0.9, 0.9, 0.9], dtype=tf.float32
    )

    ds = td.export(with_timestep_axis=False).batch(6, drop_remainder=False)
    ds_list = list(ds)
    x = ds_list[0][0]
    y = ds_list[0][1]
    w = ds_list[0][2]

    assert len(ds_list[0]) == 3
    tf.debugging.assert_equal(
        desired_x_stimulus_set, x['rank_similarity_stimulus_set']
    )
    tf.debugging.assert_equal(
        desired_x_is_select, x['rank_similarity_is_select']
    )
    tf.debugging.assert_equal(
        desired_condition_id, x['condition_id']
    )
    tf.debugging.assert_equal(desired_y, y)
    tf.debugging.assert_equal(desired_w, w)


def test_export_2a(c_rank_d_3x2, g_condition_id_3x2, o_rank_d_3x2, o_rt_a_3x2):
    """Test export.

    * Multi-output model, therefore keep dictionary keys for `y` and
    `w`.

    """
    td = TrialDataset(
        [c_rank_d_3x2, g_condition_id_3x2, o_rank_d_3x2, o_rt_a_3x2]
    )

    desired_x_stimulus_set = tf.constant(
        [
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
        ], dtype=tf.int32
    )
    desired_x_is_select = tf.expand_dims(
        tf.constant(
            [
                [
                    [False, True, False, False],
                    [False, True, False, False],
                ],
                [
                    [False, True, False, False],
                    [False, False, False, False],
                ],
                [
                    [False, True, False, False],
                    [False, True, False, False],
                ]
            ]
        ), axis=-1
    )
    desired_condition_id = tf.constant(
        [
            [[0], [0]],
            [[1], [1]],
            [[0], [0]]
        ], dtype=tf.int32
    )
    desired_y_prob = tf.constant(
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
        ], dtype=tf.float32
    )
    desired_w_prob = tf.constant(
        [
            [0.9, 0.9],
            [0.9, 0.9],
            [0.9, 0.9],
        ], dtype=tf.float32
    )
    desired_y_rt = tf.constant(
        [
            [[4.1], [4.2]],
            [[5.1], [5.2]],
            [[6.1], [6.2]],
        ], dtype=tf.float32
    )
    desired_w_rt = tf.constant(
        [
            [0.8, 0.8],
            [0.8, 0.8],
            [0.8, 0.8],
        ], dtype=tf.float32
    )

    ds = td.export().batch(4, drop_remainder=False)
    ds_list = list(ds)
    x = ds_list[0][0]
    y = ds_list[0][1]
    w = ds_list[0][2]

    assert len(ds_list[0]) == 3
    tf.debugging.assert_equal(
        desired_x_stimulus_set, x['rank_similarity_stimulus_set']
    )
    tf.debugging.assert_equal(
        desired_x_is_select, x['rank_similarity_is_select']
    )
    tf.debugging.assert_equal(
        desired_condition_id, x['condition_id']
    )
    tf.debugging.assert_equal(desired_y_prob, y['rank_prob'])
    tf.debugging.assert_equal(desired_w_prob, w['rank_prob'])
    tf.debugging.assert_equal(desired_y_rt, y['rt'])
    tf.debugging.assert_equal(desired_w_rt, w['rt'])


def test_invalid_export_0(c_rank_d_3x2, g_condition_id_3x2, o_rank_d_3x2):
    """Test export.

    Using incorrect `export_format`.

    """
    td = TrialDataset([c_rank_d_3x2, g_condition_id_3x2, o_rank_d_3x2])

    with pytest.raises(Exception) as e_info:
        td.export(export_format='garbage')
    assert e_info.type == ValueError
    assert (
        str(e_info.value) == "Unrecognized `export_format` 'garbage'."
    )


def test_subset_0(c_rank_d_3x2):
    """Test subset.

    With content only.

    """
    td = TrialDataset([c_rank_d_3x2])

    td_sub = td.subset(np.array([1, 2]))

    desired_n_sequence = 2
    desired_sequence_length = 2
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
    desired_max_outcome = 3

    assert td_sub.n_sequence == desired_n_sequence
    assert td_sub.sequence_length == desired_sequence_length
    np.testing.assert_array_equal(
        td_sub.content_list[0].stimulus_set, desired_stimulus_set
    )
    np.testing.assert_array_equal(
        td_sub.content_list[0].n_reference, desired_n_reference
    )
    np.testing.assert_array_equal(
        td_sub.content_list[0].n_select, desired_n_select
    )
    assert td_sub.content_list[0].max_outcome == desired_max_outcome
    assert len(td_sub.group_list) == 0
    assert len(td_sub.outcome_list) == 0


def test_subset_1(c_rank_d_3x2, g_condition_id_3x2, o_rank_d_3x2):
    """Test subset.

    With content, group, and outcome. Only use one of each so that we
    do not have to keep track of where items are located in the list.

    """
    td = TrialDataset(
        [c_rank_d_3x2, g_condition_id_3x2, o_rank_d_3x2]
    )

    td_sub = td.subset(np.array([1, 2]))

    desired_n_sequence = 2
    desired_sequence_length = 2
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
    desired_max_outcome = 3
    desired_condition_id = np.array(
        [
            [[1], [1]],
            [[0], [0]]
        ], dtype=np.int32
    )
    desired_outcome = np.zeros([2, 2], dtype=np.int32)
    desired_sample_weight = .9 * np.ones([2, 2])

    assert td_sub.n_sequence == desired_n_sequence
    assert td_sub.sequence_length == desired_sequence_length
    np.testing.assert_array_equal(
        td_sub.content_list[0].stimulus_set, desired_stimulus_set
    )
    np.testing.assert_array_equal(
        td_sub.content_list[0].n_reference, desired_n_reference
    )
    np.testing.assert_array_equal(
        td_sub.content_list[0].n_select, desired_n_select
    )
    assert td_sub.content_list[0].max_outcome == desired_max_outcome
    np.testing.assert_array_equal(
        td_sub.group_list[0].group_weights, desired_condition_id
    )
    np.testing.assert_array_equal(
        td_sub.outcome_list[0].index, desired_outcome
    )
    np.testing.assert_array_equal(
        td_sub.outcome_list[0].sample_weight, desired_sample_weight
    )


def test_persistence_0(c_rank_d_3x2, tmpdir):
    """Test persistence.

    * With content only.

    """
    fn = tmpdir.join('persistence_test.hdf5')

    original = TrialDataset([c_rank_d_3x2])
    original.save(fn)

    reconstructed = load_trials(fn)

    assert original.n_sequence == reconstructed.n_sequence
    assert original.sequence_length == reconstructed.sequence_length
    assert len(original.content_list) == len(reconstructed.content_list)
    assert len(original.group_list) == len(reconstructed.group_list)
    assert len(original.outcome_list) == len(reconstructed.outcome_list)

    ver = version("psiz")
    ver = '.'.join(ver.split('.')[:3])
    f = h5py.File(fn, "r")
    # pylint: disable=no-member
    reconstructed_version = f["psiz_version"].asstr()[()]
    assert ver == reconstructed_version


def test_persistence_1(c_rank_d_3x2, g_condition_id_3x2, o_rank_d_3x2, tmpdir):
    """Test persistence.

    * With content, group, and outcome.

    """
    fn = tmpdir.join('persistence_test.hdf5')

    original = TrialDataset([c_rank_d_3x2, g_condition_id_3x2, o_rank_d_3x2])
    original.save(fn)

    reconstructed = load_trials(fn)

    assert original.n_sequence == reconstructed.n_sequence
    assert original.sequence_length == reconstructed.sequence_length
    assert len(original.content_list) == len(reconstructed.content_list)
    assert len(original.group_list) == len(reconstructed.group_list)
    assert len(original.outcome_list) == len(reconstructed.outcome_list)


def test_tf_ds_concatenate(c_rank_d_3x2, c_rank_b_4x2):
    """Test concatenating two datasets"""
    td_0 = TrialDataset([c_rank_d_3x2])
    td_1 = TrialDataset([c_rank_b_4x2])

    ds_0 = td_0.export(export_format='tf')
    ds_1 = td_1.export(export_format='tf')

    # TODO TF concatenate can't handle different number of references
    ds = ds_0.concatenate(ds_1).batch(7)
    ds_list = list(ds)
    x = ds_list[0]


# TODO move to equivalent test of tf.data.Dataset.concatenate
# def test_stack_0(c_rank_d_3x2, c_rank_e_2x3):
#     """Test stack."""
#     # Creat two trial datasets.
#     outcome_idx = np.zeros(
#         [c_rank_d_3x2.n_sequence, c_rank_d_3x2.sequence_length], dtype=np.int32
#     )
#     outcome = SparseCategorical(outcome_idx, depth=c_rank_d_3x2.max_outcome)
#     groups = {
#         'anonymous_id': np.array(
#             [
#                 [[4], [4]],
#                 [[4], [4]],
#                 [[4], [4]]
#             ], dtype=np.int32
#         )
#     }
#     sample_weight = .4 * np.ones(
#         [c_rank_d_3x2.n_sequence, c_rank_d_3x2.sequence_length]
#     )
#     trials_4 = TrialDataset(
#         c_rank_d_3x2, groups=groups, outcome=outcome, sample_weight=sample_weight
#     )

#     outcome_idx = np.zeros(
#         [c_rank_e_2x3.n_sequence, c_rank_e_2x3.sequence_length], dtype=np.int32
#     )
#     outcome = SparseCategorical(outcome_idx, depth=c_rank_e_2x3.max_outcome)
#     groups = {
#         'anonymous_id': np.array(
#             [
#                 [[5], [5], [5]],
#                 [[5], [5], [5]],
#             ], dtype=np.int32
#         )
#     }
#     sample_weight = .5 * np.ones(
#         [c_rank_e_2x3.n_sequence, c_rank_e_2x3.sequence_length]
#     )
#     trials_5 = TrialDataset(
#         c_rank_e_2x3, groups=groups, outcome=outcome, sample_weight=sample_weight
#     )

#     stacked = stack((trials_4, trials_5, trials_4))

#     desired_n_sequence = 8
#     desired_sequence_length = 3
#     desired_stimulus_set = np.array(
#         [
#             [
#                 [1, 2, 3, 0],
#                 [4, 5, 6, 0],
#                 [0, 0, 0, 0]
#             ],
#             [
#                 [7, 8, 9, 0],
#                 [0, 0, 0, 0],
#                 [0, 0, 0, 0]
#             ],
#             [
#                 [10, 11, 12, 13],
#                 [14, 15, 16, 0],
#                 [0, 0, 0, 0]
#             ],
#             [
#                 [1, 2, 3, 0],
#                 [4, 5, 6, 0],
#                 [7, 8, 9, 0]
#             ],
#             [
#                 [10, 11, 12, 0],
#                 [13, 14, 15, 0],
#                 [16, 17, 18, 0]
#             ],
#             [
#                 [1, 2, 3, 0],
#                 [4, 5, 6, 0],
#                 [0, 0, 0, 0]
#             ],
#             [
#                 [7, 8, 9, 0],
#                 [0, 0, 0, 0],
#                 [0, 0, 0, 0]
#             ],
#             [
#                 [10, 11, 12, 13],
#                 [14, 15, 16, 0],
#                 [0, 0, 0, 0]
#             ],
#         ], dtype=np.int32
#     )
#     desired_n_select = np.array(
#         [
#             [1, 1, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 1, 1], [1, 1, 0],
#             [1, 0, 0], [1, 1, 0]
#         ], dtype=np.int32
#     )
#     desired_max_n_referece = 3
#     desired_sample_weight = np.array(
#         [
#             [0.4, 0.4, 0.0],
#             [0.4, 0.4, 0.0],
#             [0.4, 0.4, 0.0],
#             [0.5, 0.5, 0.5],
#             [0.5, 0.5, 0.5],
#             [0.4, 0.4, 0.0],
#             [0.4, 0.4, 0.0],
#             [0.4, 0.4, 0.0],
#         ]
#     )
#     desired_anonymous_id = np.array(
#         [
#             [
#                 [4], [4], [0]
#             ],
#             [
#                 [4], [4], [0]
#             ],
#             [
#                 [4], [4], [0]
#             ],
#             [
#                 [5], [5], [5]
#             ],
#             [
#                 [5], [5], [5]
#             ],
#             [
#                 [4], [4], [0]
#             ],
#             [
#                 [4], [4], [0]
#             ],
#             [
#                 [4], [4], [0]
#             ],
#         ], dtype=np.int32
#     )

#     assert desired_n_sequence == stacked.n_sequence
#     assert desired_sequence_length == stacked.sequence_length
#     assert desired_max_n_referece == stacked.content.max_n_reference
#     np.testing.assert_array_equal(
#         desired_stimulus_set, stacked.content.stimulus_set
#     )
#     np.testing.assert_array_equal(
#         desired_n_select, stacked.content.n_select
#     )
#     np.testing.assert_array_equal(
#         stacked.groups['anonymous_id'], desired_anonymous_id
#     )
#     np.testing.assert_array_equal(
#         desired_sample_weight, stacked.sample_weight
#     )


# def test_invalid_stack_0(c_rank_d_3x2, c_rank_e_2x3):
#     """Test invalid stack.

#     Incompatible `groups` keys.

#     """
#     # Creat two trial datasets.
#     outcome_idx = np.zeros(
#         [c_rank_d_3x2.n_sequence, c_rank_d_3x2.sequence_length], dtype=np.int32
#     )
#     outcome = SparseCategorical(outcome_idx, depth=c_rank_d_3x2.max_outcome)
#     groups = {
#         'anonymous_id': np.array(
#             [
#                 [[4], [4]],
#                 [[4], [4]],
#                 [[4], [4]]
#             ], dtype=np.int32
#         )
#     }
#     sample_weight = .4 * np.ones(
#         [c_rank_d_3x2.n_sequence, c_rank_d_3x2.sequence_length]
#     )
#     trials_4 = TrialDataset(
#         c_rank_d_3x2, groups=groups, outcome=outcome, sample_weight=sample_weight
#     )

#     outcome_idx = np.zeros(
#         [c_rank_e_2x3.n_sequence, c_rank_e_2x3.sequence_length], dtype=np.int32
#     )
#     outcome = SparseCategorical(outcome_idx, depth=c_rank_e_2x3.max_outcome)
#     groups = {
#         'condition_id': np.array(
#             [
#                 [[5], [5], [5]],
#                 [[5], [5], [5]],
#             ], dtype=np.int32
#         )
#     }
#     sample_weight = .5 * np.ones(
#         [c_rank_e_2x3.n_sequence, c_rank_e_2x3.sequence_length]
#     )
#     trials_5 = TrialDataset(
#         c_rank_e_2x3, groups=groups, outcome=outcome, sample_weight=sample_weight
#     )

#     with pytest.raises(Exception) as e_info:
#         stack((trials_4, trials_5))
#     assert e_info.type == ValueError
#     assert str(e_info.value) == (
#         "The dictionary keys of `groups` must be identical for all "
#         "TrialDatasets. Got a mismatch: dict_keys(['anonymous_id']) and "
#         "dict_keys(['condition_id'])."
#     )


# def test_invalid_stack_1(c_rank_d_3x2, c_rank_e_2x3):
#     """Test invalid stack.

#     Incompatible shapes in `groups`.

#     """
#     # Creat two trial datasets.
#     outcome_idx = np.zeros(
#         [c_rank_d_3x2.n_sequence, c_rank_d_3x2.sequence_length], dtype=np.int32
#     )
#     outcome = SparseCategorical(outcome_idx, depth=c_rank_d_3x2.max_outcome)
#     groups = {
#         'condition_id': np.array(
#             [
#                 [[4], [4]],
#                 [[4], [4]],
#                 [[4], [4]]
#             ], dtype=np.int32
#         )
#     }
#     sample_weight = .4 * np.ones(
#         [c_rank_d_3x2.n_sequence, c_rank_d_3x2.sequence_length]
#     )
#     trials_4 = TrialDataset(
#         c_rank_d_3x2, groups=groups, outcome=outcome, sample_weight=sample_weight
#     )

#     outcome_idx = np.zeros(
#         [c_rank_e_2x3.n_sequence, c_rank_e_2x3.sequence_length], dtype=np.int32
#     )
#     outcome = SparseCategorical(outcome_idx, depth=c_rank_e_2x3.max_outcome)
#     groups = {
#         'condition_id': np.array(
#             [
#                 [[0.8, 0.2], [0.8, 0.2], [0.8, 0.2]],
#                 [[0.8, 0.2], [0.8, 0.2], [0.8, 0.2]],
#             ], dtype=np.int32
#         )
#     }
#     sample_weight = .5 * np.ones(
#         [c_rank_e_2x3.n_sequence, c_rank_e_2x3.sequence_length]
#     )
#     trials_5 = TrialDataset(
#         c_rank_e_2x3, groups=groups, outcome=outcome, sample_weight=sample_weight
#     )

#     with pytest.raises(Exception) as e_info:
#         stack((trials_4, trials_5))
#     assert e_info.type == ValueError
#     assert str(e_info.value) == (
#         "The shape of 'groups's 'condition_id' is not compatible. They must "
#         "be identical on axis=2."
#     )
