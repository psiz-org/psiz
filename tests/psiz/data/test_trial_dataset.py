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

from psiz.data.groups.group import Group
from psiz.data.outcomes.sparse_categorical import SparseCategorical
from psiz.data.trial_dataset import TrialDataset
from psiz.data.trial_component import TrialComponent


class BadTrialComponent(TrialComponent):
    """Abstract class for trial content data."""

    def __init__(self):
        """Initialize."""
        TrialComponent.__init__(self)
        self.x = np.array(
            [
                [1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0],
            ]
        )
        self.n_sequence = 4
        self.sequence_length = 1

    def export(self):
        return tf.constant(self.x)


def test_init_0(c_2rank1_aa_4x1):
    """Test initialization.

    Bare minimum arguments.

    """
    td = TrialDataset([c_2rank1_aa_4x1])

    assert td.n_sequence == c_2rank1_aa_4x1.n_sequence
    assert td.sequence_length == c_2rank1_aa_4x1.sequence_length
    assert len(td.content_list) == 1
    assert len(td.group_list) == 0
    assert len(td.outcome_list) == 0


def test_init_1(c_2rank1_aa_4x1):
    """Test initialization.

    With outcome, no sample weights.

    """
    outcome_idx = np.zeros(
        [c_2rank1_aa_4x1.n_sequence, c_2rank1_aa_4x1.sequence_length],
        dtype=np.int32
    )
    rank_outcome = SparseCategorical(
        outcome_idx, depth=c_2rank1_aa_4x1.n_outcome, name='rank_outcome'
    )

    td = TrialDataset([c_2rank1_aa_4x1, rank_outcome])

    assert td.n_sequence == c_2rank1_aa_4x1.n_sequence
    assert td.sequence_length == c_2rank1_aa_4x1.sequence_length
    assert len(td.content_list) == 1
    assert len(td.group_list) == 0
    assert len(td.outcome_list) == 1


def test_init_2(c_2rank1_aa_4x1, o_2rank1_aa_4x1):
    """Test initialization.

    With outcome, including sample_weight.
    With group, mixture format.

    """
    value = np.array(
        [
            [[.1, .9]],
            [[.5, .5]],
            [[1., 0.]],
            [[.9, .1]],
        ]
    )
    group_0 = Group(
        value, name='group_id'
    )

    td = TrialDataset([c_2rank1_aa_4x1, group_0, o_2rank1_aa_4x1])

    assert td.n_sequence == c_2rank1_aa_4x1.n_sequence
    assert td.sequence_length == c_2rank1_aa_4x1.sequence_length
    assert len(td.content_list) == 1
    assert len(td.group_list) == 1
    assert len(td.outcome_list) == 1


def test_init_3(c_2rank1_aa_4x1):
    """Test initialization.

    With outcome, including sample_weight argument.
    With group, pass in sparse format.

    """
    # Create rank outcome.
    outcome_idx = np.zeros(
        [c_2rank1_aa_4x1.n_sequence, c_2rank1_aa_4x1.sequence_length],
        dtype=np.int32
    )
    sample_weight = .9 * np.ones(
        [c_2rank1_aa_4x1.n_sequence, c_2rank1_aa_4x1.sequence_length]
    )
    rank_outcome = SparseCategorical(
        outcome_idx,
        depth=c_2rank1_aa_4x1.n_outcome,
        sample_weight=sample_weight,
        name='rank_outcome'
    )

    value = np.array(
        [
            [[0]],
            [[1]],
            [[0]],
            [[0]],
        ]
    )
    group_0 = Group(
        value, name='condition_idx'
    )

    td = TrialDataset([c_2rank1_aa_4x1, group_0, rank_outcome])

    assert td.n_sequence == c_2rank1_aa_4x1.n_sequence
    assert td.sequence_length == c_2rank1_aa_4x1.sequence_length
    assert len(td.content_list) == 1
    assert len(td.group_list) == 1
    assert len(td.outcome_list) == 1


def test_init_4(c_2rank1_d_3x2, o_2rank1_d_3x2, o_rt_a_3x2):
    """Test initialization.

    One content, two outcomes.

    """
    td = TrialDataset([c_2rank1_d_3x2, o_2rank1_d_3x2, o_rt_a_3x2])

    assert td.n_sequence == 3
    assert td.sequence_length == 2
    assert len(td.content_list) == 1
    assert len(td.group_list) == 0
    assert len(td.outcome_list) == 2


def test_init_5(c_2rank1_d_3x2, o_2rank1_d_3x2, c_rate2_e_3x2, o_rate2_a_3x2):
    """Test initialization.

    * two contents
    * two outcomes

    """
    td = TrialDataset(
        [c_2rank1_d_3x2, o_2rank1_d_3x2, c_rate2_e_3x2, o_rate2_a_3x2]
    )

    assert td.n_sequence == 3
    assert td.sequence_length == 2
    assert len(td.content_list) == 2
    assert len(td.group_list) == 0
    assert len(td.outcome_list) == 2


def test_invalid_init_0(c_2rank1_aa_4x1, o_2rank1_d_3x2, o_4rank2_c_4x3):
    """Test invalid initialization.

    * Number of sequences disagrees.
    * Sequence length disagrees.

    """
    with pytest.raises(Exception) as e_info:
        TrialDataset([c_2rank1_aa_4x1, o_2rank1_d_3x2])
    assert e_info.type == ValueError
    assert str(e_info.value) == (
        "All user-provided 'TrialComponent' objects must have the same "
        "`n_sequence`. The 'TrialComponent' in position 1 does not match "
        "the previous components."
    )

    with pytest.raises(Exception) as e_info:
        TrialDataset([c_2rank1_aa_4x1, o_4rank2_c_4x3])
    assert e_info.type == ValueError
    assert str(e_info.value) == (
        "All user-provided 'TrialComponent' objects must have the same "
        "`sequence_length`. The 'TrialComponent' in position 1 does not "
        "match the previous components."
    )

    bad_component_4x1 = BadTrialComponent()
    with pytest.raises(Exception) as e_info:
        TrialDataset([c_2rank1_aa_4x1, bad_component_4x1])
    assert e_info.type == ValueError
    assert str(e_info.value) == (
        "The `TrialComponent` in position 1 must be an  instance of "
        "`psiz.data.Content`, `psiz.data.Outcome`, or `psiz.data.Group` to "
        "use `TrialDataset`."
    )


def test_export_0(c_2rank1_d_3x2, g_condition_idx_3x2):
    """Test export.

    * Include content and group only.

    """
    td = TrialDataset([c_2rank1_d_3x2, g_condition_idx_3x2])

    desired_x_stimulus_set = tf.constant(
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
    desired_x_is_select = tf.constant(
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
        desired_x_stimulus_set, x['2rank1/stimulus_set']
    )
    tf.debugging.assert_equal(
        desired_x_is_select, x['2rank1/is_select']
    )
    tf.debugging.assert_equal(
        desired_condition_id, x['condition_idx']
    )


def test_export_1a(c_2rank1_d_3x2, g_condition_idx_3x2, o_2rank1_d_3x2):
    """Test as_dataset.

    * Include content, group, and outcome.
    * A single output model, therefore drop dictionary keys of `y` and
        `w`.

    """
    td = TrialDataset([c_2rank1_d_3x2, g_condition_idx_3x2, o_2rank1_d_3x2])

    desired_x_stimulus_set = tf.constant(
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
    desired_x_is_select = tf.constant(
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
                [1., 0.],
                [1., 0.],
            ],
            [
                [1., 0.],
                [1., 0.],
            ],
            [
                [1., 0.],
                [1., 0.],
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
        desired_x_stimulus_set, x['2rank1/stimulus_set']
    )
    tf.debugging.assert_equal(
        desired_x_is_select, x['2rank1/is_select']
    )
    tf.debugging.assert_equal(
        desired_condition_id, x['condition_idx']
    )
    tf.debugging.assert_equal(desired_y, y)
    tf.debugging.assert_equal(desired_w, w)
    tf.debugging.assert_equal(desired_condition_id, x['condition_idx'])

    # Export inputs only.
    ds = td.export(inputs_only=True).batch(4, drop_remainder=False)
    ds_list = list(ds)
    x = ds_list[0]
    assert len(ds_list) == 1
    tf.debugging.assert_equal(
        desired_x_stimulus_set, x['2rank1/stimulus_set']
    )
    tf.debugging.assert_equal(
        desired_x_is_select, x['2rank1/is_select']
    )
    tf.debugging.assert_equal(
        desired_condition_id, x['condition_idx']
    )


def test_export_1b(c_2rank1_d_3x2, g_condition_idx_3x2, o_2rank1_d_3x2):
    """Test export.

    Return dataset using `with_timestep_axis=False`.

    """
    td = TrialDataset([c_2rank1_d_3x2, g_condition_idx_3x2, o_2rank1_d_3x2])

    desired_x_stimulus_set = tf.constant(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [0, 0, 0],
            [10, 11, 12],
            [14, 15, 16],
        ], dtype=tf.int32
    )
    desired_x_is_select = tf.constant(
        [
            [
                [False, True, False],
                [False, True, False],
                [False, True, False],
                [False, False, False],
                [False, True, False],
                [False, True, False],
            ]
        ], dtype=tf.bool
    )
    desired_condition_id = tf.constant(
        [
            [0], [0], [1], [1], [0], [0]
        ], dtype=tf.int32
    )
    desired_y = tf.constant(
        [
                [1., 0.],
                [1., 0.],
                [1., 0.],
                [1., 0.],
                [1., 0.],
                [1., 0.],
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
        desired_x_stimulus_set, x['2rank1/stimulus_set']
    )
    tf.debugging.assert_equal(
        desired_x_is_select, x['2rank1/is_select']
    )
    tf.debugging.assert_equal(
        desired_condition_id, x['condition_idx']
    )
    tf.debugging.assert_equal(desired_y, y)
    tf.debugging.assert_equal(desired_w, w)


def test_export_2a(
    c_2rank1_d_3x2, g_condition_idx_3x2, o_2rank1_d_3x2, o_rt_a_3x2
):
    """Test export.

    * Multi-output model, therefore keep dictionary keys for `y` and
    `w`.

    """
    td = TrialDataset(
        [c_2rank1_d_3x2, g_condition_idx_3x2, o_2rank1_d_3x2, o_rt_a_3x2]
    )

    desired_x_stimulus_set = tf.constant(
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
    desired_x_is_select = tf.constant(
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
                [1., 0.],
                [1., 0.],
            ],
            [
                [1., 0.],
                [1., 0.],
            ],
            [
                [1., 0.],
                [1., 0.],
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
        desired_x_stimulus_set, x['2rank1/stimulus_set']
    )
    tf.debugging.assert_equal(
        desired_x_is_select, x['2rank1/is_select']
    )
    tf.debugging.assert_equal(
        desired_condition_id, x['condition_idx']
    )
    tf.debugging.assert_equal(desired_y_prob, y['rank_prob'])
    tf.debugging.assert_equal(desired_w_prob, w['rank_prob'])
    tf.debugging.assert_equal(desired_y_rt, y['rt'])
    tf.debugging.assert_equal(desired_w_rt, w['rt'])


def test_export_3(c_rate2_a_4x1, g_condition_label_4x1, o_continuous_a_4x1):
    """Test export with `StringLookup`."""
    td = TrialDataset(
        [c_rate2_a_4x1, g_condition_label_4x1, o_continuous_a_4x1]
    )
    ds = td.export(export_format='tfds')

    # Map strings to indices.
    condition_lookup_layer = tf.keras.layers.StringLookup(
        vocabulary=['block', 'interleave'], num_oov_indices=0
    )

    def parse_inputs(x):
        condition_label = x.pop('condition_label')
        condition_idx = condition_lookup_layer(condition_label)
        x['condition_idx'] = condition_idx
        return x

    ds2 = ds.map(lambda x, y, w: (parse_inputs(x), y, w))
    ds2 = ds2.batch(4)
    ds2_list = list(ds2)

    desired_condition_idx = tf.constant(
        [
            [[0]],
            [[1]],
            [[0]],
            [[0]],
        ], dtype=tf.int64
    )
    tf.debugging.assert_equal(
        ds2_list[0][0]['condition_idx'], desired_condition_idx
    )


def test_invalid_export_0(c_2rank1_d_3x2, g_condition_idx_3x2, o_2rank1_d_3x2):
    """Test export.

    Using incorrect `export_format`.

    """
    td = TrialDataset([c_2rank1_d_3x2, g_condition_idx_3x2, o_2rank1_d_3x2])

    with pytest.raises(Exception) as e_info:
        td.export(export_format='garbage')
    assert e_info.type == ValueError
    assert (
        str(e_info.value) == "Unrecognized `export_format` 'garbage'."
    )


def test_invalid_export_1(c_2rank1_d_3x2, o_2rank1_d_3x2, o_rt_a_3x2_noname):
    """Test export.

    Using incorrect `export_format`.

    """
    td = TrialDataset([c_2rank1_d_3x2, o_2rank1_d_3x2, o_rt_a_3x2_noname])

    with pytest.raises(Exception) as e_info:
        td.export(export_format='tfds')
    assert e_info.type == ValueError
    assert (
        str(e_info.value) == (
            "When a `TrialDataset` has multiple outputs, all "
            "outputs must be created with the `name` argument."
        )
    )


def test_tf_ds_concatenate(c_2rank1_d_3x2, c_2rank1_e_3x2):
    """Test concatenating two datasets"""
    td_0 = TrialDataset([c_2rank1_d_3x2])
    td_1 = TrialDataset([c_2rank1_e_3x2])

    ds_0 = td_0.export(export_format='tfds')
    ds_1 = td_1.export(export_format='tfds')

    ds = ds_0.concatenate(ds_1).batch(6)
    ds_list = list(ds)
    _ = ds_list[0]
