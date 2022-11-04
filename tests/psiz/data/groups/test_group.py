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
import pytest

import numpy as np
import tensorflow as tf

from psiz.data.groups.group import Group


def test_init_0(g_condition_id_4x1):
    """Test initalization."""
    desired_group_weights = np.array(
        [
            [[0]],
            [[1]],
            [[0]],
            [[0]],
        ]
    )
    assert g_condition_id_4x1.name == 'condition_id'
    assert g_condition_id_4x1.n_sequence == 4
    assert g_condition_id_4x1.sequence_length == 1
    np.testing.assert_array_equal(
        g_condition_id_4x1.group_weights, desired_group_weights
    )


def test_init_1(g_condition_id_4x3):
    """Test initalization."""
    desired_group_weights = np.array(
        [
            [[0], [0], [0]],
            [[1], [1], [1]],
            [[0], [0], [0]],
            [[0], [0], [0]],
        ]
    )
    assert g_condition_id_4x3.name == 'condition_id'
    assert g_condition_id_4x3.n_sequence == 4
    assert g_condition_id_4x3.sequence_length == 3
    np.testing.assert_array_equal(
        g_condition_id_4x3.group_weights, desired_group_weights
    )


def test_init_2(g_mix2_4x3):
    """Test initalization."""
    desired_group_weights = np.array(
       [
            [[0.5, 0.5], [0.6, 0.4], [0.7, 0.3]],
            [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            [[0.8, 0.2], [0.8, 0.2], [0.8, 0.2]],
            [[0.2, 0.8], [0.2, 0.8], [0.2, 0.8]],
        ], dtype=np.float32
    )
    assert g_mix2_4x3.name == 'mix2'
    assert g_mix2_4x3.n_sequence == 4
    assert g_mix2_4x3.sequence_length == 3
    np.testing.assert_array_equal(
        g_mix2_4x3.group_weights, desired_group_weights
    )


def test_export_0a(g_mix2_4x3):
    desired_group_weight = tf.constant(
        [
            [[0.5, 0.5], [0.6, 0.4], [0.7, 0.3]],
            [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            [[0.8, 0.2], [0.8, 0.2], [0.8, 0.2]],
            [[0.2, 0.8], [0.2, 0.8], [0.2, 0.8]],
        ], dtype=tf.float32
    )
    desired_name = 'mix2'

    x = g_mix2_4x3.export(export_format='tf')
    tf.debugging.assert_equal(desired_group_weight, x[desired_name])


def test_export_0b(g_mix2_4x3):
    desired_group_weight = tf.constant(
        [
            [0.5, 0.5], [0.6, 0.4], [0.7, 0.3],
            [0.0, 1.0], [0.0, 1.0], [0.0, 1.0],
            [0.8, 0.2], [0.8, 0.2], [0.8, 0.2],
            [0.2, 0.8], [0.2, 0.8], [0.2, 0.8],
        ], dtype=tf.float32
    )
    desired_name = 'mix2'

    x = g_mix2_4x3.export(
        export_format='tf', with_timestep_axis=False
    )
    tf.debugging.assert_equal(desired_group_weight, x[desired_name])


def test_export_wrong(g_mix2_4x3):
    """Test export.

    Using incorrect `export_format`.

    """
    with pytest.raises(Exception) as e_info:
        g_mix2_4x3.export(export_format='garbage')
    assert e_info.type == ValueError
    assert (
        str(e_info.value) == "Unrecognized `export_format` 'garbage'."
    )


def test_subset_0(g_mix2_4x3):
    """Test subset."""
    desired_n_sequence = 2
    desired_sequence_length = 3
    desired_group_weights = np.array(
        [
            [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            [[0.8, 0.2], [0.8, 0.2], [0.8, 0.2]],
        ], dtype=np.float32
    )
    desired_name = 'mix2'

    sub = g_mix2_4x3.subset(np.array([1, 2]))

    assert desired_name == sub.name
    assert desired_n_sequence == sub.n_sequence
    assert desired_sequence_length == sub.sequence_length
    np.testing.assert_array_equal(
        desired_group_weights, sub.group_weights
    )


def test_persistence_0(g_mix2_4x3, tmpdir):
    """Test save and load."""
    h5_grp_name = "sparsecat"

    original = g_mix2_4x3
    fn = tmpdir.join('persistence_test.hdf5')

    # Save group.
    f = h5py.File(fn, "w")
    h5_grp = f.create_group(h5_grp_name)
    original.save(h5_grp)
    f.close()

    # Load group.
    f = h5py.File(fn, "r")
    h5_grp = f[h5_grp_name]
    # Encoding/read rules changed in h5py 3.0, requiring asstr() call.
    try:
        class_name = h5_grp["class_name"].asstr()[()]
    except AttributeError:
        class_name = h5_grp["class_name"][()]
    reconstructed = Group.load(h5_grp)
    f.close()

    # Check for equivalency.
    assert class_name == "psiz.data.Group"
    assert original.name == reconstructed.name
    assert original.n_sequence == reconstructed.n_sequence
    assert original.sequence_length == reconstructed.sequence_length
    np.testing.assert_array_equal(
        original.group_weights, reconstructed.group_weights
    )


# TODO delete or finish
# def test_stack_0


# TODO moved from test_trial_dataset
# def test_invalid_init_0(c_rank_aa_4x1, o_rank_aa_4x1):
#     """Test invalid groups initialization.

#     Bad group shapes:
#     * invalid rank 4 groups.
#     * mismatch in n_sequence
#     * mismatch in sequence_length

#     """
#     content = c_rank_aa_4x1
#     outcome = o_rank_aa_4x1

#     rng = default_rng()
#     groups = {
#         'condition_id': rng.choice(
#             2, size=[content.n_sequence, content.sequence_length, 1, 1]
#         ).astype(dtype=int)
#     }
#     with pytest.raises(Exception) as e_info:
#         TrialDataset(
#             content,
#             outcome=outcome,
#             groups=groups,
#             sample_weight=sample_weight
#         )
#     assert e_info.type == ValueError
#     assert str(e_info.value) == (
#         "The group weights for the dictionary key 'condition_id' must be a "
#         "rank-2 or rank-3 ND array. If using a sparse coding format, make "
#         "sure you have a trailing singleton dimension to meet this "
#         "requirement."
#     )

#     groups = {
#         'condition_id': rng.choice(
#             2, size=[content.n_sequence + 1, content.sequence_length, 1]
#         ).astype(dtype=int)
#     }
#     with pytest.raises(Exception) as e_info:
#         TrialDataset(
#             content,
#             outcome=outcome,
#             groups=groups,
#             sample_weight=sample_weight
#         )
#     assert e_info.type == ValueError
#     assert str(e_info.value) == (
#         "The group weights for the dictionary key 'condition_id' must have a "
#         "shape that agrees with 'n_squence' of the 'content'."
#     )

#     groups = {
#         'condition_id': rng.choice(
#             2, size=[content.n_sequence, content.sequence_length + 1, 1]
#         ).astype(dtype=int)
#     }
#     with pytest.raises(Exception) as e_info:
#         TrialDataset(
#             content,
#             outcome=outcome,
#             groups=groups,
#             sample_weight=sample_weight
#         )
#     assert e_info.type == ValueError
#     assert str(e_info.value) == (
#         "The group weights for the dictionary key 'condition_id' must have a "
#         "shape that agrees with 'sequence_length' of the 'content'."
#     )

#     groups = {
#         'condition_id': np.array([
#                 [[0]], [[-2]], [[1]], [[-1]]
#         ])
#     }
#     with pytest.raises(Exception) as e_info:
#         TrialDataset(
#             content,
#             outcome=outcome,
#             groups=groups,
#             sample_weight=sample_weight
#         )
#     assert e_info.type == ValueError
#     assert str(e_info.value) == (
#         "The group weights for 'condition_id' contain "
#         "values less than 0. Found 2 bad trial(s)."
#     )
