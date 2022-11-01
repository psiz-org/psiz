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

from psiz.data.outcomes.sparse_categorical import SparseCategorical
from psiz.trials import stack


def test_init_0(sparse_cat_0):
    """Test initialization."""
    desired_n_sequence = 4
    desired_max_timestep = 1
    desired_index = np.array(
        [[0], [2], [0], [1]], dtype=np.int32
    )
    desired_depth = 3

    assert sparse_cat_0.n_sequence == desired_n_sequence
    assert sparse_cat_0.sequence_length == desired_max_timestep
    np.testing.assert_array_equal(
        desired_index, sparse_cat_0.index
    )
    assert desired_depth == sparse_cat_0.depth


def test_init_1(sparse_cat_1):
    """Test initialization."""
    desired_n_sequence = 4
    desired_max_timestep = 1
    desired_index = np.array(
        [[0], [2], [0], [1]], dtype=np.int32
    )
    desired_depth = 5

    assert sparse_cat_1.n_sequence == desired_n_sequence
    assert sparse_cat_1.sequence_length == desired_max_timestep
    np.testing.assert_array_equal(
        desired_index, sparse_cat_1.index
    )
    assert desired_depth == sparse_cat_1.depth


def test_init_2(sparse_cat_2):
    """Test initialization."""
    desired_n_sequence = 4
    desired_max_timestep = 3
    desired_index = np.array(
        [[0, 0, 0], [2, 0, 0], [0, 1, 0], [1, 1, 1]], dtype=np.int32
    )
    desired_depth = 3

    assert sparse_cat_2.n_sequence == desired_n_sequence
    assert sparse_cat_2.sequence_length == desired_max_timestep
    np.testing.assert_array_equal(
        desired_index, sparse_cat_2.index
    )
    assert desired_depth == sparse_cat_2.depth


def test_init_wrong_0():
    """Test initialization.

    Indices are not integers.

    """
    outcome_idx = np.array(
        [
            [0., 2., 1.],
            [1., 2., 2.],
        ], dtype=np.float32
    )
    with pytest.raises(Exception) as e_info:
        SparseCategorical(outcome_idx, depth=3)
    assert e_info.type == ValueError


def test_init_wrong_1():
    """Test initialization.

    Indices are not greater than placeholder.

    """
    outcome_idx = np.array(
        [
            [0, 2, 1],
            [-1, 2, 2],
        ], dtype=np.int32
    )
    with pytest.raises(Exception) as e_info:
        SparseCategorical(outcome_idx, depth=3)
    assert e_info.type == ValueError


def test_init_wrong_2():
    """Test initialization.

    Indices are not rank-2 ndarray.

    """
    outcome_idx = np.array(
        [
            [
                [0, 2, 1],
                [1, 2, 2],
            ],
            [
                [2, 0, 1],
                [1, 0, 2],
            ],
        ], dtype=np.int32
    )
    with pytest.raises(Exception) as e_info:
        SparseCategorical(outcome_idx, depth=3)
    assert e_info.type == ValueError


def test_export_0(sparse_cat_0):
    desired_y = tf.constant(
        np.array(
            [
                [[1., 0., 0.]],
                [[0., 0., 1.]],
                [[1., 0., 0.]],
                [[0., 1., 0.]],
            ], dtype=np.float32
        )
    )

    tf.debugging.assert_equal(desired_y, sparse_cat_0.export())


def test_export_1(sparse_cat_1):
    desired_y = tf.constant(
        np.array(
            [
                [[1., 0., 0., 0., 0.]],
                [[0., 0., 1., 0., 0.]],
                [[1., 0., 0., 0., 0.]],
                [[0., 1., 0., 0., 0.]],
            ], dtype=np.float32
        )
    )

    tf.debugging.assert_equal(desired_y, sparse_cat_1.export())


def test_export_2(sparse_cat_1):
    """Test export.

    Use with_timestep_axis=False

    """
    desired_y = tf.constant(
        np.array(
            [
                [1., 0., 0., 0., 0.],
                [0., 0., 1., 0., 0.],
                [1., 0., 0., 0., 0.],
                [0., 1., 0., 0., 0.],
            ], dtype=np.float32
        )
    )
    y = sparse_cat_1.export(with_timestep_axis=False)
    tf.debugging.assert_equal(desired_y, y)


def test_export_3(sparse_cat_2):
    """Test export.

    Use with_timestep_axis=False

    """
    desired_y = tf.constant(
        np.array(
            [
                [1., 0., 0.],
                [1., 0., 0.],
                [1., 0., 0.],
                [0., 0., 1.],
                [1., 0., 0.],
                [1., 0., 0.],
                [1., 0., 0.],
                [0., 1., 0.],
                [1., 0., 0.],
                [0., 1., 0.],
                [0., 1., 0.],
                [0., 1., 0.]
            ], dtype=np.float32
        )
    )
    y = sparse_cat_2.export(with_timestep_axis=False)
    tf.debugging.assert_equal(desired_y, y)


def test_export_wrong(sparse_cat_2):
    """Test export.

    Using incorrect `export_format`.

    """
    with pytest.raises(Exception) as e_info:
        sparse_cat_2.export(export_format='garbage')
    assert e_info.type == ValueError
    assert (
        str(e_info.value) == "Unrecognized `export_format` 'garbage'."
    )


def test_for_dataset(sparse_cat_2):
    desired_y = tf.constant(
        np.array(
            [
                [[1., 0., 0.], [1., 0., 0.], [1., 0., 0.]],
                [[0., 0., 1.], [1., 0., 0.], [1., 0., 0.]],
                [[1., 0., 0.], [0., 1., 0.], [1., 0., 0.]],
                [[0., 1., 0.], [0., 1., 0.], [0., 1., 0.]]
            ], dtype=np.float32
        )
    )

    tf.debugging.assert_equal(desired_y, sparse_cat_2.export())


def test_persistence(sparse_cat_2, tmpdir):
    """Test save and load."""
    group_name = "outcome"

    original = sparse_cat_2
    fn = tmpdir.join('persistence_test.hdf5')

    # Save group.
    f = h5py.File(fn, "w")
    grp_stimulus = f.create_group(group_name)
    original.save(grp_stimulus)
    f.close()

    # Load group.
    f = h5py.File(fn, "r")
    h5_grp = f[group_name]
    # Encoding/read rules changed in h5py 3.0, requiring asstr() call.
    try:
        class_name = h5_grp["class_name"].asstr()[()]
    except AttributeError:
        class_name = h5_grp["class_name"][()]
    reconstructed = SparseCategorical.load(h5_grp)
    f.close()

    # Check for equivalency.
    assert class_name == "SparseCategorical"
    assert original.n_sequence == reconstructed.n_sequence
    assert original.sequence_length == reconstructed.sequence_length
    assert original.depth == reconstructed.depth
    np.testing.assert_array_equal(
        original.index, reconstructed.index
    )


def test_subset_0(rank_sim_4):
    """Test subset."""
    desired_n_sequence = 2
    desired_max_timestep = 2
    desired_index = np.array(
        [[0, 0], [0, 0]], dtype=np.int32
    )
    desired_depth = 3

    content = rank_sim_4
    outcome_idx = np.zeros(
        [content.n_sequence, content.sequence_length], dtype=np.int32
    )
    sparse_cat = SparseCategorical(outcome_idx, depth=content.max_outcome)

    sub = sparse_cat.subset(np.array([1, 2]))

    assert sub.n_sequence == desired_n_sequence
    assert sub.sequence_length == desired_max_timestep
    np.testing.assert_array_equal(
        desired_index, sub.index
    )
    assert desired_depth == sub.depth


def test_stack_0(sparse_cat_1, sparse_cat_2, sparse_cat_3):
    """Test stack."""
    desired_n_sequence = 10
    desired_max_timestep = 3
    desired_index = np.array(
        [
            [0, 0, 0],
            [2, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
            [0, 0, 0],
            [2, 0, 0],
            [0, 1, 0],
            [1, 1, 1],
            [0, 2, 1],
            [1, 2, 2],
        ], dtype=np.int32
    )
    desired_depth = 5

    stacked = stack((sparse_cat_1, sparse_cat_2, sparse_cat_3))

    assert desired_n_sequence == stacked.n_sequence
    assert desired_max_timestep == stacked.sequence_length
    np.testing.assert_array_equal(
        desired_index, stacked.index
    )
    assert desired_depth == stacked.depth
