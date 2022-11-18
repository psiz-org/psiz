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

from psiz.data.outcomes.sparse_categorical import SparseCategorical


def test_init_0(o_sparsecat_a_4x1):
    """Test initialization.

    * Default sample_weight

    """
    desired_name = 'sparsecat_a'
    desired_n_sequence = 4
    desired_sequence_length = 1
    desired_index = np.array(
        [[0], [2], [0], [1]], dtype=np.int32
    )
    desired_depth = 3
    desired_sample_weight = np.ones([4, 1], dtype=np.float32)

    assert desired_name == o_sparsecat_a_4x1.name
    assert o_sparsecat_a_4x1.n_sample == desired_n_sequence
    assert o_sparsecat_a_4x1.sequence_length == desired_sequence_length
    np.testing.assert_array_equal(
        desired_index, o_sparsecat_a_4x1.index
    )
    assert desired_depth == o_sparsecat_a_4x1.depth
    np.testing.assert_array_equal(
        desired_sample_weight, o_sparsecat_a_4x1.sample_weight
    )


def test_init_1(o_sparsecat_aa_4x1):
    """Test initialization."""
    desired_name = 'sparsecat_aa'
    desired_n_sequence = 4
    desired_sequence_length = 1
    desired_index = np.array(
        [[0], [2], [0], [1]], dtype=np.int32
    )
    desired_depth = 5
    desired_sample_weight = np.ones([4, 1], dtype=np.float32)

    assert o_sparsecat_aa_4x1.name == desired_name
    assert o_sparsecat_aa_4x1.n_sample == desired_n_sequence
    assert o_sparsecat_aa_4x1.sequence_length == desired_sequence_length
    np.testing.assert_array_equal(
        desired_index, o_sparsecat_aa_4x1.index
    )
    assert desired_depth == o_sparsecat_aa_4x1.depth
    np.testing.assert_array_equal(
        desired_sample_weight, o_sparsecat_aa_4x1.sample_weight
    )


def test_init_2(o_sparsecat_b_4x3):
    """Test initialization."""
    desired_name = 'sparsecat_b'
    desired_n_sequence = 4
    desired_sequence_length = 3
    desired_index = np.array(
        [[0, 0, 0], [2, 0, 0], [0, 1, 0], [1, 1, 1]], dtype=np.int32
    )
    desired_depth = 3
    desired_sample_weight = np.ones([4, 3], dtype=np.float32)

    assert o_sparsecat_b_4x3.name == desired_name
    assert o_sparsecat_b_4x3.n_sample == desired_n_sequence
    assert o_sparsecat_b_4x3.sequence_length == desired_sequence_length
    np.testing.assert_array_equal(
        desired_index, o_sparsecat_b_4x3.index
    )
    assert desired_depth == o_sparsecat_b_4x3.depth
    np.testing.assert_array_equal(
        desired_sample_weight, o_sparsecat_b_4x3.sample_weight
    )


def test_init_3(o_sparsecat_d_4x3):
    """Test initialization."""
    desired_name = 'sparsecat_d'
    desired_n_sequence = 4
    desired_sequence_length = 3
    desired_index = np.array(
        [[0, 0, 0], [2, 0, 0], [0, 1, 0], [1, 1, 1]], dtype=np.int32
    )
    desired_depth = 3
    desired_sample_weight = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
            [1.0, 1.0, 0.0],
        ], dtype=np.float32
    )

    assert o_sparsecat_d_4x3.name == desired_name
    assert o_sparsecat_d_4x3.n_sample == desired_n_sequence
    assert o_sparsecat_d_4x3.sequence_length == desired_sequence_length
    np.testing.assert_array_equal(
        desired_index, o_sparsecat_d_4x3.index
    )
    assert desired_depth == o_sparsecat_d_4x3.depth
    np.testing.assert_array_equal(
        desired_sample_weight, o_sparsecat_d_4x3.sample_weight
    )


def test_invalid_init_0():
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


def test_invalid_init_1():
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


def test_invalid_init_2():
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


def test_invalid_init_3():
    """Test invalid sample_weight initialization."""
    n_sample = 4
    sequence_length = 1
    depth = 4
    outcome_idx = np.zeros([n_sample, sequence_length], dtype=np.int32)

    sample_weight = .9 * np.ones(
        [n_sample, sequence_length, 2]
    )
    with pytest.raises(Exception) as e_info:
        SparseCategorical(
            outcome_idx, depth=depth, sample_weight=sample_weight
        )
    assert e_info.type == ValueError
    assert str(e_info.value) == (
        "The argument 'sample_weight' must be a rank-2 ND array."
    )

    sample_weight = .9 * np.ones(
        [n_sample + 1, sequence_length]
    )
    with pytest.raises(Exception) as e_info:
        SparseCategorical(
            outcome_idx, depth=depth, sample_weight=sample_weight
        )
    assert e_info.type == ValueError
    assert str(e_info.value) == (
        "The argument 'sample_weight' must have "
        "shape=(samples, sequence_length) that agrees with the rest of "
        "the object."
    )

    sample_weight = .9 * np.ones(
        [n_sample, sequence_length + 1]
    )
    with pytest.raises(Exception) as e_info:
        SparseCategorical(
            outcome_idx, depth=depth, sample_weight=sample_weight
        )
    assert e_info.type == ValueError
    assert str(e_info.value) == (
        "The argument 'sample_weight' must have "
        "shape=(samples, sequence_length) that agrees with the rest of "
        "the object."
    )


def test_export_0(o_sparsecat_a_4x1):
    desired_y = tf.constant(
        [
            [1., 0., 0.],
            [0., 0., 1.],
            [1., 0., 0.],
            [0., 1., 0.],
        ], dtype=tf.float32
    )
    desired_w = tf.constant(
        [1.0, 1.0, 1.0, 1.0]
    )
    desired_name = 'sparsecat_a'

    y, w = o_sparsecat_a_4x1.export(export_format='tfds')
    tf.debugging.assert_equal(desired_y, y[desired_name])
    tf.debugging.assert_equal(desired_w, w[desired_name])


def test_export_1(o_sparsecat_aa_4x1):
    desired_y = tf.constant(
        [
            [[1., 0., 0., 0., 0.]],
            [[0., 0., 1., 0., 0.]],
            [[1., 0., 0., 0., 0.]],
            [[0., 1., 0., 0., 0.]],
        ], dtype=tf.float32
    )
    desired_w = tf.constant(
        [[1.0], [1.0], [1.0], [1.0]]
    )
    desired_name = 'sparsecat_aa'

    y, w = o_sparsecat_aa_4x1.export(export_format='tfds')
    tf.debugging.assert_equal(desired_y, y[desired_name])
    tf.debugging.assert_equal(desired_w, w[desired_name])


def test_export_2(o_sparsecat_aa_4x1):
    """Test export.

    Use override `with_timestep_axis=False`

    """
    desired_y = tf.constant(
        [
            [1., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0.],
            [1., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0.],
        ], dtype=tf.float32
    )
    desired_w = tf.constant(
        [1.0, 1.0, 1.0, 1.0]
    )
    desired_name = 'sparsecat_aa'

    y, w = o_sparsecat_aa_4x1.export(
        export_format='tfds', with_timestep_axis=False
    )
    tf.debugging.assert_equal(desired_y, y[desired_name])
    tf.debugging.assert_equal(desired_w, w[desired_name])


def test_export_3a(o_sparsecat_b_4x3):
    desired_y = tf.constant(
        [
            [[1., 0., 0.], [1., 0., 0.], [1., 0., 0.]],
            [[0., 0., 1.], [1., 0., 0.], [1., 0., 0.]],
            [[1., 0., 0.], [0., 1., 0.], [1., 0., 0.]],
            [[0., 1., 0.], [0., 1., 0.], [0., 1., 0.]]
        ], dtype=tf.float32
    )
    desired_w = tf.ones([4, 3], dtype=tf.float32)
    desired_name = 'sparsecat_b'

    y, w = o_sparsecat_b_4x3.export(export_format='tfds')
    tf.debugging.assert_equal(desired_y, y[desired_name])
    tf.debugging.assert_equal(desired_w, w[desired_name])


def test_export_3b(o_sparsecat_b_4x3):
    """Test export.

    Use override `with_timestep_axis=False`

    """
    desired_y = tf.constant(
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
        ], dtype=tf.float32
    )
    desired_w = tf.ones([12], dtype=tf.float32)
    desired_name = 'sparsecat_b'

    y, w = o_sparsecat_b_4x3.export(
        export_format='tfds', with_timestep_axis=False
    )
    tf.debugging.assert_equal(desired_y, y[desired_name])
    tf.debugging.assert_equal(desired_w, w[desired_name])


def test_export_4a(o_sparsecat_d_4x3):
    desired_y = tf.constant(
        [
            [[1, 0, 0], [1, 0, 0], [1, 0, 0]],
            [[0, 0, 1], [1, 0, 0], [1, 0, 0]],
            [[1, 0, 0], [0, 1, 0], [1, 0, 0]],
            [[0, 1, 0], [0, 1, 0], [0, 1, 0]],
        ], dtype=tf.float32
    )
    desired_w = tf.constant(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
            [1.0, 1.0, 0.0],
        ], dtype=tf.float32
    )
    desired_name = 'sparsecat_d'

    y, w = o_sparsecat_d_4x3.export(export_format='tfds')
    tf.debugging.assert_equal(desired_y, y[desired_name])
    tf.debugging.assert_equal(desired_w, w[desired_name])


def test_invalid_export_0(o_sparsecat_b_4x3):
    """Test export.

    Using incorrect `export_format`.

    """
    with pytest.raises(Exception) as e_info:
        o_sparsecat_b_4x3.export(export_format='garbage')
    assert e_info.type == ValueError
    assert (
        str(e_info.value) == "Unrecognized `export_format` 'garbage'."
    )
