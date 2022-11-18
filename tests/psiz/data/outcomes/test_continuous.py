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

from psiz.data.outcomes.continuous import Continuous


def test_init_0(o_continuous_a_4x1):
    """Test initialization.

    * Default sample_weight

    """
    desired_name = 'continuous_a'
    desired_n_sequence = 4
    desired_sequence_length = 1
    desired_n_unit = 1
    desired_value = np.array(
        [[[0.0]], [[2.0]], [[-0.1]], [[1.3]]], dtype=np.float32
    )
    desired_sample_weight = np.ones([4, 1], dtype=np.float32)

    assert desired_name == o_continuous_a_4x1.name
    assert desired_n_sequence == o_continuous_a_4x1.n_sample
    assert desired_sequence_length == o_continuous_a_4x1.sequence_length
    assert desired_n_unit == o_continuous_a_4x1.n_unit
    np.testing.assert_array_equal(
        desired_value, o_continuous_a_4x1.value
    )
    np.testing.assert_array_equal(
        desired_sample_weight, o_continuous_a_4x1.sample_weight
    )


def test_init_1(o_continuous_aa_4x1):
    """Test initialization."""
    desired_name = 'continuous_aa'
    desired_n_sequence = 4
    desired_sequence_length = 1
    desired_n_unit = 1
    desired_value = np.array(
        [[[0.0]], [[2.0]], [[-0.1]], [[1.3]]], dtype=np.float32
    )
    desired_sample_weight = np.ones([4, 1], dtype=np.float32)

    assert desired_name == o_continuous_aa_4x1.name
    assert desired_n_sequence == o_continuous_aa_4x1.n_sample
    assert desired_sequence_length == o_continuous_aa_4x1.sequence_length
    assert desired_n_unit == o_continuous_aa_4x1.n_unit
    np.testing.assert_array_equal(
        desired_value, o_continuous_aa_4x1.value
    )
    np.testing.assert_array_equal(
        desired_sample_weight, o_continuous_aa_4x1.sample_weight
    )


def test_init_2(o_continuous_b_4x3):
    """Test initialization."""
    desired_name = 'continuous_b'
    desired_n_sequence = 4
    desired_sequence_length = 3
    desired_n_unit = 1
    desired_value = np.array(
        [
            [[0.0], [0.0], [0.0]],
            [[2.0], [0.0], [0.0]],
            [[-0.1], [-1.0], [0.3]],
            [[1.0], [1.0], [1.0]],
        ], dtype=np.float32
    )
    desired_sample_weight = np.ones([4, 3], dtype=np.float32)

    assert desired_name == o_continuous_b_4x3.name
    assert desired_n_sequence == o_continuous_b_4x3.n_sample
    assert desired_sequence_length == o_continuous_b_4x3.sequence_length
    assert desired_n_unit == o_continuous_b_4x3.n_unit
    np.testing.assert_array_equal(
        desired_value, o_continuous_b_4x3.value
    )
    np.testing.assert_array_equal(
        desired_sample_weight, o_continuous_b_4x3.sample_weight
    )


def test_init_3(o_continuous_c_4x3):
    """Test initialization."""
    desired_name = 'continuous_c'
    desired_n_sequence = 4
    desired_sequence_length = 3
    desired_n_unit = 2
    desired_value = np.array(
        [
            [[0.0, 0.1], [0.0, 0.2], [0.0, 0.3]],
            [[2.0, 0.4], [0.0, 0.5], [0.0, 0.6]],
            [[-0.1, 0.7], [-1.0, 0.8], [0.3, 0.9]],
            [[1.0, 1.1], [1.0, 1.2], [1.0, 1.3]],
        ], dtype=np.float32
    )
    desired_sample_weight = np.ones([4, 3], dtype=np.float32)

    assert desired_name == o_continuous_c_4x3.name
    assert desired_n_sequence == o_continuous_c_4x3.n_sample
    assert desired_sequence_length == o_continuous_c_4x3.sequence_length
    assert desired_n_unit == o_continuous_c_4x3.n_unit
    np.testing.assert_array_equal(
        desired_value, o_continuous_c_4x3.value
    )
    np.testing.assert_array_equal(
        desired_sample_weight, o_continuous_c_4x3.sample_weight
    )


def test_init_4(o_continuous_e_4x3):
    """Test initialization."""
    desired_name = 'continuous_e'
    desired_n_sequence = 4
    desired_sequence_length = 3
    desired_n_unit = 1
    desired_value = np.array(
        [
            [[0.0], [0.0], [0.0]],
            [[2.0], [0.0], [0.0]],
            [[-0.1], [-1.0], [0.3]],
            [[1.0], [1.0], [1.0]],
        ], dtype=np.float32
    )
    desired_sample_weight = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
            [1.0, 1.0, 0.0],
        ], dtype=np.float32
    )

    assert desired_name == o_continuous_e_4x3.name
    assert desired_n_sequence == o_continuous_e_4x3.n_sample
    assert desired_sequence_length == o_continuous_e_4x3.sequence_length
    assert desired_n_unit == o_continuous_e_4x3.n_unit
    np.testing.assert_array_equal(
        desired_value, o_continuous_e_4x3.value
    )
    np.testing.assert_array_equal(
        desired_sample_weight, o_continuous_e_4x3.sample_weight
    )


def test_invalid_init_0():
    """Test invalid initialization.

    * An invalid rank-4 argument.

    """
    outcome = np.array(
        [
            [
                [[0.0, 0.1], [0.0, 0.2], [0.0, 0.3]],
                [[2.0, 0.4], [0.0, 0.5], [0.0, 0.6]],
            ],
            [
                [[-0.1, 0.7], [-1.0, 0.8], [0.3, 0.9]],
                [[1.0, 1.1], [1.0, 1.2], [1.0, 1.3]],
            ]
        ], dtype=np.float32
    )
    with pytest.raises(Exception) as e_info:
        Continuous(outcome, name='bad')
    assert e_info.type == ValueError
    assert str(e_info.value) == (
        "The argument `value` must be a rank-2 or rank-3 ndarray with a shape "
        "corresponding to (samples, [sequence_length,] n_unit)."
    )


def test_export_0(o_continuous_a_4x1):
    desired_y = tf.constant(
        [[0.0], [2.0], [-.1], [1.3]], dtype=tf.float32
    )
    desired_w = tf.constant(
        [1.0, 1.0, 1.0, 1.0]
    )
    desired_name = 'continuous_a'

    y, w = o_continuous_a_4x1.export(export_format='tfds')
    tf.debugging.assert_equal(desired_y, y[desired_name])
    tf.debugging.assert_equal(desired_w, w[desired_name])


def test_export_1(o_continuous_aa_4x1):
    desired_y = tf.constant(
            [[[0.0]], [[2.0]], [[-0.1]], [[1.3]]], dtype=tf.float32
    )
    desired_w = tf.constant(
        [[1.0], [1.0], [1.0], [1.0]]
    )
    desired_name = 'continuous_aa'

    y, w = o_continuous_aa_4x1.export(export_format='tfds')
    tf.debugging.assert_equal(desired_y, y[desired_name])
    tf.debugging.assert_equal(desired_w, w[desired_name])


def test_export_2a(o_continuous_b_4x3):
    desired_y = tf.constant(
        [
            [[0.0], [0.0], [0.0]],
            [[2.0], [0.0], [0.0]],
            [[-0.1], [-1.0], [0.3]],
            [[1.0], [1.0], [1.0]],
        ], dtype=tf.float32
    )
    desired_w = tf.constant(
        [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ], dtype=tf.float32
    )
    desired_name = 'continuous_b'

    y, w = o_continuous_b_4x3.export(export_format='tfds')
    tf.debugging.assert_equal(desired_y, y[desired_name])
    tf.debugging.assert_equal(desired_w, w[desired_name])


def test_export_2b(o_continuous_b_4x3):
    """Test for_dataset

    Use override `with_timestep_axis=False`

    """
    desired_y = tf.constant(
        [
            [0.0], [0.0], [0.0], [2.0], [0.0], [0.0], [-0.1], [-1.0],
            [0.3], [1.0], [1.0], [1.0]
        ], dtype=tf.float32
    )
    desired_w = tf.ones([12], dtype=tf.float32)
    desired_name = 'continuous_b'

    y, w = o_continuous_b_4x3.export(
        export_format='tfds', with_timestep_axis=False
    )
    tf.debugging.assert_equal(desired_y, y[desired_name])
    tf.debugging.assert_equal(desired_w, w[desired_name])


def test_export_3a(o_continuous_c_4x3):
    desired_y = tf.constant(
        [
            [[0.0, 0.1], [0.0, 0.2], [0.0, 0.3]],
            [[2.0, 0.4], [0.0, 0.5], [0.0, 0.6]],
            [[-0.1, 0.7], [-1.0, 0.8], [0.3, 0.9]],
            [[1.0, 1.1], [1.0, 1.2], [1.0, 1.3]],
        ], dtype=tf.float32
    )
    desired_w = tf.ones([4, 3], dtype=tf.float32)
    desired_name = 'continuous_c'

    y, w = o_continuous_c_4x3.export(export_format='tfds')
    tf.debugging.assert_equal(desired_y, y[desired_name])
    tf.debugging.assert_equal(desired_w, w[desired_name])


def test_export_3b(o_continuous_c_4x3):
    """Test for_dataset

    Use override `with_timestep_axis=False`

    """
    desired_y = tf.constant(
        [
            [0.0, 0.1], [0.0, 0.2], [0.0, 0.3],
            [2.0, 0.4], [0.0, 0.5], [0.0, 0.6],
            [-0.1, 0.7], [-1.0, 0.8], [0.3, 0.9],
            [1.0, 1.1], [1.0, 1.2], [1.0, 1.3],
        ], dtype=tf.float32
    )
    desired_w = tf.ones([12], dtype=tf.float32)
    desired_name = 'continuous_c'

    y, w = o_continuous_c_4x3.export(
        export_format='tfds', with_timestep_axis=False
    )
    tf.debugging.assert_equal(desired_y, y[desired_name])
    tf.debugging.assert_equal(desired_w, w[desired_name])


def test_export_4a(o_continuous_e_4x3):
    desired_y = tf.constant(
        [
            [[0.0], [0.0], [0.0]],
            [[2.0], [0.0], [0.0]],
            [[-0.1], [-1.0], [0.3]],
            [[1.0], [1.0], [1.0]],
        ], dtype=tf.float32
    )
    desired_w = tf.constant([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
        [1.0, 1.0, 0.0],
    ], dtype=tf.float32)
    desired_name = 'continuous_e'

    y, w = o_continuous_e_4x3.export(export_format='tfds')
    tf.debugging.assert_equal(desired_y, y[desired_name])
    tf.debugging.assert_equal(desired_w, w[desired_name])


def test_export_wrong(o_continuous_c_4x3):
    """Test export.

    Using incorrect `export_format`.

    """
    with pytest.raises(Exception) as e_info:
        o_continuous_c_4x3.export(export_format='garbage')
    assert e_info.type == ValueError
    assert (
        str(e_info.value) == "Unrecognized `export_format` 'garbage'."
    )
