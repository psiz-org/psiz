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


def test_init_0(c_categorize_a_4x10):
    """Test initialization with minimal rank arguments."""
    desired_n_sequence = 4
    desired_sequence_length = 10
    desired_stimulus_set = np.array(
        [
            [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]],
            [[11], [12], [13], [14], [15], [16], [17], [18], [19], [20]],
            [[1], [3], [5], [7], [9], [11], [13], [15], [17], [19]],
            [[2], [4], [6], [8], [10], [12], [14], [16], [0], [0]],
        ], dtype=np.int32
    )
    desired_objective_query_label = np.array(
        [
            [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0]],
            [[1], [1], [1], [1], [1], [2], [2], [2], [2], [2]],
            [[0], [0], [0], [0], [0], [1], [1], [1], [2], [2]],
            [[0], [0], [0], [0], [0], [1], [1], [2], [0], [0]],
        ], dtype=np.int32
    )
    desired_objective_query_label = tf.keras.utils.to_categorical(
        desired_objective_query_label, num_classes=3
    )
    desired_is_actual = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        ], dtype=bool
    )

    assert c_categorize_a_4x10.n_sample == desired_n_sequence
    assert c_categorize_a_4x10.sequence_length == desired_sequence_length
    assert c_categorize_a_4x10.mask_zero
    np.testing.assert_array_equal(
        desired_stimulus_set, c_categorize_a_4x10.stimulus_set
    )
    np.testing.assert_array_equal(
        desired_objective_query_label,
        c_categorize_a_4x10.objective_query_label
    )
    np.testing.assert_array_equal(
        desired_is_actual, c_categorize_a_4x10.is_actual
    )


def test_export_0(c_categorize_b_4x3):
    """Test export."""
    x = c_categorize_b_4x3.export()
    desired_stimulus_set = tf.constant(
        [
            [[1], [2], [3]],
            [[11], [12], [13]],
            [[1], [3], [5]],
            [[2], [4], [6]],
        ], dtype=tf.int32
    )
    desired_objective_query_label = tf.constant(
        [
            [[0], [0], [0]],
            [[1], [1], [2]],
            [[0], [1], [2]],
            [[2], [2], [0]],
        ], dtype=tf.int32
    )
    desired_objective_query_label = tf.keras.utils.to_categorical(
        desired_objective_query_label, num_classes=3
    )
    tf.debugging.assert_equal(
        desired_stimulus_set, x['categorize_stimulus_set']
    )
    tf.debugging.assert_equal(
        desired_objective_query_label, x['categorize_objective_query_label']
    )


def test_export_1(c_categorize_b_4x3):
    """Test export.

    Use override `with_timestep_axis=False`.

    """
    x = c_categorize_b_4x3.export(with_timestep_axis=False)
    desired_stimulus_set = tf.constant(
        [
            [1], [2], [3], [11], [12], [13], [1], [3], [5], [2], [4], [6]
        ], dtype=tf.int32
    )
    desired_objective_query_label = tf.constant(
        [
            [0], [0], [0], [1], [1], [2], [0], [1], [2], [2], [2], [0]
        ], dtype=tf.int32
    )
    desired_objective_query_label = tf.keras.utils.to_categorical(
        desired_objective_query_label, num_classes=3
    )
    tf.debugging.assert_equal(
        desired_stimulus_set, x['categorize_stimulus_set']
    )
    tf.debugging.assert_equal(
        desired_objective_query_label, x['categorize_objective_query_label']
    )


def test_export_wrong(c_categorize_b_4x3):
    """Test export.

    Using incorrect `export_format`.

    """
    with pytest.raises(Exception) as e_info:
        c_categorize_b_4x3.export(export_format='garbage')
    assert e_info.type == ValueError
    assert (
        str(e_info.value) == "Unrecognized `export_format` 'garbage'."
    )
