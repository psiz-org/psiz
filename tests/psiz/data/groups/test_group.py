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

import pytest

import numpy as np
import tensorflow as tf


def test_init_0(g_condition_idx_4x1):
    """Test initalization."""
    desired_value = np.array(
        [
            [[0]],
            [[1]],
            [[0]],
            [[0]],
        ]
    )
    assert g_condition_idx_4x1.name == 'condition_idx'
    assert g_condition_idx_4x1.n_sample == 4
    assert g_condition_idx_4x1.sequence_length == 1
    np.testing.assert_array_equal(
        g_condition_idx_4x1.value, desired_value
    )


def test_init_1(g_condition_idx_4x3):
    """Test initalization."""
    desired_value = np.array(
        [
            [[0], [0], [0]],
            [[1], [1], [1]],
            [[0], [0], [0]],
            [[0], [0], [0]],
        ]
    )
    assert g_condition_idx_4x3.name == 'condition_idx'
    assert g_condition_idx_4x3.n_sample == 4
    assert g_condition_idx_4x3.sequence_length == 3
    np.testing.assert_array_equal(
        g_condition_idx_4x3.value, desired_value
    )


def test_init_2(g_mix2_4x3):
    """Test initalization."""
    desired_value = np.array(
       [
            [[0.5, 0.5], [0.6, 0.4], [0.7, 0.3]],
            [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            [[0.8, 0.2], [0.8, 0.2], [0.8, 0.2]],
            [[0.2, 0.8], [0.2, 0.8], [0.2, 0.8]],
        ], dtype=np.float32
    )
    assert g_mix2_4x3.name == 'mix2'
    assert g_mix2_4x3.n_sample == 4
    assert g_mix2_4x3.sequence_length == 3
    np.testing.assert_array_equal(
        g_mix2_4x3.value, desired_value
    )


def test_init_3(g_condition_label_4x1):
    """Test initalization."""
    desired_value = np.array(
        [
            [['block']],
            [['interleave']],
            [['block']],
            [['block']],
        ]
    )
    assert g_condition_label_4x1.name == 'condition_label'
    assert g_condition_label_4x1.n_sample == 4
    assert g_condition_label_4x1.sequence_length == 1
    np.testing.assert_array_equal(
        g_condition_label_4x1.value, desired_value
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

    x = g_mix2_4x3.export(export_format='tfds')
    tf.debugging.assert_equal(desired_group_weight, x[desired_name])


def test_export_0b(g_mix2_4x3):
    """Test export.

    Use override `with_timestep_axis=False`.

    """
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
        export_format='tfds', with_timestep_axis=False
    )
    tf.debugging.assert_equal(desired_group_weight, x[desired_name])


def test_export_3a(g_condition_label_4x1):
    """Test export."""
    desired_value = tf.constant(
        [
            [['block']],
            [['interleave']],
            [['block']],
            [['block']],
        ]
    )
    desired_name = 'condition_label'

    x = g_condition_label_4x1.export(export_format='tfds')
    tf.debugging.assert_equal(desired_value, x[desired_name])


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
