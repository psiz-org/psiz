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

from psiz.data.outcomes.continuous import Continuous
from psiz.trials import stack


def test_init_0(continuous_0):
    """Test initialization."""
    desired_n_sequence = 4
    desired_max_timestep = 1
    desired_n_unit = 1
    desired_value = np.array(
        [[[0.0]], [[2.0]], [[-0.1]], [[1.3]]], dtype=np.float32
    )

    assert desired_n_sequence == continuous_0.n_sequence
    assert desired_max_timestep == continuous_0.sequence_length
    assert desired_n_unit == continuous_0.n_unit
    np.testing.assert_array_equal(
        desired_value, continuous_0.value
    )


def test_init_1(continuous_1):
    """Test initialization."""
    desired_n_sequence = 4
    desired_max_timestep = 1
    desired_n_unit = 1
    desired_value = np.array(
        [[[0.0]], [[2.0]], [[-0.1]], [[1.3]]], dtype=np.float32
    )

    assert desired_n_sequence == continuous_1.n_sequence
    assert desired_max_timestep == continuous_1.sequence_length
    assert desired_n_unit == continuous_1.n_unit
    np.testing.assert_array_equal(
        desired_value, continuous_1.value
    )


def test_init_2(continuous_2):
    """Test initialization."""
    desired_n_sequence = 4
    desired_max_timestep = 3
    desired_n_unit = 1
    desired_value = np.array(
        [
            [[0.0], [0.0], [0.0]],
            [[2.0], [0.0], [0.0]],
            [[-0.1], [-1.0], [0.3]],
            [[1.0], [1.0], [1.0]],
        ], dtype=np.float32
    )

    assert desired_n_sequence == continuous_2.n_sequence
    assert desired_max_timestep == continuous_2.sequence_length
    assert desired_n_unit == continuous_2.n_unit
    np.testing.assert_array_equal(
        desired_value, continuous_2.value
    )


def test_init_3(continuous_3):
    """Test initialization."""
    desired_n_sequence = 4
    desired_max_timestep = 3
    desired_n_unit = 2
    desired_value = np.array(
        [
            [[0.0, 0.1], [0.0, 0.2], [0.0, 0.3]],
            [[2.0, 0.4], [0.0, 0.5], [0.0, 0.6]],
            [[-0.1, 0.7], [-1.0, 0.8], [0.3, 0.9]],
            [[1.0, 1.1], [1.0, 1.2], [1.0, 1.3]],
        ], dtype=np.float32
    )

    assert desired_n_sequence == continuous_3.n_sequence
    assert desired_max_timestep == continuous_3.sequence_length
    assert desired_n_unit == continuous_3.n_unit
    np.testing.assert_array_equal(
        desired_value, continuous_3.value
    )


def test_init_wrong():
    """Test initialization."""
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
        Continuous(outcome)
    assert e_info.type == ValueError


def test_export_0(continuous_0):
    desired_y = tf.constant(
        np.array(
            [[[0.0]], [[2.0]], [[-.1]], [[1.3]]],
            dtype=np.float32
        )
    )

    tf.debugging.assert_equal(desired_y, continuous_0.export())


def test_export_1(continuous_1):
    desired_y = tf.constant(
        np.array(
            [[[0.0]], [[2.0]], [[-0.1]], [[1.3]]], dtype=np.float32
        )
    )

    tf.debugging.assert_equal(desired_y, continuous_1.export())


def test_export_2a(continuous_2):
    desired_y = tf.constant(
        np.array(
            [
                [[0.0], [0.0], [0.0]],
                [[2.0], [0.0], [0.0]],
                [[-0.1], [-1.0], [0.3]],
                [[1.0], [1.0], [1.0]],
            ], dtype=np.float32
        )
    )

    tf.debugging.assert_equal(desired_y, continuous_2.export())


def test_export_2b(continuous_2):
    """Test for_dataset

    Use with_timestep_axis=False

    """
    desired_y = tf.constant(
        np.array(
            [
                [0.0], [0.0], [0.0], [2.0], [0.0], [0.0], [-0.1], [-1.0],
                [0.3], [1.0], [1.0], [1.0]
            ], dtype=np.float32
        )
    )

    y = continuous_2.export(with_timestep_axis=False)
    tf.debugging.assert_equal(desired_y, y)


def test_export_3a(continuous_3):
    desired_y = tf.constant(
        np.array(
            [
                [[0.0, 0.1], [0.0, 0.2], [0.0, 0.3]],
                [[2.0, 0.4], [0.0, 0.5], [0.0, 0.6]],
                [[-0.1, 0.7], [-1.0, 0.8], [0.3, 0.9]],
                [[1.0, 1.1], [1.0, 1.2], [1.0, 1.3]],
            ], dtype=np.float32
        )
    )

    tf.debugging.assert_equal(desired_y, continuous_3.export())


def test_export_3b(continuous_3):
    """Test for_dataset

    Use with_timestep_axis=False

    """
    desired_y = tf.constant(
        np.array(
            [
                [0.0, 0.1], [0.0, 0.2], [0.0, 0.3],
                [2.0, 0.4], [0.0, 0.5], [0.0, 0.6],
                [-0.1, 0.7], [-1.0, 0.8], [0.3, 0.9],
                [1.0, 1.1], [1.0, 1.2], [1.0, 1.3],
            ], dtype=np.float32
        )
    )

    y = continuous_3.export(with_timestep_axis=False)
    tf.debugging.assert_equal(desired_y, y)


def test_export_wrong(continuous_3):
    """Test export.

    Using incorrect `export_format`.

    """
    with pytest.raises(Exception) as e_info:
        continuous_3.export(export_format='garbage')
    assert e_info.type == ValueError
    assert (
        str(e_info.value) == "Unrecognized `export_format` 'garbage'."
    )


def test_persistence(continuous_2, tmpdir):
    """Test save and load."""
    group_name = "value"

    original = continuous_2
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
    reconstructed = Continuous.load(h5_grp)
    f.close()

    # Check for equivalency.
    assert class_name == "Continuous"
    assert original.n_sequence == reconstructed.n_sequence
    assert original.sequence_length == reconstructed.sequence_length
    np.testing.assert_array_equal(
        original.value, reconstructed.value
    )


def test_subset_3(continuous_3):
    """Test subset."""
    desired_n_sequence = 2
    desired_max_timestep = 3
    desired_value = np.array(
        [
            [[2.0, 0.4], [0.0, 0.5], [0.0, 0.6]],
            [[-0.1, 0.7], [-1.0, 0.8], [0.3, 0.9]],
        ], dtype=np.float32
    )
    desired_n_unit = 2

    sub = continuous_3.subset(np.array([1, 2]))

    assert desired_n_sequence == sub.n_sequence
    assert desired_max_timestep == sub.sequence_length
    assert desired_n_unit == sub.n_unit
    np.testing.assert_array_equal(
        desired_value, sub.value
    )


def test_stack_0(continuous_3, continuous_4):
    """Test stack."""
    desired_n_sequence = 6
    desired_max_timestep = 3
    desired_value = np.array(
        [
            [[0.0, 0.1], [0.0, 0.2], [0.0, 0.3]],
            [[2.0, 0.4], [0.0, 0.5], [0.0, 0.6]],
            [[-0.1, 0.7], [-1.0, 0.8], [0.3, 0.9]],
            [[1.0, 1.1], [1.0, 1.2], [1.0, 1.3]],
            [[2.0, 2.1], [2.0, 2.2], [2.0, 2.3]],
            [[3.0, 3.4], [3.0, 3.5], [3.0, 3.6]],
        ], dtype=np.float32
    )
    desired_n_unit = 2

    stacked = stack((continuous_3, continuous_4))

    assert desired_n_sequence == stacked.n_sequence
    assert desired_max_timestep == stacked.sequence_length
    np.testing.assert_array_equal(
        desired_value, stacked.value
    )
    assert desired_n_unit == stacked.n_unit


def test_stack_1(continuous_2, continuous_3):
    """Test stack.

    Incompatible `n_unit`.

    """
    with pytest.raises(Exception) as e_info:
        stack((continuous_2, continuous_3))
    assert e_info.type == ValueError
