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

from psiz.data.contents.rate import Rate


def test_init_0(c_rate2_a_4x1):
    """Test initialization with minimal rank arguments."""
    desired_n_sequence = 4
    desired_sequence_length = 1
    desired_stimulus_set = np.array([
        [[3, 1]],
        [[9, 12]],
        [[3, 4]],
        [[3, 4]]
    ], dtype=np.int32)

    assert c_rate2_a_4x1.n_sample == desired_n_sequence
    assert c_rate2_a_4x1.sequence_length == desired_sequence_length
    np.testing.assert_array_equal(
        desired_stimulus_set, c_rate2_a_4x1.stimulus_set
    )
    assert c_rate2_a_4x1.mask_zero


def test_init_1(c_rate2_aa_4x1):
    """Test initialization with true rank arguments."""
    desired_n_sequence = 4
    desired_sequence_length = 1
    desired_stimulus_set = np.array([
        [[3, 1]],
        [[9, 12]],
        [[3, 4]],
        [[3, 4]]
    ], dtype=np.int32)

    assert c_rate2_aa_4x1.n_sample == desired_n_sequence
    assert c_rate2_aa_4x1.sequence_length == desired_sequence_length
    np.testing.assert_array_equal(
        desired_stimulus_set, c_rate2_aa_4x1.stimulus_set
    )
    assert c_rate2_aa_4x1.mask_zero


def test_init_2(c_rate2_b_4x2):
    """Test initialization with true rank arguments."""
    desired_n_sequence = 4
    desired_sequence_length = 2
    desired_stimulus_set = np.array(
        [
            [
                [3, 1],
                [3, 1]
            ],
            [
                [9, 12],
                [0, 0]
            ],
            [
                [3, 4],
                [3, 4]
            ],
            [
                [3, 4],
                [3, 4]
            ]
        ], dtype=np.int32
    )

    assert c_rate2_b_4x2.n_sample == desired_n_sequence
    assert c_rate2_b_4x2.sequence_length == desired_sequence_length
    np.testing.assert_array_equal(
        desired_stimulus_set, c_rate2_b_4x2.stimulus_set
    )
    assert c_rate2_b_4x2.mask_zero


def test_init_3(c_rate2_c_4x3):
    """Test initialization.

    Do not auto-trim empty sequence.

    """
    desired_n_sequence = 4
    desired_sequence_length = 3
    desired_stimulus_set = np.array(
        [
            [
                [3, 1],
                [3, 1],
                [0, 0]
            ],
            [
                [9, 12],
                [0, 0],
                [0, 0]
            ],
            [
                [3, 4],
                [3, 4],
                [0, 0]
            ],
            [
                [3, 4],
                [3, 4],
                [0, 0]
            ]
        ], dtype=np.int32
    )

    assert c_rate2_c_4x3.n_sample == desired_n_sequence
    assert c_rate2_c_4x3.sequence_length == desired_sequence_length
    np.testing.assert_array_equal(
        desired_stimulus_set, c_rate2_c_4x3.stimulus_set
    )
    assert c_rate2_c_4x3.mask_zero


def test_init_4():
    """Test initialization.

    Do not raise error even though some trials do not have stimuli.

    """
    stimulus_set = np.array([
        [
            [3, 1],
            [9, 12]
        ],
        [
            [0, 0],
            [0, 0]
        ],
        [
            [3, 4],
            [2, 4]
        ],
        [
            [3, 4],
            [2, 4]
        ]
    ])
    Rate(stimulus_set)


def test_invalid_stimulus_set():
    """Test handling of invalid `stimulus_set` argument."""
    # Non-integer input.
    stimulus_set = np.array((
        (3.1, 1),
        (9, 12),
        (3, 4),
        (3, 4)
    ))
    with pytest.raises(Exception) as e_info:
        Rate(stimulus_set)
    assert e_info.type == ValueError

    # Contains negative integers.
    stimulus_set = np.array((
        (3, -1),
        (9, 12),
        (3, 4),
        (3, 4)
    ))
    with pytest.raises(Exception) as e_info:
        Rate(stimulus_set)
    assert e_info.type == ValueError

    # Incorrect shape.
    stimulus_set = np.array([
        [
            [
                [3, 1],
                [9, 12]
            ],
            [
                [3, 1],
                [9, 12]
            ],
        ],
        [
            [
                [3, 4],
                [2, 4]
            ],
            [
                [3, 4],
                [2, 4]
            ]
        ]
    ])
    with pytest.raises(Exception) as e_info:
        Rate(stimulus_set)
    assert e_info.type == ValueError

    # TODO enforce or delete
    # Integer is too large.
    # ii32 = np.iinfo(np.int32)
    # too_large = ii32.max + 1
    # stimulus_set = np.array((
    #     (3, too_large),
    #     (9, 12),
    #     (3, 4),
    #     (3, 4)
    # ))
    # with pytest.raises(Exception) as e_info:
    #     Rate(stimulus_set)
    # assert e_info.type == ValueError


def test_is_actual(c_rate2_b_4x2):
    """Test is_actual method."""
    desired_is_actual = np.array(
        [
            [1, 1],
            [1, 0],
            [1, 1],
            [1, 1]
        ], dtype=bool
    )
    np.testing.assert_array_equal(desired_is_actual, c_rate2_b_4x2.is_actual)


def test_unique_configurations(c_rate2_b_4x2):
    """Test unique configurations."""
    config_idx, df_config = c_rate2_b_4x2.unique_configurations
    config_idx_desired = np.array(
        [
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
        ], dtype=np.int32
    )
    np.testing.assert_array_equal(
        config_idx, config_idx_desired
    )
    assert df_config is None


def test_export_0(c_rate2_c_4x3):
    """Test export."""
    x = c_rate2_c_4x3.export()
    desired_stimulus_set = np.array(
        [
            [
                [3, 1],
                [3, 1],
                [0, 0]
            ],
            [
                [9, 12],
                [0, 0],
                [0, 0]
            ],
            [
                [3, 4],
                [3, 4],
                [0, 0]
            ],
            [
                [3, 4],
                [3, 4],
                [0, 0]
            ]
        ], dtype=np.int32
    )
    tf.debugging.assert_equal(
        desired_stimulus_set, x['rate2_stimulus_set']
    )


def test_export_1(c_rate2_c_4x3):
    """Test export.

    Use override `with_timestep_axis=False`.

    """
    x = c_rate2_c_4x3.export(with_timestep_axis=False)
    desired_stimulus_set = np.array(
        [
            [3, 1],
            [3, 1],
            [0, 0],
            [9, 12],
            [0, 0],
            [0, 0],
            [3, 4],
            [3, 4],
            [0, 0],
            [3, 4],
            [3, 4],
            [0, 0],
        ], dtype=np.int32
    )
    tf.debugging.assert_equal(
        desired_stimulus_set, x['rate2_stimulus_set']
    )


def test_export_wrong(c_rate2_c_4x3):
    """Test export.

    Using incorrect `export_format`.

    """
    with pytest.raises(Exception) as e_info:
        c_rate2_c_4x3.export(export_format='garbage')
    assert e_info.type == ValueError
    assert (
        str(e_info.value) == "Unrecognized `export_format` 'garbage'."
    )
