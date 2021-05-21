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

from psiz.trials.experimental.contents.rate_similarity import RateSimilarity
from psiz.trials import stack


def test_init_0(rate_sim_0):
    """Test initialization with minimal rank arguments."""
    desired_n_sequence = 4
    desired_max_timestep = 1
    desired_stimulus_set = np.array([
        [[3, 1]],
        [[9, 12]],
        [[3, 4]],
        [[3, 4]]
    ], dtype=np.int32)
    desired_n_timestep = np.array([1, 1, 1, 1], dtype=np.int32)

    assert rate_sim_0.n_sequence == desired_n_sequence
    assert rate_sim_0.max_timestep == desired_max_timestep
    np.testing.assert_array_equal(
        desired_stimulus_set, rate_sim_0.stimulus_set
    )
    np.testing.assert_array_equal(
        desired_n_timestep, rate_sim_0.n_timestep
    )


def test_init_1(rate_sim_1):
    """Test initialization with true rank arguments."""
    desired_n_sequence = 4
    desired_max_timestep = 1
    desired_stimulus_set = np.array([
        [[3, 1]],
        [[9, 12]],
        [[3, 4]],
        [[3, 4]]
    ], dtype=np.int32)
    desired_n_timestep = np.array([1, 1, 1, 1], dtype=np.int32)

    assert rate_sim_1.n_sequence == desired_n_sequence
    assert rate_sim_1.max_timestep == desired_max_timestep
    np.testing.assert_array_equal(
        desired_stimulus_set, rate_sim_1.stimulus_set
    )
    np.testing.assert_array_equal(
        desired_n_timestep, rate_sim_1.n_timestep
    )


def test_init_2(rate_sim_2):
    """Test initialization with true rank arguments."""
    desired_n_sequence = 4
    desired_max_timestep = 2
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
    desired_n_timestep = np.array([2, 1, 2, 2], dtype=np.int32)

    assert rate_sim_2.n_sequence == desired_n_sequence
    assert rate_sim_2.max_timestep == desired_max_timestep
    np.testing.assert_array_equal(
        desired_stimulus_set, rate_sim_2.stimulus_set
    )
    np.testing.assert_array_equal(
        desired_n_timestep, rate_sim_2.n_timestep
    )


def test_init_3(rate_sim_3):
    """Test initialization with true rank arguments."""
    desired_n_sequence = 4
    desired_max_timestep = 2
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
    desired_n_timestep = np.array([2, 1, 2, 2], dtype=np.int32)

    assert rate_sim_3.n_sequence == desired_n_sequence
    assert rate_sim_3.max_timestep == desired_max_timestep
    np.testing.assert_array_equal(
        desired_stimulus_set, rate_sim_3.stimulus_set
    )
    np.testing.assert_array_equal(
        desired_n_timestep, rate_sim_3.n_timestep
    )


def test_invalid_stimulus_set():
    """Test handling of invalid `stimulus_set` argument."""
    # Non-integer input.
    stimulus_set = np.array((
        (3., 1),
        (9, 12),
        (3, 4),
        (3, 4)
    ))
    with pytest.raises(Exception) as e_info:
        obs = RateSimilarity(stimulus_set)

    # Contains negative integers.
    stimulus_set = np.array((
        (3, -1),
        (9, 12),
        (3, 4),
        (3, 4)
    ))
    with pytest.raises(Exception) as e_info:
        obs = RateSimilarity(stimulus_set)

    # TODO
    # # Does not contain enough stimuli for each trial.
    # stimulus_set = np.array((
    #     (3, 0),
    #     (9, 12),
    #     (3, 4),
    #     (3, 4)
    # ))
    # with pytest.raises(Exception) as e_info:
    #     obs = RateSimilarity(stimulus_set)


def test_is_actual(rate_sim_2):
    """Test is_actual method."""
    desired_is_actual = np.array(
        [
            [1, 1],
            [1, 0],
            [1, 1],
            [1, 1]
        ], dtype=bool
    )
    np.testing.assert_array_equal(
        desired_is_actual,
        rate_sim_2.is_actual()
    )


def test_for_dataset_0(rate_sim_3):
    """Test _for_dataset."""
    x = rate_sim_3._for_dataset()
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
    tf.debugging.assert_equal(desired_stimulus_set, x['stimulus_set'])


def test_for_dataset_1(rate_sim_3):
    """Test _for_dataset.

    Use timestep=False.

    """
    x = rate_sim_3._for_dataset(timestep=False)
    desired_stimulus_set = np.array(
        [
            [3, 1],
            [3, 1],
            [9, 12],
            [0, 0],
            [3, 4],
            [3, 4],
            [3, 4],
            [3, 4]
        ], dtype=np.int32
    )
    tf.debugging.assert_equal(desired_stimulus_set, x['stimulus_set'])


def test_persistence(rate_sim_3, tmpdir):
    """Test _save and _load."""
    group_name = "content"

    original = rate_sim_3
    fn = tmpdir.join('content_test.hdf5')

    # Save group.
    f = h5py.File(fn, "w")
    grp_stimulus = f.create_group(group_name)
    original._save(grp_stimulus)
    f.close()

    # Load group.
    f = h5py.File(fn, "r")
    grp = f[group_name]
    # Encoding/read rules changed in h5py 3.0, requiring asstr() call.
    try:
        class_name = grp["class_name"].asstr()[()]
    except AttributeError:
        class_name = grp["class_name"][()]
    reconstructed = RateSimilarity._load(grp)
    f.close()

    # Check for equivalency.
    assert class_name == "RateSimilarity"
    assert original.n_sequence == reconstructed.n_sequence
    assert original.max_timestep == reconstructed.max_timestep
    np.testing.assert_array_equal(
        original.stimulus_set, reconstructed.stimulus_set
    )
    np.testing.assert_array_equal(
        original.n_timestep, reconstructed.n_timestep
    )


def test_subset_0(rate_sim_3):
    """Test subset."""
    desired_n_sequence = 2
    desired_max_timestep = 2
    desired_stimulus_set = np.array(
        [
            [
                [9, 12],
                [0, 0],
            ],
            [
                [3, 4],
                [3, 4],
            ]
        ], dtype=np.int32
    )
    desired_n_timestep = np.array([1, 2], dtype=np.int32)

    sub = rate_sim_3.subset(np.array([1, 2]))
    assert sub.n_sequence == desired_n_sequence
    assert sub.max_timestep == desired_max_timestep
    np.testing.assert_array_equal(
        desired_stimulus_set, sub.stimulus_set
    )
    np.testing.assert_array_equal(
        desired_n_timestep, sub.n_timestep
    )


def test_stack_0(rate_sim_3, rate_sim_4):
    """Test stack."""
    desired_n_sequence = 6
    desired_max_timestep = 3
    desired_stimulus_set = stimulus_set = np.array(
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
            ],
            [
                [5, 6],
                [7, 8],
                [9, 10]
            ],
            [
                [1, 2],
                [3, 4],
                [0, 0]
            ],
        ], dtype=np.int32
    )
    stacked = stack((rate_sim_3, rate_sim_4))

    assert desired_n_sequence == stacked.n_sequence
    assert desired_max_timestep == stacked.max_timestep
    np.testing.assert_array_equal(
        desired_stimulus_set, stacked.stimulus_set
    )
