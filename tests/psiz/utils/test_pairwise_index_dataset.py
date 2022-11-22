# -*- coding: utf-8 -*-
# Copyright 2021 The PsiZ Authors. All Rights Reserved.
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
"""Module for testing utils.py."""

import numpy as np
import pytest

from psiz.utils import pairwise_index_dataset


def test_scalar_defaults():
    """Test default optional arguments.

    Scalar argument.

    """
    n_stimuli = 5
    ds_pairs, ds_info = pairwise_index_dataset(n_stimuli, batch_size=10)
    pairs = list(ds_pairs.as_numpy_iterator())
    pairs_0 = pairs[0][0]
    pairs_1 = pairs[0][1]

    pairs_0_desired = np.array(
        [0, 0, 0, 0, 1, 1, 1, 2, 2, 3], dtype=np.int32
    )
    pairs_1_desired = np.array(
        [1, 2, 3, 4, 2, 3, 4, 3, 4, 4], dtype=np.int32
    )
    np.testing.assert_array_equal(pairs_0, pairs_0_desired)
    np.testing.assert_array_equal(pairs_1, pairs_1_desired)

    assert ds_info['n_pair'] == 10
    assert ds_info['batch_size'] == 10
    assert ds_info['n_batch'] == 1.0
    assert ds_info['elements'] == 'upper'


def test_array_defaults():
    """Test default optional arguments.

    Array argument.

    """
    n_stimuli = 5
    ds_pairs, ds_info = pairwise_index_dataset(
        np.arange(n_stimuli), batch_size=10
    )
    pairs = list(ds_pairs.as_numpy_iterator())
    pairs_0 = pairs[0][0]
    pairs_1 = pairs[0][1]

    pairs_0_desired = np.array(
        [0, 0, 0, 0, 1, 1, 1, 2, 2, 3], dtype=np.int32
    )
    pairs_1_desired = np.array(
        [1, 2, 3, 4, 2, 3, 4, 3, 4, 4], dtype=np.int32
    )
    np.testing.assert_array_equal(pairs_0, pairs_0_desired)
    np.testing.assert_array_equal(pairs_1, pairs_1_desired)

    assert ds_info['n_pair'] == 10
    assert ds_info['batch_size'] == 10
    assert ds_info['n_batch'] == 1.0
    assert ds_info['elements'] == 'upper'


def test_list_defaults():
    """Test default optional arguments.

    List argument.

    """
    indices = [0, 1, 2, 3, 4]
    ds_pairs, ds_info = pairwise_index_dataset(indices, batch_size=10)
    pairs = list(ds_pairs.as_numpy_iterator())
    pairs_0 = pairs[0][0]
    pairs_1 = pairs[0][1]

    pairs_0_desired = np.array(
        [0, 0, 0, 0, 1, 1, 1, 2, 2, 3], dtype=np.int32
    )
    pairs_1_desired = np.array(
        [1, 2, 3, 4, 2, 3, 4, 3, 4, 4], dtype=np.int32
    )
    np.testing.assert_array_equal(pairs_0, pairs_0_desired)
    np.testing.assert_array_equal(pairs_1, pairs_1_desired)

    assert ds_info['n_pair'] == 10
    assert ds_info['batch_size'] == 10
    assert ds_info['n_batch'] == 1.0
    assert ds_info['elements'] == 'upper'


def test_all():
    """Test default optional arguments."""
    n_stimuli = 5
    ds_pairs, ds_info = pairwise_index_dataset(
        n_stimuli, elements='all', batch_size=25
    )
    pairs = list(ds_pairs.as_numpy_iterator())
    pairs_0 = pairs[0][0]
    pairs_1 = pairs[0][1]

    pairs_0_desired = np.array(
        [
            0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1,
            2, 3, 4
        ], dtype=np.int32
    )
    pairs_1_desired = np.array(
        [
            0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4,
            4, 4, 4
        ], dtype=np.int32
    )
    np.testing.assert_array_equal(pairs_0, pairs_0_desired)
    np.testing.assert_array_equal(pairs_1, pairs_1_desired)

    assert ds_info['n_pair'] == 25
    assert ds_info['batch_size'] == 25
    assert ds_info['n_batch'] == 1.0
    assert ds_info['elements'] == 'all'


def test_lower():
    """Test default optional arguments."""
    n_stimuli = 5
    ds_pairs, ds_info = pairwise_index_dataset(
        n_stimuli, elements='lower', batch_size=10
    )
    pairs = list(ds_pairs.as_numpy_iterator())
    pairs_0 = pairs[0][0]
    pairs_1 = pairs[0][1]

    pairs_0_desired = np.array(
        [1, 2, 2, 3, 3, 3, 4, 4, 4, 4], dtype=np.int32
    )
    pairs_1_desired = np.array(
        [0, 0, 1, 0, 1, 2, 0, 1, 2, 3], dtype=np.int32
    )
    np.testing.assert_array_equal(pairs_0, pairs_0_desired)
    np.testing.assert_array_equal(pairs_1, pairs_1_desired)

    assert ds_info['n_pair'] == 10
    assert ds_info['batch_size'] == 10
    assert ds_info['n_batch'] == 1.0
    assert ds_info['elements'] == 'lower'


def test_off():
    """Test default optional arguments."""
    n_stimuli = 5
    ds_pairs, ds_info = pairwise_index_dataset(
        n_stimuli, elements='off', batch_size=20
    )
    pairs = list(ds_pairs.as_numpy_iterator())
    pairs_0 = pairs[0][0]
    pairs_1 = pairs[0][1]

    pairs_0_desired = np.array(
        [0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4],
        dtype=np.int32
    )
    pairs_1_desired = np.array(
        [1, 2, 3, 4, 2, 3, 4, 3, 4, 4, 0, 0, 1, 0, 1, 2, 0, 1, 2, 3],
        dtype=np.int32
    )
    np.testing.assert_array_equal(pairs_0, pairs_0_desired)
    np.testing.assert_array_equal(pairs_1, pairs_1_desired)

    assert ds_info['n_pair'] == 20
    assert ds_info['batch_size'] == 20
    assert ds_info['n_batch'] == 1.0
    assert ds_info['elements'] == 'off'


def test_wrong():
    """Test default optional arguments."""
    n_stimuli = 5

    # Raise error because elements argument is not implemented.
    with pytest.raises(Exception) as e_info:
        _, _ = pairwise_index_dataset(n_stimuli, elements='garbage')
    assert e_info.type == NotImplementedError

    # Raise error because indices argument must be scalar or 1D.
    with pytest.raises(Exception) as e_info:
        indices = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        _, _ = pairwise_index_dataset(indices)
    assert e_info.type == ValueError


def test_subsample_0():
    """Test subsample."""
    n_stimuli = 10
    ds_pairs, ds_info = pairwise_index_dataset(
        n_stimuli, elements='all', subsample=.1, batch_size=10
    )
    pairs = list(ds_pairs.as_numpy_iterator())
    pairs_0 = pairs[0][0]
    pairs_1 = pairs[0][1]

    pairs_0_desired = np.array(
        [2, 8, 6, 4, 7, 3, 0, 9, 2, 5], dtype=np.int32
    )
    pairs_1_desired = np.array(
        [5, 7, 2, 2, 5, 9, 9, 6, 6, 7], dtype=np.int32
    )
    np.testing.assert_array_equal(pairs_0, pairs_0_desired)
    np.testing.assert_array_equal(pairs_1, pairs_1_desired)

    assert ds_info['n_pair'] == 10
    assert ds_info['batch_size'] == 10
    assert ds_info['n_batch'] == 1.0
    assert ds_info['elements'] == 'all'


def test_subsample_1():
    """Test subsample."""
    n_stimuli = 5
    # Test correct, but edge case.
    ds_pairs, ds_info = pairwise_index_dataset(
        n_stimuli, elements='upper', subsample=1, batch_size=10
    )
    pairs = list(ds_pairs.as_numpy_iterator())
    pairs_0 = pairs[0][0]
    pairs_1 = pairs[0][1]

    # Due to subsample, order is permuted.
    pairs_0_desired = np.array(
        [1, 0, 0, 0, 0, 3, 2, 2, 1, 1], dtype=np.int32
    )
    pairs_1_desired = np.array(
        [2, 1, 3, 4, 2, 4, 3, 4, 3, 4], dtype=np.int32
    )
    np.testing.assert_array_equal(pairs_0, pairs_0_desired)
    np.testing.assert_array_equal(pairs_1, pairs_1_desired)

    assert ds_info['n_pair'] == 10
    assert ds_info['batch_size'] == 10
    assert ds_info['n_batch'] == 1.0
    assert ds_info['elements'] == 'upper'


def test_subsample_wrong():
    """Test subsample with invalid arguments."""
    n_stimuli = 5

    # Raise error because subsample is not in ]0,1].
    with pytest.raises(Exception) as e_info:
        ds_pairs, ds_info = pairwise_index_dataset(
            n_stimuli, elements='all', subsample=0
        )
    assert e_info.type == ValueError

    # Raise error because subsample is not in ]0,1].
    with pytest.raises(Exception) as e_info:
        ds_pairs, ds_info = pairwise_index_dataset(
            n_stimuli, elements='all', subsample=5
        )
    assert e_info.type == ValueError


def test_batch_size():
    """Test non-default index and batch_size arguments."""
    n_stimuli = 5
    ds_pairs, ds_info = pairwise_index_dataset(
        np.arange(n_stimuli) + 1, elements='all', batch_size=5
    )
    pairs = list(ds_pairs.as_numpy_iterator())

    # Check that there are 5 batches.
    assert len(pairs) == 5

    # Check values of first batch.
    pairs_0 = pairs[0][0]
    pairs_1 = pairs[0][1]

    pairs_0_desired = np.array(
        [1, 2, 3, 4, 5], dtype=np.int32
    )
    pairs_1_desired = np.array(
        [1, 1, 1, 1, 1], dtype=np.int32
    )
    np.testing.assert_array_equal(pairs_0, pairs_0_desired)
    np.testing.assert_array_equal(pairs_1, pairs_1_desired)

    assert ds_info['n_pair'] == 25
    assert ds_info['batch_size'] == 5
    assert ds_info['n_batch'] == 5.0
    assert ds_info['elements'] == 'all'


def test_groups():
    """Test non-default group arguments."""
    n_stimuli = 5
    ds_pairs, ds_info = pairwise_index_dataset(
        n_stimuli, groups=[0], batch_size=10
    )
    pairs = list(ds_pairs.as_numpy_iterator())
    pairs_0 = pairs[0][0]
    pairs_1 = pairs[0][1]
    groups = pairs[0][2]

    pairs_0_desired = np.array(
        [0, 0, 0, 0, 1, 1, 1, 2, 2, 3], dtype=np.int32
    )
    pairs_1_desired = np.array(
        [1, 2, 3, 4, 2, 3, 4, 3, 4, 4], dtype=np.int32
    )
    groups_desired = np.array(
        [
            [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]
        ], dtype=np.int32
    )
    np.testing.assert_array_equal(pairs_0, pairs_0_desired)
    np.testing.assert_array_equal(pairs_1, pairs_1_desired)
    np.testing.assert_array_equal(groups, groups_desired)

    assert ds_info['n_pair'] == 10
    assert ds_info['batch_size'] == 10
    assert ds_info['n_batch'] == 1.0
    assert ds_info['elements'] == 'upper'
