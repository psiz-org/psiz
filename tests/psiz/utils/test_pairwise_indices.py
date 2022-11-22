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

from psiz.utils import pairwise_indices


def test_scalar_defaults():
    """Test default optional arguments.

    Scalar argument.

    """
    n_stimuli = 5
    pairs = pairwise_indices(n_stimuli)
    pairs_desired = np.array(
        [
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],
            [1, 2],
            [1, 3],
            [1, 4],
            [2, 3],
            [2, 4],
            [3, 4]
        ], dtype=np.int32
    )
    np.testing.assert_array_equal(pairs, pairs_desired)


def test_array_defaults():
    """Test default optional arguments.

    Array argument.

    """
    n_stimuli = 5
    pairs = pairwise_indices(np.arange(n_stimuli))
    pairs_desired = np.array(
        [
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],
            [1, 2],
            [1, 3],
            [1, 4],
            [2, 3],
            [2, 4],
            [3, 4]
        ], dtype=np.int32
    )
    np.testing.assert_array_equal(pairs, pairs_desired)


def test_list_defaults():
    """Test default optional arguments.

    List argument.

    """
    indices = [0, 1, 2, 3, 4]
    pairs = pairwise_indices(indices)
    pairs_desired = np.array(
        [
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],
            [1, 2],
            [1, 3],
            [1, 4],
            [2, 3],
            [2, 4],
            [3, 4]
        ], dtype=np.int32
    )
    np.testing.assert_array_equal(pairs, pairs_desired)


def test_all():
    """Test default optional arguments."""
    n_stimuli = 5
    pairs = pairwise_indices(n_stimuli, elements='all')
    pairs_desired = np.array(
        [
            [0, 0],
            [1, 0],
            [2, 0],
            [3, 0],
            [4, 0],
            [0, 1],
            [1, 1],
            [2, 1],
            [3, 1],
            [4, 1],
            [0, 2],
            [1, 2],
            [2, 2],
            [3, 2],
            [4, 2],
            [0, 3],
            [1, 3],
            [2, 3],
            [3, 3],
            [4, 3],
            [0, 4],
            [1, 4],
            [2, 4],
            [3, 4],
            [4, 4]
        ], dtype=np.int32
    )
    np.testing.assert_array_equal(pairs, pairs_desired)


def test_lower():
    """Test default optional arguments."""
    n_stimuli = 5
    pairs = pairwise_indices(n_stimuli, elements='lower')
    pairs_desired = np.array(
        [
            [1, 0],
            [2, 0],
            [2, 1],
            [3, 0],
            [3, 1],
            [3, 2],
            [4, 0],
            [4, 1],
            [4, 2],
            [4, 3],
        ], dtype=np.int32
    )
    np.testing.assert_array_equal(pairs, pairs_desired)


def test_off():
    """Test default optional arguments."""
    n_stimuli = 5
    pairs = pairwise_indices(n_stimuli, elements='off')
    pairs_desired = np.array(
        [
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],
            [1, 2],
            [1, 3],
            [1, 4],
            [2, 3],
            [2, 4],
            [3, 4],
            [1, 0],
            [2, 0],
            [2, 1],
            [3, 0],
            [3, 1],
            [3, 2],
            [4, 0],
            [4, 1],
            [4, 2],
            [4, 3],
        ], dtype=np.int32
    )
    np.testing.assert_array_equal(pairs, pairs_desired)


def test_wrong():
    """Test default optional arguments."""
    n_stimuli = 5

    # Raise error because elements argument is not implemented.
    with pytest.raises(Exception) as e_info:
        _ = pairwise_indices(n_stimuli, elements='garbage')
    assert e_info.type == NotImplementedError

    # Raise error because indices argument must be scalar or 1D.
    with pytest.raises(Exception) as e_info:
        indices = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        _ = pairwise_indices(indices)
    assert e_info.type == ValueError


def test_subsample_0():
    """Test subsample."""
    n_stimuli = 10
    rng = np.random.default_rng(seed=252)
    pairs = pairwise_indices(n_stimuli, elements='all', subsample=.1, rng=rng)
    pairs_desired = np.array(
        [
            [7, 0],
            [1, 1],
            [6, 3],
            [9, 3],
            [2, 4],
            [1, 5],
            [0, 6],
            [3, 6],
            [3, 8],
            [5, 9]
        ], dtype=np.int32
    )
    np.testing.assert_array_equal(pairs, pairs_desired)


def test_subsample_1():
    """Test subsample.

    Request all samples using optional subsample keyword argument.

    """
    n_stimuli = 5
    pairs = pairwise_indices(n_stimuli, elements='upper', subsample=1)
    pairs_desired = np.array(
        [
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],
            [1, 2],
            [1, 3],
            [1, 4],
            [2, 3],
            [2, 4],
            [3, 4]
        ], dtype=np.int32
    )
    np.testing.assert_array_equal(pairs, pairs_desired)


def test_subsample_wrong():
    """Test subsample with invalid arguments."""
    n_stimuli = 5

    # Raise error because subsample is not in ]0,1].
    with pytest.raises(Exception) as e_info:
        _ = pairwise_indices(
            n_stimuli, elements='all', subsample=0
        )
    assert e_info.type == ValueError

    # Raise error because subsample is not in ]0,1].
    with pytest.raises(Exception) as e_info:
        _ = pairwise_indices(
            n_stimuli, elements='all', subsample=5
        )
    assert e_info.type == ValueError
