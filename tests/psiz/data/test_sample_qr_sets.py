# -*- coding: utf-8 -*-
# Copyright 2024 The PsiZ Authors. All Rights Reserved.
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
# test seed
# test exhaustive
# test sampled
# test with n_highest

import numpy as np

from psiz.data import sample_qr_sets


def test_wo_replace_exhaustive():
    """Test.

    9 choose 2 = 36

    """
    n_stimuli = 10
    n_reference = 2
    ref_prob = np.ones([n_stimuli]) / n_stimuli

    # Sample query-reference sets.
    query_idx = 2
    n_sample = 36
    qr_sets = sample_qr_sets(query_idx, n_reference, n_sample, ref_prob, replace=False)

    qr_sets_desired = np.array(
        [
            [2, 0, 1],
            [2, 0, 3],
            [2, 0, 4],
            [2, 0, 5],
            [2, 0, 6],
            [2, 0, 7],
            [2, 0, 8],
            [2, 0, 9],
            [2, 1, 3],
            [2, 1, 4],
            [2, 1, 5],
            [2, 1, 6],
            [2, 1, 7],
            [2, 1, 8],
            [2, 1, 9],
            [2, 3, 4],
            [2, 3, 5],
            [2, 3, 6],
            [2, 3, 7],
            [2, 3, 8],
            [2, 3, 9],
            [2, 4, 5],
            [2, 4, 6],
            [2, 4, 7],
            [2, 4, 8],
            [2, 4, 9],
            [2, 5, 6],
            [2, 5, 7],
            [2, 5, 8],
            [2, 5, 9],
            [2, 6, 7],
            [2, 6, 8],
            [2, 6, 9],
            [2, 7, 8],
            [2, 7, 9],
            [2, 8, 9],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(qr_sets, qr_sets_desired)

    # Check when n_sample is larger that unique possibilties.
    n_sample = 40
    qr_sets = sample_qr_sets(query_idx, n_reference, n_sample, ref_prob, replace=False)
    np.testing.assert_array_equal(qr_sets, qr_sets_desired)


def test_wo_replace_subset_seed():
    """Test.

    9 choose 2 = 36

    """
    n_stimuli = 20
    n_reference = 8
    ref_prob = np.ones([n_stimuli]) / n_stimuli

    # Sample query-reference sets.
    query_idx = 13
    n_sample = 3
    rng0 = np.random.default_rng(seed=989)
    qr_sets0 = sample_qr_sets(
        query_idx, n_reference, n_sample, ref_prob, replace=False, rng=rng0
    )
    rng1 = np.random.default_rng(seed=989)
    qr_sets1 = sample_qr_sets(
        query_idx, n_reference, n_sample, ref_prob, replace=False, rng=rng1
    )

    # Same seed should reproduce the exact same sample within an environment.
    np.testing.assert_array_equal(qr_sets0, qr_sets1)

    # Basic invariants for sampled query-reference sets.
    np.testing.assert_array_equal(qr_sets0[:, 0], np.full([n_sample], query_idx))
    assert qr_sets0.shape == (n_sample, n_reference + 1)
    assert np.all(np.sort(qr_sets0[:, 1:], axis=1) == qr_sets0[:, 1:])
    assert np.unique(qr_sets0, axis=0).shape[0] == n_sample
    assert not np.any(qr_sets0[:, 1:] == query_idx)


def test_replace():
    """Test."""
    n_stimuli = 10
    n_reference = 2
    ref_prob = np.ones([n_stimuli]) / n_stimuli

    # Sample query-reference sets.
    query_idx = 2
    n_sample = 40
    qr_sets = sample_qr_sets(query_idx, n_reference, n_sample, ref_prob)

    # Check correct query index.
    query_arr_desired = np.full([n_sample], query_idx)
    np.testing.assert_array_equal(qr_sets[:, 0], query_arr_desired)


def test_replace_seed():
    """Test.

    Check with seed.

    """
    n_stimuli = 100
    n_reference = 8
    ref_prob = np.ones([n_stimuli]) / n_stimuli

    # Sample query-reference sets.
    query_idx = 88
    n_sample = 2
    rng0 = np.random.default_rng(seed=989)
    qr_sets0 = sample_qr_sets(query_idx, n_reference, n_sample, ref_prob, rng=rng0)
    rng1 = np.random.default_rng(seed=989)
    qr_sets1 = sample_qr_sets(query_idx, n_reference, n_sample, ref_prob, rng=rng1)

    # Same seed should reproduce the exact same sample within an environment.
    np.testing.assert_array_equal(qr_sets0, qr_sets1)

    # Basic invariants for sampled query-reference sets.
    np.testing.assert_array_equal(qr_sets0[:, 0], np.full([n_sample], query_idx))
    assert qr_sets0.shape == (n_sample, n_reference + 1)
    assert not np.any(qr_sets0[:, 1:] == query_idx)
    assert all(len(np.unique(row)) == n_reference for row in qr_sets0[:, 1:])
