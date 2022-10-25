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
"""Test utils module."""

import numpy as np

from psiz.utils import random_combinations


def test_w_replace_probs():
    """Test with replacement and probabilities."""
    n_option = 20
    arr = np.arange(n_option)
    probs = np.array([
        0.04787656, 0.01988875, 0.08106771, 0.08468775, 0.07918673,
        0.05087084, 0.00922816, 0.08663405, 0.00707334, 0.02254985,
        0.01820681, 0.01532338, 0.07702897, 0.06774214, 0.09976408,
        0.05369049, 0.01056261, 0.07500489, 0.05508777, 0.03852514
    ])
    k = 8

    # Draw samples.
    n_sample = 10000
    rng = np.random.default_rng(seed=560897)
    samples = random_combinations(arr, k, n_sample, p=probs, rng=rng)
    bin_counts, _ = np.histogram(samples.flatten(), bins=n_option)
    drawn_prob = bin_counts / np.sum(bin_counts)

    # Check each sample does not have repeated values.
    for i_sample in range(n_sample):
        assert len(np.unique(samples[i_sample])) == k

    # Check that sampling distribution matches original probabilites.
    np.testing.assert_array_almost_equal(probs, drawn_prob, decimal=2)


def test_w_replace_seeded():
    """Test seed."""
    n_option = 20
    arr = np.arange(n_option)
    probs = np.array([
        0.04787656, 0.01988875, 0.08106771, 0.08468775, 0.07918673,
        0.05087084, 0.00922816, 0.08663405, 0.00707334, 0.02254985,
        0.01820681, 0.01532338, 0.07702897, 0.06774214, 0.09976408,
        0.05369049, 0.01056261, 0.07500489, 0.05508777, 0.03852514
    ])
    k = 3

    # Draw samples using seed.
    n_sample = 2
    rng = np.random.default_rng(seed=560897)
    samples = random_combinations(arr, k, n_sample, p=probs, rng=rng)
    samples_desired = np.array(
        [
            [2, 14, 3],
            [12, 13, 15]
        ], dtype=int
    )
    np.testing.assert_array_equal(samples, samples_desired)


def test_wo_replace_exhaustive():
    """Test without replacement.

    Exhaustive.

    """
    # Request exactly the number of unique k-combinations.
    arr = np.arange(5)
    k = 2
    n_sample = 10

    # Draw samples.
    rng = np.random.default_rng(seed=560897)
    samples = random_combinations(arr, k, n_sample, replace=False, rng=rng)

    samples_desired = np.array(
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
        ]
    )
    np.testing.assert_equal(samples, samples_desired)

    # Request more than the number of k-combinations.
    arr = np.arange(5)
    k = 2
    n_sample = 20

    # Draw samples.
    rng = np.random.default_rng(seed=560897)
    samples = random_combinations(arr, k, n_sample, replace=False, rng=rng)

    samples_desired = np.array(
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
        ]
    )
    np.testing.assert_equal(samples, samples_desired)


def test_wo_replace_subsample():
    """Test without replacement.

    Subsample.

    """
    arr = np.arange(5)
    k = 2
    n_sample = 5

    # Draw samples.
    rng = np.random.default_rng(seed=560897)
    samples = random_combinations(arr, k, n_sample, replace=False, rng=rng)

    samples_desired = np.array(
        [
            [0, 1],
            [0, 2],
            [0, 3],
            [1, 2],
            [1, 3]
        ]
    )
    np.testing.assert_equal(samples, samples_desired)
