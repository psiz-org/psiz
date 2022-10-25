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

from psiz.utils import choice_wo_replace


def test_choice_wo_replace():
    """Test choice_wo_replace."""
    n_trial = 10000
    n_reference = 8
    n_option = 20

    candidate_idx = np.arange(n_option)
    candidate_prob = np.array([
        0.04787656, 0.01988875, 0.08106771, 0.08468775, 0.07918673,
        0.05087084, 0.00922816, 0.08663405, 0.00707334, 0.02254985,
        0.01820681, 0.01532338, 0.07702897, 0.06774214, 0.09976408,
        0.05369049, 0.01056261, 0.07500489, 0.05508777, 0.03852514
    ])

    # Draw samples.
    np.random.seed(560897)
    drawn_idx = choice_wo_replace(
        candidate_idx, (n_trial, n_reference), candidate_prob
    )
    bin_counts, bin_edges = np.histogram(drawn_idx.flatten(), bins=n_option)
    drawn_prob = bin_counts / np.sum(bin_counts)

    # Check that sampling was done without replacement for all trials.
    for i_trial in range(n_trial):
        assert len(np.unique(drawn_idx[i_trial])) == n_reference

    # Check that sampling distribution matches original probabilites.
    np.testing.assert_array_almost_equal(candidate_prob, drawn_prob, decimal=2)


def test_seed():
    """Test seed."""
    n_trial = 2
    n_reference = 3
    n_option = 20

    candidate_idx = np.arange(n_option)
    candidate_prob = np.array([
        0.04787656, 0.01988875, 0.08106771, 0.08468775, 0.07918673,
        0.05087084, 0.00922816, 0.08663405, 0.00707334, 0.02254985,
        0.01820681, 0.01532338, 0.07702897, 0.06774214, 0.09976408,
        0.05369049, 0.01056261, 0.07500489, 0.05508777, 0.03852514
    ])

    # Draw samples using seed.
    rng = np.random.default_rng(seed=560897)
    drawn_idx = choice_wo_replace(
        candidate_idx, (n_trial, n_reference), candidate_prob, rng=rng
    )
    drawn_idx_desired = np.array(
        [
            [2, 14, 3],
            [12, 13, 15]
        ], dtype=int
    )
    np.testing.assert_array_equal(drawn_idx, drawn_idx_desired)
