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
"""Module for testing utils.py."""

import numpy as np

from psiz import utils
import psiz.keras.models


def test_generate_group_matrix():
    """Test generate_group_matrix."""
    n_row = 3

    # Test default.
    group_matrix = psiz.utils.generate_group_matrix(
        n_row
    )
    desired_group_matrix = np.array([
        [0],
        [0],
        [0]
    ])
    np.testing.assert_array_equal(
        group_matrix, desired_group_matrix
    )

    # Test one-level hierarchy.
    groups = [0, 0]
    group_matrix = psiz.utils.generate_group_matrix(
        n_row, groups=groups
    )
    desired_group_matrix = np.array([
        [0, 0],
        [0, 0],
        [0, 0]
    ])
    np.testing.assert_array_equal(
        group_matrix, desired_group_matrix
    )

    # Test three-level hierarchy.
    groups = [0, 6, 7, 3]
    group_matrix = psiz.utils.generate_group_matrix(
        n_row, groups=groups
    )
    desired_group_matrix = np.array([
        [0, 6, 7, 3],
        [0, 6, 7, 3],
        [0, 6, 7, 3]
    ])
    np.testing.assert_array_equal(
        group_matrix, desired_group_matrix
    )


def test_pairwise_matrix(rank_2g_mle_determ):
    """Test similarity matrix."""
    n_stimuli = 3
    desired_simmat0 = np.array((
        (1., 0.35035481, 0.00776613),
        (0.35035481, 1., 0.0216217),
        (0.00776613, 0.0216217, 1.)
    ))
    desired_simmat1 = np.array((
        (1., 0.29685964, 0.00548485),
        (0.29685964, 1., 0.01814493),
        (0.00548485, 0.01814493, 1.)
    ))

    ds_pairs_0, ds_info_0 = psiz.utils.pairwise_index_dataset(
        n_stimuli, elements='all', mask_zero=True, groups=[0]
    )

    ds_pairs_1, ds_info_1 = psiz.utils.pairwise_index_dataset(
        n_stimuli, elements='all', mask_zero=True, groups=[1]
    )

    computed_simmat0 = psiz.utils.pairwise_similarity(
        rank_2g_mle_determ.stimuli, rank_2g_mle_determ.kernel, ds_pairs_0,
        use_group_kernel=True
    ).numpy()

    computed_simmat1 = psiz.utils.pairwise_similarity(
        rank_2g_mle_determ.stimuli, rank_2g_mle_determ.kernel, ds_pairs_1,
        use_group_kernel=True
    ).numpy()

    # Use fact that n_sample=1 and elements='all' to take advantage of
    # np.reshape.
    computed_simmat0 = np.reshape(computed_simmat0, [n_stimuli, n_stimuli])
    computed_simmat1 = np.reshape(computed_simmat1, [n_stimuli, n_stimuli])

    np.testing.assert_array_almost_equal(desired_simmat0, computed_simmat0)
    np.testing.assert_array_almost_equal(desired_simmat1, computed_simmat1)


def test_matrix_comparison():
    """Test matrix correlation."""
    a = np.array((
        (1.0, .50, .90, .13),
        (.50, 1.0, .10, .80),
        (.90, .10, 1.0, .12),
        (.13, .80, .12, 1.0)
    ))

    b = np.array((
        (1.0, .45, .90, .11),
        (.45, 1.0, .20, .82),
        (.90, .20, 1.0, .02),
        (.11, .82, .02, 1.0)
    ))

    r2_score_1 = utils.matrix_comparison(a, b, score='r2')
    np.testing.assert_almost_equal(r2_score_1, 0.96723696)


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
    drawn_idx = psiz.utils.choice_wo_replace(
        candidate_idx, (n_trial, n_reference), candidate_prob
    )
    bin_counts, bin_edges = np.histogram(drawn_idx.flatten(), bins=n_option)
    drawn_prob = bin_counts / np.sum(bin_counts)

    # Check that sampling was done without replacement for all trials.
    for i_trial in range(n_trial):
        assert len(np.unique(drawn_idx[i_trial])) == n_reference

    # Check that sampling distribution matches original probabilites.
    np.testing.assert_array_almost_equal(candidate_prob, drawn_prob, decimal=2)
