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

from psiz.utils import pairwise_index_dataset
from psiz.utils import pairwise_similarity


def test_1g_default_all_nosample(rank_1g_mle_determ):
    """Test similarity matrix."""
    n_stimuli = 3
    desired_simmat0 = np.array([
        1., 0.35035481, 0.00776613, 0.35035481, 1., 0.0216217, 0.00776613,
        0.0216217, 1.
    ])

    ds_pairs_0, ds_info_0 = pairwise_index_dataset(
        n_stimuli, elements='all', mask_zero=True
    )

    computed_simmat0 = pairwise_similarity(
        rank_1g_mle_determ.stimuli, rank_1g_mle_determ.kernel, ds_pairs_0
    ).numpy()

    np.testing.assert_array_almost_equal(desired_simmat0, computed_simmat0)


def test_all_nosample(rank_2g_mle_determ):
    """Test similarity matrix."""
    n_stimuli = 3
    desired_simmat0 = np.array([
        1., 0.35035481, 0.00776613, 0.35035481, 1., 0.0216217, 0.00776613,
        0.0216217, 1.
    ])
    desired_simmat1 = np.array([
        1., 0.29685964, 0.00548485, 0.29685964, 1., 0.01814493, 0.00548485,
        0.01814493, 1.
    ])

    ds_pairs_0, ds_info_0 = pairwise_index_dataset(
        n_stimuli, elements='all', mask_zero=True, groups=[0]
    )

    ds_pairs_1, ds_info_1 = pairwise_index_dataset(
        n_stimuli, elements='all', mask_zero=True, groups=[1]
    )

    computed_simmat0 = pairwise_similarity(
        rank_2g_mle_determ.stimuli, rank_2g_mle_determ.kernel, ds_pairs_0,
        use_group_kernel=True
    ).numpy()

    computed_simmat1 = pairwise_similarity(
        rank_2g_mle_determ.stimuli, rank_2g_mle_determ.kernel, ds_pairs_1,
        use_group_kernel=True
    ).numpy()

    np.testing.assert_array_almost_equal(desired_simmat0, computed_simmat0)
    np.testing.assert_array_almost_equal(desired_simmat1, computed_simmat1)


def test_all_1sample(rank_2g_mle_determ):
    """Test similarity matrix."""
    n_stimuli = 3
    n_sample = 1

    desired_simmat0 = np.array([
        [1.], [0.35035481], [0.00776613], [0.35035481], [1.], [0.0216217],
        [0.00776613], [0.0216217], [1.]
    ])
    desired_simmat1 = np.array([
        [1.], [0.29685964], [0.00548485], [0.29685964], [1.], [0.01814493], 
        [0.00548485], [0.01814493], [1.]
    ])

    ds_pairs_0, ds_info_0 = pairwise_index_dataset(
        n_stimuli, elements='all', mask_zero=True, groups=[0]
    )

    ds_pairs_1, ds_info_1 = pairwise_index_dataset(
        n_stimuli, elements='all', mask_zero=True, groups=[1]
    )

    computed_simmat0 = pairwise_similarity(
        rank_2g_mle_determ.stimuli, rank_2g_mle_determ.kernel, ds_pairs_0,
        n_sample=1, use_group_kernel=True
    ).numpy()

    computed_simmat1 = pairwise_similarity(
        rank_2g_mle_determ.stimuli, rank_2g_mle_determ.kernel, ds_pairs_1,
        n_sample=1, use_group_kernel=True
    ).numpy()

    np.testing.assert_array_almost_equal(desired_simmat0, computed_simmat0)
    np.testing.assert_array_almost_equal(desired_simmat1, computed_simmat1)


def test_all_3sample(rank_2g_mle_determ):
    """Test similarity matrix."""
    n_stimuli = 3
    n_sample = 1

    desired_simmat0 = np.array([
        [1., 1., 1.],
        [0.35035481, 0.35035481, 0.35035481],
        [0.00776613, 0.00776613, 0.00776613],
        [0.35035481, 0.35035481, 0.35035481],
        [1., 1., 1.],
        [0.0216217, 0.0216217, 0.0216217],
        [0.00776613, 0.00776613, 0.00776613],
        [0.0216217, 0.0216217, 0.0216217],
        [1., 1., 1.]
    ])
    desired_simmat1 = np.array([
        [1., 1., 1.],
        [0.29685964, 0.29685964, 0.29685964],
        [0.00548485, 0.00548485, 0.00548485],
        [0.29685964, 0.29685964, 0.29685964],
        [1., 1., 1.],
        [0.01814493, 0.01814493, 0.01814493], 
        [0.00548485, 0.00548485, 0.00548485],
        [0.01814493, 0.01814493, 0.01814493],
        [1., 1., 1.]
    ])

    ds_pairs_0, ds_info_0 = pairwise_index_dataset(
        n_stimuli, elements='all', mask_zero=True, groups=[0]
    )

    ds_pairs_1, ds_info_1 = pairwise_index_dataset(
        n_stimuli, elements='all', mask_zero=True, groups=[1]
    )

    computed_simmat0 = pairwise_similarity(
        rank_2g_mle_determ.stimuli, rank_2g_mle_determ.kernel, ds_pairs_0,
        n_sample=3, use_group_kernel=True
    ).numpy()

    computed_simmat1 = pairwise_similarity(
        rank_2g_mle_determ.stimuli, rank_2g_mle_determ.kernel, ds_pairs_1,
        n_sample=3, use_group_kernel=True
    ).numpy()

    np.testing.assert_array_almost_equal(desired_simmat0, computed_simmat0)
    np.testing.assert_array_almost_equal(desired_simmat1, computed_simmat1)


def test_upper_3sample(rank_2g_mle_determ):
    """Test similarity matrix."""
    n_stimuli = 3
    n_sample = 1

    desired_simmat0 = np.array([
        [0.35035481, 0.35035481, 0.35035481],
        [0.00776613, 0.00776613, 0.00776613],
        [0.0216217, 0.0216217, 0.0216217]
    ])
    desired_simmat1 = np.array([
        [0.29685964, 0.29685964, 0.29685964],
        [0.00548485, 0.00548485, 0.00548485],
        [0.01814493, 0.01814493, 0.01814493]
    ])

    ds_pairs_0, ds_info_0 = pairwise_index_dataset(
        n_stimuli, elements='upper', mask_zero=True, groups=[0]
    )

    ds_pairs_1, ds_info_1 = pairwise_index_dataset(
        n_stimuli, elements='upper', mask_zero=True, groups=[1]
    )

    computed_simmat0 = pairwise_similarity(
        rank_2g_mle_determ.stimuli, rank_2g_mle_determ.kernel, ds_pairs_0,
        n_sample=3, use_group_kernel=True
    ).numpy()

    computed_simmat1 = pairwise_similarity(
        rank_2g_mle_determ.stimuli, rank_2g_mle_determ.kernel, ds_pairs_1,
        n_sample=3, use_group_kernel=True
    ).numpy()

    np.testing.assert_array_almost_equal(desired_simmat0, computed_simmat0)
    np.testing.assert_array_almost_equal(desired_simmat1, computed_simmat1)


def test_lower_3sample(rank_2g_mle_determ):
    """Test similarity matrix."""
    n_stimuli = 3
    n_sample = 1

    desired_simmat0 = np.array([
        [0.35035481, 0.35035481, 0.35035481],
        [0.00776613, 0.00776613, 0.00776613],
        [0.0216217, 0.0216217, 0.0216217]
    ])
    desired_simmat1 = np.array([
        [0.29685964, 0.29685964, 0.29685964],
        [0.00548485, 0.00548485, 0.00548485],
        [0.01814493, 0.01814493, 0.01814493]
    ])

    ds_pairs_0, ds_info_0 = pairwise_index_dataset(
        n_stimuli, elements='lower', mask_zero=True, groups=[0]
    )

    ds_pairs_1, ds_info_1 = pairwise_index_dataset(
        n_stimuli, elements='lower', mask_zero=True, groups=[1]
    )

    computed_simmat0 = pairwise_similarity(
        rank_2g_mle_determ.stimuli, rank_2g_mle_determ.kernel, ds_pairs_0,
        n_sample=3, use_group_kernel=True
    ).numpy()

    computed_simmat1 = pairwise_similarity(
        rank_2g_mle_determ.stimuli, rank_2g_mle_determ.kernel, ds_pairs_1,
        n_sample=3, use_group_kernel=True
    ).numpy()

    np.testing.assert_array_almost_equal(desired_simmat0, computed_simmat0)
    np.testing.assert_array_almost_equal(desired_simmat1, computed_simmat1)


def test_off_3sample(rank_2g_mle_determ):
    """Test similarity matrix."""
    n_stimuli = 3
    n_sample = 1

    desired_simmat0 = np.array([
        [0.35035481, 0.35035481, 0.35035481],
        [0.00776613, 0.00776613, 0.00776613],
        [0.0216217, 0.0216217, 0.0216217],
        [0.35035481, 0.35035481, 0.35035481],
        [0.00776613, 0.00776613, 0.00776613],
        [0.0216217, 0.0216217, 0.0216217]
    ])
    desired_simmat1 = np.array([
        [0.29685964, 0.29685964, 0.29685964],
        [0.00548485, 0.00548485, 0.00548485],
        [0.01814493, 0.01814493, 0.01814493],
        [0.29685964, 0.29685964, 0.29685964],
        [0.00548485, 0.00548485, 0.00548485],
        [0.01814493, 0.01814493, 0.01814493]
    ])

    ds_pairs_0, ds_info_0 = pairwise_index_dataset(
        n_stimuli, elements='off', mask_zero=True, groups=[0]
    )

    ds_pairs_1, ds_info_1 = pairwise_index_dataset(
        n_stimuli, elements='off', mask_zero=True, groups=[1]
    )

    computed_simmat0 = pairwise_similarity(
        rank_2g_mle_determ.stimuli, rank_2g_mle_determ.kernel, ds_pairs_0,
        n_sample=3, use_group_kernel=True
    ).numpy()

    computed_simmat1 = pairwise_similarity(
        rank_2g_mle_determ.stimuli, rank_2g_mle_determ.kernel, ds_pairs_1,
        n_sample=3, use_group_kernel=True
    ).numpy()

    np.testing.assert_array_almost_equal(desired_simmat0, computed_simmat0)
    np.testing.assert_array_almost_equal(desired_simmat1, computed_simmat1)
