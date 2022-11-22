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

import pytest

import numpy as np

from psiz.utils import pairwise_index_dataset
from psiz.utils import pairwise_similarity


@pytest.mark.xfail(
    reason="pairwise_similarity deprecated."
)
def test_1g_all_defaults(rank_1g_mle_determ):
    """Test similarity matrix.

    Use all defaults.

    """
    n_stimuli = 3
    desired_simmat0 = np.array([
        1., 0.35035481, 0.00776613, 0.35035481, 1., 0.0216217, 0.00776613,
        0.0216217, 1.
    ])

    ds_pairs_0, ds_info_0 = pairwise_index_dataset(
        np.arange(n_stimuli) + 1, elements='all'
    )

    computed_simmat0 = pairwise_similarity(
        rank_1g_mle_determ.stimuli, rank_1g_mle_determ.kernel, ds_pairs_0
    ).numpy()

    np.testing.assert_array_almost_equal(desired_simmat0, computed_simmat0)


@pytest.mark.xfail(
    reason="pairwise_similarity deprecated."
)
def test_verbose(rank_1g_mle_determ, capsys):
    """Test similarity matrix.

    Test basic format of verbose output.

    """
    n_stimuli = 3

    ds_pairs_0, ds_info_0 = pairwise_index_dataset(
        np.arange(n_stimuli) + 1, elements='all'
    )

    _ = pairwise_similarity(
        rank_1g_mle_determ.stimuli, rank_1g_mle_determ.kernel, ds_pairs_0,
        verbose=1
    )
    captured = capsys.readouterr()
    out_desired = (
        '\r    Similarity: |-------------------------------------------------'
        '-| 0.0% | ETA: 0:00:00 | ETT: 0:00:00\r\r    Similarity: |██████████'
        '████████████████████████████████████████| 100.0% | ETA: 0:00:00 | ET'
        'T: 0:00:00\r\n\r    Similarity: |███████████████████████████████████'
        '███████████████| 100.0% | ETA: 0:00:00 | ETT: 0:00:00\r\n'
    )
    assert captured.out == out_desired


@pytest.mark.xfail(
    reason="pairwise_similarity deprecated."
)
def test_verbose_v1(rank_1g_mle_random, capsys):
    """Test similarity matrix.

    Test verbose when number of batches is greater than the default
    window size (i.e., 50 chars).

    """
    n_stimuli = 10

    ds_pairs_0, ds_info_0 = pairwise_index_dataset(
        np.arange(n_stimuli) + 1, elements='all', batch_size=1
    )

    _ = pairwise_similarity(
        rank_1g_mle_random.stimuli, rank_1g_mle_random.kernel, ds_pairs_0,
        verbose=1
    )
    captured = capsys.readouterr()
    assert len(captured.out) == 5508


@pytest.mark.xfail(
    reason="pairwise_similarity deprecated."
)
def test_all_nosample(rank_2g_mle_determ):
    """Test similarity matrix.

    Use default sample argument (None).

    """
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
        np.arange(n_stimuli) + 1, elements='all', groups=[0]
    )

    ds_pairs_1, ds_info_1 = pairwise_index_dataset(
        np.arange(n_stimuli) + 1, elements='all', groups=[1]
    )

    computed_simmat0 = pairwise_similarity(
        rank_2g_mle_determ.stimuli, rank_2g_mle_determ.kernel, ds_pairs_0,
        use_group_kernel=True
    ).numpy()

    computed_simmat1 = pairwise_similarity(
        rank_2g_mle_determ.stimuli, rank_2g_mle_determ.kernel, ds_pairs_1,
        use_group_kernel=True
    ).numpy()

    # Could use the fact that n_sample=1 and elements='all' to take advantage
    # of np.reshape.
    # computed_simmat0 = np.reshape(computed_simmat0, [n_stimuli, n_stimuli])
    # computed_simmat1 = np.reshape(computed_simmat1, [n_stimuli, n_stimuli])

    np.testing.assert_array_almost_equal(desired_simmat0, computed_simmat0)
    np.testing.assert_array_almost_equal(desired_simmat1, computed_simmat1)


@pytest.mark.xfail(
    reason="pairwise_similarity deprecated."
)
def test_all_1sample(rank_2g_mle_determ):
    """Test similarity matrix.

    Use one sample.

    """
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
        np.arange(n_stimuli) + 1, elements='all', groups=[0]
    )

    ds_pairs_1, ds_info_1 = pairwise_index_dataset(
        np.arange(n_stimuli) + 1, elements='all', groups=[1]
    )

    computed_simmat0 = pairwise_similarity(
        rank_2g_mle_determ.stimuli, rank_2g_mle_determ.kernel, ds_pairs_0,
        n_sample=n_sample, use_group_kernel=True
    ).numpy()

    computed_simmat1 = pairwise_similarity(
        rank_2g_mle_determ.stimuli, rank_2g_mle_determ.kernel, ds_pairs_1,
        n_sample=n_sample, use_group_kernel=True
    ).numpy()

    np.testing.assert_array_almost_equal(desired_simmat0, computed_simmat0)
    np.testing.assert_array_almost_equal(desired_simmat1, computed_simmat1)


@pytest.mark.xfail(
    reason="pairwise_similarity deprecated."
)
def test_all_3sample(rank_2g_mle_determ):
    """Test similarity matrix."""
    n_stimuli = 3
    n_sample = 3

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
        np.arange(n_stimuli) + 1, elements='all', groups=[0]
    )

    ds_pairs_1, ds_info_1 = pairwise_index_dataset(
        np.arange(n_stimuli) + 1, elements='all', groups=[1]
    )

    computed_simmat0 = pairwise_similarity(
        rank_2g_mle_determ.stimuli, rank_2g_mle_determ.kernel, ds_pairs_0,
        n_sample=n_sample, use_group_kernel=True
    ).numpy()

    computed_simmat1 = pairwise_similarity(
        rank_2g_mle_determ.stimuli, rank_2g_mle_determ.kernel, ds_pairs_1,
        n_sample=n_sample, use_group_kernel=True
    ).numpy()

    np.testing.assert_array_almost_equal(desired_simmat0, computed_simmat0)
    np.testing.assert_array_almost_equal(desired_simmat1, computed_simmat1)


@pytest.mark.xfail(
    reason="pairwise_similarity deprecated."
)
def test_upper_3sample(rank_2g_mle_determ):
    """Test similarity matrix."""
    n_stimuli = 3
    n_sample = 3

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
        np.arange(n_stimuli) + 1, elements='upper', groups=[0]
    )

    ds_pairs_1, ds_info_1 = pairwise_index_dataset(
        np.arange(n_stimuli) + 1, elements='upper', groups=[1]
    )

    computed_simmat0 = pairwise_similarity(
        rank_2g_mle_determ.stimuli, rank_2g_mle_determ.kernel, ds_pairs_0,
        n_sample=n_sample, use_group_kernel=True
    ).numpy()

    computed_simmat1 = pairwise_similarity(
        rank_2g_mle_determ.stimuli, rank_2g_mle_determ.kernel, ds_pairs_1,
        n_sample=n_sample, use_group_kernel=True
    ).numpy()

    np.testing.assert_array_almost_equal(desired_simmat0, computed_simmat0)
    np.testing.assert_array_almost_equal(desired_simmat1, computed_simmat1)


@pytest.mark.xfail(
    reason="pairwise_similarity deprecated."
)
def test_upper_3sample_avg(rank_2g_mle_determ):
    """Test similarity matrix."""
    n_stimuli = 3
    n_sample = 3

    desired_simmat0 = np.array(
        [0.35035481, 0.00776613, 0.0216217]
    )
    desired_simmat1 = np.array(
        [0.29685964, 0.00548485, 0.01814493]
    )

    ds_pairs_0, ds_info_0 = pairwise_index_dataset(
        np.arange(n_stimuli) + 1, elements='upper', groups=[0]
    )

    ds_pairs_1, ds_info_1 = pairwise_index_dataset(
        np.arange(n_stimuli) + 1, elements='upper', groups=[1]
    )

    computed_simmat0 = pairwise_similarity(
        rank_2g_mle_determ.stimuli, rank_2g_mle_determ.kernel, ds_pairs_0,
        n_sample=n_sample, use_group_kernel=True, compute_average=True
    ).numpy()

    computed_simmat1 = pairwise_similarity(
        rank_2g_mle_determ.stimuli, rank_2g_mle_determ.kernel, ds_pairs_1,
        n_sample=n_sample, use_group_kernel=True, compute_average=True
    ).numpy()

    np.testing.assert_array_almost_equal(desired_simmat0, computed_simmat0)
    np.testing.assert_array_almost_equal(desired_simmat1, computed_simmat1)


@pytest.mark.xfail(
    reason="pairwise_similarity deprecated."
)
def test_lower_3sample(rank_2g_mle_determ):
    """Test similarity matrix."""
    n_stimuli = 3
    n_sample = 3

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
        np.arange(n_stimuli) + 1, elements='lower', groups=[0]
    )

    ds_pairs_1, ds_info_1 = pairwise_index_dataset(
        np.arange(n_stimuli) + 1, elements='lower', groups=[1]
    )

    computed_simmat0 = pairwise_similarity(
        rank_2g_mle_determ.stimuli, rank_2g_mle_determ.kernel, ds_pairs_0,
        n_sample=n_sample, use_group_kernel=True
    ).numpy()

    computed_simmat1 = pairwise_similarity(
        rank_2g_mle_determ.stimuli, rank_2g_mle_determ.kernel, ds_pairs_1,
        n_sample=n_sample, use_group_kernel=True
    ).numpy()

    np.testing.assert_array_almost_equal(desired_simmat0, computed_simmat0)
    np.testing.assert_array_almost_equal(desired_simmat1, computed_simmat1)


@pytest.mark.xfail(
    reason="pairwise_similarity deprecated."
)
def test_off_3sample(rank_2g_mle_determ):
    """Test similarity matrix."""
    n_stimuli = 3
    n_sample = 3

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
        np.arange(n_stimuli) + 1, elements='off', groups=[0]
    )

    ds_pairs_1, ds_info_1 = pairwise_index_dataset(
        np.arange(n_stimuli) + 1, elements='off', groups=[1]
    )

    computed_simmat0 = pairwise_similarity(
        rank_2g_mle_determ.stimuli, rank_2g_mle_determ.kernel, ds_pairs_0,
        n_sample=n_sample, use_group_kernel=True
    ).numpy()

    computed_simmat1 = pairwise_similarity(
        rank_2g_mle_determ.stimuli, rank_2g_mle_determ.kernel, ds_pairs_1,
        n_sample=n_sample, use_group_kernel=True
    ).numpy()

    np.testing.assert_array_almost_equal(desired_simmat0, computed_simmat0)
    np.testing.assert_array_almost_equal(desired_simmat1, computed_simmat1)


@pytest.mark.xfail(
    reason="pairwise_similarity deprecated."
)
def test_group_stimuli_and_kernel(rank_2stim_2kern_determ):
    """Test similarity matrix."""
    n_stimuli = 3
    n_sample = 3

    desired_simmat0 = np.array([
        [0.35035481, 0.35035481, 0.35035481],
        [0.00776613, 0.00776613, 0.00776613],
        [0.0216217, 0.0216217, 0.0216217],
        [0.35035481, 0.35035481, 0.35035481],
        [0.00776613, 0.00776613, 0.00776613],
        [0.0216217, 0.0216217, 0.0216217]
    ])
    desired_simmat1 = np.array(
        [
            [0.01814493, 0.01814493, 0.01814493],
            [0.29685965, 0.29685965, 0.29685965],
            [0.00548485, 0.00548485, 0.00548485],
            [0.01814493, 0.01814493, 0.01814493],
            [0.29685965, 0.29685965, 0.29685965],
            [0.00548485, 0.00548485, 0.00548485]
        ]
    )

    ds_pairs_0, ds_info_0 = pairwise_index_dataset(
        np.arange(n_stimuli) + 1, elements='off', groups=[0]
    )

    ds_pairs_1, ds_info_1 = pairwise_index_dataset(
        np.arange(n_stimuli) + 1, elements='off', groups=[1]
    )

    computed_simmat0 = pairwise_similarity(
        rank_2stim_2kern_determ.stimuli, rank_2stim_2kern_determ.kernel,
        ds_pairs_0, n_sample=n_sample, use_group_stimuli=True,
        use_group_kernel=True
    ).numpy()

    computed_simmat1 = pairwise_similarity(
        rank_2stim_2kern_determ.stimuli, rank_2stim_2kern_determ.kernel,
        ds_pairs_1, n_sample=n_sample, use_group_stimuli=True,
        use_group_kernel=True
    ).numpy()

    np.testing.assert_array_almost_equal(desired_simmat0, computed_simmat0)
    np.testing.assert_array_almost_equal(desired_simmat1, computed_simmat1)
