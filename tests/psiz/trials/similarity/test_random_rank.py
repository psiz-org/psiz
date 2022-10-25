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

import numpy as np
import pytest

from psiz.trials import RandomRank


def test_generate_with_integer():
    """Test random generator."""
    n_stimuli_desired = 10
    n_trial_desired = 50
    n_reference_desired = 4
    n_select_desired = 2
    is_ranked_desired = True
    gen = RandomRank(
        n_stimuli_desired, n_reference=n_reference_desired,
        n_select=n_select_desired
    )
    docket = gen.generate(n_trial_desired)

    assert docket.n_trial == n_trial_desired
    assert sum(docket.n_reference == n_reference_desired) == n_trial_desired
    assert docket.stimulus_set.shape[0] == n_trial_desired
    assert docket.stimulus_set.shape[1] == n_reference_desired + 1

    min_actual = np.min(docket.stimulus_set)
    max_actual = np.max(docket.stimulus_set)
    assert min_actual >= 0
    assert max_actual < 10
    n_unique_desired = n_reference_desired + 1
    for i_trial in range(n_trial_desired):
        # NOTE: The padding padding values (-1)'s are counted as unique, thus
        # the indexing into stimulus set.
        assert (
            len(np.unique(
                docket.stimulus_set[i_trial, 0:n_reference_desired + 1])
                ) == n_unique_desired
        )
    assert sum(docket.n_select == n_select_desired) == n_trial_desired
    assert sum(docket.is_ranked == is_ranked_desired) == n_trial_desired


def test_generate_with_array():
    """Test random generator."""
    eligible_indices = np.arange(1, 11)
    n_trial_desired = 50
    n_reference_desired = 4
    n_select_desired = 2
    is_ranked_desired = True
    gen = RandomRank(
        eligible_indices, n_reference=n_reference_desired,
        n_select=n_select_desired
    )
    docket = gen.generate(n_trial_desired)

    assert docket.n_trial == n_trial_desired
    assert sum(docket.n_reference == n_reference_desired) == n_trial_desired
    assert docket.stimulus_set.shape[0] == n_trial_desired
    assert docket.stimulus_set.shape[1] == n_reference_desired + 1

    min_actual = np.min(docket.stimulus_set)
    max_actual = np.max(docket.stimulus_set)
    assert min_actual >= 1
    assert max_actual <= 10
    n_unique_desired = n_reference_desired + 1
    for i_trial in range(n_trial_desired):
        # NOTE: The padding padding values (-1)'s are counted as unique, thus
        # the indexing into stimulus set.
        assert (
            len(np.unique(
                docket.stimulus_set[i_trial, 0:n_reference_desired + 1])
                ) == n_unique_desired
        )
    assert sum(docket.n_select == n_select_desired) == n_trial_desired
    assert sum(docket.is_ranked == is_ranked_desired) == n_trial_desired


def test_per_query_0():
    """Test.

    Default trial configuration.
    Without weight.
    One worker.

    """
    eligible_indices = np.arange(1, 11)
    n_highest = None
    replace = False
    verbose = 1
    gen = RandomRank(
        eligible_indices, replace=replace, n_highest=n_highest, verbose=verbose
    )

    # Generate.
    n_trial_per_query = 10
    for idx in [2, 3, 4, 5, 6, 7]:
        gen.w[idx, idx] = 0.
    docket = gen.generate(n_trial_per_query, per_query=True)

    # Count query occurence.
    used_query_idx, query_count = np.unique(
        docket.stimulus_set[:, 0], return_counts=True
    )
    used_query_idx_desired = np.array([1, 2, 9, 10], dtype=np.int32)
    query_count_desired = np.array([10, 10, 10, 10], dtype=np.int32)
    np.testing.assert_array_equal(used_query_idx, used_query_idx_desired)
    np.testing.assert_array_equal(query_count, query_count_desired)

    # Check that trials are unique.
    n_unique_desired = 4 * n_trial_per_query
    n_unique = len(np.unique(docket.stimulus_set, axis=0))
    assert n_unique == n_unique_desired

    # Check trial configuration
    np.testing.assert_array_equal(
        docket.n_select, np.ones([n_unique_desired], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        docket.n_reference, np.full([n_unique_desired], 2, dtype=np.int32)
    )


def test_query_with_weight():
    """Test.

    With weight.

    """
    eligible_indices = np.arange(1, 21)
    n_stimuli = 20
    n_reference = 8
    n_select = 2
    rng = np.random.default_rng(34)
    w = rng.random([n_stimuli, n_stimuli])
    w[np.eye(n_stimuli, dtype=bool)] = 0
    w = w / np.sum(w, axis=1, keepdims=True)
    for idx in [0, 1, 8, 9]:
        w[idx, idx] = 0.25
    replace = False
    n_worker = 2

    gen = RandomRank(
        eligible_indices, n_reference=n_reference, n_select=n_select, w=w,
        replace=replace, n_worker=n_worker
    )

    # Generate.
    n_trial_per_query = 10
    docket = gen.generate(n_trial_per_query, per_query=True)

    # Count query occurence.
    used_query_idx, query_count = np.unique(
        docket.stimulus_set[:, 0], return_counts=True
    )
    used_query_idx_desired = np.array([1, 2, 9, 10], dtype=np.int32)
    query_count_desired = np.array([10, 10, 10, 10], dtype=np.int32)
    np.testing.assert_array_equal(used_query_idx, used_query_idx_desired)
    np.testing.assert_array_equal(query_count, query_count_desired)

    # Check that trials are unique.
    n_unique_desired = 4 * n_trial_per_query
    n_unique = len(np.unique(docket.stimulus_set, axis=0))
    assert n_unique == n_unique_desired


def test_query_with_highest():
    """Test.

    With weight.

    """
    n_stimuli = 10
    eligible_indices = np.arange(1, n_stimuli + 1)
    n_reference = 2
    n_select = 1
    w = np.array(
        [
            [0.33, 0.2, 0.2, 0.2, 0.2, 0.03, 0.01, 0.01, 0.01, 0.01],
            [0.01, 0.0, 0.2, 0.2, 0.2, 0.2, 0.03, 0.01, 0.01, 0.01],
            [0.01, 0.01, 0.0, 0.2, 0.2, 0.2, 0.2, 0.03, 0.01, 0.01],
            [0.01, 0.01, 0.01, 0.0, 0.2, 0.2, 0.2, 0.2, 0.03, 0.01],
            [0.01, 0.01, 0.01, 0.01, 0.33, 0.2, 0.2, 0.2, 0.2, 0.03],
            [0.03, 0.01, 0.01, 0.01, 0.01, 0.0, 0.2, 0.2, 0.2, 0.2],
            [0.2, 0.03, 0.01, 0.01, 0.01, 0.01, 0.0, 0.2, 0.2, 0.2],
            [0.2, 0.2, 0.03, 0.01, 0.01, 0.01, 0.01, 0.0, 0.2, 0.2],
            [0.2, 0.2, 0.2, 0.03, 0.01, 0.01, 0.01, 0.01, 0.0, 0.2],
            [0.2, 0.2, 0.2, 0.2, 0.03, 0.01, 0.01, 0.01, 0.01, 0.33],
        ]
    )
    n_highest = 5
    replace = False
    n_worker = 2
    verbose = 1

    gen = RandomRank(
        eligible_indices, n_reference=n_reference, n_select=n_select, w=w,
        replace=replace, n_highest=n_highest, n_worker=n_worker,
        verbose=verbose
    )

    # Generate.
    n_trial_per_query = 10
    docket = gen.generate(n_trial_per_query, per_query=True)

    # Check references within highest.
    refs_sub_desired = {
        1: {2, 3, 4, 5, 6},
        5: {6, 7, 8, 9, 10},
        10: {1, 2, 3, 4, 5},
    }
    query_idx_list = np.array([1, 5, 10], dtype=int)
    for i_query in query_idx_list:
        bidx = np.equal(docket.stimulus_set[:, 0], i_query)
        qr_set_sub = docket.stimulus_set[bidx]
        refs_unique = set(np.unique(qr_set_sub[:, 1:]))
        assert refs_unique.issubset(refs_sub_desired[i_query])

    # Check that trials are unique.
    n_unique_desired = 3 * n_trial_per_query
    n_unique = len(np.unique(docket.stimulus_set, axis=0))
    assert n_unique == n_unique_desired


def test_query_other_nomaskzero():
    """Test.

    Default initialization.
    Alternative generation with replace=True, verbose=1, default query
    list.

    """
    n_stimuli = 10
    gen = RandomRank(n_stimuli, mask_zero=False, verbose=1)

    # Generate.
    n_trial_per_query = 10
    docket = gen.generate(n_trial_per_query, per_query=True)

    # Count query occurence.
    used_query_idx, query_count = np.unique(
        docket.stimulus_set[:, 0], return_counts=True
    )
    used_query_idx_desired = np.arange(n_stimuli, dtype=np.int32)
    query_count_desired = np.full([10], n_trial_per_query, dtype=np.int32)
    np.testing.assert_array_equal(used_query_idx, used_query_idx_desired)
    np.testing.assert_array_equal(query_count, query_count_desired)

    # Check trial configuration
    n_trial_desired = n_stimuli * n_trial_per_query
    np.testing.assert_array_equal(
        docket.n_select, np.ones([n_trial_desired], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        docket.n_reference, np.full([n_trial_desired], 2, dtype=np.int32)
    )


def test_query_other_maskzero():
    """Test.

    Default initialization.
    Alternative generation with replace=True, verbose=1, default query
    list.

    """
    n_stimuli = 10
    gen = RandomRank(np.arange(n_stimuli) + 1, mask_zero=True, verbose=1)

    # Generate.
    n_trial_per_query = 10
    docket = gen.generate(n_trial_per_query, per_query=True)

    # Count query occurence.
    used_query_idx, query_count = np.unique(
        docket.stimulus_set[:, 0], return_counts=True
    )
    used_query_idx_desired = np.arange(1, n_stimuli + 1, dtype=np.int32)
    query_count_desired = np.full([10], n_trial_per_query, dtype=np.int32)
    np.testing.assert_array_equal(used_query_idx, used_query_idx_desired)
    np.testing.assert_array_equal(query_count, query_count_desired)

    # Check trial configuration
    n_trial_desired = n_stimuli * n_trial_per_query
    np.testing.assert_array_equal(
        docket.n_select, np.ones([n_trial_desired], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        docket.n_reference, np.full([n_trial_desired], 2, dtype=np.int32)
    )


def test_raise_exception():
    """Test.

    Default initialization.
    Alternative generation with replace=True, default query
    list.

    """
    n_stimuli = 10
    gen = RandomRank(n_stimuli, replace=False)

    # Force bad state to raise error in worker subprocess.
    gen.n_reference = 11

    # Generate.
    n_trial_per_query = 10
    with pytest.raises(Exception) as e_info:
        gen.generate(n_trial_per_query, per_query=True)
    assert e_info.type == ValueError
