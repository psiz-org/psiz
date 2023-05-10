# -*- coding: utf-8 -*-
# Copyright 2023 The PsiZ Authors. All Rights Reserved.
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

"""Test WeightedRank object.

NOTE: When `mask_zero=True`, the element IDs start at 1, not 0.

"""


import numpy as np
import pytest

from psiz.data.generators import WeightedRank

from psiz.utils.sort_based_mask import sort_based_mask


def test_generate():
    """Test random generator."""
    n_stimuli_desired = 10
    n_sample_desired = 50
    n_reference_desired = 4
    n_select_desired = 2

    elements = np.arange(n_stimuli_desired, dtype=np.int32) + 1
    gen = WeightedRank(
        elements=elements, n_reference=n_reference_desired, n_select=n_select_desired
    )
    content = gen.generate(n_sample_desired)

    assert content.n_sample == n_sample_desired
    assert content.n_reference == n_reference_desired
    assert content.stimulus_set.shape[0] == n_sample_desired
    assert content.stimulus_set.shape[1] == 1
    assert content.stimulus_set.shape[2] == n_reference_desired + 1
    min_actual = np.min(content.stimulus_set)
    max_actual = np.max(content.stimulus_set)
    assert min_actual >= 1  # NOTE: inequality bc of stochasticity
    assert max_actual <= n_stimuli_desired  # NOTE: inequality bc of stochasticity
    n_unique_desired = n_reference_desired + 1
    for i_sample in range(n_sample_desired):
        assert len(np.unique(content.stimulus_set[i_sample, 0, :])) == n_unique_desired


def test_per_query_0():
    """Test.

    With non-uniform query weights and uniform reference weights.

    """
    n_stimuli = 10
    replace = False
    elements = np.arange(n_stimuli, dtype=np.int32) + 1
    gen = WeightedRank(elements=elements, n_reference=2, n_select=1)

    # Generate.
    n_sample_per_query = 10
    w = np.ones([n_stimuli, n_stimuli])
    w = w / np.sum(w, axis=1, keepdims=True)
    for idx in [2, 3, 4, 5, 6, 7]:
        w[idx, idx] = 0.0
    content = gen.generate(n_sample_per_query, w=w, per_query=True, replace=replace)

    # Check basic content configuration.
    assert content.n_select == 1
    assert content.n_reference == 2

    # Count query occurence.
    used_query_idx, query_count = np.unique(
        content.stimulus_set[:, 0, 0], return_counts=True
    )
    used_query_idx_desired = np.array([1, 2, 9, 10], dtype=np.int32)
    query_count_desired = np.array([10, 10, 10, 10], dtype=np.int32)
    np.testing.assert_array_equal(used_query_idx, used_query_idx_desired)
    np.testing.assert_array_equal(query_count, query_count_desired)

    # Check that samples are unique.
    n_unique_desired = 4 * n_sample_per_query
    n_unique = len(np.unique(content.stimulus_set, axis=0))
    assert n_unique == n_unique_desired


def test_query_with_weight():
    """Test.

    With non-uniform query and reference weights.

    """
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

    elements = np.arange(n_stimuli, dtype=np.int32) + 1
    gen = WeightedRank(
        elements=elements,
        n_reference=n_reference,
        n_select=n_select,
    )

    # Generate.
    n_sample_per_query = 10
    content = gen.generate(n_sample_per_query, w=w, per_query=True, replace=replace)

    # Count query occurence.
    used_query_idx, query_count = np.unique(
        content.stimulus_set[:, 0, 0], return_counts=True
    )
    used_query_idx_desired = np.array([1, 2, 9, 10], dtype=np.int32)
    query_count_desired = np.array([10, 10, 10, 10], dtype=np.int32)
    np.testing.assert_array_equal(used_query_idx, used_query_idx_desired)
    np.testing.assert_array_equal(query_count, query_count_desired)

    # Check that samples are unique.
    n_unique_desired = 4 * n_sample_per_query
    n_unique = len(np.unique(content.stimulus_set, axis=0))
    assert n_unique == n_unique_desired


def test_query_with_mask():
    """Test.

    With weight.

    """
    n_stimuli = 10
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

    elements = np.arange(n_stimuli, dtype=np.int32) + 1
    gen = WeightedRank(
        elements=elements,
        n_reference=n_reference,
        n_select=n_select,
    )
    # gen.truncate_w(n_highest=n_highest) TODO remove

    # Generate.
    n_sample_per_query = 10
    # NOTE: Add one to take into account query.
    w_mask = sort_based_mask(-w, n_highest + 1)
    w_masked = w * w_mask.astype(w.dtype)
    content = gen.generate(
        n_sample_per_query, w=w_masked, per_query=True, replace=replace
    )

    # Since we applied truncate_w, only the closest `n_highest` neighbors
    # are eligibile to serve as references. Diagonal elements (i.e., queries)
    # with zero weight are not eligible at all.
    refs_sub_desired = {
        1: {2, 3, 4, 5, 6},
        2: {},
        3: {},
        4: {},
        5: {6, 7, 8, 9, 10},
        6: {},
        7: {},
        8: {},
        9: {},
        10: {1, 2, 3, 4, 5},
    }
    query_idx_list = np.array([1, 2, 5, 10], dtype=np.int32)
    for i_query in query_idx_list:
        bidx = np.equal(content.stimulus_set[:, 0, 0], i_query)
        qr_set_sub = content.stimulus_set[bidx, 0, :]
        refs_unique = set(np.unique(qr_set_sub[:, 1:]))
        assert refs_unique.issubset(refs_sub_desired[i_query])

    # Check that samples are unique.
    n_unique_desired = 3 * n_sample_per_query
    n_unique = len(np.unique(content.stimulus_set, axis=0))
    assert n_unique == n_unique_desired


def test_query_other():
    """Test.

    Default weight initialization, per_query=True, replace=True

    """
    n_stimuli = 10
    elements = np.arange(n_stimuli, dtype=np.int32) + 1
    gen = WeightedRank(elements=elements, n_reference=2, n_select=1)

    # Generate.
    n_sample_per_query = 10
    content = gen.generate(n_sample_per_query, per_query=True, replace=True)

    # Count query occurence.
    used_query_idx, query_count = np.unique(
        content.stimulus_set[:, 0, 0], return_counts=True
    )
    used_query_idx_desired = np.arange(n_stimuli, dtype=np.int32) + 1
    query_count_desired = np.full([10], n_sample_per_query, dtype=np.int32)
    np.testing.assert_array_equal(used_query_idx, used_query_idx_desired)
    np.testing.assert_array_equal(query_count, query_count_desired)

    # Check content configuration.
    n_sample_desired = n_stimuli * n_sample_per_query
    assert content.n_sample == n_sample_desired
    assert content.n_select == 1
    assert content.n_reference == 2


def test_raise_exception():
    """Test exceptions."""
    n_stimuli = 10
    elements = np.arange(n_stimuli, dtype=np.int32) + 1

    with pytest.raises(Exception) as e_info:
        _ = WeightedRank(elements=elements, n_reference=11, n_select=2)
    assert e_info.type == ValueError

    with pytest.raises(Exception) as e_info:
        _ = WeightedRank(elements=elements, n_reference=11, n_select=2)
    assert e_info.type == ValueError
