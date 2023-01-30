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
"""Base generator functionaltiy.

Functions:
    sample_qr_sets

"""
import copy

import numpy as np
from psiz.utils import random_combinations


def sample_qr_sets(
    query_idx, n_reference, n_sample, reference_probability, replace=True, rng=None
):
    """Sample query-reference sets for a specific query.

    For problems involving more than a few stimuli, it is infeasible to
    generate an exhaustive list of stimlus sets. As an alternative,
    references can be stochastically sampled based on user-provide
    probabilities.

    Args:
        query_idx: An integer indicating the query index.
        n_reference: An integer indicating the number of references in
            each trial.
        n_sample: Integer indicating the number of unique combinations
            to sample. If `replace=False`, then there is a limit on the
            number of unique samples; if `n_sample` is greater than
            number of unique k-combinations, an exhaustive list of all
            k-combinations is returned (which will be less than the
            requested number of samples).
        reference_probability: An array of nonnegative values
            indicating the probability of selecting each stimulus as a
            reference given the query indicated by `query_idx`. It is
            assumed, but not checked, that all values are nonnegative.
            This array must include the probability of the query index
            so that index position semantics are preserved. The values
            do not need to sum to 1 since the array is normalized
            internally.
            shape=(n_stimuli,)
        replace (optional): Boolean indicating if the sampling is with
            or without replacement. The default is `True`, meaning that
            a particular k-combination can be sampled multiple times.
            If unique samples are desired, set `replace=False`.
        rng (optional): A NumPy random number generator that can be
            used to control stochasticity.

    Returns:
        samples: A set of query-reference samples.
            shape=(n_samples, n_reference + 1)

    """
    n_stimuli = len(reference_probability)

    # Zero out `query_idx` in `reference_probability`, so that it is not
    # eligible to be selected as a reference. We use the term "priority" to
    # make it clear that the values may not sum to one.
    ref_priority = copy.copy(reference_probability)
    ref_priority[query_idx] = 0

    # Find references with priorities greater than zero.
    bidx_eligable = np.greater(ref_priority, 0)
    ref_idx_eligable = np.arange(n_stimuli)
    ref_idx_eligable = ref_idx_eligable[bidx_eligable]

    # Convert priorities into proper probabilities.
    ref_priority = ref_priority[bidx_eligable]
    ref_probability = ref_priority / np.sum(ref_priority)

    # Sample references using eligible references only.
    ref_samples = random_combinations(
        ref_idx_eligable,
        n_reference,
        n_sample,
        p=ref_probability,
        replace=replace,
        rng=rng,
    )

    # Add refernces to query index being careful that number of obtained
    # samples may not match the number of requested samples.
    n_samples_obtained = ref_samples.shape[0]
    stimulus_set = np.hstack([np.full([n_samples_obtained, 1], query_idx), ref_samples])

    return stimulus_set
