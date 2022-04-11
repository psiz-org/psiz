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

import itertools
from math import comb

import numpy as np

from psiz.utils import choice_wo_replace


def random_combinations(a, k, n_sample, p=None, replace=True, rng=None):
    """Sample from the possible k-combinations of `a`.

    Args:
        a: The elements used to create combinations.
            shape=(n_element,)
        k: Integer indicating the number of elements in each
            k-combination.
        n_sample: Integer indicating the number of unique combinations
            to sample. If `replace=False`, then there is a limit on the
            number of unique samples; if `n_sample` is greater than
            number of unique k-combinations, an exhaustive list of all
            k-combinations is returned (which will be less than the
            requested number of samples).
        p (optional): The sampling probability associated with each
            element in `a`. If not given, all elements are given equal
            probability.
            shape=(n_element,)
        replace (optional): Boolean indicating if the sampling is with
            or without replacement. The default is `True`, meaning that
            a particular k-combination can be sampled multiple times.
            If unique samples are desired, set `replace=False`.
        rng (optional): A NumPy random number generator that can be
            used to control stochasticity.

    Returns:
        samples: A set of k-combinations. If `replace=False`, the
            returned array is sorted both within a sample and across
            samples.
            shape=(n_sample, k)

    """
    n_element = len(a)

    if p is None:
        p = np.ones([n_element]) / n_element
    else:
        # Make sure probabilities sum to one.
        p = p / np.sum(p)

    if replace:
        # Sample with replacement.
        samples = choice_wo_replace(a, [n_sample, k], p, rng=rng)
    else:
        # Sample without replacement.
        n_unique = comb(n_element, k)
        if n_sample > n_unique:
            n_sample = n_unique

        if n_sample == n_unique:
            # Sample exhaustively.
            samples_iter = itertools.combinations(a, k)
            samples = np.stack(list(samples_iter), axis=0)
        else:
            # Sample iteratively to assemble a sufficient number of unique
            # samples.
            is_sufficient = False
            samples = np.empty([0, k], dtype=int)
            while not is_sufficient:
                new_samples = choice_wo_replace(a, [n_sample, k], p, rng=rng)
                new_samples = np.sort(new_samples, axis=1)
                samples = np.vstack([samples, new_samples])
                samples = np.unique(samples, axis=0)
                if samples.shape[0] >= n_sample:
                    is_sufficient = True
                    samples = samples[0:n_sample]

    return samples
