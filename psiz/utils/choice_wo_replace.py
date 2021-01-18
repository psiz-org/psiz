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
"""Module of utility functions.

Functions:
    choice_wo_replace: Efficient sampling without replacement.

"""

import numpy as np


def choice_wo_replace(a, size, p):
    """Fast sampling without replacement.

    Arguments:
        a: An array indicating the eligable elements.
        size: A tuple indicating the number of independent samples and
            the number of draws (without replacement) for each sample.
            The tuple is ordered such that
            size = (n_sample, sample_size).
        p: An array indicating the probabilites associated with drawing
            a particular element. User provided probabilities are
            already assumed to sum to one. Probability p[i] indicates
            the probability of drawing index a[i].

    Returns:
        result: A 2D array containing the drawn elements.
            shape=(n_sample, sample_size)

    See: https://medium.com/ibm-watson/
        incredibly-fast-random-sampling-in-python-baf154bd836a

    """
    n_sample = size[0]
    sample_size = size[1]

    # Replicate probabilities as many times as `n_sample`
    replicated_probabilities = np.tile(p, (n_sample, 1))

    # Get random shifting numbers & scale them correctly.
    random_shifts = np.random.random(replicated_probabilities.shape)
    random_shifts /= random_shifts.sum(axis=1)[:, np.newaxis]

    # Shift by numbers and find largest (by finding the smallest of the
    # negative).
    shifted_probabilities = random_shifts - replicated_probabilities
    samples = np.argpartition(
        shifted_probabilities, sample_size, axis=1
    )[:, :sample_size]

    return a[samples]
