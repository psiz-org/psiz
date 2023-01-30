# -*- coding: utf-8 -*-
# Copyright 2022 The PsiZ Authors. All Rights Reserved.
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
    pairwise_indices: Generate an array of pairwise indices.

"""

import numpy as np


def pairwise_indices(indices, elements="upper", subsample=None, rng=None):
    """Generate an array of pairwise indices.

    Args:
        indices: An scalar integer or an array-like of integers
            indicating indices. If a scalar, indices are
            `np.arange(n)`.
        elements (optional): Determines which combinations in the
            pairwise matrix will be used. Can be one of 'all', 'upper',
            'lower', or 'off'.
        subsample (optional): A float ]0,1] indicating the proportion
            of all pairs that should be retained. By default all pairs
            are retained.
        rng (optional): A `numpy.random.Generator` object. Controls
            which pairs are subsampled. Only used if subsample is not
            `None`.

    Returns:
        An array of pairwise indices.
            shape=(n_sample, 2)

    """

    # Check if scalar or array-lie.
    indices = np.array(indices, copy=False)
    if indices.ndim == 0:
        indices = np.arange(indices)
    elif indices.ndim != 1:
        raise ValueError("Argument `indices` must be scalar or 1D.")
    n_idx = len(indices)

    # Start by determine pairs of indices using relative indices.
    if elements == "all":
        idx = np.meshgrid(np.arange(n_idx), np.arange(n_idx))
        idx_0 = idx[0].flatten()
        idx_1 = idx[1].flatten()
    elif elements == "upper":
        idx = np.triu_indices(n_idx, 1)
        idx_0 = idx[0]
        idx_1 = idx[1]
    elif elements == "lower":
        idx = np.tril_indices(n_idx, -1)
        idx_0 = idx[0]
        idx_1 = idx[1]
    elif elements == "off":
        idx_upper = np.triu_indices(n_idx, 1)
        idx_lower = np.tril_indices(n_idx, -1)
        idx = (
            np.hstack((idx_upper[0], idx_lower[0])),
            np.hstack((idx_upper[1], idx_lower[1])),
        )
        idx_0 = idx[0]
        idx_1 = idx[1]
    else:
        raise NotImplementedError
    del idx

    n_pair = len(idx_0)
    if subsample is not None:
        # Make sure subsample is valid subsample value.
        if subsample <= 0 or subsample > 1.0:
            raise ValueError("Argument `subsample` must be in ]0,1]")
        if rng is None:
            rng = np.random.default_rng()
        idx_rand = rng.permutation(n_pair)
        n_pair = int(np.ceil(n_pair * subsample))
        idx_rand = np.sort(idx_rand[0:n_pair])
        idx_0 = idx_0[idx_rand]
        idx_1 = idx_1[idx_rand]

    # Covert to user-provided indices.
    idx_0 = indices[idx_0]
    idx_1 = indices[idx_1]

    return np.stack((idx_0, idx_1), axis=1)
