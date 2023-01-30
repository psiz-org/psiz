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
    affine_mvn: Affine transformation of multivariate normal
        distribution.

"""

import numpy as np


def affine_mvn(loc, cov, r=None, t=None):
    """Affine transformation of multivariate normal.

    Performs the following operations:
        loc_affine = loc @ r + t
        cov_affine = r^T @ cov @ r

    Args:
        loc: Location parameters.
            shape=(n_dim,) or (1, n_dim)
        cov: Covariance.
            shape=(n_dim, n_dim)
        r: Rotation matrix.
            shape=(n_dim, n_dim)
        t: Transformation vector.
            shape=(n_dim,) or (1, n_dim)

    Returns:
        loc_affine: Rotated location parameters.
        cov_affine: Rotated covariance.

    Notes:
        1) np.matmul will prepend (postpend) singleton dimensions if
        the first (second) argument is a 1D array. This function
        therefore allows for 1D or 2D `loc` input. The translation
        vector `t` can be 1D or 2D since the additional operation will
        broadcast it appropriately.
        2) This implementation hits the means with a rotation matrix on
        the RHS, allowing the rows to correspond to an instance and
        columns to correspond to dimensionality. The more conventional
        pattern has rows corresponding to dimensionality, in which case
        the code would be implemented as:

        ```
        loc_affine = np.matmul(r, loc) + t
        cov_affine = np.matmul(r, np.matmul(cov, np.transpose(r)))
        ```

    """
    if t is None:
        # Default to no translation.
        t = 0.0
    if r is None:
        # Default to identity matrix (no rotation).
        n_dim = loc.shape[-1]
        r = np.eye(n_dim)

    loc_affine = np.matmul(loc, r) + t
    cov_affine = np.matmul(np.transpose(r), np.matmul(cov, r))
    return loc_affine, cov_affine
