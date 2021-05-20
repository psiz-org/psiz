
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
"""Procrustes superimposition.

Functions:
    procrustes_rotation: Determine rotation matrix for Procrustes
        superimposition.

"""

import numpy as np


def procrustes_rotation(z0, z1, scale=True):
    """Perform Procrustes superimposition.

    Align two sets of coordinates (`z0` and `z1`) by finding the
    optimal rotation matrix `r` that rotates `z0` into `z1`. Both `z0`
    and `z1` are centered first.

    `z0_rot = z0 @ r`

    Arguments:
        z0: The first set of points.
            shape = (n_point, n_dim)
        z1: The second set of points. The data matrices `z0` and `z1`
            must have the same shape.
            shape = (n_point, n_dim)
        n_restart (optional): A scalar indicating the number of
            restarts for the optimization routine.
        scale (optional): Boolean indicating if scaling is permitted
            in the affine transformation. By default scaling is
            allowed, generating a full Procrustes superimposition. Set
            to false to yield a partial Procrustes superimposition.

    Returns:
        r: A rotation matrix that operates on the *centered* data.
            shape=(n_dim, n_dim)

    """
    n_dim = z0.shape[1]

    # Ensure data matrices are centered.
    z0 = z0 - np.mean(z0, axis=0, keepdims=True)
    z1 = z1 - np.mean(z1, axis=0, keepdims=True)

    # Compute cross-covariance matrix.
    m = np.matmul(np.transpose(z0), z1)

    # Compute SVD of covariance matrix.
    # NOTE: h = u @ np.diag(s) @ vh = (u * s) @ vh
    u, s, vh = np.linalg.svd(m, hermitian=False)

    # Aseemble rotation matrix (does not include scaling).
    r = u @ vh

    if scale:
        # Determine scaling factor.
        z0_rot = np.matmul(z0, r)
        norm_z0 = np.sqrt(np.sum((z0_rot**2), axis=1))
        norm_z1 = np.sqrt(np.sum((z1**2), axis=1))
        scale_factor = norm_z1 / norm_z0
        scale_factor = np.mean(scale_factor)
        r = r @ (np.eye(n_dim) * scale_factor)

    return r
