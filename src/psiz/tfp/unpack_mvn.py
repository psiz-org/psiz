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

"""Module of TensorFlow Probability objects.

Functions:
    unpack_mvn

"""


import numpy as np


def unpack_mvn(dist):
    """Unpack multivariate normal distribution.

    Gracefully handles case where set of multi-variate normal
    distributions are encapsulated by an `Independent` distribution by
    inferring full covariance from the diagonal variance elements.

    Args:
        dist: A set of multi-variate normal distributions.

    Returns:
        A tuple (loc, cov) where loc is a 2D array of locations and
            cov is a 3D array of covariance matrices.

    """
    # The location parameters are always accessed the same way.
    loc = dist.mean().numpy()

    # However, accessing the covariance parameters depends on the nature of
    # the incoming distribution.
    try:
        cov = dist.covariance().numpy()
    except NotImplementedError:
        # Assume `Independent` distribution with diagonal covariance.
        v = dist.variance().numpy()
        cov = _diag_to_full_cov(v)

    return loc, cov


def _diag_to_full_cov(v):
    """Convert diagonal variance to full covariance matrix.

    Args:
        v: An array represention diagonal variance elements only.

    Returns:
        cov: A array of fully-specified covariance matrices.

    """
    n_stimuli = v.shape[0]
    n_dim = v.shape[1]
    cov = np.zeros([n_stimuli, n_dim, n_dim])
    for i_stimulus in range(n_stimuli):
        cov[i_stimulus] = np.eye(n_dim) * v[i_stimulus]
    return cov
