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


import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from psiz.tfp.unpack_mvn import unpack_mvn


def test_unpack_diagonal_covariance_mvn():
    n_stimuli = 3
    n_dim = 2
    scale_desired = 0.17
    loc_desired = np.array(
        [
            [0.1, 0.2],
            [1.1, 1.2],
            [2.1, 2.2],
        ],
        dtype=np.float32,
    )
    cov_desired = np.array(
        [
            [[0.028900036588311195, 0.0], [0.0, 0.028900036588311195]],
            [[0.028900036588311195, 0.0], [0.0, 0.028900036588311195]],
            [[0.028900036588311195, 0.0], [0.0, 0.028900036588311195]],
        ],
        dtype=np.float32,
    )

    # Create diagonal covariance MVNs.
    dist = tfp.distributions.Normal(loc=loc_desired, scale=scale_desired)
    batch_ndims = tf.size(dist.batch_shape_tensor())
    dist = tfp.distributions.Independent(dist, reinterpreted_batch_ndims=batch_ndims)

    loc, cov = unpack_mvn(dist)

    assert len(loc.shape) == 2
    assert loc.shape[0] == n_stimuli
    assert loc.shape[1] == n_dim
    np.testing.assert_almost_equal(loc, loc_desired)

    assert len(cov.shape) == 3
    assert cov.shape[0] == n_stimuli
    assert cov.shape[1] == n_dim
    assert cov.shape[2] == n_dim
    np.testing.assert_almost_equal(cov, cov_desired)


def test_unpack_full_covariance_mvn():
    n_stimuli = 3
    n_dim = 2

    loc_desired = np.array(
        [
            [0.1, 0.2],
            [1.1, 1.2],
            [2.1, 2.2],
        ],
        dtype=np.float32,
    )
    cov_desired = np.array(
        [
            [[0.028900036588311195, 0.0], [0.0, 0.028900036588311195]],
            [[0.028900036588311195, 0.0], [0.0, 0.028900036588311195]],
            [[0.028900036588311195, 0.0], [0.0, 0.028900036588311195]],
        ],
        dtype=np.float32,
    )

    dist = tfp.distributions.MultivariateNormalFullCovariance(
        loc=loc_desired, covariance_matrix=cov_desired
    )

    loc, cov = unpack_mvn(dist)

    assert len(loc.shape) == 2
    assert loc.shape[0] == n_stimuli
    assert loc.shape[1] == n_dim
    np.testing.assert_almost_equal(loc, loc_desired)

    assert len(cov.shape) == 3
    assert cov.shape[0] == n_stimuli
    assert cov.shape[1] == n_dim
    assert cov.shape[2] == n_dim
    np.testing.assert_almost_equal(cov, cov_desired)
