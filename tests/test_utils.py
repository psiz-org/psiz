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
"""Module for testing utils.py.

Todo:
    - test compare_models

"""


import numpy as np
import pytest
import tensorflow as tf

from psiz import utils
import psiz.models
import psiz.keras.layers


@pytest.fixture(scope="module")
def ground_truth():
    """Return a ground truth model."""
    n_stimuli = 3
    n_dim = 2
    n_group = 2

    # Define embedding.
    stimuli = tf.keras.layers.Embedding(n_stimuli+1, n_dim, mask_zero=True)
    stimuli.build([None, None, None])

    # Define kernel.
    attention = psiz.keras.layers.GroupAttention(n_dim=n_dim, n_group=n_group)
    similarity = psiz.keras.layers.ExponentialSimilarity()
    kernel = psiz.keras.layers.AttentionKernel(
        attention=attention, similarity=similarity
    )

    # Build model.
    model = psiz.models.Rank(
        stimuli=stimuli, kernel=kernel
    )
    proxy = psiz.models.Proxy(model=model)

    # Set parameters.
    z = np.array((
        (.1, .1), (.15, .2), (.4, .5)
    ))
    attention = np.array((
        (1.2, .8),
        (.7, 1.3)
    ))
    proxy.z = z
    proxy.theta = {
        'rho': 2,
        'tau': 1,
        'beta': 10,
        'gamma': 0
    }
    proxy.w = attention

    return proxy


def test_pairwise_matrix(ground_truth):
    """Test similarity matrix."""
    actual_simmat1 = np.array((
        (1., 0.35035481, 0.00776613),
        (0.35035481, 1., 0.0216217),
        (0.00776613, 0.0216217, 1.)
    ))
    actual_simmat2 = np.array((
        (1., 0.29685964, 0.00548485),
        (0.29685964, 1., 0.01814493),
        (0.00548485, 0.01814493, 1.)
    ))

    computed_simmat0 = utils.pairwise_matrix(
        ground_truth.similarity, ground_truth.z)

    # Check explicit use of first set of attention weights.
    def similarity_func1(z_q, z_ref):
        sim_func = ground_truth.similarity(z_q, z_ref, group_id=0)
        return sim_func

    # Check without passing in explicit attention.
    computed_simmat1 = utils.pairwise_matrix(
        similarity_func1, ground_truth.z)

    # Check explicit use of second set of attention weights.
    def similarity_func2(z_q, z_ref):
        sim_func = ground_truth.similarity(z_q, z_ref, group_id=1)
        return sim_func

    computed_simmat2 = utils.pairwise_matrix(
        similarity_func2, ground_truth.z)

    np.testing.assert_array_almost_equal(actual_simmat1, computed_simmat0)
    np.testing.assert_array_almost_equal(actual_simmat1, computed_simmat1)
    np.testing.assert_array_almost_equal(actual_simmat2, computed_simmat2)


def test_matrix_comparison():
    """Test matrix correlation."""
    a = np.array((
        (1.0, .50, .90, .13),
        (.50, 1.0, .10, .80),
        (.90, .10, 1.0, .12),
        (.13, .80, .12, 1.0)
    ))

    b = np.array((
        (1.0, .45, .90, .11),
        (.45, 1.0, .20, .82),
        (.90, .20, 1.0, .02),
        (.11, .82, .02, 1.0)
    ))

    r2_score_1 = utils.matrix_comparison(a, b, score='r2')
    np.testing.assert_almost_equal(r2_score_1, 0.96723696)


def test_procrustean_solution():
    """Test procrustean solution for simple problem."""
    # Assemble rotation matrix (with scaling and reflection).
    s = np.array([[-2, 0], [0, 2]])
    r = psiz.utils.rotation_matrix(np.pi/4)
    r = np.matmul(s, r)

    # Assemble translation vector.
    t = np.array([-.8, 1])
    t = np.expand_dims(t, axis=0)

    # Create random set of points.
    x = np.random.rand(10, 2)
    # Apply affine transformation.
    y = np.matmul(x, r) + t
    # Attempt to recover original set of points.
    r_recov, t_recov = utils.procrustes_2d(x, y, n_restart=10)
    x_recov = np.matmul(y, r_recov) + t_recov

    np.testing.assert_almost_equal(x, x_recov, decimal=2)
