# -*- coding: utf-8 -*-
# Copyright 2018 The PsiZ Authors. All Rights Reserved.
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
# ==============================================================================

"""Module for testing models.py.

Todo
    - test init
    - test freeze and thaw
    - test general run
"""


import numpy as np
import pytest
import tensorflow as tf

from psiz.trials import UnjudgedTrials
from psiz.models import Exponential, HeavyTailed, StudentsT


@pytest.fixture(scope="module")
def ground_truth():
    """Return a ground truth embedding."""
    n_stimuli = 10
    dimensionality = 2

    model = Exponential(n_stimuli)
    mean = np.ones((dimensionality))
    cov = np.identity(dimensionality)
    Z = np.random.multivariate_normal(mean, cov, (n_stimuli))
    freeze_options = {
        'rho': 2,
        'tau': 1,
        'beta': 1,
        'gamma': 0,
        'Z': Z
    }
    model.freeze(freeze_options)
    return model


def test_private_exponential_similarity():
    """Test exponential similarity function."""
    n_stimuli = 10
    n_dim = 3
    model = Exponential(n_stimuli, dimensionality=n_dim)

    z_q_in = np.array((
        (.11, -.13, .28),
        (.45, .09, -1.45)
    ))

    z_ref_in = np.array((
        (.203, -.78, .120),
        (-.105, -.34, -.278)
    ))

    rho = tf.constant(1.9, dtype=tf.float32)
    tau = tf.constant(2.1, dtype=tf.float32)
    beta = tf.constant(1.11, dtype=tf.float32)
    gamma = tf.constant(.001, dtype=tf.float32)
    sim_params = {'rho': rho, 'tau': tau, 'gamma': gamma, 'beta': beta}

    attention_weights = tf.constant(1., shape=[2, 3])

    z_q = tf.placeholder(tf.float32, [None, n_dim], name='z_q')
    z_ref = tf.placeholder(tf.float32, [None, n_dim], name='z_ref')

    s = model._similarity(z_q, z_ref, sim_params, attention_weights)

    sess = tf.Session()
    s_actual = sess.run(
                s, feed_dict={
                    z_q: z_q_in,
                    z_ref: z_ref_in}
                )

    s_desired = np.array([0.60972816, 0.10853130])
    np.testing.assert_allclose(s_actual, s_desired)
