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
    - freeze and thaw        
    - general run
    - attention
    - heavy-tailed similarity
    - Student's t similarity
    - simple hard-coded example for similarity_matrix
"""


import numpy as np
import pytest
import tensorflow as tf

from psiz.trials import UnjudgedTrials
from psiz.models import Exponential, HeavyTailed, StudentsT
from psiz.simulate import Agent


@pytest.fixture(scope="module")
def ground_truth():
    """Return a ground truth embedding."""
    n_stimuli = 10
    n_dim = 2

    model = Exponential(n_stimuli)
    mean = np.ones((n_dim))
    cov = np.identity(n_dim)
    z = np.random.multivariate_normal(mean, cov, (n_stimuli))
    freeze_options = {
        'rho': 2,
        'tau': 1,
        'beta': 1,
        'gamma': 0,
        'z': z
    }
    model.freeze(freeze_options)
    return model


@pytest.fixture(scope="module")
def unjudged_trials():
    """Return a set of unjudged trials."""
    stimulus_set = np.array((
        (0, 1, 2, 7, 3),
        (3, 4, 5, 9, 1),
        (1, 8, 9, 2, -1),
        (7, 3, 2, 7, -1),
        (6, 7, 5, 0, -1),
        (2, 1, 0, 6, -1),
        (3, 0, 2, 6, -1),
    ))
    n_selected = np.array((
        2, 2, 2, 2, 1, 1, 1
        ), dtype=np.int64)
    unjudged_trials = UnjudgedTrials(stimulus_set, n_selected=n_selected)
    return unjudged_trials


def test_private_exponential_similarity():
    """Test exponential similarity function."""
    n_stimuli = 10
    n_dim = 3
    model = Exponential(n_stimuli, n_dim=n_dim)

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

    s = model._tf_similarity(z_q, z_ref, sim_params, attention_weights)

    sess = tf.Session()
    s_actual = sess.run(
        s, feed_dict={z_q: z_q_in, z_ref: z_ref_in}
    )
    sess.close()

    s_desired = np.array([0.60972816, 0.10853130])
    np.testing.assert_allclose(s_actual, s_desired)


def test_private_exponential_similarity_broadcast():
    """Test exponential similarity function with multiple references."""
    n_stimuli = 10
    n_dim = 3
    model = Exponential(n_stimuli, n_dim=n_dim)

    z_q_in = np.array((
        (.11, -.13, .28),
        (.45, .09, -1.45)
    ))
    z_q_in = np.expand_dims(z_q_in, axis=2)

    z_ref_0 = np.array((
        (.203, -.78, .120),
        (-.105, -.34, -.278)
    ))
    z_ref_1 = np.array((
        (.302, -.87, .021),
        (-.501, -.43, -.872)
    ))
    z_ref_in = np.stack((z_ref_0, z_ref_1), axis=2)

    attention_weights = tf.constant(1., shape=[2, 3, 1])

    rho = tf.constant(1.9, dtype=tf.float32)
    tau = tf.constant(2.1, dtype=tf.float32)
    beta = tf.constant(1.11, dtype=tf.float32)
    gamma = tf.constant(.001, dtype=tf.float32)
    sim_params = {'rho': rho, 'tau': tau, 'gamma': gamma, 'beta': beta}

    z_q = tf.placeholder(tf.float32, name='z_q')
    z_ref = tf.placeholder(tf.float32, name='z_ref')

    s = model._tf_similarity(z_q, z_ref, sim_params, attention_weights)

    sess = tf.Session()
    s_actual = sess.run(
        s, feed_dict={z_q: z_q_in, z_ref: z_ref_in}
    )
    sess.close()

    s_desired = np.array((
        [0.60972816, 0.48281544],
        [0.10853130, 0.16589911]
    ))
    # print('s_actual:', s_actual)
    # print('s_desired:', s_desired)
    np.testing.assert_allclose(s_actual, s_desired)


def test_public_exponential_similarity():
    """Test similarity function."""
    # Create Exponential model.
    n_stimuli = 10
    n_dim = 3
    model = Exponential(n_stimuli, n_dim=n_dim)
    freeze_options = {
        'rho': 1.9, 'tau': 2.1, 'beta': 1.11, 'gamma': .001
    }
    model.freeze(freeze_options)

    z_q = np.array((
        (.11, -.13, .28),
        (.45, .09, -1.45)
    ))
    z_ref = np.array((
        (.203, -.78, .120),
        (-.105, -.34, -.278)
    ))
    # TODO test both cases where attention_weights are and are not provided.
    # attention_weights = np.ones((2, 3))
    # s_actual = model.similarity(z_q, z_ref, attention_weights)
    s_actual = model.similarity(z_q, z_ref)
    s_desired = np.array([0.60972816, 0.10853130])
    np.testing.assert_allclose(s_actual, s_desired)


def test_public_exponential_similarity_broadcast():
    """Test similarity function."""
    # Create Exponential model.
    n_stimuli = 10
    n_dim = 3
    model = Exponential(n_stimuli, n_dim=n_dim)
    freeze_options = {
        'rho': 1.9, 'tau': 2.1, 'beta': 1.11, 'gamma': .001
    }
    model.freeze(freeze_options)

    z_q = np.array((
        (.11, -.13, .28),
        (.45, .09, -1.45)
    ))
    z_q = np.expand_dims(z_q, axis=2)

    z_ref_0 = np.array((
        (.203, -.78, .120),
        (-.105, -.34, -.278)
    ))
    z_ref_1 = np.array((
        (.302, -.87, .021),
        (-.501, -.43, -.872)
    ))
    z_ref = np.stack((z_ref_0, z_ref_1), axis=2)

    # TODO test both cases where attention_weights are and are not provided.
    # attention_weights = np.ones((2, 3))
    # s_actual = model.similarity(z_q, z_ref, attention_weights)
    s_actual = model.similarity(z_q, z_ref)
    s_desired = np.array((
        [0.60972816, 0.48281544],
        [0.10853130, 0.16589911]
    ))
    np.testing.assert_allclose(s_actual, s_desired)


def test_weight_projections():
    """Test projection of attention weights."""
    # Create Exponential model.
    n_stimuli = 10
    n_dim = 3
    model = Exponential(n_stimuli, n_dim=n_dim)
    attention = np.array(
        (
            (1., 1., 1.),
            (2., 1., 1.)
        ), ndmin=2
    )
    # attention = np.array(((2., 1., 1.)), ndmin=2) # TODO

    # Project attention weights.
    total = np.sum(attention, axis=1, keepdims=True)
    attention_desired = n_dim * attention / total

    tf_attention = tf.convert_to_tensor(
        attention, dtype=tf.float32
    )
    attention_actual_op = model._project_attention(tf_attention)
    sess = tf.Session()
    attention_actual = sess.run(attention_actual_op)
    sess.close()

    np.testing.assert_allclose(attention_actual, attention_desired)


def test_freeze():
    """Test freeze method."""
    n_stimuli = 10
    n_dim = 2
    n_group = 2
    model = Exponential(n_stimuli, n_dim, n_group)

    model.freeze({'z': np.ones((n_stimuli, n_dim))})
    with pytest.raises(Exception):
        model.freeze({'z': np.ones((n_stimuli-1, n_dim))})
    with pytest.raises(Exception):
        model.freeze({'z': np.ones((n_stimuli, n_dim-1))})
    
    model.freeze({'attention': np.ones((n_group, n_dim))})
    with pytest.raises(Exception):
        model.freeze({'attention': np.ones((n_group+1, n_dim))})
    with pytest.raises(Exception):
        model.freeze({'attention': np.ones((n_group, n_dim-1))})


def test_probability(ground_truth, unjudged_trials):
    """Test probability method."""
    (outcome_idx_list, prob) = ground_truth.probability(unjudged_trials)
    prob_actual = np.sum(prob, axis=1)
    prob_desired = np.ones((unjudged_trials.n_trial))
    np.testing.assert_allclose(prob_actual, prob_desired)


def test_tf_probability(ground_truth, unjudged_trials):
    """Test tf_probability method."""
    prob_desired = np.ones((unjudged_trials.n_trial))

    (outcome_idx_list, prob_1) = ground_truth.probability(unjudged_trials)
    prob_actual_1 = np.sum(prob_1, axis=1)

    z_tf = ground_truth.z['value']
    z_tf = tf.convert_to_tensor(
        z_tf, dtype=tf.float32
    )
    # TODO clean this up
    tf_theta = {}
    for param_name in ground_truth.theta:
        tf_theta[param_name] = tf.constant(
            ground_truth.theta[param_name]['value'], dtype=tf.float32)
    (outcome_idx_list, prob_2_tf) = ground_truth.tf_probability(
        unjudged_trials, z_tf, tf_theta)

    sess = tf.Session()
    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        prob_2 = prob_2_tf.eval()

    np.testing.assert_allclose(prob_actual_1, prob_desired)
    np.testing.assert_allclose(prob_1, prob_2, rtol=1e-6)