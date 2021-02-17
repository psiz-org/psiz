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
"""Module for testing simulate.py.

Todo:
    - test outcome_idx_list
    - test resultant groups of judged trials
    - test simulate and _select for a Trial object with different
        configurations.

"""

import numpy as np
import numpy.ma as ma
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

import psiz


@pytest.fixture(scope="module")
def rank_1g_mle_rand():
    n_stimuli = 10
    n_dim = 2

    stimuli = psiz.keras.layers.EmbeddingNormalDiag(
        n_stimuli+1, n_dim, mask_zero=True,
        loc_initializer=tf.keras.initializers.Constant(
            1.
        ),
        scale_initializer=tf.keras.initializers.Constant(
            tfp.math.softplus_inverse(1.).numpy()
        )
    )

    kernel = psiz.keras.layers.DistanceBased(
        distance=psiz.keras.layers.Minkowski(
            rho_initializer=tf.keras.initializers.Constant(2.),
            w_initializer=tf.keras.initializers.Constant(1.),
            trainable=False
        ),
        similarity=psiz.keras.layers.ExponentialSimilarity(
            trainable=False,
            beta_initializer=tf.keras.initializers.Constant(1.),
            tau_initializer=tf.keras.initializers.Constant(1.),
            gamma_initializer=tf.keras.initializers.Constant(0.001),
        )
    )

    behavior = psiz.keras.layers.RankBehavior()
    model = psiz.keras.models.Rank(
        stimuli=stimuli, kernel=kernel, behavior=behavior
    )
    return model


@pytest.fixture(scope="module")
def docket_0():
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
    n_select = np.array((
        2, 2, 2, 2, 1, 1, 1
        ), dtype=np.int32)
    docket = psiz.trials.RankDocket(stimulus_set, n_select=n_select)
    return docket


def test_simulate(rank_1g_mle_rand, docket_0):
    """Test simulation of agent."""
    agent = psiz.agents.RankAgent(rank_1g_mle_rand)
    obs = agent.simulate(docket_0)
    np.testing.assert_array_equal(
        obs.stimulus_set[:, 0], docket_0.stimulus_set[:, 0]
    )


def test_rank_sample(rank_1g_mle_rand):
    """Test _rank_sample method."""
    # TODO broader method test
    # stimulus_set = np.array((
    #     (0, 1, 2, 7, 3),
    #     (3, 4, 5, 9, 1),
    #     (1, 8, 9, 2, -1),
    #     (7, 3, 2, 7, -1),
    #     (6, 7, 5, 0, -1),
    #     (2, 1, 0, 6, -1),
    # ))
    # n_select = np.array((
    #     2, 2, 2, 2, 1, 1
    #     ), dtype=np.int32)

    n_trial = 100000
    stimulus_set = np.array((
        (0, 1, 2, 7, 3),
        (3, 4, 5, 9, 1),
    ))
    stimulus_set = np.tile(stimulus_set, (int(n_trial/2), 1))
    n_select = 2 * np.ones(n_trial, dtype=np.int32)
    docket = psiz.trials.RankDocket(stimulus_set, n_select=n_select)

    agent = psiz.agents.RankAgent(rank_1g_mle_rand)
    probs = np.array((
        (.01, .01, .01, .01, .01, .8, .1, .01, .01, .01, .01, .01),
        (.01, .01, .01, .01, .01, .8, .1, .01, .01, .01, .01, .01),
    ))
    probs = np.tile(probs, (int(n_trial/2), 1))
    # probs = np.array((
    #     (.01, .01, .01, .01, .01, .8, .1, .01, .01, .01, .01, .01),
    #     (.01, .01, .01, .01, .01, .8, .1, .01, .01, .01, .01, .01),
    #     (.1, .8, .07, .01, .01, .01, -1, -1, -1, -1, -1, -1),
    #     (.1, .8, .07, .01, .01, .01, -1, -1, -1, -1, -1, -1),
    #     (.1, .8, .1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    #     (.1, .8, .1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    #     (.1, .8, .1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    # ))
    outcome_distribution = tfp.distributions.Categorical(
        probs=probs
    )
    chosen_outcome_idx = outcome_distribution.sample().numpy()

    # (_, chosen_outcome_idx) = agent._rank_sample(
    #     docket.all_outcomes(), prob_all
    # )
    _, counts = np.unique(chosen_outcome_idx, return_counts=True)
    prop = counts / np.sum(counts)

    prop_desired = np.array(
        [.01, .01, .01, .01, .01, .8, .1, .01, .01, .01, .01, .01]
    )
    np.testing.assert_allclose(prop_desired, prop, rtol=1e-6, atol=.005)
