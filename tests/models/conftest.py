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
"""Fixtures for models."""

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

import psiz


@pytest.fixture(scope="module")
def ds_rank_docket():
    """Rank docket dataset."""
    stimulus_set = np.array((
        (0, 1, 2, -1, -1, -1, -1, -1, -1),
        (9, 12, 7, -1, -1, -1, -1, -1, -1),
        (3, 4, 5, 6, 7, -1, -1, -1, -1),
        (3, 4, 5, 6, 13, 14, 15, 16, 17)
    ), dtype=np.int32)

    n_trial = 4
    n_select = np.array((1, 1, 1, 2), dtype=np.int32)
    docket = psiz.trials.RankDocket(stimulus_set, n_select=n_select)

    ds_docket = docket.as_dataset(
        np.zeros([n_trial, 1])
    ).batch(n_trial, drop_remainder=False)

    return ds_docket


@pytest.fixture(scope="module")
def ds_rank_obs():
    """Rank observations dataset."""
    stimulus_set = np.array((
        (0, 1, 2, -1, -1, -1, -1, -1, -1),
        (9, 12, 7, -1, -1, -1, -1, -1, -1),
        (3, 4, 5, 6, 7, -1, -1, -1, -1),
        (3, 4, 5, 6, 13, 14, 15, 16, 17)
    ), dtype=np.int32)

    n_trial = 4
    n_select = np.array((1, 1, 1, 2), dtype=np.int32)
    n_reference = np.array((2, 2, 4, 8), dtype=np.int32)
    is_ranked = np.array((True, True, True, True))
    group_id = np.array((0, 0, 1, 1), dtype=np.int32)

    obs = psiz.trials.RankObservations(
        stimulus_set, n_select=n_select, group_id=group_id
    )
    ds_obs = obs.as_dataset().batch(n_trial, drop_remainder=False)

    return ds_obs


@pytest.fixture(scope="module")
def rank_1g_vi():
    n_stimuli = 30
    n_dim = 10
    kl_weight = 0.

    embedding_posterior = psiz.keras.layers.EmbeddingNormalDiag(
        n_stimuli+1, n_dim, mask_zero=True,
        scale_initializer=tf.keras.initializers.Constant(
            tfp.math.softplus_inverse(.01).numpy()
        )
    )

    prior_scale = .2
    embedding_prior = psiz.keras.layers.EmbeddingShared(
        n_stimuli+1, n_dim, mask_zero=True,
        embedding=psiz.keras.layers.EmbeddingNormalDiag(
            1, 1,
            loc_initializer=tf.keras.initializers.Constant(0.),
            scale_initializer=tf.keras.initializers.Constant(
                tfp.math.softplus_inverse(prior_scale).numpy()
            ),
            loc_trainable=False,
        )
    )
    embedding_variational = psiz.keras.layers.EmbeddingVariational(
        posterior=embedding_posterior, prior=embedding_prior,
        kl_weight=kl_weight, kl_n_sample=30
    )
    stimuli = psiz.keras.layers.Stimuli(embedding=embedding_variational)

    kernel = psiz.keras.layers.Kernel(
        distance=psiz.keras.layers.WeightedMinkowski(
            rho_initializer=tf.keras.initializers.Constant(2.),
            trainable=False,
        ),
        similarity=psiz.keras.layers.ExponentialSimilarity(
            fit_tau=False, fit_gamma=False,
            tau_initializer=tf.keras.initializers.Constant(1.),
            gamma_initializer=tf.keras.initializers.Constant(0.),
        )
    )

    behavior = psiz.keras.layers.RankBehavior()
    model = psiz.models.Rank(
        stimuli=stimuli, kernel=kernel, behavior=behavior
    )
    return model


@pytest.fixture(scope="module")
def rank_1g_mle():
    """A MLE rank model."""
    n_stimuli = 30
    n_dim = 10

    stimuli = psiz.keras.layers.Stimuli(
        embedding=psiz.keras.layers.EmbeddingDeterministic(
            n_stimuli+1, n_dim, mask_zero=True
        )
    )

    kernel = psiz.keras.layers.Kernel(
        distance=psiz.keras.layers.WeightedMinkowski(
            rho_initializer=tf.keras.initializers.Constant(2.),
            trainable=False,
        ),
        similarity=psiz.keras.layers.ExponentialSimilarity(
            fit_tau=False, fit_gamma=False,
            tau_initializer=tf.keras.initializers.Constant(1.),
            gamma_initializer=tf.keras.initializers.Constant(0.),
        )
    )

    behavior = psiz.keras.layers.RankBehavior()

    model = psiz.models.Rank(
        stimuli=stimuli, kernel=kernel, behavior=behavior
    )
    return model


@pytest.fixture(scope="module")
def ds_rate_docket():
    """Rate docket dataset."""
    stimulus_set = np.array((
        (0, 1),
        (9, 12),
        (3, 4),
        (3, 17)
    ), dtype=np.int32)

    n_trial = 4
    docket = psiz.trials.RateDocket(stimulus_set)

    ds_docket = docket.as_dataset(
        np.zeros([n_trial, 1])
    ).batch(n_trial, drop_remainder=False)

    return ds_docket


@pytest.fixture(scope="module")
def ds_rate_obs():
    """Rate observations dataset."""
    n_trial = 4
    stimulus_set = np.array((
        (0, 1),
        (9, 12),
        (3, 4),
        (3, 17)
    ), dtype=np.int32)
    rating = np.array([0.1, .4, .8, .9])
    group_id = np.array((0, 0, 1, 1), dtype=np.int32)

    obs = psiz.trials.RateObservations(
        stimulus_set, rating, group_id=group_id
    )
    ds_obs = obs.as_dataset().batch(n_trial, drop_remainder=False)

    return ds_obs


@pytest.fixture(scope="module")
def rate_1g_mle():
    """A MLE rate model."""
    n_stimuli = 30
    n_dim = 10

    stimuli = psiz.keras.layers.Stimuli(
        embedding=psiz.keras.layers.EmbeddingDeterministic(
            n_stimuli+1, n_dim, mask_zero=True
        )
    )

    kernel = psiz.keras.layers.Kernel(
        distance=psiz.keras.layers.WeightedMinkowski(
            rho_initializer=tf.keras.initializers.Constant(2.),
            trainable=False,
        ),
        similarity=psiz.keras.layers.ExponentialSimilarity(
            fit_tau=False, fit_gamma=False, fit_beta=False,
            beta_initializer=tf.keras.initializers.Constant(3.),
            tau_initializer=tf.keras.initializers.Constant(1.),
            gamma_initializer=tf.keras.initializers.Constant(0.),
        )
    )

    behavior = psiz.keras.layers.RateBehavior()

    model = psiz.models.Rate(
        stimuli=stimuli, kernel=kernel, behavior=behavior
    )
    return model
