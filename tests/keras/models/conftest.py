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
def ds_rank_docket_2g():
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
        np.array([
            [0], [0], [1], [1],
        ])
    ).batch(n_trial, drop_remainder=False)

    return ds_docket


@pytest.fixture(scope="module")
def ds_rank_obs_2g():
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
    groups = np.array(([0], [0], [1], [1]), dtype=np.int32)

    obs = psiz.trials.RankObservations(
        stimulus_set, n_select=n_select, groups=groups
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
    stimuli = psiz.keras.layers.EmbeddingVariational(
        posterior=embedding_posterior, prior=embedding_prior,
        kl_weight=kl_weight, kl_n_sample=30
    )

    kernel = psiz.keras.layers.DistanceBased(
        distance=psiz.keras.layers.Minkowski(
            rho_initializer=tf.keras.initializers.Constant(2.),
            w_initializer=tf.keras.initializers.Constant(1.),
            trainable=False
        ),
        similarity=psiz.keras.layers.ExponentialSimilarity(
            trainable=False,
            beta_initializer=tf.keras.initializers.Constant(10.),
            tau_initializer=tf.keras.initializers.Constant(1.),
            gamma_initializer=tf.keras.initializers.Constant(0.),
        )
    )

    behavior = psiz.keras.layers.RankBehavior()
    model = psiz.keras.models.Rank(
        stimuli=stimuli, kernel=kernel, behavior=behavior
    )
    return model


@pytest.fixture(scope="module")
def rank_1g_mle():
    """A MLE rank model."""
    n_stimuli = 30
    n_dim = 10

    stimuli = tf.keras.layers.Embedding(
        n_stimuli+1, n_dim, mask_zero=True
    )

    kernel = psiz.keras.layers.DistanceBased(
        distance=psiz.keras.layers.Minkowski(
            rho_initializer=tf.keras.initializers.Constant(2.),
            w_initializer=tf.keras.initializers.Constant(1.),
            trainable=False
        ),
        similarity=psiz.keras.layers.ExponentialSimilarity(
            trainable=False,
            beta_initializer=tf.keras.initializers.Constant(10.),
            tau_initializer=tf.keras.initializers.Constant(1.),
            gamma_initializer=tf.keras.initializers.Constant(0.),
        )
    )

    behavior = psiz.keras.layers.RankBehavior()

    model = psiz.keras.models.Rank(
        stimuli=stimuli, kernel=kernel, behavior=behavior
    )
    return model


@pytest.fixture(scope="module")
def rank_2g_mle():
    """A MLE rank model for two groups."""
    n_stimuli = 30
    n_dim = 10
    n_group = 2

    stimuli = tf.keras.layers.Embedding(
        n_stimuli+1, n_dim, mask_zero=True
    )

    shared_similarity = psiz.keras.layers.ExponentialSimilarity(
        beta_initializer=tf.keras.initializers.Constant(1.),
        tau_initializer=tf.keras.initializers.Constant(1.),
        gamma_initializer=tf.keras.initializers.Constant(0.),
        trainable=False
    )
    # Define group-specific kernels.
    kernel_0 = psiz.keras.layers.DistanceBased(
        distance=psiz.keras.layers.Minkowski(
            rho_trainable=False,
            rho_initializer=tf.keras.initializers.Constant(2.),
            w_initializer=tf.keras.initializers.Constant(
                [1.2, 1., .1, .1, .1, .1, .1, .1, .1, .1]
            ),
        ),
        similarity=shared_similarity
    )

    kernel_1 = psiz.keras.layers.DistanceBased(
        distance=psiz.keras.layers.Minkowski(
            rho_trainable=False,
            rho_initializer=tf.keras.initializers.Constant(2.),
            w_initializer=tf.keras.initializers.Constant(
                [.1, .1, .1, .1, .1, .1, .1, .1, 1., 1.2]
            ),
        ),
        similarity=shared_similarity
    )

    kernel_group = psiz.keras.layers.GateMulti(
        subnets=[kernel_0, kernel_1], group_col=0
    )

    behavior = psiz.keras.layers.RankBehavior()

    model = psiz.keras.models.Rank(
        stimuli=stimuli, kernel=kernel_group, behavior=behavior,
        use_group_kernel=True
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
def ds_rate_docket_2g():
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
        np.array([
            [0], [0], [1], [1],
        ])
    ).batch(n_trial, drop_remainder=False)

    return ds_docket


@pytest.fixture(scope="module")
def ds_rate_obs_2g():
    """Rate observations dataset."""
    n_trial = 4
    stimulus_set = np.array((
        (0, 1),
        (9, 12),
        (3, 4),
        (3, 17)
    ), dtype=np.int32)
    rating = np.array([0.1, .4, .8, .9])
    groups = np.array(([0], [0], [1], [1]), dtype=np.int32)

    obs = psiz.trials.RateObservations(
        stimulus_set, rating, groups=groups
    )
    ds_obs = obs.as_dataset().batch(n_trial, drop_remainder=False)

    return ds_obs


@pytest.fixture(scope="module")
def rate_1g_mle():
    """A MLE rate model."""
    n_stimuli = 30
    n_dim = 10

    stimuli = tf.keras.layers.Embedding(
        n_stimuli+1, n_dim, mask_zero=True
    )

    kernel = psiz.keras.layers.DistanceBased(
        distance=psiz.keras.layers.Minkowski(
            rho_initializer=tf.keras.initializers.Constant(2.),
            w_initializer=tf.keras.initializers.Constant(1.),
            trainable=False
        ),
        similarity=psiz.keras.layers.ExponentialSimilarity(
            trainable=False,
            beta_initializer=tf.keras.initializers.Constant(3.),
            tau_initializer=tf.keras.initializers.Constant(1.),
            gamma_initializer=tf.keras.initializers.Constant(0.),
        )
    )

    behavior = psiz.keras.layers.RateBehavior()

    model = psiz.keras.models.Rate(
        stimuli=stimuli, kernel=kernel, behavior=behavior
    )
    return model


@pytest.fixture(scope="module")
def rate_2g_mle():
    """A MLE rate model with group-specific kernel."""
    n_stimuli = 30
    n_dim = 10
    n_group = 2

    stimuli = tf.keras.layers.Embedding(
        n_stimuli+1, n_dim, mask_zero=True
    )

    shared_similarity = psiz.keras.layers.ExponentialSimilarity(
        beta_initializer=tf.keras.initializers.Constant(1.),
        tau_initializer=tf.keras.initializers.Constant(1.),
        gamma_initializer=tf.keras.initializers.Constant(0.),
        trainable=False
    )
    # Define group-specific kernels.
    kernel_0 = psiz.keras.layers.DistanceBased(
        distance=psiz.keras.layers.Minkowski(
            rho_trainable=False,
            rho_initializer=tf.keras.initializers.Constant(2.),
            w_initializer=tf.keras.initializers.Constant(
                [1.2, 1., .1, .1, .1, .1, .1, .1, .1, .1]
            ),
        ),
        similarity=shared_similarity
    )

    kernel_1 = psiz.keras.layers.DistanceBased(
        distance=psiz.keras.layers.Minkowski(
            rho_trainable=False,
            rho_initializer=tf.keras.initializers.Constant(2.),
            w_initializer=tf.keras.initializers.Constant(
                [.1, .1, .1, .1, .1, .1, .1, .1, 1., 1.2]
            ),
        ),
        similarity=shared_similarity
    )

    kernel_group = psiz.keras.layers.GateMulti(
        subnets=[kernel_0, kernel_1], group_col=0
    )

    behavior = psiz.keras.layers.RateBehavior()

    model = psiz.keras.models.Rate(
        stimuli=stimuli, kernel=kernel_group, behavior=behavior,
        use_group_kernel=True
    )
    return model


@pytest.fixture(scope="module")
def rate_1g_vi():
    """A VI rate model."""
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
    stimuli = psiz.keras.layers.EmbeddingVariational(
        posterior=embedding_posterior, prior=embedding_prior,
        kl_weight=kl_weight, kl_n_sample=30
    )

    kernel = psiz.keras.layers.DistanceBased(
        distance=psiz.keras.layers.Minkowski(
            rho_initializer=tf.keras.initializers.Constant(2.),
            w_initializer=tf.keras.initializers.Constant(1.),
            trainable=False
        ),
        similarity=psiz.keras.layers.ExponentialSimilarity(
            trainable=False,
            beta_initializer=tf.keras.initializers.Constant(3.),
            tau_initializer=tf.keras.initializers.Constant(1.),
            gamma_initializer=tf.keras.initializers.Constant(0.),
        )
    )

    behavior = psiz.keras.layers.RateBehavior()

    model = psiz.keras.models.Rate(
        stimuli=stimuli, kernel=kernel, behavior=behavior
    )
    return model
