# -*- coding: utf-8 -*-
# Copyright 2022 The PsiZ Authors. All Rights Reserved.
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
"""Module for testing models.py."""

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

import psiz


def call_fit_evaluate_predict(model, ds2_docket, ds2_obs):
    """Simple test of call, fit, evaluate, and predict."""
    # Test call.
    for data in ds2_docket:
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        _ = model(x, training=False)

    # Test fit.
    model.fit(ds2_obs, epochs=2)

    # Test evaluate.
    model.evaluate(ds2_obs)

    # Test predict.
    model.predict(ds2_docket)


def build_mle_kernel(similarity, n_dim):
    """Build kernel for single group."""
    mink = psiz.keras.layers.Minkowski(
        rho_trainable=False,
        rho_initializer=tf.keras.initializers.Constant(2.),
        w_constraint=psiz.keras.constraints.NonNegNorm(
            scale=n_dim, p=1.
        ),
    )

    kernel = psiz.keras.layers.DistanceBased(
        distance=mink,
        similarity=similarity
    )
    return kernel


@pytest.fixture(scope="module")
def ds2_rank_docket_2g():
    """Rank docket dataset."""
    stimulus_set = np.array((
        (1, 2, 3, 0, 0, 0, 0, 0, 0),
        (10, 13, 8, 0, 0, 0, 0, 0, 0),
        (4, 5, 6, 7, 8, 0, 0, 0, 0),
        (4, 5, 6, 7, 14, 15, 16, 17, 18)
    ), dtype=np.int32)

    n_trial = 4
    n_select = np.array((1, 1, 1, 2), dtype=np.int32)
    docket = psiz.trials.RankDocket(
        stimulus_set, n_select=n_select, mask_zero=True
    )

    groups = np.array(([0], [0], [1], [1]), dtype=np.int32)
    ds2_docket = docket.as_dataset(
        groups=groups
    ).batch(n_trial, drop_remainder=False)

    return ds2_docket


@pytest.fixture(scope="module")
def ds2_rank_obs_2g():
    """Rank observations dataset."""
    stimulus_set = np.array((
        (1, 2, 3, 0, 0, 0, 0, 0, 0),
        (10, 13, 8, 0, 0, 0, 0, 0, 0),
        (4, 5, 6, 7, 8, 0, 0, 0, 0),
        (4, 5, 6, 7, 14, 15, 16, 17, 18)
    ), dtype=np.int32)

    n_trial = 4
    n_select = np.array((1, 1, 1, 2), dtype=np.int32)
    groups = np.array(([0], [0], [1], [1]), dtype=np.int32)

    obs = psiz.trials.RankObservations(
        stimulus_set, n_select=n_select, groups=groups, mask_zero=True
    )
    ds2_obs = obs.as_dataset().batch(n_trial, drop_remainder=False)

    return ds2_obs


@pytest.fixture(scope="module")
def ds_rank_obs_3g():
    """Rank observations dataset."""
    # TODO same as tests/keras/models/test_rank:ds_rank_obs_3g
    stimulus_set = np.array((
        (1, 2, 3, 0, 0, 0, 0, 0, 0),
        (10, 13, 8, 0, 0, 0, 0, 0, 0),
        (4, 5, 6, 7, 8, 0, 0, 0, 0),
        (4, 5, 6, 7, 14, 15, 16, 17, 18)
    ), dtype=np.int32)

    n_trial = 4
    n_select = np.array((1, 1, 1, 2), dtype=np.int32)
    groups = np.array(([0], [0], [1], [2]), dtype=np.int32)

    obs = psiz.trials.RankObservations(
        stimulus_set, n_select=n_select, groups=groups
    )
    ds_obs = obs.as_dataset().batch(n_trial, drop_remainder=False)

    return ds_obs


@pytest.fixture(scope="module")
def ds2_rate_docket_2g():
    """Rate docket dataset."""
    stimulus_set = np.array((
        (1, 2),
        (10, 13),
        (4, 5),
        (4, 18)
    ), dtype=np.int32)

    n_trial = 4
    docket = psiz.trials.RateDocket(stimulus_set)

    groups = np.array(([0], [0], [1], [1]), dtype=np.int32)
    ds2_docket = docket.as_dataset(
        groups=groups
    ).batch(n_trial, drop_remainder=False)

    return ds2_docket


@pytest.fixture(scope="module")
def ds2_rate_obs_2g():
    """Rate observations dataset."""
    n_trial = 4
    stimulus_set = np.array((
        (1, 2),
        (10, 13),
        (4, 5),
        (4, 18)
    ), dtype=np.int32)
    rating = np.array([0.1, .4, .8, .9])
    groups = np.array(([0], [0], [1], [1]), dtype=np.int32)

    obs = psiz.trials.RateObservations(
        stimulus_set, rating, groups=groups
    )
    ds2_obs = obs.as_dataset().batch(n_trial, drop_remainder=False)

    return ds2_obs


@pytest.fixture(scope="module")
def ds2_rank_rate_docket():
    return None


@pytest.fixture(scope="module")
def ds2_rank_rate_obs():
    return None


@pytest.fixture(scope="module")
def bb_rank_1g_1g_mle():
    """Backbone model: Rank, one group, MLE."""
    n_stimuli = 20
    n_dim = 3

    stimuli = tf.keras.layers.Embedding(
        n_stimuli + 1, n_dim, mask_zero=True
    )
    kernel = psiz.keras.layers.DistanceBased(
        distance=psiz.keras.layers.Minkowski(
            rho_initializer=tf.keras.initializers.Constant(2.),
            w_initializer=tf.keras.initializers.Constant(1.),
            trainable=False,
        ),
        similarity=psiz.keras.layers.ExponentialSimilarity(
            beta_initializer=tf.keras.initializers.Constant(10.),
            tau_initializer=tf.keras.initializers.Constant(1.),
            gamma_initializer=tf.keras.initializers.Constant(0.001),
            trainable=False,
        )
    )
    rank = psiz.keras.layers.RankSimilarity()

    model = psiz.keras.models.Backbone(
        stimuli=stimuli, kernel=kernel, behavior=rank
    )
    return model


@pytest.fixture(scope="module")
def bb_rank_1g_2g_mle():
    """A MLE rank model for two groups."""
    # TODO same as tests/keras/models/conftest:rank_2g_mle
    n_stimuli = 30
    n_dim = 10

    stimuli = tf.keras.layers.Embedding(
        n_stimuli + 1, n_dim, mask_zero=True
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

    kernel_group = psiz.keras.layers.BraidGate(
        subnets=[kernel_0, kernel_1], group_col=0
    )

    rank = psiz.keras.layers.RankSimilarity()

    model = psiz.keras.models.Backbone(
        stimuli=stimuli, kernel=kernel_group, behavior=rank
    )
    return model


@pytest.fixture(scope="module")
def bb_rank_1g_3g_mle():
    """Rank, three groups, MLE."""
    # TODO same as tests/keras/models/test_rank:rank_3g_mle_v2
    n_stimuli = 20
    n_dim = 3

    stimuli = tf.keras.layers.Embedding(
        n_stimuli + 1, n_dim, mask_zero=True,
    )

    shared_similarity = psiz.keras.layers.ExponentialSimilarity(
        trainable=False,
        beta_initializer=tf.keras.initializers.Constant(10.),
        tau_initializer=tf.keras.initializers.Constant(1.),
        gamma_initializer=tf.keras.initializers.Constant(0.)
    )

    # Define group-specific kernels.
    kernel_0 = build_mle_kernel(shared_similarity, n_dim)
    kernel_1 = build_mle_kernel(shared_similarity, n_dim)
    kernel_2 = build_mle_kernel(shared_similarity, n_dim)
    kernel_group = psiz.keras.layers.BraidGate(
        subnets=[kernel_0, kernel_1, kernel_2], group_col=0
    )

    rank = psiz.keras.layers.RankSimilarity()

    model = psiz.keras.models.Backbone(
        stimuli=stimuli, kernel=kernel_group, behavior=rank
    )
    return model


@pytest.fixture(scope="module")
def bb_rank_2g_2g_mle():
    """Backbone model: """
    # TODO same as tests/conftest:rank_2stim_2kern_determ
    n_stimuli = 3
    n_dim = 2
    stimuli_0 = tf.keras.layers.Embedding(
        n_stimuli + 1, n_dim, mask_zero=True,
        embeddings_initializer=tf.keras.initializers.Constant(
            np.array(
                [
                    [0.0, 0.0], [.1, .1], [.15, .2], [.4, .5]
                ], dtype=np.float32
            )
        )
    )

    stimuli_1 = tf.keras.layers.Embedding(
        n_stimuli + 1, n_dim, mask_zero=True,
        embeddings_initializer=tf.keras.initializers.Constant(
            np.array(
                [
                    [0.0, 0.0], [.15, .2], [.4, .5], [.1, .1]
                ], dtype=np.float32
            )
        )
    )

    stimuli_group = psiz.keras.layers.BraidGate(
        subnets=[stimuli_0, stimuli_1], group_col=0
    )

    shared_similarity = psiz.keras.layers.ExponentialSimilarity(
        trainable=False,
        beta_initializer=tf.keras.initializers.Constant(10.),
        tau_initializer=tf.keras.initializers.Constant(1.),
        gamma_initializer=tf.keras.initializers.Constant(0.)
    )

    # Define group-specific kernels.
    kernel_0 = psiz.keras.layers.DistanceBased(
        distance=psiz.keras.layers.Minkowski(
            rho_trainable=False,
            rho_initializer=tf.keras.initializers.Constant(2.),
            w_initializer=tf.keras.initializers.Constant(
                [1.2, .8]
            ),
            w_constraint=psiz.keras.constraints.NonNegNorm(
                scale=n_dim, p=1.
            ),
        ),
        similarity=shared_similarity
    )

    kernel_1 = psiz.keras.layers.DistanceBased(
        distance=psiz.keras.layers.Minkowski(
            rho_trainable=False,
            rho_initializer=tf.keras.initializers.Constant(2.),
            w_initializer=tf.keras.initializers.Constant(
                [.7, 1.3]
            ),
            w_constraint=psiz.keras.constraints.NonNegNorm(
                scale=n_dim, p=1.
            ),
        ),
        similarity=shared_similarity
    )

    kernel_group = psiz.keras.layers.BraidGate(
        subnets=[kernel_0, kernel_1], group_col=0
    )

    behavior = psiz.keras.layers.RankSimilarity()

    model = psiz.keras.models.Backbone(
        stimuli=stimuli_group, kernel=kernel_group, behavior=behavior
    )
    return model


@pytest.fixture(scope="module")
def bb_rank_2g_2g_2g_mle():
    """Backbone model."""
    # TODO same as tests/conftest:rank_2stim_2kern_2behav
    n_stimuli = 20
    n_dim = 2
    stimuli_0 = tf.keras.layers.Embedding(
        n_stimuli + 1, n_dim, mask_zero=True
    )

    stimuli_1 = tf.keras.layers.Embedding(
        n_stimuli + 1, n_dim, mask_zero=True
    )

    stimuli_group = psiz.keras.layers.BraidGate(
        subnets=[stimuli_0, stimuli_1], group_col=0
    )

    shared_similarity = psiz.keras.layers.ExponentialSimilarity(
        trainable=False,
        beta_initializer=tf.keras.initializers.Constant(10.),
        tau_initializer=tf.keras.initializers.Constant(1.),
        gamma_initializer=tf.keras.initializers.Constant(0.)
    )

    # Define group-specific kernels.
    kernel_0 = psiz.keras.layers.DistanceBased(
        distance=psiz.keras.layers.Minkowski(
            rho_trainable=False,
            rho_initializer=tf.keras.initializers.Constant(2.),
            w_initializer=tf.keras.initializers.Constant(
                [1.2, .8]
            ),
            w_constraint=psiz.keras.constraints.NonNegNorm(
                scale=n_dim, p=1.
            ),
        ),
        similarity=shared_similarity
    )

    kernel_1 = psiz.keras.layers.DistanceBased(
        distance=psiz.keras.layers.Minkowski(
            rho_trainable=False,
            rho_initializer=tf.keras.initializers.Constant(2.),
            w_initializer=tf.keras.initializers.Constant(
                [.7, 1.3]
            ),
            w_constraint=psiz.keras.constraints.NonNegNorm(
                scale=n_dim, p=1.
            ),
        ),
        similarity=shared_similarity
    )

    kernel_group = psiz.keras.layers.BraidGate(
        subnets=[kernel_0, kernel_1], group_col=0
    )

    behavior_0 = psiz.keras.layers.RankSimilarity()
    behavior_1 = psiz.keras.layers.RankSimilarity()
    behavior_group = psiz.keras.layers.BraidGate(
        subnets=[behavior_0, behavior_1], group_col=0
    )

    model = psiz.keras.models.Backbone(
        stimuli=stimuli_group, kernel=kernel_group, behavior=behavior_group
    )
    return model


@pytest.fixture(scope="module")
def bb_rank_1g_vi():
    """Backbone model: Rank, one group, VI."""
    n_stimuli = 20
    n_dim = 3
    kl_weight = .1

    prior_scale = .2
    embedding_posterior = psiz.keras.layers.EmbeddingNormalDiag(
        n_stimuli + 1, n_dim, mask_zero=True,
        scale_initializer=tf.keras.initializers.Constant(
            tfp.math.softplus_inverse(prior_scale).numpy()
        )
    )
    embedding_prior = psiz.keras.layers.EmbeddingShared(
        n_stimuli + 1, n_dim, mask_zero=True,
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

    mink = psiz.keras.layers.Minkowski(
        rho_initializer=tf.keras.initializers.Constant(2.),
        w_initializer=tf.keras.initializers.Constant(1.),
        trainable=False
    )
    kernel = psiz.keras.layers.DistanceBased(
        distance=mink,
        similarity=psiz.keras.layers.ExponentialSimilarity(
            trainable=False,
            beta_initializer=tf.keras.initializers.Constant(10.),
            tau_initializer=tf.keras.initializers.Constant(1.),
            gamma_initializer=tf.keras.initializers.Constant(0.),
        )
    )

    rank = psiz.keras.layers.RankSimilarity()

    model = psiz.keras.models.Backbone(
        stimuli=stimuli, kernel=kernel, behavior=rank
    )
    return model


@pytest.fixture(scope="module")
def bb_rate_1g_mle():
    """Backbone model: A MLE rate model."""
    n_stimuli = 30
    n_dim = 10

    stimuli = tf.keras.layers.Embedding(
        n_stimuli + 1, n_dim, mask_zero=True
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

    behavior = psiz.keras.layers.RateSimilarity()

    model = psiz.keras.models.Backbone(
        stimuli=stimuli, kernel=kernel, behavior=behavior
    )
    return model


@pytest.fixture(scope="module")
def bb_rank_rate_1g_mle():
    return None


class TestRankSimilarity:
    """Test w/ RankSimilarity behavior only."""

    @pytest.mark.parametrize(
        "is_eager", [True, False]
    )
    def test_mle_1g_1g(
        self, bb_rank_1g_1g_mle, ds2_rank_docket_2g, ds2_rank_obs_2g, is_eager
    ):
        """Test MLE, one group."""
        tf.config.run_functions_eagerly(is_eager)

        model = bb_rank_1g_1g_mle
        compile_kwargs = {
            'loss': tf.keras.losses.CategoricalCrossentropy(),
            'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
            'weighted_metrics': [
                tf.keras.metrics.CategoricalCrossentropy(name='cce')
            ]
        }
        model.compile(**compile_kwargs)
        call_fit_evaluate_predict(model, ds2_rank_docket_2g, ds2_rank_obs_2g)

    def test_mle_1g_2g(
        self, bb_rank_1g_2g_mle, ds2_rank_docket_2g, ds2_rank_obs_2g
    ):
        """Test MLE, 2 groups."""
        model = bb_rank_1g_2g_mle
        compile_kwargs = {
            'loss': tf.keras.losses.CategoricalCrossentropy(),
            'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
            'weighted_metrics': [
                tf.keras.metrics.CategoricalCrossentropy(name='cce')
            ]
        }
        model.compile(**compile_kwargs)
        call_fit_evaluate_predict(model, ds2_rank_docket_2g, ds2_rank_obs_2g)

    def test_mle_2g_2g(
        self, bb_rank_2g_2g_mle, ds2_rank_docket_2g, ds2_rank_obs_2g
    ):
        """Test MLE, 2 groups."""
        model = bb_rank_2g_2g_mle
        compile_kwargs = {
            'loss': tf.keras.losses.CategoricalCrossentropy(),
            'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
            'weighted_metrics': [
                tf.keras.metrics.CategoricalCrossentropy(name='cce')
            ]
        }
        model.compile(**compile_kwargs)
        call_fit_evaluate_predict(model, ds2_rank_docket_2g, ds2_rank_obs_2g)

    def test_mle_2g_2g_2g(
        self, bb_rank_2g_2g_2g_mle, ds2_rank_docket_2g, ds2_rank_obs_2g
    ):
        """Test MLE, 2 groups."""
        model = bb_rank_2g_2g_2g_mle
        compile_kwargs = {
            'loss': tf.keras.losses.CategoricalCrossentropy(),
            'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
            'weighted_metrics': [
                tf.keras.metrics.CategoricalCrossentropy(name='cce')
            ]
        }
        model.compile(**compile_kwargs)
        call_fit_evaluate_predict(model, ds2_rank_docket_2g, ds2_rank_obs_2g)

    @pytest.mark.parametrize(
        "is_eager", [True, False]
    )
    def test_vi_1g_1g(
        self, bb_rank_1g_vi, ds2_rank_docket_2g, ds2_rank_obs_2g, is_eager
    ):
        """Test VI, one group."""
        tf.config.run_functions_eagerly(is_eager)

        model = bb_rank_1g_vi

        # Compile
        compile_kwargs = {
            'loss': tf.keras.losses.CategoricalCrossentropy(),
            'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
            'weighted_metrics': [
                tf.keras.metrics.CategoricalCrossentropy(name='cce')
            ]
        }
        model.compile(**compile_kwargs)
        call_fit_evaluate_predict(model, ds2_rank_docket_2g, ds2_rank_obs_2g)


class TestRateSimilarity:
    """Test w/ RateSimilarity behavior only."""

    def test_rate(self, bb_rate_1g_mle, ds2_rate_docket_2g, ds2_rate_obs_2g):
        """Test rate behavior only."""
        # TODO
        is_eager = True
        tf.config.run_functions_eagerly(is_eager)

        model = bb_rate_1g_mle
        # Compile
        compile_kwargs = {
            'loss': tf.keras.losses.MeanSquaredError(),
            'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
            'weighted_metrics': [
                tf.keras.metrics.MeanSquaredError(name='mse')
            ]
        }
        model.compile(**compile_kwargs)
        call_fit_evaluate_predict(model, ds2_rate_docket_2g, ds2_rate_obs_2g)


class TestMultiBehavior:
    """Test with multiple types of behavior."""

    def test_rank_rate(
        self, bb_rank_rate_1g_mle, ds2_rank_rate_docket, ds2_rank_rate_obs
    ):
        """Test rank and rate."""
