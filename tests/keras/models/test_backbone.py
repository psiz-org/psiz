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
def ds2_rank_rate_docket_2g():
    return None


@pytest.fixture(scope="module")
def ds2_rank_rate_obs_2g():
    n_trial = 8
    # TODO
    # stimulus_set = np.array(
    #     [
    #         # Rank trials.
    #         [1, 2, 3, 0, 0, 0, 0, 0, 0],
    #         [10, 13, 8, 0, 0, 0, 0, 0, 0],
    #         [4, 5, 6, 7, 8, 0, 0, 0, 0],
    #         [4, 5, 6, 7, 14, 15, 16, 17, 18],
    #         # Rate trials.
    #         [1, 2, 0, 0, 0, 0, 0, 0, 0],
    #         [10, 13, 0, 0, 0, 0, 0, 0, 0],
    #         [4, 5, 0, 0, 0, 0, 0, 0, 0],
    #         [4, 18, 0, 0, 0, 0, 0, 0, 0],
    #     ], dtype=np.int32
    # )

    # # Additional info for rank trials.
    # n_select = np.array((1, 1, 1, 2, 0, 0, 0, 0), dtype=np.int32)
    stimulus_set = np.array((
        (1, 2, 3, 0, 0, 0, 0, 0, 0),
        (10, 13, 8, 0, 0, 0, 0, 0, 0),
        (4, 5, 6, 7, 8, 0, 0, 0, 0),
        (4, 5, 6, 7, 14, 15, 16, 17, 18)
    ), dtype=np.int32)
    n_select = np.array((1, 1, 1, 2), dtype=np.int32)
    groups = np.array(([0], [0], [1], [1]), dtype=np.int32)
    obs_rank = psiz.trials.RankObservations(
        stimulus_set, n_select=n_select, groups=groups, mask_zero=True
    )

    # Rank data for dataset.
    stimulus_set_rank = obs_rank.all_outcomes()
    is_select_rank = np.expand_dims(obs_rank.is_select(compress=False), axis=2)
    y_rank = np.zeros([obs_rank.n_trial, stimulus_set_rank.shape[2]])
    y_rank[:, 0] = 1
    y_rank = np.concatenate([y_rank, np.zeros_like(y_rank)], axis=0)

    # Add on rate data.
    stimulus_set_rate = np.array(
        [
            [1, 2, 0, 0, 0, 0, 0, 0, 0],
            [10, 13, 0, 0, 0, 0, 0, 0, 0],
            [4, 5, 0, 0, 0, 0, 0, 0, 0],
            [4, 18, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    stimulus_set_rate = np.expand_dims(stimulus_set_rate, axis=2)
    stimulus_set_rate = np.repeat(stimulus_set_rate, 56, axis=2)
    stimulus_set = np.concatenate(
        [stimulus_set_rank, stimulus_set_rate], axis=0
    )
    # placeholder
    is_select_rank = np.concatenate(
        (is_select_rank, np.zeros_like(is_select_rank)), axis=0
    )

    groups = np.array(
        [
            [0, 0], [0, 0], [0, 0], [0, 0], [0, 1], [0, 1], [0, 1], [0, 1]
        ], dtype=np.int32
    )

    x = {
        'stimulus_set': stimulus_set,
        'is_select': is_select_rank,
        'groups': groups,
    }

    y_rate = np.array([0.0, 0.0, 0.0, 0.0, 0.1, .4, .8, .9])  # ratings
    y = {
        'rank_branch': tf.constant(y_rank, dtype=tf.float32),
        'rate_branch': tf.constant(y_rate, dtype=tf.float32)
    }

    # Define sample weights for each branch.
    w_rank = tf.constant([1., 1., 1., 1., 0., 0., 0., 0.])
    w_rate = tf.constant([0., 0., 0., 0., 1., 1., 1., 1.])
    w = {
        'rank_branch': w_rank,
        'rate_branch': w_rate
    }
    ds = tf.data.Dataset.from_tensor_slices((x, y, w)).batch(
        n_trial, drop_remainder=False
    )

    return ds


@pytest.fixture(scope="module")
def bb_rank_1g_1g_mle():
    """Backbone model: Rank, one group, MLE."""
    n_stimuli = 20
    n_dim = 3

    percept = tf.keras.layers.Embedding(
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
    rank = psiz.keras.layers.RankSimilarity(kernel=kernel)

    model = psiz.keras.models.Backbone(percept=percept, behavior=rank)
    return model


# TODO same as tests/keras/models/conftest:rank_2g_mle
@pytest.fixture(scope="module")
def bb_rank_1g_2g_mle():
    """A MLE rank model for two groups."""
    n_stimuli = 30
    n_dim = 10

    percept = tf.keras.layers.Embedding(
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
    kernel = psiz.keras.layers.BraidGate(
        subnets=[kernel_0, kernel_1], groups_subset=0
    )

    rank = psiz.keras.layers.RankSimilarity(kernel=kernel)

    model = psiz.keras.models.Backbone(percept=percept, behavior=rank)
    return model


@pytest.fixture(scope="module")
def bb_rank_1g_3g_mle():
    """Rank, three groups, MLE."""
    # TODO same as tests/keras/models/test_rank:rank_3g_mle_v2
    n_stimuli = 20
    n_dim = 3

    percept = tf.keras.layers.Embedding(
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
    kernel = psiz.keras.layers.BraidGate(
        subnets=[kernel_0, kernel_1, kernel_2], groups_subset=0
    )

    rank = psiz.keras.layers.RankSimilarity(kernel=kernel)

    model = psiz.keras.models.Backbone(percept=percept, behavior=rank)
    return model


@pytest.fixture(scope="module")
def bb_rank_2g_2g_mle():
    """Backbone model: """
    # TODO same as tests/conftest:rank_2stim_2kern_determ
    n_stimuli = 3
    n_dim = 2
    percept_0 = tf.keras.layers.Embedding(
        n_stimuli + 1, n_dim, mask_zero=True,
        embeddings_initializer=tf.keras.initializers.Constant(
            np.array(
                [
                    [0.0, 0.0], [.1, .1], [.15, .2], [.4, .5]
                ], dtype=np.float32
            )
        )
    )
    percept_1 = tf.keras.layers.Embedding(
        n_stimuli + 1, n_dim, mask_zero=True,
        embeddings_initializer=tf.keras.initializers.Constant(
            np.array(
                [
                    [0.0, 0.0], [.15, .2], [.4, .5], [.1, .1]
                ], dtype=np.float32
            )
        )
    )
    percept = psiz.keras.layers.BraidGate(
        subnets=[percept_0, percept_1], groups_subset=0
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
    kernel = psiz.keras.layers.BraidGate(
        subnets=[kernel_0, kernel_1], groups_subset=0
    )

    rank = psiz.keras.layers.RankSimilarity(kernel=kernel)

    model = psiz.keras.models.Backbone(percept=percept, behavior=rank)
    return model


@pytest.fixture(scope="module")
def bb_rank_2g_2g_2g_mle():
    """Backbone model."""
    # TODO same as tests/conftest:rank_2stim_2kern_2behav
    n_stimuli = 20
    n_dim = 2
    percept_0 = tf.keras.layers.Embedding(
        n_stimuli + 1, n_dim, mask_zero=True
    )
    percept_1 = tf.keras.layers.Embedding(
        n_stimuli + 1, n_dim, mask_zero=True
    )
    percept = psiz.keras.layers.BraidGate(
        subnets=[percept_0, percept_1], groups_subset=0
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
    kernel = psiz.keras.layers.BraidGate(
        subnets=[kernel_0, kernel_1], groups_subset=0
    )

    behavior_0 = psiz.keras.layers.RankSimilarity(kernel=kernel)
    behavior_1 = psiz.keras.layers.RankSimilarity(kernel=kernel)
    behavior = psiz.keras.layers.BraidGate(
        subnets=[behavior_0, behavior_1], groups_subset=0
    )

    model = psiz.keras.models.Backbone(percept=percept, behavior=behavior)
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
    percept = psiz.keras.layers.EmbeddingVariational(
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

    rank = psiz.keras.layers.RankSimilarity(kernel=kernel)

    model = psiz.keras.models.Backbone(
        percept=percept, behavior=rank
    )
    return model


@pytest.fixture(scope="module")
def bb_rate_1g_mle():
    """Backbone model: A MLE rate model."""
    n_stimuli = 30
    n_dim = 10

    percept = tf.keras.layers.Embedding(
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

    rate = psiz.keras.layers.RateSimilarity(kernel=kernel)

    model = psiz.keras.models.Backbone(percept=percept, behavior=rate)
    return model


@pytest.fixture(scope="module")
def bb_rank_rate_1g_mle():
    n_stimuli = 20
    n_dim = 3

    percept = tf.keras.layers.Embedding(
        n_stimuli + 1, n_dim, mask_zero=True
    )

    # Define a kernel that will be shared across behaviors.
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

    # Define a multi-behavior module
    rank = psiz.keras.layers.RankSimilarity(kernel=kernel)
    rate = psiz.keras.layers.RateSimilarity(kernel=kernel)
    behav_branch = psiz.keras.layers.BranchGate(
        subnets=[rank, rate], groups_subset=1, name="behav_branch",
        output_names=['rank_branch', 'rate_branch']
    )

    model = psiz.keras.models.Backbone(
        percept=percept, behavior=behav_branch
    )
    return model


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

        # Get coverage of Stochastic model.
        # TODO These tests probably belong somewhere else.
        # Test setting `n_sample`.
        model.n_sample = 2
        assert model.n_sample == 2

        # Test serialization.
        config = model.get_config()
        assert config['n_sample'] == 2
        recon_model = psiz.keras.models.Backbone.from_config(config)
        assert recon_model.n_sample == 2

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
        self, bb_rank_rate_1g_mle, ds2_rank_rate_docket_2g,
        ds2_rank_rate_obs_2g
    ):
        """Test rank and rate."""
        is_eager = True  # TODO
        tf.config.run_functions_eagerly(is_eager)

        model = bb_rank_rate_1g_mle
        # Compile
        compile_kwargs = {
            'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
            'loss': {
                'rank_branch': tf.keras.losses.CategoricalCrossentropy(
                    name='rank_loss'
                ),
                'rate_branch': tf.keras.losses.MeanSquaredError(
                    name='rate_loss'
                ),
            },
            'loss_weights': {'rank_branch': 1.0, 'rate_branch': 1.0},
        }
        model.compile(**compile_kwargs)
        # TODO one docket and one obs.
        call_fit_evaluate_predict(
            model, ds2_rank_rate_obs_2g, ds2_rank_rate_obs_2g
        )
