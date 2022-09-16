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
from psiz.data.outcomes.continuous import Continuous
from psiz.data.contents.rank_similarity import (
    RankSimilarity
)
from psiz.data.contents.rate_similarity import (
    RateSimilarity
)
from psiz.data.outcomes.sparse_categorical import (
    SparseCategorical
)
from psiz.data.trial_dataset import TrialDataset


def call_fit_evaluate_predict(model, ds2_obs):
    """Simple test of call, fit, evaluate, and predict."""
    # Test isolated call.
    for data in ds2_obs:
        x, _, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        _ = model(x, training=False)

    # Test fit.
    model.fit(ds2_obs, epochs=3)

    # Test evaluate.
    model.evaluate(ds2_obs)

    # Test predict.
    model.predict(ds2_obs)


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
def ds2_rank_2g():
    """Rank observations dataset."""
    n_sequence = 4
    stimulus_set = np.array((
        (1, 2, 3, 0, 0, 0, 0, 0, 0),
        (10, 13, 8, 0, 0, 0, 0, 0, 0),
        (4, 5, 6, 7, 8, 0, 0, 0, 0),
        (4, 5, 6, 7, 14, 15, 16, 17, 18)
    ), dtype=np.int32)

    n_select = np.array((1, 1, 1, 2), dtype=np.int32)
    groups = np.array(([0], [0], [1], [1]), dtype=np.int32)

    content = RankSimilarity(stimulus_set, n_select=n_select)
    outcome_idx = np.zeros(
        [content.n_sequence, content.max_timestep], dtype=np.int32
    )
    outcome = SparseCategorical(outcome_idx, depth=content.max_outcome)
    ds = TrialDataset(content, outcome=outcome, groups=groups).export()
    ds = ds.batch(n_sequence, drop_remainder=False)

    return ds


@pytest.fixture(scope="module")
def ds2_rank_3g():
    """Rank observations dataset."""
    # TODO copied from tests/keras/models/test_rank:ds_rank_obs_3g
    n_sequence = 4
    stimulus_set = np.array((
        (1, 2, 3, 0, 0, 0, 0, 0, 0),
        (10, 13, 8, 0, 0, 0, 0, 0, 0),
        (4, 5, 6, 7, 8, 0, 0, 0, 0),
        (4, 5, 6, 7, 14, 15, 16, 17, 18)
    ), dtype=np.int32)

    n_select = np.array((1, 1, 1, 2), dtype=np.int32)
    groups = np.array(([0], [0], [1], [2]), dtype=np.int32)

    content = RankSimilarity(stimulus_set, n_select=n_select)
    outcome_idx = np.zeros(
        [content.n_sequence, content.max_timestep], dtype=np.int32
    )
    outcome = SparseCategorical(outcome_idx, depth=content.max_outcome)
    ds = TrialDataset(content, outcome=outcome, groups=groups).export()
    ds = ds.batch(n_sequence, drop_remainder=False)

    return ds


@pytest.fixture(scope="module")
def ds2_rate_2g():
    """Rate observations dataset."""
    n_sequence = 4
    stimulus_set = np.array((
        (1, 2),
        (10, 13),
        (4, 5),
        (4, 18)
    ), dtype=np.int32)
    rating = np.array([[0.1], [.4], [.8], [.9]])
    groups = np.array(([0], [0], [1], [1]), dtype=np.int32)

    content = RateSimilarity(stimulus_set)
    outcome = Continuous(rating)
    ds = TrialDataset(content, outcome=outcome, groups=groups).export()
    ds = ds.batch(n_sequence, drop_remainder=False)
    return ds


@pytest.fixture(scope="module")
def ds2_rank_rate_2g():
    n_trial = 8

    # Rank data for dataset.
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
    y_rank = np.expand_dims(y_rank, axis=1)

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
        'stimulus_set': np.expand_dims(stimulus_set, axis=1),
        'is_select': np.expand_dims(is_select_rank, axis=1),
        'groups': np.expand_dims(groups, axis=1),
    }

    y_rate = np.array([0.0, 0.0, 0.0, 0.0, 0.1, .4, .8, .9])  # ratings
    y_rate = np.expand_dims(y_rate, axis=1)
    y = {
        'rank_branch': tf.constant(y_rank, dtype=tf.float32),
        'rate_branch': tf.constant(y_rate, dtype=tf.float32)
    }

    # Define sample weights for each branch.
    w_rank = tf.constant([[1.], [1.], [1.], [1.], [0.], [0.], [0.], [0.]])
    w_rate = tf.constant([[0.], [0.], [0.], [0.], [1.], [1.], [1.], [1.]])
    w = {
        'rank_branch': w_rank,
        'rate_branch': w_rate
    }
    ds = tf.data.Dataset.from_tensor_slices((x, y, w)).batch(
        n_trial, drop_remainder=False
    )

    return ds


@pytest.fixture(scope="module")
def ds2_categorize_2g():
    """A dummy dataset for categorize behavior.

    Assumes:
    * `mask_zero=True`
    * 20 stimuli (not including 0 index)
    * Three classes
        * class 0: 1-10
        * class 1: 11-15
        * class 0: 16-20

    """
    n_sequence = 4
    sequence_length = 10
    # n_stimuli = 20
    n_output = 3

    groups = np.zeros([n_sequence, sequence_length, 1])

    stimulus_set = np.array(
        [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
            [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        ]
    )
    y = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            [0, 0, 0, 0, 0, 1, 1, 1, 2, 2],
            [0, 0, 0, 0, 0, 1, 1, 2, 2, 2],
        ]
    )
    y_onehot = tf.one_hot(
        y, n_output, on_value=1.0, off_value=0.0
    )
    x = {
        'stimulus_set': tf.constant(stimulus_set),
        'correct_label': tf.constant(np.expand_dims(y, axis=2)),
        'groups': tf.constant(groups),
    }
    w = tf.constant(np.ones_like(y))
    ds2_obs = tf.data.Dataset.from_tensor_slices((x, y_onehot, w)).batch(
        n_sequence, drop_remainder=False
    )
    return ds2_obs


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
    rank_cell = psiz.keras.layers.RankSimilarityCellV2(
        percept=percept, kernel=kernel
    )
    rank = tf.keras.layers.RNN(rank_cell, return_sequences=True)

    model = psiz.keras.models.BackboneV2(net=rank)

    compile_kwargs = {
        'loss': tf.keras.losses.CategoricalCrossentropy(),
        'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
        'weighted_metrics': [
            tf.keras.metrics.CategoricalCrossentropy(name='cce')
        ]
    }
    model.compile(**compile_kwargs)
    return model


@pytest.fixture(scope="module")
def bb_rank_1g_2g_mle():
    """A MLE rank model for two groups."""
    # TODO copied from tests/keras/models/conftest:rank_2g_mle
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

    rank_cell = psiz.keras.layers.RankSimilarityCellV2(
        percept=percept, kernel=kernel
    )
    rank = tf.keras.layers.RNN(rank_cell, return_sequences=True)

    model = psiz.keras.models.BackboneV2(net=rank)
    compile_kwargs = {
        'loss': tf.keras.losses.CategoricalCrossentropy(),
        'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
        'weighted_metrics': [
            tf.keras.metrics.CategoricalCrossentropy(name='cce')
        ]
    }
    model.compile(**compile_kwargs)
    return model


@pytest.fixture(scope="module")
def bb_rank_1g_3g_mle():
    """Rank, three groups, MLE."""
    # TODO copied from tests/keras/models/test_rank:rank_3g_mle_v2
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

    rank_cell = psiz.keras.layers.RankSimilarityCellV2(kernel=kernel)
    rank = tf.keras.layers.RNN(rank_cell, return_sequences=True)

    model = psiz.keras.models.BackboneV2(percept=percept, behavior=rank)
    return model


@pytest.fixture(scope="module")
def bb_rank_2g_2g_mle():
    """Backbone model: """
    # TODO copied from tests/conftest:rank_2stim_2kern_determ
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

    # Define group-specific kernels.
    shared_similarity = psiz.keras.layers.ExponentialSimilarity(
        trainable=False,
        beta_initializer=tf.keras.initializers.Constant(10.),
        tau_initializer=tf.keras.initializers.Constant(1.),
        gamma_initializer=tf.keras.initializers.Constant(0.)
    )
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

    rank_cell = psiz.keras.layers.RankSimilarityCellV2(
        percept=percept, kernel=kernel
    )
    rank = tf.keras.layers.RNN(rank_cell, return_sequences=True)

    model = psiz.keras.models.BackboneV2(net=rank)

    compile_kwargs = {
        'loss': tf.keras.losses.CategoricalCrossentropy(),
        'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
        'weighted_metrics': [
            tf.keras.metrics.CategoricalCrossentropy(name='cce')
        ]
    }
    model.compile(**compile_kwargs)
    return model


@pytest.fixture(scope="module")
def bb_rank_2g_2g_2g_mle():
    """Backbone model."""
    # TODO copied from tests/conftest:rank_2stim_2kern_2behav
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

    # Define group-specific behavior cells.
    behavior_cell_0 = psiz.keras.layers.RankSimilarityCellV2(
        percept=percept, kernel=kernel
    )
    behavior_cell_1 = psiz.keras.layers.RankSimilarityCellV2(
        percept=percept, kernel=kernel
    )
    behavior_0 = tf.keras.layers.RNN(behavior_cell_0, return_sequences=True)
    behavior_1 = tf.keras.layers.RNN(behavior_cell_1, return_sequences=True)
    behavior = psiz.keras.layers.BraidGate(
        subnets=[behavior_0, behavior_1], groups_subset=0
    )

    model = psiz.keras.models.BackboneV2(net=behavior)

    compile_kwargs = {
        'loss': tf.keras.losses.CategoricalCrossentropy(),
        'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
        'weighted_metrics': [
            tf.keras.metrics.CategoricalCrossentropy(name='cce')
        ]
    }
    model.compile(**compile_kwargs)
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

    rank_cell = psiz.keras.layers.RankSimilarityCellV2(
        percept=percept, kernel=kernel
    )
    rank = tf.keras.layers.RNN(rank_cell, return_sequences=True)

    model = psiz.keras.models.BackboneV2(net=rank)

    compile_kwargs = {
        'loss': tf.keras.losses.CategoricalCrossentropy(),
        'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
        'weighted_metrics': [
            tf.keras.metrics.CategoricalCrossentropy(name='cce')
        ]
    }
    model.compile(**compile_kwargs)
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

    rate_cell = psiz.keras.layers.RateSimilarityCellV2(
        percept=percept, kernel=kernel
    )
    rate = tf.keras.layers.RNN(rate_cell, return_sequences=True)

    model = psiz.keras.models.BackboneV2(net=rate)

    compile_kwargs = {
        'loss': tf.keras.losses.MeanSquaredError(),
        'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
        'weighted_metrics': [
            tf.keras.metrics.MeanSquaredError(name='mse')
        ]
    }
    model.compile(**compile_kwargs)
    return model


@pytest.fixture(scope="module")
def bb_categorize_1g_1g_mle():
    """Backbone model: Rank, one group, MLE."""
    n_stimuli = 20
    n_dim = 4
    n_output = 3

    alcove_embedding = tf.keras.layers.Embedding(
        n_stimuli + 1, n_dim, mask_zero=True,
        trainable=False,
    )

    similarity = psiz.keras.layers.ExponentialSimilarity(
        beta_initializer=tf.keras.initializers.Constant(3.0),
        tau_initializer=tf.keras.initializers.Constant(1.0),
        gamma_initializer=tf.keras.initializers.Constant(0.0),
        trainable=False,
    )

    cell = psiz.keras.layers.ALCOVECellV2(
        n_output, percept=alcove_embedding, similarity=similarity,
        rho_initializer=tf.keras.initializers.Constant(2.0),
        temperature_initializer=tf.keras.initializers.Constant(1.0),
        lr_attention_initializer=tf.keras.initializers.Constant(.03),
        lr_association_initializer=tf.keras.initializers.Constant(.03),
        trainable=False
    )
    categorize = tf.keras.layers.RNN(
        cell, return_sequences=True, stateful=False
    )

    model = psiz.keras.models.BackboneV2(net=categorize)

    compile_kwargs = {
        'loss': tf.keras.losses.CategoricalCrossentropy(),
        'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
        'weighted_metrics': [
            tf.keras.metrics.CategoricalAccuracy(name='accuracy')
        ]
    }
    model.compile(**compile_kwargs)
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
    rank_cell = psiz.keras.layers.RankSimilarityCellV2(
        percept=percept, kernel=kernel
    )
    rank = tf.keras.layers.RNN(rank_cell, return_sequences=True)
    rate_cell = psiz.keras.layers.RateSimilarityCellV2(
        percept=percept, kernel=kernel
    )
    rate = tf.keras.layers.RNN(rate_cell, return_sequences=True)
    behav_branch = psiz.keras.layers.BranchGate(
        subnets=[rank, rate], groups_subset=1, name="behav_branch",
        output_names=['rank_branch', 'rate_branch']
    )

    model = psiz.keras.models.BackboneV2(net=behav_branch)

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
    return model


class TestRankSimilarity:
    """Test w/ RankSimilarity behavior only."""

    def test_n_sample_and_serialization(self, bb_rank_1g_1g_mle):
        """Test n_sample attribute and serialization."""
        model = bb_rank_1g_1g_mle

        # Get coverage of Stochastic model.
        # Test setting `n_sample`.
        model.n_sample = 2
        assert model.n_sample == 2
        # TODO add test assertion for sub layers

        # Test serialization.
        config = model.get_config()
        assert config['n_sample'] == 2
        recon_model = psiz.keras.models.BackboneV2.from_config(config)
        assert recon_model.n_sample == 2

    @pytest.mark.parametrize(
        "is_eager", [True, False]
    )
    def test_mle_1g_1g(self, bb_rank_1g_1g_mle, ds2_rank_2g, is_eager):
        """Test MLE, one group."""
        tf.config.run_functions_eagerly(is_eager)

        model = bb_rank_1g_1g_mle
        call_fit_evaluate_predict(model, ds2_rank_2g)

    @pytest.mark.parametrize(
        "is_eager", [True, False]
    )
    def test_mle_1g_2g(self, bb_rank_1g_2g_mle, ds2_rank_2g, is_eager):
        """Test MLE, 2 groups."""
        tf.config.run_functions_eagerly(is_eager)
        model = bb_rank_1g_2g_mle
        call_fit_evaluate_predict(model, ds2_rank_2g)

    def test_mle_2g_2g(self, bb_rank_2g_2g_mle, ds2_rank_2g):
        """Test MLE, 2 groups."""
        model = bb_rank_2g_2g_mle
        call_fit_evaluate_predict(model, ds2_rank_2g)

    def test_mle_2g_2g_2g(self, bb_rank_2g_2g_2g_mle, ds2_rank_2g):
        """Test MLE, 2 groups."""
        model = bb_rank_2g_2g_2g_mle
        call_fit_evaluate_predict(model, ds2_rank_2g)

    @pytest.mark.parametrize(
        "is_eager", [True, False]
    )
    def test_vi_1g_1g(self, bb_rank_1g_vi, ds2_rank_2g, is_eager):
        """Test VI, one group."""
        tf.config.run_functions_eagerly(is_eager)

        model = bb_rank_1g_vi
        call_fit_evaluate_predict(model, ds2_rank_2g)


class TestRateSimilarity:
    """Test w/ RateSimilarity behavior only."""

    @pytest.mark.parametrize(
        "is_eager", [True, False]
    )
    def test_rate(self, bb_rate_1g_mle, ds2_rate_2g, is_eager):
        """Test rate behavior only."""
        tf.config.run_functions_eagerly(is_eager)
        model = bb_rate_1g_mle
        call_fit_evaluate_predict(model, ds2_rate_2g)


class TestCategorizeSequences:
    """Test with sequence inputs and RNN cell."""

    @pytest.mark.parametrize(
        "is_eager", [True, False]
    )
    def test_categorize_sequence(
        self, bb_categorize_1g_1g_mle, ds2_categorize_2g, is_eager
    ):
        """Test MLE, one group."""
        tf.config.run_functions_eagerly(is_eager)
        model = bb_categorize_1g_1g_mle
        ds_obs = ds2_categorize_2g
        call_fit_evaluate_predict(model, ds_obs)


class TestMultiBehavior:
    """Test with multiple types of behavior."""

    @pytest.mark.parametrize(
        "is_eager", [True, False]
    )
    def test_rank_rate(
        self, bb_rank_rate_1g_mle, ds2_rank_rate_2g, is_eager
    ):
        """Test rank and rate."""
        tf.config.run_functions_eagerly(is_eager)
        model = bb_rank_rate_1g_mle
        call_fit_evaluate_predict(model, ds2_rank_rate_2g)
