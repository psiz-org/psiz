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
def ds_ranksim_v0():
    """Dataset.

    Rank similarity, no timestep, no gate weights.

    """
    n_sequence = 4
    stimulus_set = np.array((
        (1, 2, 3, 0, 0, 0, 0, 0, 0),
        (10, 13, 8, 0, 0, 0, 0, 0, 0),
        (4, 5, 6, 7, 8, 0, 0, 0, 0),
        (4, 5, 6, 7, 14, 15, 16, 17, 18)
    ), dtype=np.int32)

    n_select = np.array((1, 1, 1, 2), dtype=np.int32)
    groups = np.array(([0], [0], [1], [1]), dtype=np.int32)

    content = psiz.data.RankSimilarity(stimulus_set, n_select=n_select)
    outcome_idx = np.zeros(
        [content.n_sequence, content.max_timestep], dtype=np.int32
    )
    outcome = psiz.data.SparseCategorical(
        outcome_idx, depth=content.max_outcome
    )

    # TODO HACK intercept and modify
    (x, y, w) = psiz.data.TrialDataset(
        content, outcome=outcome, groups=groups
    ).export(with_timestep_axis=False, export_format='tensors')
    x.pop('groups')
    ds = tf.data.Dataset.from_tensor_slices((x, y, w))

    ds = ds.batch(n_sequence, drop_remainder=False)
    return ds


@pytest.fixture(scope="module")
def ds_ranksim_v1():
    """Dataset.

    Rank similarity, no timestep, with kernel gate weights.

    """
    n_sequence = 4
    stimulus_set = np.array((
        (1, 2, 3, 0, 0, 0, 0, 0, 0),
        (10, 13, 8, 0, 0, 0, 0, 0, 0),
        (4, 5, 6, 7, 8, 0, 0, 0, 0),
        (4, 5, 6, 7, 14, 15, 16, 17, 18)
    ), dtype=np.int32)

    n_select = np.array((1, 1, 1, 2), dtype=np.int32)
    groups = np.array(([0], [0], [1], [1]), dtype=np.int32)

    content = psiz.data.RankSimilarity(stimulus_set, n_select=n_select)
    outcome_idx = np.zeros(
        [content.n_sequence, content.max_timestep], dtype=np.int32
    )
    outcome = psiz.data.SparseCategorical(
        outcome_idx, depth=content.max_outcome
    )

    # TODO HACK intercept and modify
    (x, y, w) = psiz.data.TrialDataset(
        content, outcome=outcome, groups=groups
    ).export(with_timestep_axis=False, export_format='tensors')
    kernel_gate_weights = x.pop('groups')
    x['kernel_gate_weights'] = kernel_gate_weights
    ds = tf.data.Dataset.from_tensor_slices((x, y, w))

    ds = ds.batch(n_sequence, drop_remainder=False)
    return ds


@pytest.fixture(scope="module")
def ds_ranksim_v2():
    """Dataset.

    Rank similarity, no timestep, with percept and kernel gate weights.

    """
    n_sequence = 4
    stimulus_set = np.array((
        (1, 2, 3, 0, 0, 0, 0, 0, 0),
        (10, 13, 8, 0, 0, 0, 0, 0, 0),
        (4, 5, 6, 7, 8, 0, 0, 0, 0),
        (4, 5, 6, 7, 14, 15, 16, 17, 18)
    ), dtype=np.int32)

    n_select = np.array((1, 1, 1, 2), dtype=np.int32)
    groups = np.array(([0], [0], [1], [1]), dtype=np.int32)

    content = psiz.data.RankSimilarity(stimulus_set, n_select=n_select)
    outcome_idx = np.zeros(
        [content.n_sequence, content.max_timestep], dtype=np.int32
    )
    outcome = psiz.data.SparseCategorical(
        outcome_idx, depth=content.max_outcome
    )

    # TODO HACK intercept and modify
    (x, y, w) = psiz.data.TrialDataset(
        content, outcome=outcome, groups=groups
    ).export(with_timestep_axis=False, export_format='tensors')
    gate_weights = x.pop('groups')
    x['kernel_gate_weights'] = gate_weights
    x['percept_gate_weights'] = gate_weights
    ds = tf.data.Dataset.from_tensor_slices((x, y, w))

    ds = ds.batch(n_sequence, drop_remainder=False)
    return ds


@pytest.fixture(scope="module")
def ds_ranksim_v3():
    """Dataset.

    Rank similarity, no timestep, with percept and kernel gate weights.

    """
    n_sequence = 4
    stimulus_set = np.array((
        (1, 2, 3, 0, 0, 0, 0, 0, 0),
        (10, 13, 8, 0, 0, 0, 0, 0, 0),
        (4, 5, 6, 7, 8, 0, 0, 0, 0),
        (4, 5, 6, 7, 14, 15, 16, 17, 18)
    ), dtype=np.int32)

    n_select = np.array((1, 1, 1, 2), dtype=np.int32)
    groups = np.array(([0], [0], [1], [1]), dtype=np.int32)

    content = psiz.data.RankSimilarity(stimulus_set, n_select=n_select)
    outcome_idx = np.zeros(
        [content.n_sequence, content.max_timestep], dtype=np.int32
    )
    outcome = psiz.data.SparseCategorical(
        outcome_idx, depth=content.max_outcome
    )

    # TODO HACK intercept and modify
    (x, y, w) = psiz.data.TrialDataset(
        content, outcome=outcome, groups=groups
    ).export(with_timestep_axis=False, export_format='tensors')
    gate_weights = x.pop('groups')
    x['percept_gate_weights_0'] = gate_weights
    x['percept_gate_weights_1'] = gate_weights
    ds = tf.data.Dataset.from_tensor_slices((x, y, w))

    ds = ds.batch(n_sequence, drop_remainder=False)
    return ds


@pytest.fixture(scope="module")
def ds_ranksimcell_v0():
    """Dataset.

    Rank similarity, with timestep, no gate weights.

    """
    n_sequence = 4
    stimulus_set = np.array((
        (1, 2, 3, 0, 0, 0, 0, 0, 0),
        (10, 13, 8, 0, 0, 0, 0, 0, 0),
        (4, 5, 6, 7, 8, 0, 0, 0, 0),
        (4, 5, 6, 7, 14, 15, 16, 17, 18)
    ), dtype=np.int32)

    n_select = np.array((1, 1, 1, 2), dtype=np.int32)
    groups = np.array(([0], [0], [1], [1]), dtype=np.int32)

    content = psiz.data.RankSimilarity(stimulus_set, n_select=n_select)
    outcome_idx = np.zeros(
        [content.n_sequence, content.max_timestep], dtype=np.int32
    )
    outcome = psiz.data.SparseCategorical(
        outcome_idx, depth=content.max_outcome
    )

    # TODO HACK intercept and modify
    (x, y, w) = psiz.data.TrialDataset(
        content, outcome=outcome, groups=groups
    ).export(with_timestep_axis=True, export_format='tensors')
    x.pop('groups')
    ds = tf.data.Dataset.from_tensor_slices((x, y, w))

    ds = ds.batch(n_sequence, drop_remainder=False)
    return ds


@pytest.fixture(scope="module")
def ds_ratesim_v0():
    """Dataset.

    Rate similarity, no timestep, no gate weights.

    """
    n_sequence = 4
    stimulus_set = np.array((
        (1, 2),
        (10, 13),
        (4, 5),
        (4, 18)
    ), dtype=np.int32)
    rating = np.array([[0.1], [0.4], [0.8], [0.9]])
    groups = np.array(([0], [0], [1], [1]), dtype=np.int32)

    content = psiz.data.RateSimilarity(stimulus_set)
    outcome = psiz.data.Continuous(rating)

    # TODO HACK intercept and modify
    (x, y, w) = psiz.data.TrialDataset(
        content, outcome=outcome, groups=groups
    ).export(with_timestep_axis=False, export_format='tensors')
    x.pop('groups')
    ds = tf.data.Dataset.from_tensor_slices((x, y, w))

    ds = ds.batch(n_sequence, drop_remainder=False)
    return ds


@pytest.fixture(scope="module")
def ds_ratesimcell_v0():
    """Dataset.

    Rate similarity, no timestep, no gate weights.

    """
    n_sequence = 4
    stimulus_set = np.array((
        (1, 2),
        (10, 13),
        (4, 5),
        (4, 18)
    ), dtype=np.int32)
    rating = np.array([[0.1], [.4], [.8], [.9]])
    groups = np.array(([0], [0], [1], [1]), dtype=np.int32)

    content = psiz.data.RateSimilarity(stimulus_set)
    outcome = psiz.data.Continuous(rating)

    # TODO HACK intercept and modify
    (x, y, w) = psiz.data.TrialDataset(
        content, outcome=outcome, groups=groups
    ).export(with_timestep_axis=True, export_format='tensors')
    x.pop('groups')
    ds = tf.data.Dataset.from_tensor_slices((x, y, w))

    ds = ds.batch(n_sequence, drop_remainder=False)
    return ds


@pytest.fixture(scope="module")
def ds_ratesim_v1():
    """Dataset.

    Rate similarity, no timestep, with behavior gate weights.

    """
    n_sequence = 4
    stimulus_set = np.array((
        (1, 2),
        (10, 13),
        (4, 5),
        (4, 18)
    ), dtype=np.int32)
    rating = np.array([[0.1], [.4], [.8], [.9]])
    groups = np.array(([0], [0], [1], [1]), dtype=np.int32)

    content = psiz.data.RateSimilarity(stimulus_set)
    outcome = psiz.data.Continuous(rating)

    # TODO HACK intercept and modify
    (x, y, w) = psiz.data.TrialDataset(
        content, outcome=outcome, groups=groups
    ).export(with_timestep_axis=False, export_format='tensors')
    gate_weights = x.pop('groups')
    x['behavior_gate_weights'] = gate_weights
    ds = tf.data.Dataset.from_tensor_slices((x, y, w))

    ds = ds.batch(n_sequence, drop_remainder=False)
    return ds


@pytest.fixture(scope="module")
def ds_categorize_v0():
    """Dataset.

    Categorize, with timestep

    """
    n_sequence = 4
    # sequence_length = 10
    # n_stimuli = 20
    n_output = 3

    stimulus_set = tf.constant(
        [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
            [2, 4, 6, 8, 10, 12, 14, 16, 0, 0],  # NOTE: 2 masked trials
        ], dtype=tf.int32
    )
    y = tf.constant(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            [0, 0, 0, 0, 0, 1, 1, 1, 2, 2],
            [0, 0, 0, 0, 0, 1, 1, 2, 0, 0],
        ], dtype=tf.int32
    )
    y_onehot = tf.one_hot(y, n_output, on_value=1.0, off_value=0.0)
    # NOTE: We add a trailing axis to represent "stimuli axis" and because
    # RNN layer complains if tensor rank is less than 3.
    x = {
        'categorize_stimulus_set': tf.expand_dims(stimulus_set, axis=2),
        'categorize_correct_label': tf.expand_dims(y, axis=2),
    }
    w = tf.constant(
        [
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 0., 0.]
        ]
    )
    ds = tf.data.Dataset.from_tensor_slices((x, y_onehot, w)).batch(
        n_sequence, drop_remainder=False
    )
    return ds


@pytest.fixture(scope="module")
def ds_ranksim_ratesim_v0():
    """Dataset.

    RankSimilarity and RateSimilarity, no timestep, with behavior gate
    weights.

    """
    n_trial = 8

    # Rank data for dataset.
    stimulus_set = np.array((
        (1, 2, 3, 0, 0, 0, 0, 0, 0),
        (10, 13, 8, 0, 0, 0, 0, 0, 0),
        (4, 5, 6, 7, 8, 0, 0, 0, 0),
        (4, 5, 6, 7, 14, 15, 16, 17, 18)
    ), dtype=np.int32)
    n_select = np.array((1, 1, 1, 2), dtype=np.int32)
    gate_weights_behavior = np.array(([0], [0], [1], [1]), dtype=np.int32)
    obs_rank = psiz.trials.RankObservations(
        stimulus_set,
        n_select=n_select,
        groups=gate_weights_behavior,
        mask_zero=True
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
            [0, 0],  # Placeholder, no rate trial.
            [0, 0],  # Placeholder, no rate trial.
            [0, 0],  # Placeholder, no rate trial.
            [0, 0],  # Placeholder, no rate trial.
            [1, 2],
            [10, 13],
            [4, 5],
            [4, 18],
        ]
    )

    # Add placeholder for rate trials.
    stimulus_set_rank = np.concatenate(
        (stimulus_set_rank, np.zeros_like(stimulus_set_rank)), axis=0
    )
    is_select_rank = np.concatenate(
        (is_select_rank, np.zeros_like(is_select_rank)), axis=0
    )

    gate_weights_behavior = np.array(
        [
            [0], [0], [0], [0], [1], [1], [1], [1]
        ], dtype=np.int32
    )

    # Expand dimensions to account for timestep axis.
    x = {
        'rank_similarity_stimulus_set': tf.constant(stimulus_set_rank),
        'rank_similarity_is_select': tf.constant(is_select_rank),
        'rate_similarity_stimulus_set': tf.constant(stimulus_set_rate),
        'gate_weights_behavior': tf.constant(gate_weights_behavior),
    }

    y_rate = np.array([0.0, 0.0, 0.0, 0.0, 0.1, .4, .8, .9])  # ratings
    y_rate = np.expand_dims(y_rate, axis=1)
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
def ds_rank_docket():
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

    ds_docket = docket.as_dataset(
        np.zeros([n_trial, 1])
    ).batch(n_trial, drop_remainder=False)

    return ds_docket


@pytest.fixture(scope="module")
def ds_rank_docket_2g():
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
    ds_obs = obs.as_dataset().batch(n_trial, drop_remainder=False)

    return ds_obs


@pytest.fixture(scope="module")
def rank_1g_vi():
    n_stimuli = 30
    n_dim = 10
    kl_weight = 0.

    embedding_posterior = psiz.keras.layers.EmbeddingNormalDiag(
        n_stimuli + 1, n_dim, mask_zero=True,
        scale_initializer=tf.keras.initializers.Constant(
            tfp.math.softplus_inverse(.01).numpy()
        )
    )

    prior_scale = .2
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
        subnets=[kernel_0, kernel_1], gating_index=-1
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
        (1, 2),
        (10, 13),
        (4, 5),
        (4, 18)
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
        (1, 2),
        (10, 13),
        (4, 5),
        (4, 18)
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
    ds_obs = obs.as_dataset().batch(n_trial, drop_remainder=False)

    return ds_obs


@pytest.fixture(scope="module")
def rate_default_1g_mle():
    """A MLE rate model."""
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

    model = psiz.keras.models.Rate(stimuli=stimuli, kernel=kernel)
    return model


@pytest.fixture(scope="module")
def rate_1g_mle():
    """A MLE rate model."""
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
        subnets=[kernel_0, kernel_1], gating_index=-1
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
        n_stimuli + 1, n_dim, mask_zero=True,
        scale_initializer=tf.keras.initializers.Constant(
            tfp.math.softplus_inverse(.01).numpy()
        )
    )

    prior_scale = .2
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
