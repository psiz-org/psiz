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
"""Fixtures for models."""

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

import psiz


@pytest.fixture(scope="module")
def ds_2rank1_v0():
    """Dataset.

    Rank similarity
    * no timestep
    * no groups

    """
    n_sample = 4
    stimulus_set = np.array((
        (1, 2, 3),
        (18, 16, 14),
        (17, 15, 13),
        (4, 5, 6)
    ), dtype=np.int32)
    n_select = 1
    content = psiz.data.Rank(stimulus_set, n_select=n_select)

    outcome_idx = np.zeros(
        [content.n_sample, content.sequence_length], dtype=np.int32
    )
    outcome = psiz.data.SparseCategorical(
        outcome_idx, depth=content.n_outcome
    )

    tfds = psiz.data.Dataset([content, outcome]).export(
        export_format='tfds', with_timestep_axis=False
    )
    tfds = tfds.batch(n_sample, drop_remainder=False)
    return tfds


@pytest.fixture(scope="module")
def ds_4rank1_v0():
    """Dataset.

    Rank similarity
    * no timestep
    * no groups

    """
    n_sample = 4
    stimulus_set = np.array((
        (1, 2, 3, 4, 5),
        (10, 13, 8, 9, 12),
        (4, 5, 6, 7, 8),
        (14, 15, 16, 17, 18)
    ), dtype=np.int32)
    n_select = 1
    content = psiz.data.Rank(stimulus_set, n_select=n_select)

    outcome_idx = np.zeros(
        [content.n_sample, content.sequence_length], dtype=np.int32
    )
    outcome = psiz.data.SparseCategorical(
        outcome_idx, depth=content.n_outcome
    )

    tfds = psiz.data.Dataset([content, outcome]).export(
        export_format='tfds', with_timestep_axis=False
    )
    tfds = tfds.batch(n_sample, drop_remainder=False)
    return tfds


@pytest.fixture(scope="module")
def ds_4rank1_v1():
    """Dataset.

    Rank similarity
    * no timestep
    * groups

    """
    n_sample = 4
    stimulus_set = np.array((
        (1, 2, 3, 4, 5),
        (10, 13, 8, 9, 12),
        (4, 5, 6, 7, 8),
        (14, 15, 16, 17, 18)
    ), dtype=np.int32)
    n_select = 1
    content = psiz.data.Rank(stimulus_set, n_select=n_select)

    groups = psiz.data.Group(
        np.array(([0], [0], [1], [1]), dtype=np.int32),
        name='kernel_gate_weights'
    )

    outcome_idx = np.zeros(
        [content.n_sample, content.sequence_length], dtype=np.int32
    )
    outcome = psiz.data.SparseCategorical(
        outcome_idx, depth=content.n_outcome
    )

    tfds = psiz.data.Dataset([content, groups, outcome]).export(
        export_format='tfds', with_timestep_axis=False
    )
    tfds = tfds.batch(n_sample, drop_remainder=False)
    return tfds


@pytest.fixture(scope="module")
def ds_4rank1_v2():
    """Dataset.

    Rank similarity
    * no timestep
    * groups

    """
    n_sample = 4
    stimulus_set = np.array((
        (1, 2, 3, 4, 5),
        (10, 13, 8, 9, 12),
        (4, 5, 6, 7, 8),
        (14, 15, 16, 17, 18)
    ), dtype=np.int32)
    n_select = 1
    content = psiz.data.Rank(stimulus_set, n_select=n_select)

    kernel_groups = psiz.data.Group(
        np.array(([0], [0], [1], [1]), dtype=np.int32),
        name='kernel_gate_weights'
    )
    percept_groups = psiz.data.Group(
        np.array(([0], [0], [1], [1]), dtype=np.int32),
        name='percept_gate_weights'
    )

    outcome_idx = np.zeros(
        [content.n_sample, content.sequence_length], dtype=np.int32
    )
    outcome = psiz.data.SparseCategorical(
        outcome_idx, depth=content.n_outcome
    )

    tfds = psiz.data.Dataset(
        [content, kernel_groups, percept_groups, outcome]
    ).export(
        export_format='tfds', with_timestep_axis=False
    )
    tfds = tfds.batch(n_sample, drop_remainder=False)
    return tfds


@pytest.fixture(scope="module")
def ds_4rank1_v3():
    """Dataset.

    Rank similarity
    * no timestep
    * groups

    """
    n_sample = 4
    stimulus_set = np.array((
        (1, 2, 3, 4, 5),
        (10, 13, 8, 9, 12),
        (4, 5, 6, 7, 8),
        (14, 15, 16, 17, 18)
    ), dtype=np.int32)
    n_select = 1
    content = psiz.data.Rank(stimulus_set, n_select=n_select)

    percept_groups_0 = psiz.data.Group(
        np.array(([0], [0], [1], [1]), dtype=np.int32),
        name='percept_gate_weights_0'
    )
    percept_groups_1 = psiz.data.Group(
        np.array(([0], [0], [1], [1]), dtype=np.int32),
        name='percept_gate_weights_1'
    )

    outcome_idx = np.zeros(
        [content.n_sample, content.sequence_length], dtype=np.int32
    )
    outcome = psiz.data.SparseCategorical(
        outcome_idx, depth=content.n_outcome
    )

    tfds = psiz.data.Dataset(
        [content, percept_groups_0, percept_groups_1, outcome]
    ).export(
        export_format='tfds', with_timestep_axis=False
    )
    tfds = tfds.batch(n_sample, drop_remainder=False)
    return tfds


@pytest.fixture(scope="module")
def ds_8rank2_v0():
    """Dataset.

    Rank similarity
    * no timestep
    * no groups

    """
    n_sample = 4
    stimulus_set = np.array((
        (1, 2, 3, 4, 5, 6, 7, 8, 9),
        (18, 16, 14, 12, 10, 8, 6, 4, 2),
        (17, 15, 13, 11, 9, 7, 5, 3, 1),
        (1, 2, 3, 4, 14, 15, 16, 17, 18)
    ), dtype=np.int32)
    n_select = 2
    content = psiz.data.Rank(stimulus_set, n_select=n_select)

    outcome_idx = np.zeros(
        [content.n_sample, content.sequence_length], dtype=np.int32
    )
    outcome = psiz.data.SparseCategorical(
        outcome_idx, depth=content.n_outcome
    )

    tfds = psiz.data.Dataset([content, outcome]).export(
        export_format='tfds', with_timestep_axis=False
    )
    tfds = tfds.batch(n_sample, drop_remainder=False)
    return tfds


@pytest.fixture(scope="module")
def ds_2rank1_8rank2_v0():
    """Dataset.

    Rank similarity
    * no timestep
    * no groups

    """
    n_sample = 8
    stimulus_set = np.array((
        (1, 2, 3),
        (18, 16, 14),
        (17, 15, 13),
        (4, 5, 6),
        (0, 0, 0),
        (0, 0, 0),
        (0, 0, 0),
        (0, 0, 0),
    ), dtype=np.int32)
    n_select = 1
    content_2rank1 = psiz.data.Rank(stimulus_set, n_select=n_select)

    outcome_idx = np.zeros(
        [content_2rank1.n_sample, content_2rank1.sequence_length],
        dtype=np.int32
    )
    sample_weight = np.array(
        [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32
    )
    outcome_2rank1 = psiz.data.SparseCategorical(
        outcome_idx,
        depth=content_2rank1.n_outcome,
        name='given2rank1',
        sample_weight=sample_weight
    )

    stimulus_set = np.array((
        (0, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0, 0),
        (1, 2, 3, 4, 5, 6, 7, 8, 9),
        (18, 16, 14, 12, 10, 8, 6, 4, 2),
        (17, 15, 13, 11, 9, 7, 5, 3, 1),
        (1, 2, 3, 4, 14, 15, 16, 17, 18)
    ), dtype=np.int32)
    n_select = 2
    content_8rank2 = psiz.data.Rank(stimulus_set, n_select=n_select)

    outcome_idx = np.zeros(
        [content_8rank2.n_sample, content_8rank2.sequence_length],
        dtype=np.int32
    )
    sample_weight = np.array(
        [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32
    )
    outcome_8rank2 = psiz.data.SparseCategorical(
        outcome_idx,
        depth=content_8rank2.n_outcome,
        name='given8rank2',
        sample_weight=sample_weight
    )

    rank_config_val = np.array(
        [[0], [0], [0], [0], [1], [1], [1], [1]], dtype=np.int32
    )
    rank_config = psiz.data.Group(
        rank_config_val, name='rank_config'
    )
    tfds = psiz.data.Dataset(
        [
            content_2rank1,
            outcome_2rank1,
            content_8rank2,
            outcome_8rank2,
            rank_config
        ]
    ).export(
        export_format='tfds', with_timestep_axis=False
    )
    tfds = tfds.batch(n_sample, drop_remainder=False)
    return tfds


@pytest.fixture(scope="module")
def ds_time_8rank2_v0():
    """Dataset.

    Rank similarity
    * with timestep
    * no groups

    """
    n_sample = 4
    stimulus_set = np.array((
        (1, 2, 3, 4, 5, 6, 7, 8, 9),
        (10, 13, 8, 14, 15, 16, 11, 12, 9),
        (4, 5, 6, 7, 8, 9, 10, 11, 12),
        (4, 5, 6, 7, 14, 15, 16, 17, 18)
    ), dtype=np.int32)
    n_select = 2
    content = psiz.data.Rank(stimulus_set, n_select=n_select)

    outcome_idx = np.zeros(
        [content.n_sample, content.sequence_length], dtype=np.int32
    )
    outcome = psiz.data.SparseCategorical(
        outcome_idx, depth=content.n_outcome
    )

    tfds = psiz.data.Dataset([content, outcome]).export(
        with_timestep_axis=True, export_format='tfds'
    )
    tfds = tfds.batch(n_sample, drop_remainder=False)
    return tfds


@pytest.fixture(scope="module")
def ds_rate2_v0():
    """Dataset.

    Rate similarity
    * no timestep
    * no groups

    """
    n_sample = 4
    stimulus_set = np.array((
        (1, 2),
        (10, 13),
        (4, 5),
        (4, 18)
    ), dtype=np.int32)
    rating = np.array([[0.1], [0.4], [0.8], [0.9]])
    content = psiz.data.Rate(stimulus_set)

    outcome = psiz.data.Continuous(rating)

    tfds = psiz.data.Dataset([content, outcome]).export(
        with_timestep_axis=False, export_format='tfds'
    )
    tfds = tfds.batch(n_sample, drop_remainder=False)
    return tfds


@pytest.fixture(scope="module")
def ds_time_rate2_v0():
    """Dataset.

    Rate similarity
    * no timestep
    * no groups

    """
    n_sample = 4
    stimulus_set = np.array((
        (1, 2),
        (10, 13),
        (4, 5),
        (4, 18)
    ), dtype=np.int32)
    content = psiz.data.Rate(stimulus_set)

    rating = np.array([[0.1], [.4], [.8], [.9]])
    outcome = psiz.data.Continuous(rating)

    tfds = psiz.data.Dataset([content, outcome]).export(
        with_timestep_axis=True, export_format='tfds'
    )
    tfds = tfds.batch(n_sample, drop_remainder=False)
    return tfds


@pytest.fixture(scope="module")
def ds_rate2_v1():
    """Dataset.

    Rate similarity
    * no timestep
    * with groups (behavior gate weights)

    """
    n_sample = 4
    stimulus_set = np.array((
        (1, 2),
        (10, 13),
        (4, 5),
        (4, 18)
    ), dtype=np.int32)
    content = psiz.data.Rate(stimulus_set)

    groups = psiz.data.Group(
        np.array([[0], [0], [1], [1]], dtype=np.int32),
        name='behavior_gate_weights'
    )

    rating = np.array([[0.1], [.4], [.8], [.9]])
    outcome = psiz.data.Continuous(rating)

    tfds = psiz.data.Dataset([content, outcome, groups]).export(
        with_timestep_axis=False, export_format='tfds'
    )
    tfds = tfds.batch(n_sample, drop_remainder=False)
    return tfds


@pytest.fixture(scope="module")
def ds_time_categorize_v0():
    """Dataset.

    Categorize, with timestep

    """
    n_sample = 4
    # sequence_length = 10
    # n_stimuli = 20
    n_output = 3

    stimulus_set = np.array(
        [
            [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]],
            [[11], [12], [13], [14], [15], [16], [17], [18], [19], [20]],
            [[1], [3], [5], [7], [9], [11], [13], [15], [17], [19]],
            # NOTE: 2 masked trials
            [[2], [4], [6], [8], [10], [12], [14], [16], [0], [0]],
        ], dtype=np.int32
    )
    objective_query_label = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            [0, 0, 0, 0, 0, 1, 1, 1, 2, 2],
            [0, 0, 0, 0, 0, 1, 1, 2, 0, 0],
        ], dtype=np.int32
    )
    sample_weight = np.array(
        [
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 0., 0.]
        ]
    )
    outcome = psiz.data.SparseCategorical(
        objective_query_label, depth=n_output, sample_weight=sample_weight
    )
    objective_query_label = tf.keras.utils.to_categorical(
        objective_query_label, num_classes=3
    )
    content = psiz.data.Categorize(
        stimulus_set=stimulus_set, objective_query_label=objective_query_label
    )
    pds = psiz.data.Dataset([content, outcome])
    tfds = pds.export(export_format='tfds').batch(
        n_sample, drop_remainder=False
    )
    return tfds


@pytest.fixture(scope="module")
def ds_4rank2_rate2_v0():
    """Dataset.

    Rank and Rate, no timestep, with behavior gate
    weights.

    """
    n_sample = 4

    # Rank data for dataset.
    stimulus_set = np.array((
        (1, 2, 3, 4, 5),
        (10, 13, 8, 11, 12),
        (4, 5, 6, 7, 8),
        (14, 15, 16, 17, 18)
    ), dtype=np.int32)
    n_select = 2
    content_rank = psiz.data.Rank(stimulus_set, n_select=n_select)
    outcome_idx = np.zeros(
        [content_rank.n_sample, content_rank.sequence_length], dtype=np.int32
    )
    outcome_rank = psiz.data.SparseCategorical(
        outcome_idx, depth=content_rank.n_outcome, name='rank_branch'
    )

    gate_weights = psiz.data.Group(
        np.array(
            ([1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]),
            dtype=np.float32
        ),
        name='gate_weights_behavior'
    )

    # Rate data.
    stimulus_set_rate = np.array(
        [
            [1, 2],
            [10, 13],
            [4, 5],
            [4, 18],
        ]
    )
    content_rate = psiz.data.Rate(stimulus_set_rate)
    outcome_rate = psiz.data.Continuous(
        np.array([[0.1], [.4], [.8], [.9]]),
        name='rate_branch'
    )

    pds = psiz.data.Dataset(
        [content_rank, outcome_rank, content_rate, outcome_rate, gate_weights]
    )

    tfds = pds.export(export_format='tfds', with_timestep_axis=False).batch(
        n_sample, drop_remainder=False
    )
    return tfds


@pytest.fixture(scope="module")
def ds_4rank1_rt_v0():
    """Dataset.

    Rank and response time output.
    * no timestep

    """
    n_trial = 8

    # Rank data for dataset.
    stimulus_set = np.array((
        (1, 2, 3, 4, 5),
        (10, 11, 12, 13, 14),
        (4, 5, 6, 7, 8),
        (14, 15, 16, 17, 18)
    ), dtype=np.int32)
    n_select = 1
    content_rank = psiz.data.Rank(stimulus_set, n_select=n_select)
    outcome_idx = np.zeros(
        [content_rank.n_sample, content_rank.sequence_length], dtype=np.int32
    )
    outcome_rank = psiz.data.SparseCategorical(
        outcome_idx, depth=content_rank.n_outcome, name='rank_choice_branch'
    )
    outcome_rt = psiz.data.Continuous(
        np.array([[4.0], [6.0], [7.0], [11.0]]),
        name='rank_rt_branch'
    )
    pds = psiz.data.Dataset([content_rank, outcome_rank, outcome_rt])
    tfds = pds.export(with_timestep_axis=False).batch(
        n_trial, drop_remainder=False
    )
    return tfds


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
    n_trial = 4
    stimulus_set = np.array((
        (1, 2, 3, 0, 0, 0, 0, 0, 0),
        (10, 13, 8, 0, 0, 0, 0, 0, 0),
        (4, 5, 6, 7, 8, 0, 0, 0, 0),
        (4, 5, 6, 7, 14, 15, 16, 17, 18)
    ), dtype=np.int32)
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
