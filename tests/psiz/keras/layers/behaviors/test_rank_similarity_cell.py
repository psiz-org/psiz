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
"""Module for testing models."""

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

import psiz


def percept_static_v0():
    """Static percept layer."""
    n_stimuli = 20
    n_dim = 3
    percept = tf.keras.layers.Embedding(
        n_stimuli + 1, n_dim, mask_zero=True
    )
    return percept


def percept_stochastic_v0():
    """Stochastic percept layer."""
    n_stimuli = 20
    n_dim = 3
    prior_scale = .2
    percept = psiz.keras.layers.EmbeddingNormalDiag(
        n_stimuli + 1, n_dim, mask_zero=True,
        scale_initializer=tf.keras.initializers.Constant(
            tfp.math.softplus_inverse(prior_scale).numpy()
        )
    )
    return percept


def kernel_v0():
    """A kernel layer."""
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
    return kernel


def rank_similarity_cell_call(ds, rank_cell):
    """A rank similarity cell call."""
    for data in ds:
        x, _, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        batch_size = x['rank_similarity_stimulus_set'].shape[0]
        states_t0 = rank_cell.get_initial_state(batch_size=batch_size)
        outputs, states_t1 = rank_cell(x, states_t0)
    return outputs, states_t1


@pytest.fixture(scope="module")
def ds_rank_v0():
    """Dataset.

    No timestep axis.
    No sample axis.
    No groups.

    """
    n_sequence = 4
    stimulus_set = np.array((
        (1, 2, 3, 0, 0, 0, 0, 0, 0),
        (10, 13, 8, 0, 0, 0, 0, 0, 0),
        (4, 5, 6, 7, 8, 0, 0, 0, 0),
        (4, 5, 6, 7, 14, 15, 16, 17, 18)
    ), dtype=np.int32)

    n_select = np.array((1, 1, 1, 2), dtype=np.int32)

    content = psiz.data.Rank(stimulus_set, n_select=n_select)
    outcome_idx = np.zeros(
        [content.n_sequence, content.sequence_length], dtype=np.int32
    )
    outcome = psiz.data.SparseCategorical(
        outcome_idx, depth=content.max_outcome
    )
    ds = psiz.data.TrialDataset(content, outcome=outcome).export(
        with_timestep_axis=False, export_format='tfds'
    )
    ds = ds.batch(n_sequence, drop_remainder=False)
    return ds


@pytest.fixture(scope="module")
def ds_rank_v1():
    """Dataset.

    * No timestep axis
    * Sample axis
    * No groups

    """
    n_sequence = 4
    stimulus_set = np.array((
        (1, 2, 3, 0, 0, 0, 0, 0, 0),
        (10, 13, 8, 0, 0, 0, 0, 0, 0),
        (4, 5, 6, 7, 8, 0, 0, 0, 0),
        (4, 5, 6, 7, 14, 15, 16, 17, 18)
    ), dtype=np.int32)

    n_select = np.array((1, 1, 1, 2), dtype=np.int32)
    content = psiz.data.Rank(stimulus_set, n_select=n_select)

    outcome_idx = np.zeros(
        [content.n_sequence, content.sequence_length], dtype=np.int32
    )
    outcome = psiz.data.SparseCategorical(
        outcome_idx, depth=content.max_outcome
    )

    # HACK using timestep axis for sample axis.
    ds = psiz.data.TrialDataset(content, outcome=outcome).export(
        with_timestep_axis=True, export_format='tfds'
    )
    ds = ds.batch(n_sequence, drop_remainder=False)
    return ds


# TODO write custom dataset to test non-standard inputs with extra axis.
# @pytest.fixture(scope="module")
# def ds_rank_v2():
#     """Dataset.

#     * No timestep axis
#     * Sample axis
#     * Additional axis
#     * No groups

#     """
#     n_sequence = 4
#     stimulus_set = np.array((
#         (1, 2, 3, 0, 0, 0, 0, 0, 0),
#         (10, 13, 8, 0, 0, 0, 0, 0, 0),
#         (4, 5, 6, 7, 8, 0, 0, 0, 0),
#         (4, 5, 6, 7, 14, 15, 16, 17, 18)
#     ), dtype=np.int32)

#     n_select = np.array((1, 1, 1, 2), dtype=np.int32)
#     content = psiz.data.Rank(stimulus_set, n_select=n_select)

#     outcome_idx = np.zeros(
#         [content.n_sequence, content.sequence_length], dtype=np.int32
#     )
#     outcome = psiz.data.SparseCategorical(
#         outcome_idx, depth=content.max_outcome
#     )

#     # HACK intercept and modify
#     # (x, y, w) = psiz.data.TrialDataset(content, outcome=outcome).export(
#     #     with_timestep_axis=True, export_format='tensors'
#     # )
#     # x['rank_similarity_stimulus_set'] = tf.expand_dims(
#     #     x['rank_similarity_stimulus_set'], axis=2
#     # )
#     # x['rank_similarity_is_select'] = tf.expand_dims(
#     #     x['rank_similarity_is_select'], axis=2
#     # )
#     # ds = tf.data.Dataset.from_tensor_slices((x, y, w))
#     # TODO HACK using timestep axis for sample axis
#     ds = psiz.data.TrialDataset(content, outcome=outcome).export(
#         with_timestep_axis=True, export_format='tfds'
#     )
#     ds = ds.batch(n_sequence, drop_remainder=False)
#     return ds


def test_call_v0(ds_rank_v0):
    """Test call."""
    percept = percept_static_v0()
    kernel = kernel_v0()
    rank_cell = psiz.keras.layers.RankSimilarityCell(
        percept=percept, kernel=kernel
    )
    outputs, states_t1 = rank_similarity_cell_call(ds_rank_v0, rank_cell)
    tf.debugging.assert_equal(tf.shape(outputs), tf.TensorShape([4, 56]))


def test_call_v1(ds_rank_v1):
    """Test call.

    With additional (unused) sample axis.

    """
    percept = percept_static_v0()
    kernel = kernel_v0()
    rank_cell = psiz.keras.layers.RankSimilarityCell(
        percept=percept, kernel=kernel
    )
    outputs, states_t1 = rank_similarity_cell_call(ds_rank_v1, rank_cell)
    tf.debugging.assert_equal(tf.shape(outputs), tf.TensorShape([4, 1, 56]))


def test_call_v2(ds_rank_v1):
    """Test call.

    With additional (used) sample axis.

    """
    # Imitate being insdie RNN.
    sample_axis_outermost = 2
    is_inside_rnn = True

    percept = percept_stochastic_v0()
    n_sample = 10
    percept.set_stochastic_mixin(
        sample_axis_outermost, n_sample, is_inside_rnn
    )

    kernel = kernel_v0()
    rank_cell = psiz.keras.layers.RankSimilarityCell(
        percept=percept, kernel=kernel
    )
    outputs, states_t1 = rank_similarity_cell_call(ds_rank_v1, rank_cell)
    tf.debugging.assert_equal(
        tf.shape(outputs), tf.TensorShape([4, 10, 56])
    )


# TODO used when ds_rank_v2 is ready.
# def test_call_v3(ds_rank_v2):
#     """Test call.

#     With additional (unused) sample axis and (unused) unnamed axis.

#     """
#     percept = percept_static_v0()
#     kernel = kernel_v0()
#     rank_cell = psiz.keras.layers.RankSimilarityCell(
#         percept=percept, kernel=kernel
#     )
#     outputs, states_t1 = rank_similarity_cell_call(ds_rank_v2, rank_cell)
#     tf.debugging.assert_equal(
#         tf.shape(outputs), tf.TensorShape([4, 1, 1, 56])
#     )
