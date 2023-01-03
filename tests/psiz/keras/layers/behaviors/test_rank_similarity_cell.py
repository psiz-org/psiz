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


def rank_similarity_cell_call(tfds, rank_cell):
    """A rank similarity cell call."""
    for data in tfds:
        x, _, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        batch_size = x['given3rank1_stimulus_set'].shape[0]
        states_t0 = rank_cell.get_initial_state(batch_size=batch_size)
        outputs, states_t1 = rank_cell(x, states_t0)
    return outputs, states_t1


@pytest.fixture(scope="module")
def ds_3rank1_v0():
    """Dataset.

    No timestep axis.
    No groups.

    """
    n_sample = 4
    stimulus_set = np.array((
        (1, 2, 3, 4),
        (10, 13, 16, 19),
        (4, 5, 6, 7),
        (14, 15, 16, 17)
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
        with_timestep_axis=False, export_format='tfds'
    )
    tfds = tfds.batch(n_sample, drop_remainder=False)
    return tfds


@pytest.fixture(scope="module")
def ds_3rank1_v1():
    """Dataset.

    * Timestep axis
    * No groups

    """
    n_sample = 4
    stimulus_set = np.array((
        (1, 2, 3, 4),
        (10, 13, 16, 19),
        (4, 5, 6, 7),
        (14, 15, 16, 17)
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
        with_timestep_axis=True, export_format='tfds'
    )
    tfds = tfds.batch(n_sample, drop_remainder=False)
    return tfds


def test_call_v0(ds_3rank1_v0):
    """Test call."""
    percept = percept_static_v0()
    kernel = kernel_v0()
    rank_cell = psiz.keras.layers.RankSimilarityCell(
        n_reference=3, n_select=1, percept=percept, kernel=kernel
    )
    outputs, states_t1 = rank_similarity_cell_call(ds_3rank1_v0, rank_cell)
    tf.debugging.assert_equal(tf.shape(outputs), tf.TensorShape([4, 3]))


def test_call_v1(ds_3rank1_v1):
    """Test call.

    Using timestep axis as stand additional (unused) axis.

    """
    percept = percept_static_v0()
    kernel = kernel_v0()
    rank_cell = psiz.keras.layers.RankSimilarityCell(
        n_reference=3, n_select=1, percept=percept, kernel=kernel
    )
    outputs, states_t1 = rank_similarity_cell_call(ds_3rank1_v1, rank_cell)
    tf.debugging.assert_equal(tf.shape(outputs), tf.TensorShape([4, 1, 3]))
