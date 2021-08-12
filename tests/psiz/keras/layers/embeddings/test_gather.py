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
"""Test EmbeddingGather layer."""

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from psiz.keras.layers import EmbeddingNormalDiag
from psiz.keras.layers import EmbeddingGather


def test_deterministic_0():
    """Test using (deterministic) native TF Embedding layer.

    Globally shared, n_dim=1.
    """
    n_stimuli = 4
    n_dim = 1
    prior_scale = .12

    # Create core embedding composed of one 1D point.
    embedding_core = tf.keras.layers.Embedding(
        1, 1,
        embeddings_initializer=tf.keras.initializers.Constant(0.1),
    )
    # Map one point to four points.
    embedding_gather = EmbeddingGather(
        embedding=embedding_core, input_map=np.zeros([n_stimuli])
    )

    # Test call with 1D input.
    x = tf.constant(
        np.array([0, 1, 2, 3]), dtype=tf.int32
    )
    z = embedding_gather(x)
    z_desired = tf.constant([[.1], [.1], [.1], [.1]], dtype=tf.float32)
    tf.debugging.assert_equal(
        z_desired, z
    )

    # Test call with 2D input.
    x = tf.constant(
        np.array([[0, 1], [2, 3], [0, 1], [2, 3]]), dtype=tf.int32
    )
    z = embedding_gather(x)
    z_desired = tf.constant(
        np.array(
            [[[.1], [.1]], [[.1], [.1]], [[.1], [.1]], [[.1], [.1]]]
        ), dtype=tf.float32
    )
    tf.debugging.assert_equal(
        z_desired, z
    )

    # Test embedding properties.
    embeddings_desired = tf.constant(
        [[.1], [.1], [.1], [.1]], dtype=tf.float32
    )
    embeddings = embedding_gather.embeddings
    tf.debugging.assert_equal(
        embeddings_desired, embeddings
    )

    mask_zero_desired = False
    assert mask_zero_desired == embedding_gather.mask_zero

    input_dim_desired = 4
    assert input_dim_desired == embedding_gather.input_dim

    output_dim_desired = 1
    assert output_dim_desired == embedding_gather.output_dim


def test_deterministic_1():
    """Test using (deterministic) native TF Embedding layer.

    Globally shared, n_dim=2.
    """
    n_stimuli = 4
    n_dim = 2
    prior_scale = .12

    # Create core embedding composed of one 2D point.
    embedding_core = tf.keras.layers.Embedding(
        1, 2,
        embeddings_initializer=tf.keras.initializers.Constant([0.1, 0.2]),
    )
    # Map one point to four points.
    embedding_gather = EmbeddingGather(
        embedding=embedding_core, input_map=np.zeros([n_stimuli])
    )

    # Test call with 1D input.
    x = tf.constant(
        np.array([0, 1, 2, 3]), dtype=tf.int32
    )
    z = embedding_gather(x)
    z_desired = tf.constant(
        [[.1, .2], [.1, .2], [.1, .2], [.1, .2]], dtype=tf.float32
    )
    tf.debugging.assert_equal(
        z_desired, z
    )

    # Test call with 2d input.
    x = tf.constant(
        np.array([[0, 1], [2, 3], [0, 1], [2, 3]]), dtype=tf.int32
    )
    z = embedding_gather(x)
    z_desired = tf.constant(
        [
            [[.1, .2], [.1, .2]],
            [[.1, .2], [.1, .2]],
            [[.1, .2], [.1, .2]],
            [[.1, .2], [.1, .2]],
        ], dtype=tf.float32
    )
    tf.debugging.assert_equal(
        z_desired, z
    )

    # Test embedding properties.
    embeddings_desired = tf.constant(
        [[.1, .2], [.1, .2], [.1, .2], [.1, .2]], dtype=tf.float32
    )
    embeddings = embedding_gather.embeddings
    tf.debugging.assert_equal(
        embeddings_desired, embeddings
    )

    mask_zero_desired = False
    assert mask_zero_desired == embedding_gather.mask_zero

    input_dim_desired = 4
    assert input_dim_desired == embedding_gather.input_dim

    output_dim_desired = 2
    assert output_dim_desired == embedding_gather.output_dim


def test_deterministic_2():
    """Test using (deterministic) native TF Embedding layer.

    hierarchically shared."""
    n_stimuli = 6
    n_dim = 2
    prior_scale = .12

    # Create core embedding composed of one 1D point.
    embedding_core = tf.keras.layers.Embedding(
        3, 2,
        embeddings_initializer=tf.keras.initializers.Constant(
            np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        )
    )
    # Map one point to four points.
    input_map = [0, 0, 1, 1, 2, 2]
    embedding_gather = EmbeddingGather(
        embedding=embedding_core, input_map=input_map
    )

    # Test call with 1D input.
    x = tf.constant(
        np.array([0, 1, 2, 3, 4, 5]), dtype=tf.int32
    )
    z = embedding_gather(x)
    z_desired = tf.constant(
        [[.1, .2], [.1, .2], [.3, .4], [.3, .4], [.5, .6], [.5, .6]],
        dtype=tf.float32
    )
    tf.debugging.assert_equal(
        z_desired, z
    )

    # Test embedding properties.
    embeddings_desired = tf.constant(
        [[.1, .2], [.1, .2], [.3, .4], [.3, .4], [.5, .6], [.5, .6]],
        dtype=tf.float32
    )
    embeddings = embedding_gather.embeddings
    tf.debugging.assert_equal(
        embeddings_desired, embeddings
    )

    mask_zero_desired = False
    assert mask_zero_desired == embedding_gather.mask_zero

    input_dim_desired = 6
    assert input_dim_desired == embedding_gather.input_dim

    output_dim_desired = 2
    assert output_dim_desired == embedding_gather.output_dim


def test_stochastic_0():
    """Test using stochastic embedding.

    Globally shared, n_dim=1.
    """
    n_stimuli = 4
    n_dim = 1

    # Create core embedding composed of one 1D point.
    embedding_core = EmbeddingNormalDiag(
        1, 1,
        loc_initializer=tf.keras.initializers.Constant(0.1),
        scale_initializer=tf.keras.initializers.Constant(
            tfp.math.softplus_inverse(.01).numpy()
        ),
        loc_trainable=False,
    )
    # Map one point to four points.
    embedding_gather = EmbeddingGather(
        embedding=embedding_core, input_map=np.zeros([n_stimuli])
    )

    # Test call with 1D input.
    x = tf.constant(
        np.array([0, 1, 2, 3]), dtype=tf.int32
    )
    z = embedding_gather(x).numpy()
    assert z.shape[0] == 4
    assert z.shape[1] == n_dim

    # Test call with 2D input.
    x = tf.constant(
        np.array([[0, 1], [2, 3], [0, 1], [2, 3]]), dtype=tf.int32
    )
    z = embedding_gather(x).numpy()
    assert z.shape[0] == 4
    assert z.shape[1] == 2
    assert z.shape[2] == n_dim

    # Test embedding properties.
    embeddings_loc_desired = tf.constant(
        [[.1], [.1], [.1], [.1]],
        dtype=tf.float32
    )
    embeddings_scale_desired = tf.constant(
        [[0.0100001], [0.0100001], [0.0100001], [0.0100001]],
        dtype=tf.float32
    )
    embeddings = embedding_gather.embeddings
    tf.debugging.assert_equal(
        embeddings_loc_desired, embeddings.distribution.loc
    )
    np.testing.assert_array_almost_equal(
        embeddings_scale_desired.numpy(),
        embeddings.distribution.scale.numpy()
    )

    mask_zero_desired = False
    assert mask_zero_desired == embedding_gather.mask_zero

    input_dim_desired = 4
    assert input_dim_desired == embedding_gather.input_dim

    output_dim_desired = 1
    assert output_dim_desired == embedding_gather.output_dim


def test_stochastic_1():
    """Test using stochastic embedding.

    Globally shared, n_dim=2.
    """
    n_stimuli = 4
    n_dim = 2
    prior_scale = .12

    # Create core embedding composed of one 1D point.
    embedding_core = EmbeddingNormalDiag(
        1, 2,
        loc_initializer=tf.keras.initializers.Constant(
            np.array([0.1, 0.2])
        ),
        scale_initializer=tf.keras.initializers.Constant(
            tfp.math.softplus_inverse(
                np.array([0.01, 0.02])
            ).numpy()
        ),
        loc_trainable=False,
    )
    # Map one point to four points.
    embedding_gather = EmbeddingGather(
        embedding=embedding_core, input_map=np.zeros([n_stimuli])
    )

    # Test call with 1D input.
    x = tf.constant(
        np.array([0, 1, 2, 3]), dtype=tf.int32
    )
    z = embedding_gather(x).numpy()
    assert z.shape[0] == 4
    assert z.shape[1] == n_dim

    # Test call with 2D input.
    x = tf.constant(
        np.array([[0, 1], [2, 3], [0, 1], [2, 3]]), dtype=tf.int32
    )
    z = embedding_gather(x).numpy()
    assert z.shape[0] == 4
    assert z.shape[1] == 2
    assert z.shape[2] == n_dim

    # Test embedding properties.
    embeddings_loc_desired = tf.constant(
        [[.1, .2], [.1, .2], [.1, .2], [.1, .2]],
        dtype=tf.float32
    )
    embeddings_scale_desired = tf.constant(
        [
            [0.0100001, 0.0200001],
            [0.0100001, 0.0200001],
            [0.0100001, 0.0200001],
            [0.0100001, 0.0200001],
        ],
        dtype=tf.float32
    )
    embeddings = embedding_gather.embeddings
    tf.debugging.assert_equal(
        embeddings_loc_desired, embeddings.distribution.loc
    )
    np.testing.assert_array_almost_equal(
        embeddings_scale_desired.numpy(),
        embeddings.distribution.scale.numpy()
    )

    mask_zero_desired = False
    assert mask_zero_desired == embedding_gather.mask_zero

    input_dim_desired = 4
    assert input_dim_desired == embedding_gather.input_dim

    output_dim_desired = 2
    assert output_dim_desired == embedding_gather.output_dim


def test_stochastic_2():
    """Test stochastic, hierarchically shared."""
    n_stimuli = 6
    n_dim = 2
    prior_scale = .12

    # Create core embedding composed of one 1D point.
    embedding_core = EmbeddingNormalDiag(
        3, 2,
        loc_initializer=tf.keras.initializers.Constant(
            np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        ),
        scale_initializer=tf.keras.initializers.Constant(
            tfp.math.softplus_inverse(
                np.array([[0.01, 0.02], [0.03, 0.04], [0.05, 0.06]])
            ).numpy()
        ),
        loc_trainable=False,
    )
    # Map one point to four points.
    input_map = [0, 0, 1, 2, 1, 2]
    embedding_gather = EmbeddingGather(
        embedding=embedding_core, input_map=input_map
    )

    # Test call with 1D input.
    x = tf.constant(
        np.array([0, 1, 2, 3, 4, 5]), dtype=tf.int32
    )
    z = embedding_gather(x).numpy()
    assert z.shape[0] == 6
    assert z.shape[1] == n_dim

    # Test embedding properties.
    embeddings_loc_desired = tf.constant(
        [[.1, .2], [.1, .2], [.3, .4], [.5, .6], [.3, .4], [.5, .6]],
        dtype=tf.float32
    )
    embeddings_scale_desired = tf.constant(
        [
            [0.0100001, 0.0200001],
            [0.0100001, 0.0200001],
            [0.0300001, 0.0400001],
            [0.05000011, 0.0600001],
            [0.0300001, 0.0400001],
            [0.05000011, 0.0600001],
        ],
        dtype=tf.float32
    )
    embeddings = embedding_gather.embeddings
    tf.debugging.assert_equal(
        embeddings_loc_desired, embeddings.distribution.loc
    )
    np.testing.assert_array_almost_equal(
        embeddings_scale_desired.numpy(),
        embeddings.distribution.scale.numpy()
    )

    mask_zero_desired = False
    assert mask_zero_desired == embedding_gather.mask_zero

    input_dim_desired = 6
    assert input_dim_desired == embedding_gather.input_dim

    output_dim_desired = 2
    assert output_dim_desired == embedding_gather.output_dim
