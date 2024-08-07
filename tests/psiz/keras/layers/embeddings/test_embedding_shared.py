# -*- coding: utf-8 -*-
# Copyright 2024 The PsiZ Authors. All Rights Reserved.
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
"""Test EmbeddingSharded layer."""

import keras
import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

import psiz.data
import psiz.keras.layers


@pytest.fixture
def emb_inputs_v1():
    """A minibatch of non-gate inupts."""
    # Create a simple batch (batch_size=5).
    inputs = tf.constant(
        np.array(
            [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 1, 2], [9, 1, 2]], dtype=np.int32
        )
    )
    return inputs


def test_call_approx(emb_inputs_v1):
    """Test call."""
    n_stimuli = 10
    n_dim = 3
    prior_scale = 0.2

    embedding_prior = psiz.keras.layers.EmbeddingShared(
        n_stimuli,
        n_dim,
        mask_zero=False,
        embedding=psiz.keras.layers.EmbeddingNormalDiag(
            1,
            1,
            loc_initializer=keras.initializers.Constant(0.0),
            scale_initializer=keras.initializers.Constant(
                tfp.math.softplus_inverse(prior_scale).numpy()
            ),
            loc_trainable=False,
        ),
    )

    embedding_prior.build([None, 3])
    outputs = embedding_prior(emb_inputs_v1)

    # Just check shape since stochastic.
    desired_shape = tf.TensorShape([5, 3, 3])
    tf.debugging.assert_equal(tf.shape(outputs), desired_shape)


def test_serialization():
    """Test serialization with weights."""
    n_stimuli = 10
    n_dim = 3
    prior_scale = 0.2

    orig_layer = psiz.keras.layers.EmbeddingShared(
        n_stimuli,
        n_dim,
        mask_zero=False,
        embedding=psiz.keras.layers.EmbeddingNormalDiag(
            1,
            1,
            loc_initializer=keras.initializers.Constant(0.0),
            scale_initializer=keras.initializers.Constant(
                tfp.math.softplus_inverse(prior_scale).numpy()
            ),
            loc_trainable=False,
        ),
    )
    orig_layer.build([None, n_dim])
    config = orig_layer.get_config()
    weights = orig_layer.get_weights()

    recon_layer = psiz.keras.layers.EmbeddingShared.from_config(config)
    recon_layer.build([None, 3])
    recon_layer.set_weights(weights)

    # Test prior, which should trivially be equal becuase of initialization.
    tf.debugging.assert_equal(
        orig_layer.embeddings.mode(), recon_layer.embeddings.mode()
    )
    tf.debugging.assert_equal(
        orig_layer.embeddings.variance(), recon_layer.embeddings.variance()
    )


def test_properties():
    n_stimuli = 10
    n_dim = 3
    prior_scale = 0.2

    embedding_prior = psiz.keras.layers.EmbeddingShared(
        n_stimuli,
        n_dim,
        mask_zero=False,
        embedding=psiz.keras.layers.EmbeddingNormalDiag(
            1,
            1,
            loc_initializer=keras.initializers.Constant(0.0),
            scale_initializer=keras.initializers.Constant(
                tfp.math.softplus_inverse(prior_scale).numpy()
            ),
            loc_trainable=False,
        ),
    )
    embedding_prior.build([None, 3])

    input_dim = embedding_prior.input_dim
    assert input_dim == n_stimuli

    output_dim = embedding_prior.output_dim
    assert output_dim == n_dim

    mask_zero = embedding_prior.mask_zero
    assert not mask_zero

    embeddings = embedding_prior.embeddings
    assert isinstance(embeddings, tfp.distributions.Distribution)
