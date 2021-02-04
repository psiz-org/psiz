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
"""Test EmbeddingVariational layer."""

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from psiz.keras.layers import EmbeddingNormalDiag
from psiz.keras.layers import EmbeddingShared
from psiz.keras.layers import EmbeddingVariational


@pytest.fixture
def emb_inputs_v1():
    """A minibatch of non-gate inupts."""
    # Create a simple batch (batch_size=5).
    inputs = tf.constant(
        np.array(
            [
                [0, 1, 2],
                [3, 4, 5],
                [6, 7, 8],
                [9, 1, 2],
                [9, 1, 2]
            ], dtype=np.int32
        )
    )
    return inputs


def test_call(emb_inputs_v1):
    """Test call."""
    kl_weight = .1
    n_stimuli = 10
    n_dim = 3
    prior_scale = .2

    embedding_posterior = EmbeddingNormalDiag(
        n_stimuli, n_dim, mask_zero=False,
        scale_initializer=tf.keras.initializers.Constant(
            tfp.math.softplus_inverse(prior_scale).numpy()
        )
    )
    embedding_prior = EmbeddingShared(
        n_stimuli, n_dim, mask_zero=False,
        embedding=EmbeddingNormalDiag(
            1, 1,
            loc_initializer=tf.keras.initializers.Constant(0.),
            scale_initializer=tf.keras.initializers.Constant(
                tfp.math.softplus_inverse(prior_scale).numpy()
            ),
            loc_trainable=False,
        )
    )

    embedding_variational = EmbeddingVariational(
        posterior=embedding_posterior, prior=embedding_prior,
        kl_weight=kl_weight, kl_n_sample=30
    )

    embedding_variational.build([None, 3])
    outputs = embedding_variational(emb_inputs_v1)

    # Just check shape since stochastic.
    desired_shape = tf.TensorShape([5, 3, 3])
    tf.debugging.assert_equal(tf.shape(outputs), desired_shape)


def test_output_shape(emb_inputs_v1):
    """Test output_shape method."""
    kl_weight = .1
    n_stimuli = 10
    n_dim = 3
    prior_scale = .2

    embedding_posterior = EmbeddingNormalDiag(
        n_stimuli, n_dim, mask_zero=False,
        scale_initializer=tf.keras.initializers.Constant(
            tfp.math.softplus_inverse(prior_scale).numpy()
        )
    )
    embedding_prior = EmbeddingShared(
        n_stimuli, n_dim, mask_zero=False,
        embedding=EmbeddingNormalDiag(
            1, 1,
            loc_initializer=tf.keras.initializers.Constant(0.),
            scale_initializer=tf.keras.initializers.Constant(
                tfp.math.softplus_inverse(prior_scale).numpy()
            ),
            loc_trainable=False,
        )
    )

    embedding_variational = EmbeddingVariational(
        posterior=embedding_posterior, prior=embedding_prior,
        kl_weight=kl_weight, kl_n_sample=30
    )

    input_shape = tf.shape(emb_inputs_v1).numpy().tolist()
    output_shape = embedding_variational.compute_output_shape(input_shape)
    desired_output_shape = tf.TensorShape([5, 3, 3])
    tf.debugging.assert_equal(output_shape, desired_output_shape)


def test_serialization():
    """Test serialization."""
    kl_weight = .1
    n_stimuli = 10
    n_dim = 3
    prior_scale = .2

    embedding_posterior = EmbeddingNormalDiag(
        n_stimuli, n_dim, mask_zero=False,
        scale_initializer=tf.keras.initializers.Constant(
            tfp.math.softplus_inverse(prior_scale).numpy()
        )
    )
    embedding_prior = EmbeddingShared(
        n_stimuli, n_dim, mask_zero=False,
        embedding=EmbeddingNormalDiag(
            1, 1,
            loc_initializer=tf.keras.initializers.Constant(0.),
            scale_initializer=tf.keras.initializers.Constant(
                tfp.math.softplus_inverse(prior_scale).numpy()
            ),
            loc_trainable=False,
        )
    )

    orig_layer = EmbeddingVariational(
        posterior=embedding_posterior, prior=embedding_prior,
        kl_weight=kl_weight, kl_n_sample=30
    )
    orig_layer.build([None, 3])
    config = orig_layer.get_config()

    recon_layer = EmbeddingVariational.from_config(config)
    recon_layer.build([None, 3])

    tf.debugging.assert_equal(
        tf.shape(orig_layer.posterior.embeddings.mode()),
        tf.shape(recon_layer.posterior.embeddings.mode())
    )
    tf.debugging.assert_equal(
        orig_layer.prior.embeddings.mode(),
        recon_layer.prior.embeddings.mode()
    )
