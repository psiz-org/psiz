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

import copy

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


@pytest.fixture(scope="module")
def ds_3rank1_4x1():
    """Rank observations dataset."""
    n_sample = 4
    stimulus_set = np.array(
        [
            [1, 2, 3, 4],
            [10, 13, 16, 19],
            [4, 5, 6, 7],
            [14, 15, 16, 17]
        ], dtype=np.int32
    )
    n_select = 1
    content = psiz.data.Rank(stimulus_set, n_select=n_select)

    condition_idx = psiz.data.Group(
        np.array([[0], [0], [1], [1]], dtype=np.int32),
        name='condition_idx'
    )

    outcome_idx = np.zeros(
        [content.n_sample, content.sequence_length], dtype=np.int32
    )
    outcome = psiz.data.SparseCategorical(
        outcome_idx, depth=content.n_outcome
    )

    tfds = psiz.data.Dataset([content, outcome, condition_idx]).export()
    tfds = tfds.batch(n_sample, drop_remainder=False)

    return tfds


@pytest.fixture(scope="module")
def ds_3rank1_4x2():
    """Rank observations dataset."""
    n_sample = 4
    stimulus_set = np.array([
        [[1, 2, 3, 4], [5, 6, 7, 8]],
        [[10, 13, 16, 19], [3, 6, 9, 12]],
        [[4, 5, 6, 7], [8, 9, 10, 11]],
        [[14, 15, 16, 17], [13, 12, 11, 10]],
    ], dtype=np.int32)
    n_select = 1
    content = psiz.data.Rank(stimulus_set, n_select=n_select)

    condition_idx = psiz.data.Group(
        np.array(
            [[[0], [0]], [[0], [0]], [[1], [1]], [[1], [1]]], dtype=np.int32
        ),
        name='condition_idx'
    )

    outcome_idx = np.zeros(
        [content.n_sample, content.sequence_length], dtype=np.int32
    )
    outcome = psiz.data.SparseCategorical(
        outcome_idx, depth=content.n_outcome
    )
    tfds = psiz.data.Dataset([content, outcome, condition_idx]).export()
    tfds = tfds.batch(n_sample, drop_remainder=False)

    return tfds


class TempNoRNN(psiz.keras.models.StochasticModel):
    """A basic model for testing.

    Attributes:
        net: The network.
        n_sample: Integer indicating the number of samples to draw for
            stochastic layers. Only useful if using stochastic layers
            (e.g., variational models).

    """

    def __init__(self, net=None, n_sample=1, **kwargs):
        """Initialize.

        Args:
            net: A Keras layer.
            n_sample (optional): See psiz.keras.models.StochasticModel.
            kwargs:  Additional key-word arguments.

        Raises:
            ValueError: If arguments are invalid.

        """
        super(TempNoRNN, self).__init__(n_sample=n_sample, **kwargs)

        # Assign layers.
        self.net = net

    def call(self, inputs, training=None):
        """Call.

        Args:
            inputs: A dictionary of inputs.
            training (optional): Boolean indicating if training mode.

        """
        inputs = copy.copy(inputs)
        # Initialize states placeholder since not using RNN functionality
        states = tf.constant([0.])

        #  Drop sequence axis.
        inputs['given3rank1_stimulus_set'] = (
            inputs['given3rank1_stimulus_set'][:, 0]
        )

        output, states = self.net(inputs, states, training=training)

        #  Add back sequence axis.
        output = tf.expand_dims(output, axis=1)
        return output


class TempRNN(psiz.keras.models.StochasticModel):
    """A basic model for testing.

    Attributes:
        net: The network.
        n_sample: Integer indicating the number of samples to draw for
            stochastic layers. Only useful if using stochastic layers
            (e.g., variational models).

    """

    def __init__(self, net=None, n_sample=1, **kwargs):
        """Initialize.

        Args:
            net: A Keras layer.
            n_sample (optional): See psiz.keras.models.StochasticModel.
            kwargs:  Additional key-word arguments.

        Raises:
            ValueError: If arguments are invalid.

        """
        super(TempRNN, self).__init__(n_sample=n_sample, **kwargs)

        # Assign layers.
        self.net = tf.keras.layers.RNN(net, return_sequences=True)

    def call(self, inputs, training=None):
        """Call.

        Args:
            inputs: A dictionary of inputs.
            training (optional): Boolean indicating if training mode.

        """
        return self.net(inputs, training=training)


def test_call_approx(emb_inputs_v1):
    """Test call."""
    kl_weight = .1
    n_stimuli = 10
    n_dim = 3
    prior_scale = .2

    embedding_posterior = psiz.keras.layers.EmbeddingNormalDiag(
        n_stimuli, n_dim, mask_zero=False,
        scale_initializer=tf.keras.initializers.Constant(
            tfp.math.softplus_inverse(prior_scale).numpy()
        )
    )
    embedding_prior = psiz.keras.layers.EmbeddingShared(
        n_stimuli, n_dim, mask_zero=False,
        embedding=psiz.keras.layers.EmbeddingNormalDiag(
            1, 1,
            loc_initializer=tf.keras.initializers.Constant(0.),
            scale_initializer=tf.keras.initializers.Constant(
                tfp.math.softplus_inverse(prior_scale).numpy()
            ),
            loc_trainable=False,
        )
    )

    embedding_variational = psiz.keras.layers.EmbeddingVariational(
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

    embedding_posterior = psiz.keras.layers.EmbeddingNormalDiag(
        n_stimuli, n_dim, mask_zero=False,
        scale_initializer=tf.keras.initializers.Constant(
            tfp.math.softplus_inverse(prior_scale).numpy()
        )
    )
    embedding_prior = psiz.keras.layers.EmbeddingShared(
        n_stimuli, n_dim, mask_zero=False,
        embedding=psiz.keras.layers.EmbeddingNormalDiag(
            1, 1,
            loc_initializer=tf.keras.initializers.Constant(0.),
            scale_initializer=tf.keras.initializers.Constant(
                tfp.math.softplus_inverse(prior_scale).numpy()
            ),
            loc_trainable=False,
        )
    )

    embedding_variational = psiz.keras.layers.EmbeddingVariational(
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

    embedding_posterior = psiz.keras.layers.EmbeddingNormalDiag(
        n_stimuli, n_dim, mask_zero=False,
        scale_initializer=tf.keras.initializers.Constant(
            tfp.math.softplus_inverse(prior_scale).numpy()
        )
    )
    embedding_prior = psiz.keras.layers.EmbeddingShared(
        n_stimuli, n_dim, mask_zero=False,
        embedding=psiz.keras.layers.EmbeddingNormalDiag(
            1, 1,
            loc_initializer=tf.keras.initializers.Constant(0.),
            scale_initializer=tf.keras.initializers.Constant(
                tfp.math.softplus_inverse(prior_scale).numpy()
            ),
            loc_trainable=False,
        )
    )

    orig_layer = psiz.keras.layers.EmbeddingVariational(
        posterior=embedding_posterior, prior=embedding_prior,
        kl_weight=kl_weight, kl_n_sample=30
    )
    orig_layer.build([None, 3])
    config = orig_layer.get_config()

    recon_layer = psiz.keras.layers.EmbeddingVariational.from_config(config)
    recon_layer.build([None, 3])

    tf.debugging.assert_equal(
        tf.shape(orig_layer.posterior.embeddings.mode()),
        tf.shape(recon_layer.posterior.embeddings.mode())
    )
    tf.debugging.assert_equal(
        orig_layer.prior.embeddings.mode(),
        recon_layer.prior.embeddings.mode()
    )


def test_properties():
    kl_weight = .1
    n_stimuli = 10
    n_dim = 3
    prior_scale = .2

    embedding_posterior = psiz.keras.layers.EmbeddingNormalDiag(
        n_stimuli, n_dim, mask_zero=False,
        scale_initializer=tf.keras.initializers.Constant(
            tfp.math.softplus_inverse(prior_scale).numpy()
        )
    )
    embedding_prior = psiz.keras.layers.EmbeddingShared(
        n_stimuli, n_dim, mask_zero=False,
        embedding=psiz.keras.layers.EmbeddingNormalDiag(
            1, 1,
            loc_initializer=tf.keras.initializers.Constant(0.),
            scale_initializer=tf.keras.initializers.Constant(
                tfp.math.softplus_inverse(prior_scale).numpy()
            ),
            loc_trainable=False,
        )
    )

    orig_layer = psiz.keras.layers.EmbeddingVariational(
        posterior=embedding_posterior, prior=embedding_prior,
        kl_weight=kl_weight, kl_n_sample=30
    )
    orig_layer.build([None, 3])

    input_dim = orig_layer.input_dim
    assert input_dim == n_stimuli

    output_dim = orig_layer.output_dim
    assert output_dim == n_dim

    mask_zero = orig_layer.mask_zero
    assert not mask_zero

    embeddings = orig_layer.embeddings
    assert isinstance(embeddings, tfp.distributions.Distribution)


@pytest.mark.parametrize(
    "is_eager", [True, False]
)
def test_fit_no_rnn(ds_3rank1_4x1, is_eager):
    """Test fit method (triggering backprop)."""
    tf.config.run_functions_eagerly(is_eager)

    tfds = ds_3rank1_4x1

    kl_weight = .1
    n_stimuli = 20
    n_dim = 3
    prior_scale = .2

    embedding_posterior = psiz.keras.layers.EmbeddingNormalDiag(
        n_stimuli, n_dim, mask_zero=True,
        scale_initializer=tf.keras.initializers.Constant(
            tfp.math.softplus_inverse(prior_scale).numpy()
        )
    )
    embedding_prior = psiz.keras.layers.EmbeddingShared(
        n_stimuli, n_dim, mask_zero=True,
        embedding=psiz.keras.layers.EmbeddingNormalDiag(
            1, 1,
            loc_initializer=tf.keras.initializers.Constant(0.),
            scale_initializer=tf.keras.initializers.Constant(
                tfp.math.softplus_inverse(prior_scale).numpy()
            ),
            loc_trainable=False,
        )
    )

    embedding_variational = psiz.keras.layers.EmbeddingVariational(
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

    rank_cell = psiz.keras.layers.RankSimilarityCell(
        n_reference=3, n_select=1, percept=embedding_variational, kernel=kernel
    )

    model = TempNoRNN(net=rank_cell, n_sample=1)
    compile_kwargs = {
        'loss': tf.keras.losses.CategoricalCrossentropy(),
        'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
        'weighted_metrics': [
            tf.keras.metrics.CategoricalCrossentropy(name='cce')
        ]
    }
    model.compile(**compile_kwargs)

    model.fit(tfds, epochs=3)


@pytest.mark.parametrize(
    "is_eager", [
        True,
        pytest.param(
            False,
            marks=pytest.mark.xfail(
                reason="VI's 'add_loss' does not work inside RNN cell."
            )
        )
    ]
)
def test_fit_with_rnn(ds_3rank1_4x2, is_eager):
    """Test fit method (triggering backprop)."""
    tf.config.run_functions_eagerly(is_eager)

    tfds = ds_3rank1_4x2

    kl_weight = .1
    n_stimuli = 20
    n_dim = 3
    prior_scale = .2

    embedding_posterior = psiz.keras.layers.EmbeddingNormalDiag(
        n_stimuli, n_dim, mask_zero=True,
        scale_initializer=tf.keras.initializers.Constant(
            tfp.math.softplus_inverse(prior_scale).numpy()
        )
    )
    embedding_prior = psiz.keras.layers.EmbeddingShared(
        n_stimuli, n_dim, mask_zero=True,
        embedding=psiz.keras.layers.EmbeddingNormalDiag(
            1, 1,
            loc_initializer=tf.keras.initializers.Constant(0.),
            scale_initializer=tf.keras.initializers.Constant(
                tfp.math.softplus_inverse(prior_scale).numpy()
            ),
            loc_trainable=False,
        )
    )

    embedding_variational = psiz.keras.layers.EmbeddingVariational(
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

    rank_cell = psiz.keras.layers.RankSimilarityCell(
        n_reference=3, n_select=1, percept=embedding_variational, kernel=kernel
    )

    model = TempRNN(net=rank_cell, n_sample=1)
    compile_kwargs = {
        'loss': tf.keras.losses.CategoricalCrossentropy(),
        'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
        'weighted_metrics': [
            tf.keras.metrics.CategoricalCrossentropy(name='cce')
        ]
    }
    model.compile(**compile_kwargs)

    model.fit(tfds, epochs=3)
