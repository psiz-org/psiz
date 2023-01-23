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

import pytest
import tensorflow as tf
import tensorflow_probability as tfp

import psiz.keras.layers
from psiz.keras.models.stochastic_model import StochasticModel


class LayerA(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(LayerA, self).__init__(**kwargs)
        self.units = units
        self.kernel_initializer = tf.keras.initializers.Constant(1.)

    def build(self, input_shape):
        """Build."""
        last_dim = input_shape[-1]
        self.kernel = self.add_weight(
            "kernel",
            shape=[last_dim, self.units],
            initializer=self.kernel_initializer,
            dtype=self.dtype,
            trainable=True,
        )
        self.built = True

    def call(self, inputs, training=False):
        x = tf.matmul(a=inputs, b=self.kernel)
        return x

    def get_config(self):
        config = super(LayerA, self).get_config()
        config.update({"units": self.units})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class LayerB(tf.keras.layers.Layer):
    """A simple repeat layer."""

    def __init__(self, **kwargs):
        """Initialize."""
        super(LayerB, self).__init__(**kwargs)
        self.w0_initializer = tf.keras.initializers.Constant(1.)

    def build(self, input_shape):
        """Build."""
        self.w0 = self.add_weight(
            "w0",
            shape=[],
            initializer=self.w0_initializer,
            dtype=self.dtype,
            trainable=True,
        )
        self.built = True

    def call(self, inputs, training=None):
        """Call."""
        return self.w0 * inputs

    def get_config(self):
        return super(LayerB, self).get_config()

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class CellA(tf.keras.layers.Layer):
    """A simple RNN cell."""

    def __init__(self, **kwargs):
        """Initialize."""
        super(CellA, self).__init__(**kwargs)
        self.layer_0 = LayerA(3)

        # Satisfy RNNCell contract.
        # NOTE: A placeholder state.
        self.state_size = [
            tf.TensorShape([1])
        ]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        """Get initial state."""
        initial_state = [
            tf.zeros([batch_size, 1], name='zeros_initial_state')
        ]
        return initial_state

    def call(self, inputs, states, training=None):
        """Call."""
        outputs = self.layer_0(inputs)
        return outputs, states

    def get_config(self):
        return super(CellA, self).get_config()

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class ModelControl(tf.keras.Model):
    """A non-stochastic model to use as a control case.

    Gates:
        None

    """
    def __init__(self):
        super(ModelControl, self).__init__()
        self.dense_layer = tf.keras.layers.Dense(3)

    def call(self, inputs):
        x = inputs['x_a']
        x = self.dense_layer(x)
        return x

    def get_config(self):
        return super(ModelControl, self).get_config()


class ModelA(StochasticModel):
    """A stochastic model.

    Default input handling.
    No custom layers.

    Gates:
        None

    """
    def __init__(self, **kwargs):
        super(ModelA, self).__init__(**kwargs)
        self.dense_layer = tf.keras.layers.Dense(3)

    def call(self, inputs):
        x = self.dense_layer(inputs['x_a'])
        return x

    def get_config(self):
        return super(ModelA, self).get_config()


class ModelB(StochasticModel):
    """A stochastic model with a custom layer.

    Custom layer.
    Assumes single tensor input as dictionary.

    Gates:
        None

    """
    def __init__(self, **kwargs):
        super(ModelB, self).__init__(**kwargs)
        self.custom_layer = LayerA(3)

    def call(self, inputs):
        x = self.custom_layer(inputs['x_a'])
        return x

    def get_config(self):
        """Return model configuration."""
        return super(ModelB, self).get_config()


class ModelB2(StochasticModel):
    """A stochastic model with a custom layer.

    Custom layer.
    Assumes single tensor input as tensor via methid overriding.

    Gates:
        None

    """
    def __init__(self, **kwargs):
        super(ModelB2, self).__init__(**kwargs)
        self.custom_layer = LayerA(3)

    def call(self, inputs):
        x = self.custom_layer(inputs)
        return x

    def get_config(self):
        """Return model configuration."""
        return super(ModelB2, self).get_config()


class ModelC(StochasticModel):
    """A stochastic model with a custom layer.

    Assumes dictionary of tensors input.

    Gates:
        None

    """

    def __init__(self, **kwargs):
        """Initialize."

        Args:
            kwargs:  Additional key-word arguments.

        Raises:
            ValueError: If arguments are invalid.

        """
        super(ModelC, self).__init__(**kwargs)
        self.branch_0 = LayerB()
        self.branch_1 = LayerB()
        self.add_layer = tf.keras.layers.Add()

    def call(self, inputs):
        """Call.

        Args:
            inputs: A dictionary of inputs.

        """
        x_a = inputs['x_a']
        x_b = inputs['x_b']
        x_a = self.branch_0(x_a)
        x_b = self.branch_1(x_b)
        return self.add_layer([x_a, x_b])

    def get_config(self):
        """Return model configuration."""
        return super(ModelC, self).get_config()


class ModelD(StochasticModel):
    """A stochastic model with an RNN layer.

    Assumes dictionary of tensors input.

    Gates:
        None

    """

    def __init__(self, **kwargs):
        """Initialize."

        Args:
            kwargs:  Additional key-word arguments.

        Raises:
            ValueError: If arguments are invalid.

        """
        super(ModelD, self).__init__(**kwargs)
        self.rnn_layer = tf.keras.layers.RNN(
            CellA(), return_sequences=True
        )
        self.add_layer = tf.keras.layers.Add()

    def call(self, inputs):
        """Call.

        Args:
            inputs: A dictionary of inputs.

        """
        x_a = inputs['x_a']
        x_b = inputs['x_b']
        x_a = self.rnn_layer(x_a)
        return self.add_layer([x_a, x_b])

    def get_config(self):
        """Return model configuration."""
        return super(ModelD, self).get_config()


class RankModelA(StochasticModel):
    """A `RankSimilarity` model.

    A stochastic, non-VI percept layer.

    Gates:
        None

    """

    def __init__(self, **kwargs):
        """Initialize."""
        super(RankModelA, self).__init__(**kwargs)

        n_stimuli = 20
        n_dim = 3
        prior_scale = .2

        percept = psiz.keras.layers.EmbeddingNormalDiag(
            n_stimuli + 1, n_dim, mask_zero=True,
            scale_initializer=tf.keras.initializers.Constant(
                tfp.math.softplus_inverse(prior_scale).numpy()
            )
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
        behavior = psiz.keras.layers.RankSimilarity(
            n_reference=4, n_select=1, percept=percept, kernel=kernel
        )
        self.behavior = behavior

    def call(self, inputs):
        """Call."""
        return self.behavior(inputs)

    def get_config(self):
        return super(RankModelA, self).get_config()


class RankModelB(StochasticModel):
    """A `RankSimilarity` model.

    A variational percept layer.

    Gates:
        None

    """

    def __init__(self, **kwargs):
        """Initialize."""
        super(RankModelB, self).__init__(**kwargs)

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
        behavior = psiz.keras.layers.RankSimilarity(
            n_reference=4, n_select=1, percept=percept, kernel=kernel
        )
        self.behavior = behavior

    def call(self, inputs):
        """Call."""
        return self.behavior(inputs)

    def get_config(self):
        return super(RankModelB, self).get_config()


class RankModelC(StochasticModel):
    """A `RankSimilarity` model.

    A variational percept layer.

    Gates:
        Percept layer (BraidGate:2) with shared prior.

    """

    def __init__(self, **kwargs):
        """Initialize."""
        super(RankModelC, self).__init__(**kwargs)

        n_stimuli = 20
        n_dim = 3
        kl_weight = .1
        prior_scale = .2
        embedding_posterior_0 = psiz.keras.layers.EmbeddingNormalDiag(
            n_stimuli + 1, n_dim, mask_zero=True,
            scale_initializer=tf.keras.initializers.Constant(
                tfp.math.softplus_inverse(prior_scale).numpy()
            )
        )
        embedding_posterior_1 = psiz.keras.layers.EmbeddingNormalDiag(
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
        percept_0 = psiz.keras.layers.EmbeddingVariational(
            posterior=embedding_posterior_0, prior=embedding_prior,
            kl_weight=kl_weight, kl_n_sample=30
        )
        percept_1 = psiz.keras.layers.EmbeddingVariational(
            posterior=embedding_posterior_1, prior=embedding_prior,
            kl_weight=kl_weight, kl_n_sample=30
        )
        percept = psiz.keras.layers.BraidGate(
            subnets=[percept_0, percept_1], gating_index=-1
        )
        percept_adapter = psiz.keras.layers.GateAdapter(
            gating_keys=['percept_gate_weights'],
            format_inputs_as_tuple=True
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
        behavior = psiz.keras.layers.RankSimilarity(
            n_reference=4,
            n_select=1,
            percept=percept,
            kernel=kernel,
            percept_adapter=percept_adapter
        )
        self.behavior = behavior

    def call(self, inputs):
        """Call."""
        return self.behavior(inputs)

    def get_config(self):
        return super(RankModelC, self).get_config()


class RankCellModelA(StochasticModel):
    """A VI RankSimilarityCell model.

    Variational percept layer.

    Gates:
        None

    """

    def __init__(self, **kwargs):
        """Initialize."""
        super(RankCellModelA, self).__init__(**kwargs)

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
        rank_cell = psiz.keras.layers.RankSimilarityCell(
            n_reference=8, n_select=2, percept=percept, kernel=kernel
        )
        rnn = tf.keras.layers.RNN(rank_cell, return_sequences=True)
        self.behavior = rnn

    def call(self, inputs):
        """Call."""
        return self.behavior(inputs)

    def get_config(self):
        return super(RankCellModelA, self).get_config()


class RateModelA(StochasticModel):
    """A `RateSimilarity` model.

    A variatoinal percept layer.

    Gates:
        None

    """

    def __init__(self, **kwargs):
        """Initialize."""
        super(RateModelA, self).__init__(**kwargs)

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
        behavior = psiz.keras.layers.RateSimilarity(
            percept=percept, kernel=kernel
        )
        self.behavior = behavior

    def call(self, inputs):
        """Call."""
        return self.behavior(inputs)

    def get_config(self):
        return super(RateModelA, self).get_config()


class ALCOVEModelA(StochasticModel):
    """An `ALCOVECell` model.

    Gates:
        None

    """

    def __init__(self, **kwargs):
        """Initialize."""
        super(ALCOVEModelA, self).__init__(**kwargs)

        n_stimuli = 20
        n_dim = 4
        n_output = 3
        prior_scale = .2

        percept = psiz.keras.layers.EmbeddingNormalDiag(
            n_stimuli + 1, n_dim, mask_zero=True,
            scale_initializer=tf.keras.initializers.Constant(
                tfp.math.softplus_inverse(prior_scale).numpy()
            ),
            trainable=False,
        )
        similarity = psiz.keras.layers.ExponentialSimilarity(
            beta_initializer=tf.keras.initializers.Constant(3.0),
            tau_initializer=tf.keras.initializers.Constant(1.0),
            gamma_initializer=tf.keras.initializers.Constant(0.0),
            trainable=False,
        )
        cell = psiz.keras.layers.ALCOVECell(
            n_output,
            percept=percept,
            similarity=similarity,
            rho_initializer=tf.keras.initializers.Constant(2.0),
            temperature_initializer=tf.keras.initializers.Constant(1.0),
            lr_attention_initializer=tf.keras.initializers.Constant(.03),
            lr_association_initializer=tf.keras.initializers.Constant(.03),
            trainable=False
        )
        rnn = tf.keras.layers.RNN(cell, return_sequences=True, stateful=False)
        self.behavior = rnn

    def call(self, inputs):
        """Call."""
        return self.behavior(inputs)

    def get_config(self):
        return super(ALCOVEModelA, self).get_config()


class ALCOVEModelB(StochasticModel):
    """An `ALCOVECell` model.

    VI percept layer.

    Gates:
        None

    """

    def __init__(self, **kwargs):
        """Initialize."""
        super(ALCOVEModelB, self).__init__(**kwargs)

        n_stimuli = 20
        n_dim = 4
        n_output = 3
        kl_weight = .1
        prior_scale = .2

        # VI percept layer
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
        similarity = psiz.keras.layers.ExponentialSimilarity(
            beta_initializer=tf.keras.initializers.Constant(3.0),
            tau_initializer=tf.keras.initializers.Constant(1.0),
            gamma_initializer=tf.keras.initializers.Constant(0.0),
            trainable=False,
        )
        cell = psiz.keras.layers.ALCOVECell(
            n_output,
            percept=percept,
            similarity=similarity,
            rho_initializer=tf.keras.initializers.Constant(2.0),
            temperature_initializer=tf.keras.initializers.Constant(1.0),
            lr_attention_initializer=tf.keras.initializers.Constant(.03),
            lr_association_initializer=tf.keras.initializers.Constant(.03),
            trainable=False
        )
        rnn = tf.keras.layers.RNN(cell, return_sequences=True, stateful=False)
        self.behavior = rnn

    def call(self, inputs):
        """Call."""
        return self.behavior(inputs)

    def get_config(self):
        return super(ALCOVEModelB, self).get_config()


def build_ranksim_subclass_a():
    """Build subclassed `Model`.

    RankSimilarity, one group, stochastic (non VI).

    """
    model = RankModelA(n_sample=3)
    compile_kwargs = {
        'loss': tf.keras.losses.CategoricalCrossentropy(),
        'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
        'weighted_metrics': [
            tf.keras.metrics.CategoricalCrossentropy(name='cce')
        ]
    }
    model.compile(**compile_kwargs)
    return model


def build_ranksim_subclass_b():
    """Build subclassed `Model`.

    RankSimilarity, one group, stochastic (non VI).

    """
    model = RankModelB(n_sample=3)
    compile_kwargs = {
        'loss': tf.keras.losses.CategoricalCrossentropy(),
        'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
        'weighted_metrics': [
            tf.keras.metrics.CategoricalCrossentropy(name='cce')
        ]
    }
    model.compile(**compile_kwargs)
    return model


def build_ranksim_subclass_c():
    """Build subclassed `Model`.

    RankSimilarity, gated VI percept.

    """
    model = RankModelC(n_sample=3)
    compile_kwargs = {
        'loss': tf.keras.losses.CategoricalCrossentropy(),
        'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
        'weighted_metrics': [
            tf.keras.metrics.CategoricalCrossentropy(name='cce')
        ]
    }
    model.compile(**compile_kwargs)
    return model


def build_ranksimcell_subclass_a():
    """Build subclassed `Model`.

    RankSimilarityCell, one group, stochastic (non VI).

    """
    model = RankCellModelA(n_sample=3)
    compile_kwargs = {
        'loss': tf.keras.losses.CategoricalCrossentropy(),
        'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
        'weighted_metrics': [
            tf.keras.metrics.CategoricalCrossentropy(name='cce')
        ]
    }
    model.compile(**compile_kwargs)
    return model


def build_ratesim_subclass_a():
    """Build subclassed `Model`."""
    model = RateModelA(n_sample=11)
    compile_kwargs = {
        'loss': tf.keras.losses.MeanSquaredError(),
        'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
        'weighted_metrics': [
            tf.keras.metrics.MeanSquaredError(name='mse')
        ]
    }
    model.compile(**compile_kwargs)
    return model


def build_alcove_subclass_a():
    """Build subclassed `Model`.

    ALCOVECell, one group, stochastic (non VI).

    """
    model = ALCOVEModelA(n_sample=2)
    compile_kwargs = {
        'loss': tf.keras.losses.CategoricalCrossentropy(),
        'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
        'weighted_metrics': [
            tf.keras.metrics.CategoricalAccuracy(name='accuracy')
        ]
    }
    model.compile(**compile_kwargs)
    return model


def build_alcove_subclass_b():
    """Build subclassed `Model`.

    ALCOVECell, one group, VI percept layer.

    """
    model = ALCOVEModelB(n_sample=2)
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
def ds_x2():
    """Dataset.

    x = [rank-2]

    """
    n_example = 6
    x_a = tf.constant([
        [0.1, 1.1, 2.1],
        [0.2, 1.2, 2.2],
        [0.3, 1.3, 2.3],
        [0.4, 1.4, 2.4],
        [0.5, 1.5, 2.5],
        [0.6, 1.6, 2.6]
    ], dtype=tf.float32)
    x = {'x_a': x_a}
    y = tf.constant([
        [10.1, 11.1, 12.1],
        [10.2, 11.2, 12.2],
        [10.3, 11.3, 12.3],
        [10.4, 11.4, 12.4],
        [10.5, 11.5, 12.5],
        [10.6, 11.6, 12.6]
    ], dtype=tf.float32)

    w = tf.constant([1., 1., 0.2, 1., 1., 0.8], dtype=tf.float32)
    tfds = tf.data.Dataset.from_tensor_slices((x, y, w))
    tfds = tfds.batch(n_example, drop_remainder=False)

    input_shape = {'x_a': tf.TensorShape(x_a.shape)}

    return {'tfds': tfds, 'input_shape': input_shape}


@pytest.fixture(scope="module")
def ds_x2_as_tensor():
    """Dataset.

    x = [rank-2]

    """
    n_example = 6
    x = tf.constant([
        [0.1, 1.1, 2.1],
        [0.2, 1.2, 2.2],
        [0.3, 1.3, 2.3],
        [0.4, 1.4, 2.4],
        [0.5, 1.5, 2.5],
        [0.6, 1.6, 2.6]
    ], dtype=tf.float32)
    y = tf.constant([
        [10.1, 11.1, 12.1],
        [10.2, 11.2, 12.2],
        [10.3, 11.3, 12.3],
        [10.4, 11.4, 12.4],
        [10.5, 11.5, 12.5],
        [10.6, 11.6, 12.6]
    ], dtype=tf.float32)

    w = tf.constant([1., 1., 0.2, 1., 1., 0.8], dtype=tf.float32)
    tfds = tf.data.Dataset.from_tensor_slices((x, y, w))
    tfds = tfds.batch(n_example, drop_remainder=False)

    input_shape = tf.TensorShape(x.shape)

    return {'tfds': tfds, 'input_shape': input_shape}


@pytest.fixture(scope="module")
def ds_x2_x2_x2():
    """Dataset.

    x = [rank-2,  rank-2, rank-2]

    """
    n_example = 6
    x_a = tf.constant([
        [0.1, 1.1, 2.1],
        [0.2, 1.2, 2.2],
        [0.3, 1.3, 2.3],
        [0.4, 1.4, 2.4],
        [0.5, 1.5, 2.5],
        [0.6, 1.6, 2.6]
    ], dtype=tf.float32)
    x_b = tf.constant([
        [10.1, 11.1, 12.1],
        [10.2, 11.2, 12.2],
        [10.3, 11.3, 12.3],
        [10.4, 11.4, 12.4],
        [10.5, 11.5, 12.5],
        [10.6, 11.6, 12.6]
    ], dtype=tf.float32)
    x_c = tf.constant([
        [20.1, 21.1, 22.1],
        [20.2, 21.2, 22.2],
        [20.3, 21.3, 22.3],
        [20.4, 21.4, 22.4],
        [20.5, 21.5, 22.5],
        [20.6, 21.6, 22.6]
    ], dtype=tf.float32)

    x = {
        'x_a': x_a,
        'x_b': x_b,
        'x_c': x_c,
    }
    y = tf.constant([
        [10.1, 11.1, 12.1],
        [10.2, 11.2, 12.2],
        [10.3, 11.3, 12.3],
        [10.4, 11.4, 12.4],
        [10.5, 11.5, 12.5],
        [10.6, 11.6, 12.6]
    ], dtype=tf.float32)

    w = tf.constant([1., 1., 0.2, 1., 1., 0.8], dtype=tf.float32)
    tfds = tf.data.Dataset.from_tensor_slices((x, y, w))
    tfds = tfds.batch(n_example, drop_remainder=False)

    input_shape = {
        'x_a': tf.TensorShape(x_a.shape),
        'x_b': tf.TensorShape(x_b.shape),
        'x_c': tf.TensorShape(x_c.shape),
    }

    return {'tfds': tfds, 'input_shape': input_shape}


@pytest.fixture(scope="module")
def ds_x3_x3():
    """Dataset.

    x = [rank-3, rank-3]

    """
    n_example = 6
    x_a = tf.constant([
        [[0.1, 1.1, 2.1], [3.1, 4.1, 5.1]],
        [[0.2, 1.2, 2.2], [3.2, 4.2, 5.2]],
        [[0.3, 1.3, 2.3], [3.3, 4.3, 5.3]],
        [[0.4, 1.4, 2.4], [3.4, 4.4, 5.4]],
        [[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]],
        [[0.6, 1.6, 2.6], [3.6, 4.6, 5.6]]
    ], dtype=tf.float32)
    x_b = tf.constant([
        [[10.1, 11.1, 12.1], [13.1, 14.1, 15.1]],
        [[10.2, 11.2, 12.2], [13.2, 14.2, 15.2]],
        [[10.3, 11.3, 12.3], [13.3, 14.3, 15.3]],
        [[10.4, 11.4, 12.4], [13.4, 14.4, 15.4]],
        [[10.5, 11.5, 12.5], [13.5, 14.5, 15.5]],
        [[10.6, 11.6, 12.6], [13.6, 14.6, 15.6]]
    ], dtype=tf.float32)

    x = {
        'x_a': x_a,
        'x_b': x_b,
    }
    y = tf.constant([
        [[10.1, 11.1, 12.1], [10.1, 11.1, 12.1]],
        [[10.2, 11.2, 12.2], [10.2, 11.2, 12.2]],
        [[10.3, 11.3, 12.3], [10.3, 11.3, 12.3]],
        [[10.4, 11.4, 12.4], [10.4, 11.4, 12.4]],
        [[10.5, 11.5, 12.5], [10.5, 11.5, 12.5]],
        [[10.6, 11.6, 12.6], [10.6, 11.6, 12.6]]
    ], dtype=tf.float32)

    w = tf.constant([
        [1.0, 1.0],
        [1.0, 1.0],
        [0.2, 0.2],
        [1.0, 1.0],
        [1.0, 1.0],
        [0.8, 0.8]
    ], dtype=tf.float32)
    tfds = tf.data.Dataset.from_tensor_slices((x, y, w))
    tfds = tfds.batch(n_example, drop_remainder=False)

    input_shape = {
        'x_a': tf.TensorShape(x_a.shape),
        'x_b': tf.TensorShape(x_b.shape),
    }

    return {'tfds': tfds, 'input_shape': input_shape}


def call_fit_evaluate_predict(model, tfds):
    """Simple test of call, fit, evaluate, and predict."""
    # Test isolated call.
    for data in tfds:
        x, _, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        _ = model(x, training=False)

    # Test fit.
    model.fit(tfds, epochs=3)

    # Test evaluate.
    model.evaluate(tfds)

    # Test predict.
    model.predict(tfds)


class TestControl:
    """Test non-stochastic Control Model."""

    @pytest.mark.parametrize(
        "is_eager", [True, False]
    )
    def test_save_load(self, ds_x2, is_eager, tmpdir):
        """Test model serialization."""
        tf.config.run_functions_eagerly(is_eager)
        tfds = ds_x2['tfds']

        model = ModelControl()
        compile_kwargs = {
            'loss': tf.keras.losses.MeanSquaredError(),
            'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
        }
        model.compile(**compile_kwargs)
        model.fit(tfds, epochs=2)
        result0 = model.evaluate(tfds)
        kernel0 = model.dense_layer.kernel
        bias0 = model.dense_layer.bias
        fp_model = tmpdir.join('test_model')
        model.save(fp_model)
        del model

        loaded = tf.keras.models.load_model(
            fp_model, custom_objects={"ModelControl": ModelControl}
        )
        result1 = loaded.evaluate(tfds)
        kernel1 = loaded.dense_layer.kernel
        bias1 = loaded.dense_layer.bias

        # Test for model equality.
        assert result0 == result1
        tf.debugging.assert_equal(kernel0, kernel1)
        tf.debugging.assert_equal(bias0, bias1)


class TestModelA:
    """Test custom ModelA"""

    @pytest.mark.parametrize(
        "is_eager", [True, False]
    )
    @pytest.mark.parametrize(
        "save_traces", [True, False]
    )
    def test_save_load(self, ds_x2, is_eager, save_traces, tmpdir):
        """Test model serialization."""
        tf.config.run_functions_eagerly(is_eager)
        tfds = ds_x2['tfds']

        model = ModelA(n_sample=2)
        compile_kwargs = {
            'loss': tf.keras.losses.MeanSquaredError(),
            'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
        }
        model.compile(**compile_kwargs)
        model.fit(tfds, epochs=2)
        assert model.n_sample == 2
        results_0 = model.evaluate(tfds, return_dict=True)
        kernel0 = model.dense_layer.kernel
        bias0 = model.dense_layer.bias
        fp_model = tmpdir.join('test_model')
        model.save(fp_model, save_traces=save_traces)
        del model

        loaded = tf.keras.models.load_model(
            fp_model, custom_objects={"ModelA": ModelA}
        )
        results_1 = loaded.evaluate(tfds, return_dict=True)
        kernel1 = loaded.dense_layer.kernel
        bias1 = loaded.dense_layer.bias

        # Test for model equality.
        assert loaded.n_sample == 2
        assert results_0['loss'] == results_1['loss']
        tf.debugging.assert_equal(kernel0, kernel1)
        tf.debugging.assert_equal(bias0, bias1)


class TestModelB:
    """Test custom ModelB"""

    @pytest.mark.parametrize(
        "is_eager", [True, False]
    )
    @pytest.mark.parametrize(
        "save_traces", [True, False]
    )
    def test_save_load_b1(self, ds_x2, is_eager, save_traces, tmpdir):
        """Test model serialization."""
        tf.config.run_functions_eagerly(is_eager)
        tfds = ds_x2['tfds']

        model = ModelB(n_sample=2)
        compile_kwargs = {
            'loss': tf.keras.losses.MeanSquaredError(),
            'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
        }
        model.compile(**compile_kwargs)
        model.fit(tfds, epochs=2)
        assert model.n_sample == 2
        results_0 = model.evaluate(tfds, return_dict=True)
        kernel0 = model.custom_layer.kernel
        fp_model = tmpdir.join('test_model')
        model.save(fp_model, save_traces=save_traces)
        del model

        loaded = tf.keras.models.load_model(
            fp_model, custom_objects={"ModelB": ModelB}
        )
        results_1 = loaded.evaluate(tfds, return_dict=True)
        kernel1 = loaded.custom_layer.kernel

        # Test for model equality.
        assert loaded.n_sample == 2
        assert results_0['loss'] == results_1['loss']
        tf.debugging.assert_equal(kernel0, kernel1)

    @pytest.mark.parametrize(
        "is_eager", [True, False]
    )
    @pytest.mark.parametrize(
        "save_traces", [True, False]
    )
    def test_save_load_b2(
        self, ds_x2_as_tensor, is_eager, save_traces, tmpdir
    ):
        """Test model serialization."""
        tf.config.run_functions_eagerly(is_eager)
        tfds = ds_x2_as_tensor['tfds']

        model = ModelB2(n_sample=2)
        compile_kwargs = {
            'loss': tf.keras.losses.MeanSquaredError(),
            'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
        }
        model.compile(**compile_kwargs)
        model.fit(tfds, epochs=2)
        assert model.n_sample == 2
        results_0 = model.evaluate(tfds, return_dict=True)
        kernel0 = model.custom_layer.kernel
        fp_model = tmpdir.join('test_model')
        model.save(fp_model, save_traces=save_traces)
        del model

        loaded = tf.keras.models.load_model(
            fp_model, custom_objects={"ModelB2": ModelB2}
        )
        results_1 = loaded.evaluate(tfds, return_dict=True)
        kernel1 = loaded.custom_layer.kernel

        # Test for model equality.
        assert loaded.n_sample == 2
        assert results_0['loss'] == results_1['loss']
        tf.debugging.assert_equal(kernel0, kernel1)


class TestModelC:
    """Test custom ModelC"""

    @pytest.mark.parametrize(
        "is_eager", [True, False]
    )
    def test_usage(self, ds_x2_x2_x2, is_eager):
        """Test usage."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_x2_x2_x2['tfds']
        input_shape = ds_x2_x2_x2['input_shape']
        model = ModelC(n_sample=2)
        model.build(input_shape)

        assert model.n_sample == 2

        x0_desired = tf.constant([
            [0.1, 1.1, 2.1],
            [0.1, 1.1, 2.1],
            [0.2, 1.2, 2.2],
            [0.2, 1.2, 2.2],
            [0.3, 1.3, 2.3],
            [0.3, 1.3, 2.3],
            [0.4, 1.4, 2.4],
            [0.4, 1.4, 2.4],
            [0.5, 1.5, 2.5],
            [0.5, 1.5, 2.5],
            [0.6, 1.6, 2.6],
            [0.6, 1.6, 2.6],
        ], dtype=tf.float32)
        x1_desired = tf.constant([
            [10.1, 11.1, 12.1],
            [10.1, 11.1, 12.1],
            [10.2, 11.2, 12.2],
            [10.2, 11.2, 12.2],
            [10.3, 11.3, 12.3],
            [10.3, 11.3, 12.3],
            [10.4, 11.4, 12.4],
            [10.4, 11.4, 12.4],
            [10.5, 11.5, 12.5],
            [10.5, 11.5, 12.5],
            [10.6, 11.6, 12.6],
            [10.6, 11.6, 12.6],
        ], dtype=tf.float32)
        x2_desired = tf.constant([
            [20.1, 21.1, 22.1],
            [20.1, 21.1, 22.1],
            [20.2, 21.2, 22.2],
            [20.2, 21.2, 22.2],
            [20.3, 21.3, 22.3],
            [20.3, 21.3, 22.3],
            [20.4, 21.4, 22.4],
            [20.4, 21.4, 22.4],
            [20.5, 21.5, 22.5],
            [20.5, 21.5, 22.5],
            [20.6, 21.6, 22.6],
            [20.6, 21.6, 22.6],
        ], dtype=tf.float32)

        y_desired = tf.constant([
            [10.1, 11.1, 12.1],
            [10.1, 11.1, 12.1],
            [10.2, 11.2, 12.2],
            [10.2, 11.2, 12.2],
            [10.3, 11.3, 12.3],
            [10.3, 11.3, 12.3],
            [10.4, 11.4, 12.4],
            [10.4, 11.4, 12.4],
            [10.5, 11.5, 12.5],
            [10.5, 11.5, 12.5],
            [10.6, 11.6, 12.6],
            [10.6, 11.6, 12.6]
        ], dtype=tf.float32)

        sample_weight_desired = tf.constant(
            [1., 1., 1., 1., 0.2, 0.2, 1., 1., 1., 1., 0.8, 0.8],
            dtype=tf.float32
        )

        y_pred_desired = tf.constant([
            [10.2, 12.2, 14.2],
            [10.2, 12.2, 14.2],
            [10.4, 12.4, 14.4],
            [10.4, 12.4, 14.4],
            [10.6, 12.6, 14.6],
            [10.6, 12.6, 14.6],
            [10.8, 12.8, 14.8],
            [10.8, 12.8, 14.8],
            [11., 13., 15.],
            [11., 13., 15.],
            [11.2, 13.2, 15.2],
            [11.2, 13.2, 15.2]
        ], dtype=tf.float32)

        # Perform a `test_step`.
        for data in tfds:
            x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
            # Adjust `x`, `y` and `sample_weight` batch axis to reflect
            # multiple samples.
            x = model.repeat_samples_in_batch_axis(x, model.n_sample)
            y = model.repeat_samples_in_batch_axis(y, model.n_sample)
            sample_weight = model.repeat_samples_in_batch_axis(
                sample_weight, model.n_sample
            )

            # Assert `x`, `y` and `sample_weight` handled correctly.
            tf.debugging.assert_equal(x['x_a'], x0_desired)
            tf.debugging.assert_equal(x['x_b'], x1_desired)
            tf.debugging.assert_equal(x['x_c'], x2_desired)
            tf.debugging.assert_equal(y, y_desired)
            tf.debugging.assert_equal(sample_weight, sample_weight_desired)

            y_pred = model(x, training=False)
            # Assert `y_pred` handled correctly.
            tf.debugging.assert_near(y_pred, y_pred_desired)

    @pytest.mark.parametrize(
        "is_eager", [True, False]
    )
    def test_nsample_change(self, ds_x2_x2_x2, is_eager):
        """Test model where number of samples changes between use."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_x2_x2_x2['tfds']
        model = ModelC(n_sample=2)
        compile_kwargs = {
            'loss': tf.keras.losses.MeanSquaredError(),
            'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
        }
        model.compile(**compile_kwargs)
        model.fit(tfds)

        # Change model's `n_sample` attribute.
        model.n_sample = 5

        # When running model, we now expect the following:
        y_desired = tf.constant([
            [10.1, 11.1, 12.1],
            [10.1, 11.1, 12.1],
            [10.1, 11.1, 12.1],
            [10.1, 11.1, 12.1],
            [10.1, 11.1, 12.1],
            [10.2, 11.2, 12.2],
            [10.2, 11.2, 12.2],
            [10.2, 11.2, 12.2],
            [10.2, 11.2, 12.2],
            [10.2, 11.2, 12.2],
            [10.3, 11.3, 12.3],
            [10.3, 11.3, 12.3],
            [10.3, 11.3, 12.3],
            [10.3, 11.3, 12.3],
            [10.3, 11.3, 12.3],
            [10.4, 11.4, 12.4],
            [10.4, 11.4, 12.4],
            [10.4, 11.4, 12.4],
            [10.4, 11.4, 12.4],
            [10.4, 11.4, 12.4],
            [10.5, 11.5, 12.5],
            [10.5, 11.5, 12.5],
            [10.5, 11.5, 12.5],
            [10.5, 11.5, 12.5],
            [10.5, 11.5, 12.5],
            [10.6, 11.6, 12.6],
            [10.6, 11.6, 12.6],
            [10.6, 11.6, 12.6],
            [10.6, 11.6, 12.6],
            [10.6, 11.6, 12.6],
        ], dtype=tf.float32)
        sample_weight_desired = tf.constant(
            [
                1., 1., 1., 1., 1.,
                1., 1., 1., 1., 1.,
                0.2, 0.2, 0.2, 0.2, 0.2,
                1., 1., 1., 1., 1.,
                1., 1., 1., 1., 1.,
                0.8, 0.8, 0.8, 0.8, 0.8,
            ], dtype=tf.float32
        )
        y_pred_shape_desired = tf.TensorShape([30, 3])

        # Perform a `test_step` to verify `n_sample` took effect.
        for data in tfds:
            x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
            # Adjust `x`, `y` and `sample_weight` batch axis to reflect
            # multiple samples.
            x = model.repeat_samples_in_batch_axis(x, model.n_sample)
            y = model.repeat_samples_in_batch_axis(y, model.n_sample)
            sample_weight = model.repeat_samples_in_batch_axis(
                sample_weight, model.n_sample
            )
            # Assert `y` and `sample_weight` handled correctly.
            # Assert `y` and `sample_weight` handled correctly.
            tf.debugging.assert_equal(y, y_desired)
            tf.debugging.assert_equal(sample_weight, sample_weight_desired)

            y_pred = model(x, training=False)
            # Assert `y_pred` handled correctly.
            tf.debugging.assert_equal(tf.shape(y_pred), y_pred_shape_desired)

    @pytest.mark.parametrize(
        "is_eager", [True, False]
    )
    @pytest.mark.parametrize(
        "save_traces", [True, False]
    )
    def test_save_load(self, ds_x2_x2_x2, is_eager, save_traces, tmpdir):
        """Test model serialization."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_x2_x2_x2['tfds']
        model = ModelC(n_sample=7)
        compile_kwargs = {
            'loss': tf.keras.losses.MeanSquaredError(),
            'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
        }
        model.compile(**compile_kwargs)

        model.fit(tfds, epochs=2)
        assert model.n_sample == 7
        results_0 = model.evaluate(tfds, return_dict=True)
        branch_0_w0_0 = model.branch_0.w0
        branch_1_w0_0 = model.branch_1.w0

        # Save the model.
        fp_model = tmpdir.join('test_model')
        model.save(fp_model, save_traces=save_traces)
        del model

        # Load the saved model.
        loaded = tf.keras.models.load_model(
            fp_model, custom_objects={"ModelC": ModelC}
        )
        results_1 = loaded.evaluate(tfds, return_dict=True)
        branch_0_w0_1 = loaded.branch_0.w0
        branch_1_w0_1 = loaded.branch_1.w0

        # Test for model equality.
        assert loaded.n_sample == 7
        assert results_0['loss'] == results_1['loss']
        tf.debugging.assert_equal(branch_0_w0_0, branch_0_w0_1)
        tf.debugging.assert_equal(branch_1_w0_0, branch_1_w0_1)


class TestModelD:
    """Test using subclassed `Model` `ModelD`"""

    @pytest.mark.parametrize(
        "is_eager", [True, False]
    )
    def test_usage(self, ds_x3_x3, is_eager):
        """Test with RNN layer."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_x3_x3['tfds']
        input_shape = ds_x3_x3['input_shape']
        model = ModelD(n_sample=10)
        model.build(input_shape)

        assert model.n_sample == 10

        # Do a quick test of the Tensor shapes.
        x0_shape_desired = tf.TensorShape([60, 2, 3])
        x1_shape_desired = tf.TensorShape([60, 2, 3])
        y_shape_desired = tf.TensorShape([60, 2, 3])
        w_shape_desired = tf.TensorShape([60, 2])
        y_pred_shape_desired = tf.TensorShape([60, 2, 3])

        # Perform a `test_step`.
        for data in tfds:
            x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
            # Adjust `x`, `y` and `sample_weight` batch axis to reflect
            # multiple samples.
            x = model.repeat_samples_in_batch_axis(x, model.n_sample)
            y = model.repeat_samples_in_batch_axis(y, model.n_sample)
            sample_weight = model.repeat_samples_in_batch_axis(
                sample_weight, model.n_sample
            )
            # Assert `x`, `y` and `sample_weight` handled correctly.
            tf.debugging.assert_equal(x['x_a'].shape, x0_shape_desired)
            tf.debugging.assert_equal(x['x_b'].shape, x1_shape_desired)
            tf.debugging.assert_equal(y.shape, y_shape_desired)
            tf.debugging.assert_equal(sample_weight.shape, w_shape_desired)

            y_pred = model(x, training=False)
            # Assert `y_pred` handled correctly.
            tf.debugging.assert_equal(y_pred.shape, y_pred_shape_desired)

    @pytest.mark.parametrize(
        "is_eager", [True, False]
    )
    @pytest.mark.parametrize(
        "save_traces", [True, False]
    )
    def test_save_load(self, ds_x3_x3, is_eager, save_traces, tmpdir):
        """Test model serialization."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_x3_x3['tfds']
        model = ModelD(n_sample=11)
        compile_kwargs = {
            'loss': tf.keras.losses.MeanSquaredError(),
            'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
        }
        model.compile(**compile_kwargs)

        model.fit(tfds, epochs=2)
        assert model.n_sample == 11
        results_0 = model.evaluate(tfds, return_dict=True)
        kernel_0 = model.rnn_layer.cell.layer_0.kernel

        # Save the model.
        fp_model = tmpdir.join('test_model')
        model.save(fp_model, save_traces=save_traces)
        del model

        # Load the saved model.
        loaded = tf.keras.models.load_model(
            fp_model, custom_objects={"ModelD": ModelD}
        )
        results_1 = loaded.evaluate(tfds, return_dict=True)
        kernel_1 = loaded.rnn_layer.cell.layer_0.kernel

        # Test for model equality.
        assert loaded.n_sample == 11
        assert results_0['loss'] == results_1['loss']
        tf.debugging.assert_equal(kernel_0, kernel_1)


class TestRankSimilarity:
    """Test using `RankSimilarity` layer."""

    @pytest.mark.parametrize(
        "is_eager", [True, False]
    )
    def test_usage_subclass_a(self, ds_4rank1_v0, is_eager):
        """Test subclassed `StochasticModel`."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_4rank1_v0
        model = build_ranksim_subclass_a()
        call_fit_evaluate_predict(model, tfds)
        tf.keras.backend.clear_session()

    @pytest.mark.parametrize(
        "is_eager", [True, False]
    )
    @pytest.mark.parametrize(
        "save_traces", [True, False]
    )
    def test_save_load_subclass_a(
        self, ds_4rank1_v0, is_eager, save_traces, tmpdir
    ):
        """Test save/load.

        We change default `n_sample` for a more comprehensive test.

        """
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_4rank1_v0
        model = build_ranksim_subclass_a()
        input_shape = {k: v.shape for k, v in tfds.element_spec[0].items()}
        model.build(input_shape)

        # Test initialization settings.
        assert model.n_sample == 3

        # Test propogation of setting `n_sample`.
        model.n_sample = 21
        assert model.n_sample == 21

        model.fit(tfds, epochs=1)
        _ = model.evaluate(tfds)
        percept_mean = model.behavior.percept.embeddings.mean()
        percept_variance = model.behavior.percept.embeddings.variance()

        # Test storage serialization.
        fp_model = tmpdir.join('test_model')
        model.save(fp_model, save_traces=save_traces)
        del model

        # Load the saved model.
        loaded = tf.keras.models.load_model(
            fp_model, custom_objects={"RankModelA": RankModelA}
        )
        _ = loaded.evaluate(tfds)
        loaded_percept_mean = loaded.behavior.percept.embeddings.mean()
        loaded_percept_variance = loaded.behavior.percept.embeddings.variance()

        # Test for model equality.
        # NOTE: Don't check loss for equivalence because of
        # stochasticity.
        assert loaded.n_sample == 21
        tf.debugging.assert_equal(percept_mean, loaded_percept_mean)
        tf.debugging.assert_equal(
            percept_variance, loaded_percept_variance
        )

        tf.keras.backend.clear_session()

    @pytest.mark.parametrize(
        "is_eager", [True, False]
    )
    def test_usage_subclass_b(self, ds_4rank1_v0, is_eager):
        """Test subclassed `StochasticModel`."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_4rank1_v0
        model = build_ranksim_subclass_b()
        call_fit_evaluate_predict(model, tfds)
        tf.keras.backend.clear_session()

    @pytest.mark.parametrize(
        "is_eager", [True, False]
    )
    @pytest.mark.parametrize(
        "save_traces", [True, False]
    )
    def test_save_load_subclass_b(
        self, ds_4rank1_v0, is_eager, save_traces, tmpdir
    ):
        """Test save/load."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_4rank1_v0
        model = build_ranksim_subclass_b()
        input_shape = {k: v.shape for k, v in tfds.element_spec[0].items()}
        model.build(input_shape)

        # Test initialization settings.
        assert model.n_sample == 3

        # Test propogation of setting `n_sample`.
        model.n_sample = 21
        assert model.n_sample == 21

        model.fit(tfds, epochs=1)
        _ = model.evaluate(tfds)
        percept_mean = model.behavior.percept.embeddings.mean()
        percept_variance = model.behavior.percept.embeddings.variance()

        # Test storage serialization.
        fp_model = tmpdir.join('test_model')
        model.save(fp_model, save_traces=save_traces)
        del model

        # Load the saved model.
        loaded = tf.keras.models.load_model(
            fp_model, custom_objects={"RankModelB": RankModelB}
        )
        _ = loaded.evaluate(tfds)
        loaded_percept_mean = loaded.behavior.percept.embeddings.mean()
        loaded_percept_variance = loaded.behavior.percept.embeddings.variance()

        # Test for model equality.
        assert loaded.n_sample == 21
        tf.debugging.assert_equal(percept_mean, loaded_percept_mean)
        tf.debugging.assert_equal(
            percept_variance, loaded_percept_variance
        )

        tf.keras.backend.clear_session()

    @pytest.mark.parametrize(
        "is_eager", [True, False]
    )
    def test_usage_subclass_c(self, ds_4rank1_v2, is_eager):
        """Test subclassed `StochasticModel`."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_4rank1_v2
        model = build_ranksim_subclass_c()
        call_fit_evaluate_predict(model, tfds)
        tf.keras.backend.clear_session()

    @pytest.mark.parametrize(
        "is_eager", [True, False]
    )
    def test_agent_subclass_a(self, ds_4rank1_v0, is_eager):
        """Test usage in 'agent mode'."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_4rank1_v0
        model = build_ranksim_subclass_a()

        def simulate_agent(x):
            depth = 4  # TODO programmatic version?
            n_sample = 3
            x = model.repeat_samples_in_batch_axis(x, n_sample)
            outcome_probs = model(x)
            outcome_probs = model.average_repeated_samples(
                outcome_probs, n_sample
            )
            outcome_distribution = tfp.distributions.Categorical(
                probs=outcome_probs
            )
            outcome_idx = outcome_distribution.sample()
            outcome_one_hot = tf.one_hot(outcome_idx, depth)
            return outcome_one_hot

        _ = tfds.map(lambda x, y, w: (x, simulate_agent(x), w))

        tf.keras.backend.clear_session()


class TestRankSimilarityCell:
    """Test using `RankSimilarityCell` layer."""

    @pytest.mark.parametrize(
        "is_eager", [
            True,
            pytest.param(
                False,
                marks=pytest.mark.xfail(
                    reason="'add_loss' does not work inside RNN cell."
                )
            ),
        ]
    )
    def test_usage_subclass_a(self, ds_time_8rank2_v0, is_eager):
        """Test subclassed `StochasticModel`."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_time_8rank2_v0
        model = build_ranksimcell_subclass_a()
        call_fit_evaluate_predict(model, tfds)
        tf.keras.backend.clear_session()

    @pytest.mark.xfail(
        reason="'add_loss' does not work inside RNN cell."
    )
    @pytest.mark.parametrize(
        "is_eager", [
            True, False
        ]
    )
    @pytest.mark.parametrize(
        "save_traces", [True, False]
    )
    def test_save_load_subclass_a(
        self, ds_time_8rank2_v0, is_eager, save_traces, tmpdir
    ):
        """Test save/load."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_time_8rank2_v0
        model = build_ranksimcell_subclass_a()
        input_shape = {k: v.shape for k, v in tfds.element_spec[0].items()}
        model.build(input_shape)

        # Test initialization settings.
        assert model.n_sample == 3

        # Test propogation of setting `n_sample`.
        model.n_sample = 21
        assert model.n_sample == 21

        model.fit(tfds, epochs=1)
        percept_mean = model.behavior.cell.percept.embeddings.mean()
        _ = model.evaluate(tfds)

        # Test storage serialization.
        fp_model = tmpdir.join('test_model')
        model.save(fp_model, save_traces=save_traces)
        del model

        # Load the saved model.
        loaded = tf.keras.models.load_model(
            fp_model, custom_objects={"RankModelB": RankModelB}
        )
        loaded_percept_mean = loaded.behavior.cell.percept.embeddings.mean()
        _ = loaded.evaluate(tfds)

        # Test for model equality.
        assert loaded.n_sample == 21

        # Check `percept` posterior mean the same.
        tf.debugging.assert_equal(percept_mean, loaded_percept_mean)

        tf.keras.backend.clear_session()


class TestRateSimilarity:
    """Test using `RateSimilarity` layer."""

    @pytest.mark.parametrize(
        "is_eager", [True, False]
    )
    def test_usage_subclass_a(self, ds_rate2_v0, is_eager):
        """Test subclassed `StochasticModel`."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_rate2_v0
        model = build_ratesim_subclass_a()
        call_fit_evaluate_predict(model, tfds)
        tf.keras.backend.clear_session()

    @pytest.mark.parametrize(
        "is_eager", [True, False]
    )
    def test_save_load_subclass_a(self, ds_rate2_v0, is_eager, tmpdir):
        """Test save/load."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_rate2_v0
        model = build_ratesim_subclass_a()
        model.fit(tfds, epochs=1)

        # Test stochastic attributes.
        assert model.n_sample == 11

        _ = model.evaluate(tfds)
        percept_mean = model.behavior.percept.embeddings.mean()

        # Test storage serialization.
        fp_model = tmpdir.join('test_model')
        model.save(fp_model)
        del model

        # Load the saved model.
        loaded = tf.keras.models.load_model(
            fp_model, custom_objects={"RateModelA": RateModelA}
        )
        _ = loaded.evaluate(tfds)

        # Test for model equality.
        assert loaded.n_sample == 11

        # Check `percept` posterior mean the same.
        tf.debugging.assert_equal(
            percept_mean, loaded.behavior.percept.embeddings.mean()
        )

        tf.keras.backend.clear_session()


class TestALCOVECell:
    """Test using `ALCOVECell` layer."""

    @pytest.mark.parametrize(
        "is_eager", [True, False]
    )
    def test_usage_subclass_a(self, ds_time_categorize_v0, is_eager):
        """Test subclassed model, one group."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_time_categorize_v0
        model = build_alcove_subclass_a()
        call_fit_evaluate_predict(model, tfds)
        tf.keras.backend.clear_session()

    @pytest.mark.parametrize(
        "is_eager", [True, False]
    )
    @pytest.mark.parametrize(
        "save_traces", [True, False]
    )
    def test_save_load_subclass_a(
        self, ds_time_categorize_v0, is_eager, save_traces, tmpdir
    ):
        """Test save/load."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_time_categorize_v0
        model = build_alcove_subclass_a()
        model.fit(tfds, epochs=1)

        # Test initialization settings.
        assert model.n_sample == 2

        # Update `n_sample`.
        model.n_sample = 11
        _ = model.evaluate(tfds)
        percept_mean = model.behavior.cell.percept.embeddings.mean()

        # Test storage.
        fp_model = tmpdir.join('test_model')
        model.save(fp_model, save_traces=save_traces)
        del model
        # Load the saved model.
        loaded = tf.keras.models.load_model(
            fp_model, custom_objects={"ALCOVEModelA": ALCOVEModelA}
        )
        _ = loaded.evaluate(tfds)

        # Test for model equality.
        loaded_percept_mean = loaded.behavior.cell.percept.embeddings.mean()
        assert loaded.n_sample == 11

        # Check `percept` posterior mean the same.
        tf.debugging.assert_equal(percept_mean, loaded_percept_mean)

        tf.keras.backend.clear_session()

    @pytest.mark.parametrize(
        "is_eager", [
            True,
            pytest.param(
                False,
                marks=pytest.mark.xfail(
                    reason="'add_loss' does not work inside RNN cell."
                )
            ),
        ]
    )
    def test_usage_subclass_b(self, ds_time_categorize_v0, is_eager):
        """Test subclassed model, one group."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_time_categorize_v0
        model = build_alcove_subclass_b()
        call_fit_evaluate_predict(model, tfds)
        tf.keras.backend.clear_session()

    @pytest.mark.xfail(
        reason="'add_loss' does not work inside RNN cell."
    )
    @pytest.mark.parametrize(
        "is_eager", [True, False]
    )
    @pytest.mark.parametrize(
        "save_traces", [True, False]
    )
    def test_save_load_subclass_b(
        self, ds_time_categorize_v0, is_eager, save_traces, tmpdir
    ):
        """Test save/load."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_time_categorize_v0
        model = build_alcove_subclass_b()
        model.fit(tfds, epochs=1)

        # Test initialization settings.
        assert model.n_sample == 2

        # Increase `n_sample` to get more consistent evaluations
        model.n_sample = 11
        _ = model.evaluate(tfds)
        percept_mean = model.behavior.cell.percept.embeddings.mean()

        # Test storage.
        fp_model = tmpdir.join('test_model')
        model.save(fp_model, save_traces=save_traces)
        del model
        # Load the saved model.
        loaded = tf.keras.models.load_model(
            fp_model, custom_objects={"ALCOVEModelA": ALCOVEModelA}
        )
        _ = loaded.evaluate(tfds)

        # Test for model equality.
        assert loaded.n_sample == 11

        # Check `percept` posterior mean the same.
        tf.debugging.assert_equal(
            percept_mean,
            loaded.behavior.cell.percept.embeddings.mean()
        )

        tf.keras.backend.clear_session()
