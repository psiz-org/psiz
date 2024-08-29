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
"""Module of TensorFlow embedding layers.

Classes:
    EmbeddingGammaDiag: A Gamma distribution embedding layer.

"""


import keras
import tensorflow_probability as tfp

import psiz.keras.constraints
from psiz.keras.layers.embeddings.stochastic_embedding import StochasticEmbedding


@keras.saving.register_keras_serializable(
    package="psiz.keras.layers", name="EmbeddingGammaDiag"
)
class EmbeddingGammaDiag(StochasticEmbedding):
    """Gamma distribution embedding.

    Each embedding point is characterized by a Gamma distribution.

    """

    def __init__(
        self,
        input_dim,
        output_dim,
        mask_zero=False,
        input_length=1,
        concentration_initializer=None,
        rate_initializer=None,
        concentration_regularizer=None,
        rate_regularizer=None,
        concentration_constraint=None,
        rate_constraint=None,
        concentration_trainable=True,
        rate_trainable=True,
        sample_shape=(),
        **kwargs
    ):
        """Initialize.

        Args:
            input_dim:
            output_dim:
            mask_zero (optional):
            input_length (optional):
            concentration_initializer (optional):
            rate_initializer (optional):
            concentration_regularizer (optional):
            rate_regularizer (optional):
            concentration_constraint (optional):
            rate_constraint (optional):
            concentration_trainable (optional):
            rate_trainable (optional):
            kwargs: Additional key-word arguments.

        Notes:
            The trinability of a particular variable is determined by a
                logical "and" between `self.trainable` (the
                layer-wise attribute) and `self.x_trainable` (the
                attribute that specifically controls the variable `x`).
            Uses a constraint-based approach instead of a parameter
                transformation approach in order to avoid low viscosity
                issues with small parameter values.

        """
        super(EmbeddingGammaDiag, self).__init__(
            input_dim,
            output_dim,
            mask_zero=mask_zero,
            input_length=input_length,
            sample_shape=sample_shape,
            **kwargs
        )

        # Handle initializers.
        if concentration_initializer is None:
            concentration_initializer = keras.initializers.RandomUniform(1.0, 3.0)
        self.concentration_initializer = keras.initializers.get(
            concentration_initializer
        )
        if rate_initializer is None:
            rate_initializer = keras.initializers.RandomUniform(0.0, 1.0)
        self.rate_initializer = keras.initializers.get(rate_initializer)

        # Handle regularizers.
        self.concentration_regularizer = keras.regularizers.get(
            concentration_regularizer
        )
        self.rate_regularizer = keras.regularizers.get(rate_regularizer)

        # Handle constraints.
        if concentration_constraint is None:
            concentration_constraint = psiz.keras.constraints.GreaterEqualThan(
                min_value=1.0
            )
        self.concentration_constraint = keras.constraints.get(concentration_constraint)
        if rate_constraint is None:
            rate_constraint = psiz.keras.constraints.GreaterThan(min_value=0.0)
        self.rate_constraint = keras.constraints.get(rate_constraint)

        self.concentration_trainable = self.trainable and concentration_trainable
        self.rate_trainable = self.trainable and rate_trainable

    def build(self, input_shape=None):
        """Build embeddings distribution."""
        if self.built:
            return

        # Handle concentration variables.
        self.concentration = self.add_weight(
            shape=[self.input_dim, self.output_dim],
            initializer=self.concentration_initializer,
            name="concentration",
            regularizer=self.concentration_regularizer,
            trainable=self.concentration_trainable,
            constraint=self.concentration_constraint,
        )

        # Handle rate variables.
        self.rate = self.add_weight(
            shape=[self.input_dim, self.output_dim],
            initializer=self.rate_initializer,
            name="untransformed_rate",
            regularizer=self.rate_regularizer,
            trainable=self.rate_trainable,
            constraint=self.rate_constraint,
        )

    @property
    def embeddings(self):
        """Return embeddings."""
        dist = tfp.distributions.Gamma(self.concentration, self.rate)
        batch_ndims = keras.ops.size(dist.batch_shape_tensor())
        return tfp.distributions.Independent(
            dist, reinterpreted_batch_ndims=batch_ndims
        )

    def call(self, inputs):
        """Call."""
        inputs = super().call(inputs)

        # Delay reification until end of subclass call in order to
        # generate independent samples for each instance in batch_size.
        inputs_concentration = keras.ops.take(
            self.embeddings.distribution.concentration, inputs, axis=0
        )
        inputs_rate = keras.ops.take(self.embeddings.distribution.rate, inputs, axis=0)

        # [inputs_concetration, inputs_rate] = super().call(inputs)
        # Use reparameterization trick.
        dist_batch = tfp.distributions.Gamma(inputs_concentration, inputs_rate)
        # Reify output using samples.
        return dist_batch.sample(self.sample_shape)

    def get_config(self):
        """Return layer configuration."""
        config = super(EmbeddingGammaDiag, self).get_config()
        config.update(
            {
                "concentration_initializer": keras.initializers.serialize(
                    self.concentration_initializer
                ),
                "rate_initializer": keras.initializers.serialize(self.rate_initializer),
                "concentration_regularizer": keras.regularizers.serialize(
                    self.concentration_regularizer
                ),
                "rate_regularizer": keras.regularizers.serialize(self.rate_regularizer),
                "concentration_constraint": keras.constraints.serialize(
                    self.concentration_constraint
                ),
                "rate_constraint": keras.constraints.serialize(self.rate_constraint),
                "concentration_trainable": self.concentration_trainable,
                "rate_trainable": self.rate_trainable,
            }
        )
        return config
