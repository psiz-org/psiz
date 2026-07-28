# -*- coding: utf-8 -*-
# Copyright 2026 The PsiZ Authors. All Rights Reserved.
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
    EmbeddingNonCenteredNormalDiag: A non-centered parameterization
        embedding layer with gradient scaling.

"""

import keras

from psiz.keras.layers.embeddings.stochastic_embedding import StochasticEmbedding
from psiz.keras.ops.scale_gradient import scale_gradient
from psiz.stochastic import get_stochastic_adapter
from psiz.stochastic import softplus_inverse


@keras.saving.register_keras_serializable(
    package="psiz.keras.layers", name="EmbeddingNonCenteredNormalDiag"
)
class EmbeddingNonCenteredNormalDiag(StochasticEmbedding):
    """A non-centered parameterization embedding with gradient scaling.

    This layer implements a non-centered parameterization where:
        z = mu + sigma * epsilon
    where mu and sigma come from a base embedding layer, and epsilon is learned
    by this layer.

    """

    def __init__(
        self,
        base_embedding,
        epsilon_loc_initializer=None,
        epsilon_scale_initializer=None,
        epsilon_loc_regularizer=None,
        epsilon_scale_regularizer=None,
        epsilon_loc_constraint=None,
        epsilon_scale_constraint=None,
        epsilon_loc_trainable=True,
        epsilon_scale_trainable=True,
        epsilon_loc_gradient_scale=1.0,
        epsilon_scale_gradient_scale=1.0,
        sample_shape=(),
        **kwargs,
    ):
        """Initialize."""
        if not hasattr(base_embedding, "input_dim") or not hasattr(
            base_embedding, "output_dim"
        ):
            raise ValueError(
                "base_embedding must have 'input_dim' and 'output_dim' attributes"
            )

        input_dim = kwargs.pop("input_dim", None)
        output_dim = kwargs.pop("output_dim", None)
        mask_zero = kwargs.pop("mask_zero", None)
        if input_dim is None:
            input_dim = base_embedding.input_dim
        if output_dim is None:
            output_dim = base_embedding.output_dim
        if mask_zero is None:
            mask_zero = getattr(base_embedding, "mask_zero", False)

        super(EmbeddingNonCenteredNormalDiag, self).__init__(
            input_dim,
            output_dim,
            mask_zero=mask_zero,
            sample_shape=sample_shape,
            **kwargs,
        )

        self.base_embedding = base_embedding
        self.epsilon_loc = None
        self.epsilon_untransformed_scale = None
        self.epsilon_loc_gradient_scale = epsilon_loc_gradient_scale
        self.epsilon_scale_gradient_scale = epsilon_scale_gradient_scale

        if epsilon_loc_initializer is None:
            epsilon_loc_initializer = keras.initializers.RandomNormal(
                mean=0.0, stddev=0.01
            )
        self.epsilon_loc_initializer = keras.initializers.get(epsilon_loc_initializer)
        if epsilon_scale_initializer is None:
            epsilon_scale_initializer = keras.initializers.RandomNormal(
                mean=keras.ops.convert_to_numpy(softplus_inverse(1.0)), stddev=0.001
            )
        self.epsilon_scale_initializer = keras.initializers.get(
            epsilon_scale_initializer
        )

        self.epsilon_loc_regularizer = keras.regularizers.get(epsilon_loc_regularizer)
        self.epsilon_scale_regularizer = keras.regularizers.get(
            epsilon_scale_regularizer
        )

        self.epsilon_loc_constraint = keras.constraints.get(epsilon_loc_constraint)
        self.epsilon_scale_constraint = keras.constraints.get(epsilon_scale_constraint)

        self.epsilon_loc_trainable = self.trainable and epsilon_loc_trainable
        self.epsilon_scale_trainable = self.trainable and epsilon_scale_trainable

    def build(self, input_shape=None):
        """Build the layer."""
        if self.built:
            return

        if not self.base_embedding.built:
            self.base_embedding.build(input_shape)

        self.epsilon_loc = self.add_weight(
            shape=[self.input_dim, self.output_dim],
            initializer=self.epsilon_loc_initializer,
            name="epsilon_loc",
            regularizer=self.epsilon_loc_regularizer,
            constraint=self.epsilon_loc_constraint,
            trainable=self.epsilon_loc_trainable,
        )

        self.epsilon_untransformed_scale = self.add_weight(
            shape=[self.input_dim, self.output_dim],
            initializer=self.epsilon_scale_initializer,
            name="epsilon_untransformed_scale",
            regularizer=self.epsilon_scale_regularizer,
            constraint=self.epsilon_scale_constraint,
            trainable=self.epsilon_scale_trainable,
        )

        super(EmbeddingNonCenteredNormalDiag, self).build(input_shape)

    @property
    def prior_loc(self):
        """Return the mean from base embedding."""
        return self.base_embedding.embeddings.mean()

    @property
    def prior_scale(self):
        """Return the scale from base embedding."""
        return self.base_embedding.embeddings.stddev()

    @property
    def epsilon_scale(self):
        """Return epsilon scale."""
        return keras.backend.epsilon() + keras.ops.softplus(
            self.epsilon_untransformed_scale
        )

    @property
    def loc(self):
        """Return posterior mean."""
        return self.prior_loc + self.prior_scale * self.epsilon_loc

    @property
    def scale(self):
        """Return posterior scale."""
        return self.prior_scale * self.epsilon_scale

    @property
    def embeddings(self):
        """Return posterior embeddings distribution."""
        adapter = get_stochastic_adapter()
        dist = adapter.normal(loc=self.loc, scale=self.scale)
        batch_ndims = keras.ops.size(dist.batch_shape_tensor())
        return adapter.independent(dist, reinterpreted_batch_ndims=batch_ndims)

    @property
    def epsilon_embeddings(self):
        """Return epsilon embeddings distribution."""
        adapter = get_stochastic_adapter()
        dist = adapter.normal(loc=self.epsilon_loc, scale=self.epsilon_scale)
        batch_ndims = keras.ops.size(dist.batch_shape_tensor())
        return adapter.independent(dist, reinterpreted_batch_ndims=batch_ndims)

    def call(self, inputs):
        """Call."""
        inputs = super().call(inputs)

        inputs_prior_loc = keras.ops.take(self.prior_loc, inputs, axis=0)
        inputs_prior_scale = keras.ops.take(self.prior_scale, inputs, axis=0)

        inputs_epsilon_loc = keras.ops.take(self.epsilon_loc, inputs, axis=0)
        inputs_epsilon_scale = keras.ops.take(self.epsilon_scale, inputs, axis=0)
        inputs_epsilon_scale = keras.ops.clip(inputs_epsilon_scale, 1e-4, 50.0)

        inputs_epsilon_loc = scale_gradient(
            inputs_epsilon_loc, self.epsilon_loc_gradient_scale
        )
        inputs_epsilon_scale = scale_gradient(
            inputs_epsilon_scale, self.epsilon_scale_gradient_scale
        )

        adapter = get_stochastic_adapter()
        dist_batch = adapter.normal(loc=inputs_epsilon_loc, scale=inputs_epsilon_scale)
        epsilon_samples = dist_batch.sample(self.sample_shape)

        return inputs_prior_loc + inputs_prior_scale * epsilon_samples

    def take(self, inputs):
        """Return distribution for specific indices."""
        inputs = super().call(inputs)

        inputs_loc = keras.ops.take(self.loc, inputs, axis=0)
        inputs_scale = keras.ops.take(self.scale, inputs, axis=0)

        adapter = get_stochastic_adapter()
        dist = adapter.normal(loc=inputs_loc, scale=inputs_scale)
        batch_ndims = keras.ops.size(dist.batch_shape_tensor())
        return adapter.independent(dist, reinterpreted_batch_ndims=batch_ndims)

    def get_config(self):
        """Return layer configuration."""
        config = super(EmbeddingNonCenteredNormalDiag, self).get_config()
        config.update(
            {
                "base_embedding": keras.saving.serialize_keras_object(
                    self.base_embedding
                ),
                "epsilon_loc_initializer": keras.initializers.serialize(
                    self.epsilon_loc_initializer
                ),
                "epsilon_scale_initializer": keras.initializers.serialize(
                    self.epsilon_scale_initializer
                ),
                "epsilon_loc_regularizer": keras.regularizers.serialize(
                    self.epsilon_loc_regularizer
                ),
                "epsilon_scale_regularizer": keras.regularizers.serialize(
                    self.epsilon_scale_regularizer
                ),
                "epsilon_loc_constraint": keras.constraints.serialize(
                    self.epsilon_loc_constraint
                ),
                "epsilon_scale_constraint": keras.constraints.serialize(
                    self.epsilon_scale_constraint
                ),
                "epsilon_loc_trainable": self.epsilon_loc_trainable,
                "epsilon_scale_trainable": self.epsilon_scale_trainable,
                "epsilon_loc_gradient_scale": float(self.epsilon_loc_gradient_scale),
                "epsilon_scale_gradient_scale": float(
                    self.epsilon_scale_gradient_scale
                ),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        """Create layer from config."""
        config = dict(config)
        config["base_embedding"] = keras.saving.deserialize_keras_object(
            config["base_embedding"]
        )
        return super().from_config(config)
