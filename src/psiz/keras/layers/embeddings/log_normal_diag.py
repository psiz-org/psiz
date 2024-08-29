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
    EmbeddingLogNormalDiag: A log-normal distribution embedding layer.

"""


import keras
import tensorflow_probability as tfp

from psiz.keras.layers.embeddings.loc_scale import _EmbeddingLocScale


@keras.saving.register_keras_serializable(
    package="psiz.keras.layers", name="EmbeddingLogNormalDiag"
)
class EmbeddingLogNormalDiag(_EmbeddingLocScale):
    """A distribution-based embedding.

    Each embedding point is characterized by a Log-Normal distribution
    with a diagonal scale matrix.

    """

    def __init__(self, input_dim, output_dim, **kwargs):
        """Initialize."""
        # Pass in an alternative "default" scale initializer.
        scale_initializer = kwargs.pop("scale_initializer", None)
        if scale_initializer is None:
            scale_initializer = keras.initializers.RandomNormal(
                mean=tfp.math.softplus_inverse(2.0).numpy(), stddev=0.01
            )

        self.loc = None
        self.untransformed_scale = None
        super(EmbeddingLogNormalDiag, self).__init__(
            input_dim, output_dim, scale_initializer=scale_initializer, **kwargs
        )

    def build(self, input_shape=None):
        """Build embeddings distribution."""
        if self.built:
            return

        # Handle location variables.
        self.loc = self.add_weight(
            shape=[self.input_dim, self.output_dim],
            initializer=self.loc_initializer,
            name="loc",
            regularizer=self.loc_regularizer,
            trainable=self.loc_trainable,
            constraint=self.loc_constraint,
        )

        # Handle scale variables.
        self.untransformed_scale = self.add_weight(
            shape=[self.input_dim, self.output_dim],
            initializer=self.scale_initializer,
            name="untransformed_scale",
            regularizer=self.scale_regularizer,
            trainable=self.scale_trainable,
            constraint=self.scale_constraint,
        )

    @property
    def scale(self):
        """Return embeddings."""
        scale = keras.backend.epsilon() + keras.ops.softplus(self.untransformed_scale)
        return scale

    @property
    def embeddings(self):
        """Return embeddings."""
        dist = tfp.distributions.LogNormal(loc=self.loc, scale=self.scale)
        batch_ndims = keras.ops.size(dist.batch_shape_tensor())
        return tfp.distributions.Independent(
            dist, reinterpreted_batch_ndims=batch_ndims
        )

    def call(self, inputs):
        """Call."""
        [inputs_loc, inputs_scale] = super().call(inputs)
        # Use reparameterization trick.
        dist_batch = tfp.distributions.LogNormal(loc=inputs_loc, scale=inputs_scale)
        # Reify output using samples.
        return dist_batch.sample(self.sample_shape)
