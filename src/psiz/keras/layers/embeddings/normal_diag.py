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
    EmbeddingNormalDiag: A normal distribution embedding layer.

"""

import keras

from psiz.keras.layers.embeddings.loc_scale import _EmbeddingLocScale
from psiz.keras.ops.scale_gradient import scale_gradient
from psiz.stochastic import get_stochastic_adapter


@keras.saving.register_keras_serializable(
    package="psiz.keras.layers", name="EmbeddingNormalDiag"
)
class EmbeddingNormalDiag(_EmbeddingLocScale):
    """A distribution-based embedding.

    Each embedding point is characterized by a Normal distribution with
    a diagonal scale matrix.

    """

    def __init__(
        self,
        input_dim,
        output_dim,
        loc_gradient_scale=1.0,
        scale_gradient_scale=1.0,
        **kwargs
    ):
        """Initialize."""
        self.loc = None
        self.untransformed_scale = None
        self.loc_gradient_scale = loc_gradient_scale
        self.scale_gradient_scale = scale_gradient_scale
        super(EmbeddingNormalDiag, self).__init__(input_dim, output_dim, **kwargs)

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
            constraint=self.loc_constraint,
            trainable=self.loc_trainable,
        )

        # Handle scale variables.
        self.untransformed_scale = self.add_weight(
            shape=[self.input_dim, self.output_dim],
            initializer=self.scale_initializer,
            name="untransformed_scale",
            regularizer=self.scale_regularizer,
            constraint=self.scale_constraint,
            trainable=self.scale_trainable,
        )

    @property
    def scale(self):
        """Return embeddings."""
        scale = keras.backend.epsilon() + keras.ops.softplus(self.untransformed_scale)
        return scale

    @property
    def embeddings(self):
        """Return embeddings."""
        adapter = get_stochastic_adapter()
        dist = adapter.normal(loc=self.loc, scale=self.scale)
        batch_ndims = keras.ops.size(dist.batch_shape_tensor())
        return adapter.independent(dist, reinterpreted_batch_ndims=batch_ndims)

    def call(self, inputs):
        """Call."""
        [inputs_loc, inputs_scale] = super().call(inputs)

        inputs_loc = scale_gradient(inputs_loc, self.loc_gradient_scale)
        inputs_scale = scale_gradient(inputs_scale, self.scale_gradient_scale)

        # Use reparameterization trick.
        adapter = get_stochastic_adapter()
        dist_batch = adapter.normal(loc=inputs_loc, scale=inputs_scale)
        # Reify output using samples.
        outputs = dist_batch.sample(self.sample_shape)
        # TODO(roads) MAYBE keras.ops.cast(outputs, dtype=self.compute_dtype)
        return outputs

    def take(self, inputs):
        """Take."""
        [inputs_loc, inputs_scale] = super().call(inputs)
        adapter = get_stochastic_adapter()
        dist = adapter.normal(loc=inputs_loc, scale=inputs_scale)
        batch_ndims = keras.ops.size(dist.batch_shape_tensor())
        return adapter.independent(dist, reinterpreted_batch_ndims=batch_ndims)

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        config.update(
            {
                "loc_gradient_scale": self.loc_gradient_scale,
                "scale_gradient_scale": self.scale_gradient_scale,
            }
        )
        return config
