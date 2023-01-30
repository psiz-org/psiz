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
"""Module of TensorFlow embedding layers.

Classes:
    EmbeddingTruncatedNormalDiag: A truncated normal distribution
        embedding layer.

"""

import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow_probability as tfp

from psiz.keras.layers.embeddings.loc_scale import _EmbeddingLocScale


@tf.keras.utils.register_keras_serializable(
    package="psiz.keras.layers", name="EmbeddingTruncatedNormalDiag"
)
class EmbeddingTruncatedNormalDiag(_EmbeddingLocScale):
    """A distribution-based embedding.

    Each embedding point is characterized by a Truncated Normal
    distribution with a diagonal scale matrix.

    """

    def __init__(self, input_dim, output_dim, low=0.0, high=1000000.0, **kwargs):
        """Initialize."""
        self.low = low
        self.high = high
        # Intercept constraints.
        loc_constraint = kwargs.pop("loc_constraint", None)
        if loc_constraint is None:
            loc_constraint = tf.keras.constraints.NonNeg()
        kwargs.update({"loc_constraint": loc_constraint})
        self.loc = None
        self.untransformed_scale = None
        super(EmbeddingTruncatedNormalDiag, self).__init__(
            input_dim, output_dim, **kwargs
        )

    def _build_embeddings_distribution(self, dtype):
        """Build embeddings distribution."""
        # Handle location variables.
        self.loc = self.add_weight(
            name="loc",
            shape=[self.input_dim, self.output_dim],
            dtype=dtype,
            initializer=self.loc_initializer,
            regularizer=self.loc_regularizer,
            trainable=self.loc_trainable,
            constraint=self.loc_constraint,
        )

        # Handle scale variables.
        self.untransformed_scale = self.add_weight(
            name="untransformed_scale",
            shape=[self.input_dim, self.output_dim],
            dtype=dtype,
            initializer=self.scale_initializer,
            regularizer=self.scale_regularizer,
            trainable=self.scale_trainable,
            constraint=self.scale_constraint,
        )
        scale = tfp.util.DeferredTensor(
            self.untransformed_scale, lambda x: (K.epsilon() + tf.nn.softplus(x))
        )

        dist = tfp.distributions.TruncatedNormal(self.loc, scale, self.low, self.high)
        batch_ndims = tf.size(dist.batch_shape_tensor())
        return tfp.distributions.Independent(
            dist, reinterpreted_batch_ndims=batch_ndims
        )

    def call(self, inputs):
        """Call."""
        [inputs_loc, inputs_scale] = super().call(inputs)
        # Use reparameterization trick.
        dist_batch = tfp.distributions.TruncatedNormal(
            inputs_loc, inputs_scale, self.low, self.high
        )
        # Reify output using samples.
        return dist_batch.sample(self.sample_shape)

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        config.update(
            {
                "low": float(self.low),
                "high": float(self.high),
            }
        )
        return config
