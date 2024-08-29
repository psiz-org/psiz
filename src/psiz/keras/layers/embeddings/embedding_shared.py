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
    EmbeddingShared: An embedding layer that shares weights across
        stimuli and dimensions.

"""


import keras
import tensorflow_probability as tfp


@keras.saving.register_keras_serializable(
    package="psiz.keras.layers", name="EmbeddingShared"
)
class EmbeddingShared(keras.layers.Layer):
    """A class for wrapping a shared Embedding."""

    def __init__(self, input_dim, output_dim, embedding, mask_zero=False, **kwargs):
        """Initialize.

        Args:
            input_dim:
            output_dim:
            embedding: An embedding layer.
            mask_zero (optional):
            kwargs: Additional key-word arguments.

        """
        super(EmbeddingShared, self).__init__(**kwargs)
        self._embedding = embedding
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mask_zero = mask_zero

    def build(self, input_shape):
        """Build."""
        self._embedding.build(input_shape)
        super().build(input_shape)

    def call(self, inputs):
        """Call."""
        # Intercept inputs and convert to zeros because all indices share the same underlying
        # embedding coordinate.
        inputs = keras.ops.zeros_like(inputs)

        outputs = self._embedding(inputs)
        outputs = keras.ops.repeat(outputs, self.output_dim, axis=-1)
        return outputs

    def get_config(self):
        """Return configuration."""
        config = super(EmbeddingShared, self).get_config()
        config.update(
            {
                "input_dim": int(self.input_dim),
                "output_dim": int(self.output_dim),
                "embedding": keras.saving.serialize_keras_object(self._embedding),
                "mask_zero": self.mask_zero,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        """Create layer from configuration.

        This method is the reverse of `get_config`, capable of
        instantiating the same layer from the config dictionary.

        Args:
            config: A Python dictionary, typically the output of
                `get_config`.

        Returns:
            layer: A layer instance.

        """
        config["embedding"] = keras.saving.deserialize_keras_object(config["embedding"])
        return cls(**config)

    @property
    def embeddings(self):
        """Getter method for `embeddings`.

        Return distribution that creates copies of the source
        distribution for each stimulus and dimension.

        The incoming distribution has,
        batch_shape=[] event_shape=[1, 1]

        We require a distribution with,
        event_shape=[self.input_dim, self.output_dim].

        """
        # First, reshape event_shape from [1, 1] to [].
        b = tfp.bijectors.Reshape(event_shape_out=[], event_shape_in=[1, 1])

        # Second, use Sample to expand event_shape to,
        # [self.input_dim, self.output.dim].
        dist = tfp.distributions.TransformedDistribution(
            distribution=tfp.distributions.Sample(
                self._embedding.embeddings,
                sample_shape=[self.input_dim, self.output_dim],
            ),
            bijector=b,
        )
        return dist
