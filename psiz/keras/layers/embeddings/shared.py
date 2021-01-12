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
    EmbeddingShared: An embedding layer that shares weights across
        stimuli and dimensions.

"""

import tensorflow as tf
import tensorflow_probability as tfp


@tf.keras.utils.register_keras_serializable(
    package='psiz.keras.layers', name='EmbeddingShared'
)
class EmbeddingShared(tf.keras.layers.Layer):
    """A class for wrapping a shared Embedding."""
    def __init__(
            self, input_dim, output_dim, embedding, mask_zero=False,
            **kwargs):
        """Initialize.

        Arguments:
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
        super().build(input_shape)
        self._embedding.build(input_shape)

    def call(self, inputs):
        """Call."""
        # Intercept inputs.
        inputs = tf.zeros_like(inputs)
        outputs = self._embedding(inputs)
        return outputs

    def get_config(self):
        """Return configuration."""
        config = super(EmbeddingShared, self).get_config()
        config.update({
            'input_dim': int(self.input_dim),
            'output_dim': int(self.output_dim),
            'embedding': tf.keras.utils.serialize_keras_object(
                self._embedding
            ),
            'mask_zero': self.mask_zero,
        })
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
        config['embedding'] = tf.keras.layers.deserialize(
            config['embedding']
        )
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
        b = tfp.bijectors.Reshape(
            event_shape_out=tf.TensorShape([]),
            event_shape_in=tf.TensorShape([1, 1])
        )

        # Second, use Sample to expand event_shape to,
        # [self.input_dim, self.output.dim].
        dist = tfp.distributions.TransformedDistribution(
            distribution=tfp.distributions.Sample(
                self._embedding.embeddings,
                sample_shape=[self.input_dim, self.output_dim]
            ), bijector=b,
        )
        return dist
