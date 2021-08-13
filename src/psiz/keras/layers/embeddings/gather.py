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
    EmbeddingGather: An embedding layer that remaps a source embedding.

"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


@tf.keras.utils.register_keras_serializable(
    package='psiz.keras.layers', name='EmbeddingGather'
)
class EmbeddingGather(tf.keras.layers.Layer):
    """A class for mapping Embedding inputs."""
    def __init__(
            self, embedding=None, input_map=None, **kwargs):
        """Initialize.

        Arguments:
            embedding: An embedding layer.
            input_map (optional):
            kwargs: Additional key-word arguments.

        """
        super(EmbeddingGather, self).__init__(**kwargs)
        self._embedding = embedding

        # Make sure provided `input_map` works with provided embedding.
        if np.min(input_map) < 0:
            raise ValueError(
                'Indices in `input_map` must be non-negative.'
            )
        if np.max(input_map) > (self._embedding.input_dim - 1):
            raise ValueError(
                'Indices in `input_map` must not be greater than the '
                '`input_dim` of the provided embedding.'
            )
        self.input_dim = len(input_map)
        input_map = tf.constant(input_map, dtype=tf.int32)
        self.input_map = input_map
        self._is_distribution = None

    @property
    def mask_zero(self):
        """Get `mask_zero`."""
        return self._embedding.mask_zero

    @property
    def output_dim(self):
        """Get `output_dim`."""
        return self._embedding.output_dim

    def build(self, input_shape):
        """Build."""
        super().build(input_shape)
        self._embedding.build(input_shape)
        self._is_distribution = isinstance(
            self._embedding.embeddings, tfp.distributions.Distribution
        )

    def call(self, inputs):
        """Call."""
        # Intercept inputs.
        # Flatten inputs for mapping.
        inputs_shape = tf.shape(inputs)
        inputs = tf.reshape(
            inputs, [tf.reduce_prod(inputs_shape)]
        )
        # Map inputs.
        inputs = tf.gather(self.input_map, inputs)
        # Unflatten
        inputs = tf.reshape(inputs, inputs_shape)

        outputs = self._embedding(inputs)
        return outputs

    def get_config(self):
        """Return configuration."""
        config = super(EmbeddingGather, self).get_config()
        config.update({
            'embedding': tf.keras.utils.serialize_keras_object(
                self._embedding
            ),
            'input_map': self.input_map.numpy().tolist()
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
        """Getter method for `embeddings`."""
        if self._is_distribution:
            z_mapped = self._embedding[self.input_map]
        else:
            z = self._embedding.embeddings
            z_mapped = tf.gather(z, self.input_map)

        return z_mapped
