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
    EmbeddingTake: An embedding layer that remaps a source embedding.

"""

import numpy as np
import keras

from psiz.stochastic import is_distribution


@keras.saving.register_keras_serializable(
    package="psiz.keras.layers", name="EmbeddingTake"
)
class EmbeddingTake(keras.layers.Layer):
    """A class for mapping Embedding inputs."""

    def __init__(self, embedding=None, input_map=None, **kwargs):
        """Initialize.

        Arguments:
            embedding: An embedding layer.
            input_map: Mapping from internal embedding to externally
                exposed embedding.
            kwargs: Additional key-word arguments.

        """
        super(EmbeddingTake, self).__init__(**kwargs)
        self._embedding = embedding

        # Make sure provided `input_map` works with provided embedding.
        if np.min(input_map) < 0:
            raise ValueError("Indices in `input_map` must be non-negative.")
        if np.max(input_map) > (self._embedding.input_dim - 1):
            raise ValueError(
                "Indices in `input_map` must not be greater than the "
                "`input_dim` of the provided embedding."
            )
        self.input_dim = len(input_map)
        self.input_map = input_map.astype("int32")
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
        self._is_distribution = is_distribution(self._embedding.embeddings)

    def call(self, inputs):
        """Call."""
        # Intercept inputs.
        # Flatten inputs for mapping.
        inputs_shape = keras.ops.shape(inputs)
        inputs = keras.ops.reshape(inputs, [keras.ops.prod(inputs_shape)])
        # Map inputs.
        inputs = keras.ops.take(self.input_map, inputs)
        # Unflatten
        inputs = keras.ops.reshape(inputs, inputs_shape)

        outputs = self._embedding(inputs)
        return outputs

    def get_config(self):
        """Return configuration."""
        config = super(EmbeddingTake, self).get_config()
        config.update(
            {
                "embedding": keras.saving.serialize_keras_object(self._embedding),
                "input_map": self.input_map.tolist(),
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
        config["input_map"] = np.array(config["input_map"])
        return cls(**config)

    @property
    def embeddings(self):
        """Getter method for `embeddings`."""
        if self._is_distribution:
            z_mapped = self._embedding.take(self.input_map)
        else:
            z = self._embedding.embeddings
            z_mapped = keras.ops.take(z, self.input_map, axis=0)

        return z_mapped
