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
"""Module for a TensorFlow EmbeddingND.

Classes:
    EmbeddingND: An embedding layer that accepts n-dimensional inputs.

"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


@tf.keras.utils.register_keras_serializable(
    package='psiz.keras.layers', name='EmbeddingND'
)
class EmbeddingND(tf.keras.layers.Layer):
    """An embedding layer that accepts n-dimensional input indices.

    Implments multi-dimensional input semantics by mapping multi-index
    to a linear index which is then used to access a "flat" embedding
    layer. This allows existing "flat" embedding layers to easily
    be reused. The `output_dim` stays the same as the source embedding,
    only the semantics associated with `input_dim` changes.

    """
    def __init__(self, embedding=None, input_dims=None, **kwargs):
        """Initialize.

        Arguments:
            embedding: An embedding layer that will serve as the
                "physical" embedding. The physical embedding will host
                multiple logical embeddings.
            input_dims: A list of integers indicating the input shape
                of the embedding layer. Reshape follows 'C' reshape
                rules. This list should *not* include the `output_dim`.
            kwargs (optional): Additional key-word arguments for
                tf.keras.layers.Layer initialization.

        """
        super(EmbeddingND, self).__init__(**kwargs)

        # If no `input_dims` provided, do not reshape.
        if input_dims is None:
            input_dims = [embedding.input_dim]

        # Check that `input_dims` is compatible with provided embedding
        # layer.
        input_dim_reshape = np.prod(input_dims)
        if embedding.input_dim != input_dim_reshape:
            raise ValueError(
                'The provided `input_dims` and `embedding` are not shape '
                'compatible. The provided `embedding` has input_dim={0}, '
                'which cannot be reshaped to ({1}).'.format(
                    embedding.input_dim, input_dims
                )
            )

        self.input_dims = input_dims
        self.embedding = embedding

    def build(self, input_shape):
        """Build."""
        super().build(input_shape)
        self.embedding.build(input_shape)

    def call(self, multi_index):
        """Call.

        Arguments:
            multi_index: A muti-index TF tensor where the first
                dimension corresponds to the n-dimensional indices.

        Returns:
            The embedding coordinates of the 2-dimensional indices.

        """
        # Reshape input into a 2D Tensor to prepare for linear mapping.
        mi_shape = tf.shape(multi_index)
        multi_index = tf.reshape(
            multi_index, (mi_shape[0], tf.math.reduce_prod(mi_shape[1:]))
        )

        # Map multi-index to linear index.
        indices_flat = _tf_ravel_multi_index(multi_index, self.input_dims)

        # Unflattend dimensions.
        indices_flat = tf.reshape(indices_flat, mi_shape[1:])

        # Make usual call to embedding layer.
        return self.embedding(indices_flat)

    def get_config(self):
        """Return configuration."""
        config = super(EmbeddingND, self).get_config()
        config.update({
            'embedding': tf.keras.utils.serialize_keras_object(
                self.embedding
            ),
            'input_dims': self.input_dims
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

        Returns:
            An embedding (tf.Tensor or tfp.Distribution) that reshapes
            source embedding appropriately.

        """
        # TODO assumes 2D
        z_flat = self.embedding.embeddings
        if isinstance(z_flat, tfp.distributions.Distribution):
            # Assumes Independent distribution.
            z = tfp.distributions.BatchReshape(
                z_flat.distribution,
                [self.n_logical, self.input_dim, self.output_dim]
            )
            batch_ndims = tf.size(z.batch_shape_tensor())
            z = tfp.distributions.Independent(
                z, reinterpreted_batch_ndims=batch_ndims
            )
        else:
            z = tf.reshape(
                z_flat, [self.n_logical, self.input_dim, self.output_dim]
            )
        return z

    @property
    def output_dim(self):
        return self.embedding.output_dim

    @property
    def mask_zero(self):
        return self.embedding.mask_zero


def _tf_ravel_multi_index(multi_index, input_dims):
    strides = tf.math.cumprod(input_dims, axis=0, exclusive=True, reverse=True)
    adj = multi_index * tf.expand_dims(strides, 1)
    return tf.reduce_sum(adj, axis=0)
