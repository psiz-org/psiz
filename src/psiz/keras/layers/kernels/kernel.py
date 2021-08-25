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
"""Module of TensorFlow kernel layers.

Classes:
    Kernel: DEPRECATED A kernel that allows the user to separately
        specify a distance and similarity function.
    AttentionKernel: DEPRECATED A kernel that uses group-specific
        attention weights and allows the user to separately specify a
        distance and similarity function.

"""

import numpy as np
import tensorflow as tf

import psiz.keras.constraints as pk_constraints
import psiz.keras.initializers as pk_initializers
from psiz.keras.layers.distances.minkowski import WeightedMinkowski
from psiz.keras.layers.group_level import GroupLevel
from psiz.keras.layers.similarities.exponential import ExponentialSimilarity


@tf.keras.utils.register_keras_serializable(
    package='psiz.keras.layers', name='Kernel'
)
class Kernel(GroupLevel):
    """A basic population-wide kernel."""

    def __init__(self, distance=None, similarity=None, **kwargs):
        """Initialize."""
        super(Kernel, self).__init__(**kwargs)

        if distance is None:
            distance = WeightedMinkowski()
        self.distance = distance

        if similarity is None:
            similarity = ExponentialSimilarity()
        self.similarity = similarity

    def call(self, inputs):
        """Call.

        Compute k(z_0, z_1), where `k` is the similarity kernel.

        Note: Broadcasting rules are used to compute similarity between
            `z_0` and `z_1`.

        Arguments:
            inputs:
                z_0: A tf.Tensor denoting a set of vectors.
                    shape = (batch_size, [n, m, ...] n_dim)
                z_1: A tf.Tensor denoting a set of vectors.
                    shape = (batch_size, [n, m, ...] n_dim)

        """
        z_0 = inputs[0]
        z_1 = inputs[1]
        # group = inputs[-1][:, self.group_level]

        # Create identity attention weights.
        attention = tf.ones_like(z_0)

        # Compute distance between query and references.
        dist_qr = self.distance([z_0, z_1, attention])

        # Compute similarity.
        sim_qr = self.similarity(dist_qr)
        return sim_qr

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        config.update({
            'distance': tf.keras.utils.serialize_keras_object(self.distance),
            'similarity': tf.keras.utils.serialize_keras_object(
                self.similarity
            ),
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Create from configuration."""
        config['distance'] = tf.keras.layers.deserialize(config['distance'])
        config['similarity'] = tf.keras.layers.deserialize(
            config['similarity']
        )
        return cls(**config)


@tf.keras.utils.register_keras_serializable(
    package='psiz.keras.layers', name='AttentionKernel'
)
class AttentionKernel(GroupLevel):
    """Attention kernel container."""

    def __init__(
            self, n_dim=None, attention=None, distance=None, similarity=None,
            **kwargs):
        """Initialize.

        Arguments:
            n_dim: The dimensionality of the attention weights. This
                should match the dimensionality of the embedding.
            attention: A attention layer. If this is specified, the
                argument `n_dim` is ignored.
            distance: A distance layer.
            similarity: A similarity layer.

        """
        super(AttentionKernel, self).__init__(**kwargs)

        if attention is None:
            n_group = 1
            scale = n_dim
            alpha = np.ones((n_dim))
            attention = tf.keras.layers.Embedding(
                n_group, n_dim, mask_zero=False,
                embeddings_initializer=pk_initializers.RandomAttention(
                    alpha, scale
                ),
                embeddings_constraint=pk_constraints.NonNegNorm(
                    scale=n_dim
                ),
            ),
        self.attention = attention

        if distance is None:
            distance = WeightedMinkowski()
        self.distance = distance

        if similarity is None:
            similarity = ExponentialSimilarity()
        self.similarity = similarity

    def call(self, inputs):
        """Call.

        Compute k(z_0, z_1), where `k` is the similarity kernel.

        Note: Broadcasting rules are used to compute similarity between
            `z_0` and `z_1`.

        Arguments:
            inputs:
                z_0: A tf.Tensor denoting a set of vectors.
                    shape = (batch_size, [n, m, ...] n_dim)
                z_1: A tf.Tensor denoting a set of vectors.
                    shape = (batch_size, [n, m, ...] n_dim)
                group: A tf.Tensor denoting group assignments.
                    shape = (batch_size, k)

        """
        z_0 = inputs[0]
        z_1 = inputs[1]
        groups = inputs[-1][:, self.group_level]

        # NOTE: The remainder of the method assumes that `groups` is a
        # rank-1 tensor.

        # Adjust rank of groups by adding singleton axis to match rank of
        # z[1:-1], i.e., omitting batch axis and n_dim axis.
        reshape_shape = tf.ones([tf.rank(z_1) - 2], dtype=tf.int32)
        reshape_shape = tf.concat((tf.shape(groups), reshape_shape), 0)
        groups = tf.reshape(groups, reshape_shape)

        # Tile groups to be compatible with `z_1`, again omitting batch
        # axis and n_dim axis. TODO
        # repeats = tf.shape(z_1)[1:-1]
        # repeats = tf.concat([tf.constant([1]), repeats], 0)
        # groups = tf.tile(groups, repeats)

        # Embed group indices as attention weights.
        attention = self.attention(groups)

        # Compute distance between query and references.
        dist_qr = self.distance([z_0, z_1, attention])
        # Compute similarity.
        sim_qr = self.similarity(dist_qr)
        return sim_qr

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        config.update({
            # 'n_dim': int(self.n_dim),
            'attention': tf.keras.utils.serialize_keras_object(self.attention),
            'distance': tf.keras.utils.serialize_keras_object(self.distance),
            'similarity': tf.keras.utils.serialize_keras_object(
                self.similarity
            ),
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Create from configuration."""
        config['attention'] = tf.keras.layers.deserialize(config['attention'])
        config['distance'] = tf.keras.layers.deserialize(config['distance'])
        config['similarity'] = tf.keras.layers.deserialize(
            config['similarity']
        )
        return cls(**config)
