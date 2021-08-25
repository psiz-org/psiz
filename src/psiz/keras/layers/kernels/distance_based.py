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
    DistanceBased: A distance-based similarity kernel.

"""

import tensorflow as tf

from psiz.keras.layers.distances.mink import Minkowski
from psiz.keras.layers.similarities.exponential import ExponentialSimilarity


@tf.keras.utils.register_keras_serializable(
    package='psiz.keras.layers', name='DistanceBased'
)
class DistanceBased(tf.keras.layers.Layer):
    """A distance-based kernel layer."""

    def __init__(self, distance=None, similarity=None, **kwargs):
        """Initialize."""
        super(DistanceBased, self).__init__(**kwargs)

        if distance is None:
            distance = Minkowski()
        self.distance = distance

        if similarity is None:
            similarity = ExponentialSimilarity()
        self.similarity = similarity

    def call(self, inputs):
        """Call.

        Compute k(z_0, z_1), where `k` is the pairwise kernel function.

        Arguments:
            inputs: A list of two tf.Tensor's, plus an optional third
                Tensor. The first two tensors representing coordinates
                the pairwise coordinates for which to compute
                similarity. The coordinate tensors are assumed be at
                least rank 2, where the first axis indicates the batch
                size and the last axis indicates the dimensionality of
                the coordinate space.
                shape=(batch_size, [n, m, ...] n_dim)
                The optional third tensor is assumed to be rank-2 and
                indicates group membership.
                shape=(batch_size, n_col)

        Returns:
            sim_qr: A tf.Tensor of pairwise similarites.
                shape=(batch_size, [n, m, ...])

        """
        # Compute distances (element-wise between last dimension)
        dist_qr = self.distance(inputs)

        # Compute similarity.
        return self.similarity(dist_qr)

    def build(self, input_shape):
        """Build."""
        self.distance.build(input_shape)
        distance_output_shape = self.distance.compute_output_shape(input_shape)
        # Clear any losses that were created during `compute_output_shape`.
        # pylint: disable=protected-access
        self.distance._clear_losses()
        self.similarity.build(distance_output_shape)
        super().build(input_shape)

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

    def compute_output_shape(self, input_shape):
        dist_output_shape = self.distance.compute_output_shape(input_shape)
        kernel_output_shape = self.similarity.compute_output_shape(
            dist_output_shape
        )
        return kernel_output_shape
