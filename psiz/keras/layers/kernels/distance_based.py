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
    DistanceBased: A distance-based kernel.

"""

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K

import psiz.keras.constraints as pk_constraints
import psiz.keras.initializers as pk_initializers
from psiz.keras.layers.distances.mink import Minkowski
from psiz.keras.layers.group_level import GroupLevel
from psiz.keras.layers.similarities.exponential import ExponentialSimilarity
from psiz.keras.layers.variational import Variational
from psiz.keras.layers.embeddings.deterministic import EmbeddingDeterministic


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

        Compute k(z_0, z_1), where `k` is the pairwise function.

        Arguments:
            inputs: A tf.Tensor representing coordinates. The tensor is
                assumed be at least rank 3, where the last two
                dimensions have specific semantics: the dimensionality
                of the space and the element-wise pairs.
                shape=([n_sample,] batch_size, [n, m, ...] n_dim, 2)

        Returns:
            sim_qr: A tf.Tensor of similarites.
                shape=([n_sample,] batch_size, [n, m, ...])

        """
        # Compute distances (element-wise between last dimension)
        dist_qr = self.distance(inputs)

        # Compute similarity.
        return self.similarity(dist_qr)

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
