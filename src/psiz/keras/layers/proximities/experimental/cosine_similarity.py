# -*- coding: utf-8 -*-
# Copyright 2023 The PsiZ Authors. All Rights Reserved.
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
"""CosineSimilarity pairwise proximity layer.

Classes:
    CosineSimilarity: A TensorFlow layer for computing the (weighted)
        cosine similarity between pairs of vectors.

"""

import tensorflow as tf
from tensorflow.keras import backend

from psiz.keras.layers.proximities.proximity import Proximity


@tf.keras.utils.register_keras_serializable(
    package="psiz.keras.layers", name="CosineSimilarity"
)
class CosineSimilarity(Proximity):
    """Compute the cosine similarity between pairs of vectors.

    Computes the cosine similarity:
    sum_i w_i u_i v_i / ((sum_i w_i u_i^2)(sum_i w_i v_i^2)),
    where w is a learnable vector of weights.

    Follows the `scipy` implementation of weighted cosine similarity
    (`scipy.spatial.distance.cosine`), except outputs similarity, not
    distance.

    The last axis of the input tensors is consumed in order to compute
    an inner product (see `call` method). It is assumed that both
    input tensors have the same rank, are broadcast-compatible, and
    have the same size for the last axis.

    """

    def __init__(
        self,
        w_initializer=None,
        w_regularizer=None,
        w_constraint=None,
        w_trainable=True,
        **kwargs
    ):
        """Initialize.

        Args:
            w_initializer (optional): Initializer for a vector of
            weights `w`. By default this is ones.
            w_regularizer (optional): Regularizer applied to `w`.
            w_constraint (optional): Constraint applied to `w`.
            w_trainable (optional): Boolean indicating if `w`
                is trainable.

        """
        super(CosineSimilarity, self).__init__(**kwargs)

        self.w_trainable = self.trainable and w_trainable
        if w_initializer is None:
            w_initializer = tf.keras.initializers.Ones()
        else:
            w_initializer = tf.keras.initializers.get(w_initializer)
        self.w_initializer = w_initializer
        self.w_regularizer = tf.keras.regularizers.get(w_regularizer)
        self.w_constraint = tf.keras.constraints.get(w_constraint)

    def build(self, input_shape):
        """Build."""
        n_dim = input_shape[0][-1]
        dtype = tf.as_dtype(self.dtype or backend.floatx())

        with tf.name_scope(self.name):
            self.w = self.add_weight(
                shape=[n_dim],
                initializer=self.w_initializer,
                regularizer=self.w_regularizer,
                trainable=self.w_trainable,
                name="w",
                dtype=dtype,
                constraint=self.w_constraint,
            )

    def call(self, inputs):
        """Call.

        Args:
            inputs: A list of two tf.Tensor's denoting a the set of
                vectors to compute pairwise distances. Each tensor is
                assumed to have the same shape and be at least rank-2.
                Any additional tensors in the list are ignored.
                shape = (batch_size, [n, m, ...] n_dim)

        Returns:
            shape = (batch_size, [n, m, ...])

        """
        z_0 = inputs[0]
        z_1 = inputs[1]

        # Broadcast `w` to appropriate shape.
        z_shape = tf.shape(z_0)
        w = tf.broadcast_to(self.w, z_shape)

        numer = tf.reduce_sum(w * z_0 * z_1, axis=-1)
        denom_0 = tf.sqrt(tf.reduce_sum(w * z_0 * z_0, axis=-1))
        denom_1 = tf.sqrt(tf.reduce_sum(w * z_1 * z_1, axis=-1))
        s = numer / (denom_0 * denom_1)

        return self.activation(s)

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        config.update(
            {
                "w_initializer": tf.keras.initializers.serialize(self.w_initializer),
                "w_regularizer": tf.keras.regularizers.serialize(self.w_regularizer),
                "w_constraint": tf.keras.constraints.serialize(self.w_constraint),
                "w_trainable": self.w_trainable,
            }
        )
        return config
