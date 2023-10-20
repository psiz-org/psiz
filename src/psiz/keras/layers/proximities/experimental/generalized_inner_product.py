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
"""GeneralizedInnerProduct pairwise proximity layer.

Classes:
    GeneralizedInnerProduct: A TensorFlow layer for computing the
        generalized inner product between pairs of vectors.

"""

import tensorflow as tf
from tensorflow.keras import backend

from psiz.keras.layers.proximities.proximity import Proximity


@tf.keras.utils.register_keras_serializable(
    package="psiz.keras.layers", name="GeneralizedInnerProduct"
)
class GeneralizedInnerProduct(Proximity):
    """Compute the inner product between pairs of vectors.

    Computes the inner product z_i^T W z_j, where z_i and z_j exist on
    R^n and W exists on R^n x R^n. By default, no constraints are
    placed on the matrix W.

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
            w_initializer (optional): Initializer for a kxk weight
                matrix that forms the inner part of the generalized
                inner product. By default this is initialized to the
                identity matrix.
            w_regularizer (optional): Regularizer applied to `w`.
            w_constraint (optional): Constraint applied to `w`.
            w_trainable (optional): Boolean indicating if `w`
                is trainable.

        """
        super(GeneralizedInnerProduct, self).__init__(**kwargs)

        self.w_trainable = self.trainable and w_trainable
        if w_initializer is None:
            w_initializer = tf.keras.initializers.Identity(gain=1.0)
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
                shape=[n_dim, n_dim],
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

        # Add dummy axis to achieve batch dot product.
        z_0 = tf.expand_dims(z_0, -2)
        z_1 = tf.expand_dims(
            z_1, -1
        )  # NOTE: `axis=-1` is intentional to acheive transpose
        d = tf.matmul(tf.matmul(z_0, self.w), z_1)
        # Remove dummy and vetigal axis.
        d = tf.squeeze(d, [-2, -1])

        return self.activation(d)

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
