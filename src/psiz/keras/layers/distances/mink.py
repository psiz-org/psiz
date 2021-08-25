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
"""Minkowski pairwise distance layer.

Classes:
    Minkowski: A TensorFlow layer for computing (weighted) minkowski
        distance.

"""

import tensorflow as tf
from tensorflow.python.keras import backend as K

import psiz.keras.constraints as pk_constraints
from psiz.keras.layers.ops.core import wpnorm


@tf.keras.utils.register_keras_serializable(
    package='psiz.keras.layers', name='Minkowski'
)
class Minkowski(tf.keras.layers.Layer):
    """Minkowski pairwise distance.

    A pairwise distance layer that consumes the last axis of the input
        tensors (see `call` method).

    NOTE: It is assumed that both tensors have the same rank, are
    broadcast-compatible, and have the same size for the last axis.

    """

    def __init__(
            self, rho_trainable=True, rho_initializer=None,
            rho_regularizer=None, rho_constraint=None, w_trainable=True,
            w_initializer=None, w_regularizer=None, w_constraint=None,
            **kwargs):
        """Initialize.

        Arguments:
            rho_trainable (optional):
            rho_initializer (optional):
            rho_regularizer (optional):
            rho_constraint (optional):
            w_trainable (optional):
            w_initializer (optional):
            w_regularizer (optional):
            w_constraint (optional):

        """
        super(Minkowski, self).__init__(**kwargs)

        self.rho_trainable = self.trainable and rho_trainable
        if rho_initializer is None:
            rho_initializer = tf.random_uniform_initializer(1., 2.)
        self.rho_initializer = tf.keras.initializers.get(rho_initializer)
        self.rho_regularizer = tf.keras.regularizers.get(rho_regularizer)
        if rho_constraint is None:
            rho_constraint = pk_constraints.GreaterEqualThan(min_value=1.0)
        self.rho_constraint = tf.keras.constraints.get(rho_constraint)
        with tf.name_scope(self.name):
            self.rho = self.add_weight(
                shape=[], initializer=self.rho_initializer,
                regularizer=self.rho_regularizer, trainable=self.rho_trainable,
                name="rho", dtype=K.floatx(),
                constraint=self.rho_constraint
            )

        self.w_trainable = self.trainable and w_trainable
        if w_initializer is None:
            w_initializer = tf.random_uniform_initializer(1.01, 3.)
        self.w_initializer = tf.keras.initializers.get(w_initializer)
        self.w_regularizer = tf.keras.regularizers.get(w_regularizer)
        if w_constraint is None:
            w_constraint = tf.keras.constraints.NonNeg()
        self.w_constraint = tf.keras.constraints.get(w_constraint)

        self.w = None

    def build(self, input_shape):
        """Build."""
        with tf.name_scope(self.name):
            self.w = self.add_weight(
                shape=[input_shape[0][-1]], initializer=self.w_initializer,
                regularizer=self.w_regularizer, trainable=self.w_trainable,
                name="w", dtype=K.floatx(), constraint=self.w_constraint
            )

    def call(self, inputs):
        """Call.

        Arguments:
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
        x = z_0 - z_1

        # Broadcast `rho` and `w` to appropriate shape.
        x_shape = tf.shape(x)
        # Broadcast `rho` to shape=(batch_size, [n, m, ...]).
        rho = self.rho * tf.ones(x_shape[0:-1])
        # Broadcast `w` to shape=(batch_size, [n, m, ...] n_dim).
        w = tf.broadcast_to(self.w, x_shape)

        # Weighted Minkowski distance.
        d_qr = wpnorm(x, w, rho)
        d_qr = tf.squeeze(d_qr, [-1])
        return d_qr

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        config.update({
            'rho_initializer':
                tf.keras.initializers.serialize(self.rho_initializer),
            'w_initializer':
                tf.keras.initializers.serialize(self.w_initializer),
            'rho_regularizer':
                tf.keras.regularizers.serialize(self.rho_regularizer),
            'w_regularizer':
                tf.keras.regularizers.serialize(self.w_regularizer),
            'rho_constraint':
                tf.keras.constraints.serialize(self.rho_constraint),
            'w_constraint':
                tf.keras.constraints.serialize(self.w_constraint),
            'rho_trainable': self.rho_trainable,
            'w_trainable': self.w_trainable,
        })
        return config
