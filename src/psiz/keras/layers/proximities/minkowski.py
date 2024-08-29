# -*- coding: utf-8 -*-
# Copyright 2024 The PsiZ Authors. All Rights Reserved.
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


import keras

import psiz.keras.constraints as pk_constraints
from psiz.keras.ops.wpnorm import wpnorm
from psiz.keras.layers.proximities.proximity import Proximity


@keras.saving.register_keras_serializable(package="psiz.keras.layers", name="Minkowski")
class Minkowski(Proximity):
    """Minkowski pairwise distance.

    A pairwise Minkowski distance layer that consumes the last axis of
        the input tensors (see `call` method).

    NOTE: It is assumed that both tensors have the same rank, are
    broadcast-compatible, and have the same size for the last axis.

    """

    def __init__(
        self,
        rho_trainable=True,
        rho_initializer=None,
        rho_regularizer=None,
        rho_constraint=None,
        w_trainable=True,
        w_initializer=None,
        w_regularizer=None,
        w_constraint=None,
        **kwargs
    ):
        """Initialize.

        Args:
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
            rho_initializer = keras.initializers.RandomUniform(minval=1.0, maxval=2.0)
        self.rho_initializer = keras.initializers.get(rho_initializer)
        self.rho_regularizer = keras.regularizers.get(rho_regularizer)
        if rho_constraint is None:
            rho_constraint = pk_constraints.GreaterEqualThan(min_value=1.0)
        self.rho_constraint = keras.constraints.get(rho_constraint)
        with keras.name_scope(self.name):
            self.rho = self.add_weight(
                shape=[],
                initializer=self.rho_initializer,
                regularizer=self.rho_regularizer,
                trainable=self.rho_trainable,
                name="rho",
                dtype=keras.backend.floatx(),
                constraint=self.rho_constraint,
            )

        self.w_trainable = self.trainable and w_trainable
        if w_initializer is None:
            w_initializer = keras.initializers.RandomUniform(minval=1.01, maxval=3.0)
        self.w_initializer = keras.initializers.get(w_initializer)
        self.w_regularizer = keras.regularizers.get(w_regularizer)
        if w_constraint is None:
            w_constraint = keras.constraints.NonNeg()
        self.w_constraint = keras.constraints.get(w_constraint)

    def build(self, input_shape):
        """Build."""
        with keras.name_scope(self.name):
            self.w = self.add_weight(
                shape=[input_shape[0][-1]],
                initializer=self.w_initializer,
                regularizer=self.w_regularizer,
                trainable=self.w_trainable,
                name="w",
                constraint=self.w_constraint,
            )
        #  NOTE: Calling super because Proximity.build() builds activation layer.
        super().build(input_shape)

    def call(self, inputs):
        """Call.

        Args:
            inputs: A list of two tensors denoting a the set of
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
        x_shape = keras.ops.shape(x)
        # Broadcast `rho` to shape=(batch_size, [n, m, ...]).
        rho = self.rho * keras.ops.ones(x_shape[0:-1])
        # Broadcast `w` to shape=(batch_size, [n, m, ...] n_dim).
        w = keras.ops.broadcast_to(self.w, x_shape)

        # Weighted Minkowski distance.
        d_qr = wpnorm(x, w, rho)
        return self.activation(d_qr)

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        config.update(
            {
                "rho_initializer": keras.initializers.serialize(self.rho_initializer),
                "w_initializer": keras.initializers.serialize(self.w_initializer),
                "rho_regularizer": keras.regularizers.serialize(self.rho_regularizer),
                "w_regularizer": keras.regularizers.serialize(self.w_regularizer),
                "rho_constraint": keras.constraints.serialize(self.rho_constraint),
                "w_constraint": keras.constraints.serialize(self.w_constraint),
                "rho_trainable": self.rho_trainable,
                "w_trainable": self.w_trainable,
            }
        )
        return config
