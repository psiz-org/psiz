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
"""Weighted Minkowski distance layer.

Classes:
    WeightedMinkowski: DEPRECATED A TensorFlow layer for computing
        weighted minkowski distance.

"""

import tensorflow as tf
from tensorflow.python.keras import backend as K

import psiz.keras.constraints as pk_constraints
from psiz.keras.layers.ops.core import wpnorm


@tf.keras.utils.register_keras_serializable(
    package='psiz.keras.layers', name='WeightedMinkowski'
)
class WeightedMinkowski(tf.keras.layers.Layer):
    """Weighted Minkowski distance."""

    def __init__(self, rho_initializer=None, **kwargs):
        """Initialize.

        Arguments:
            rho_initializer (optional): Initializer for rho.

        """
        super(WeightedMinkowski, self).__init__(**kwargs)

        if rho_initializer is None:
            rho_initializer = tf.random_uniform_initializer(1.01, 3.)
        self.rho_initializer = tf.keras.initializers.get(rho_initializer)
        self.rho = self.add_weight(
            shape=[], initializer=self.rho_initializer,
            trainable=self.trainable, name="rho", dtype=K.floatx(),
            constraint=pk_constraints.GreaterThan(min_value=1.0)
        )

    def call(self, inputs):
        """Call.

        Arguments:
            inputs:
                z_0: A tf.Tensor denoting a set of vectors.
                    shape = (batch_size, [n, m, ...] n_dim)
                z_1: A tf.Tensor denoting a set of vectors.
                    shape = (batch_size, [n, m, ...] n_dim)
                w: The weights allocated to each dimension
                    in a weighted minkowski metric.
                    shape = (batch_size, [n, m, ...] n_dim)

        Returns:
            shape = (batch_size, [n, m, ...])

        """
        z_0 = inputs[0]  # Query.
        z_1 = inputs[1]  # References.
        w = inputs[2]    # Dimension weights.

        # Expand rho to shape=(sample_size, batch_size, [n, m, ...]).
        rho = self.rho * tf.ones(tf.shape(z_0)[0:-1])

        # Weighted Minkowski distance.
        x = z_0 - z_1
        d_qr = wpnorm(x, w, rho)
        d_qr = tf.squeeze(d_qr, [-1])
        return d_qr

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        config.update({
            'rho_initializer': tf.keras.initializers.serialize(
                self.rho_initializer
            )
        })
        return config
