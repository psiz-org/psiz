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
    InverseSimilarity: A parameterized inverse similarity layer.

"""

import tensorflow as tf
from tensorflow.python.keras import backend as K

import psiz.keras.constraints as pk_constraints


@tf.keras.utils.register_keras_serializable(
    package='psiz.keras.layers', name='InverseSimilarity'
)
class InverseSimilarity(tf.keras.layers.Layer):
    """Inverse-distance similarity function.

    The inverse-distance similarity function is parameterized as:
        s(x,y) = 1 / (d(x,y)**tau + mu),
    where x and y are n-dimensional vectors.

    """

    def __init__(
            self, fit_tau=True, fit_mu=True, tau_initializer=None,
            mu_initializer=None, **kwargs):
        """Initialize.

        Arguments:
            fit_tau (optional): Boolean indicating if variable is
                trainable.
            fit_gamma (optional): Boolean indicating if variable is
                trainable.
            fit_beta (optional): Boolean indicating if variable is
                trainable.

        """
        super(InverseSimilarity, self).__init__(**kwargs)

        self.fit_tau = fit_tau
        if tau_initializer is None:
            tau_initializer = tf.random_uniform_initializer(1., 2.)
        self.tau_initializer = tf.keras.initializers.get(tau_initializer)
        tau_trainable = self.trainable and self.fit_tau
        with tf.name_scope(self.name):
            self.tau = self.add_weight(
                shape=[], initializer=self.tau_initializer,
                trainable=tau_trainable, name="tau", dtype=K.floatx(),
                constraint=pk_constraints.GreaterEqualThan(min_value=1.0)
            )

        self.fit_mu = fit_mu
        if mu_initializer is None:
            mu_initializer = tf.random_uniform_initializer(0.0000000001, .001)
        self.mu_initializer = tf.keras.initializers.get(tau_initializer)
        mu_trainable = self.trainable and self.fit_mu
        with tf.name_scope(self.name):
            self.mu = self.add_weight(
                shape=[], initializer=self.tau_initializer,
                trainable=mu_trainable,
                name="mu", dtype=K.floatx(),
                constraint=pk_constraints.GreaterEqualThan(
                    min_value=2.2204e-16
                )
            )

    def call(self, inputs):
        """Call.

        Arguments:
            inputs: A tensor of distances.

        Returns:
            A tensor of similarities.

        """
        return 1 / (tf.pow(inputs, self.tau) + self.mu)

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        config.update({
            'fit_tau': self.fit_tau,
            'fit_mu': self.fit_mu,
            'tau_initializer': tf.keras.initializers.serialize(
                self.tau_initializer
            ),
            'mu_initializer': tf.keras.initializers.serialize(
                self.mu_initializer
            ),
        })
        return config
