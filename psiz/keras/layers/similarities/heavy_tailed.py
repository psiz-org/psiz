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
    HeavyTailedSimilarity: A parameterized heavy-tailed similarity
        layer.

"""

import tensorflow as tf
from tensorflow.python.keras import backend as K

import psiz.keras.constraints as pk_constraints


@tf.keras.utils.register_keras_serializable(
    package='psiz.keras.layers', name='HeavyTailedSimilarity'
)
class HeavyTailedSimilarity(tf.keras.layers.Layer):
    """Heavy-tailed family similarity function.

    The heavy-tailed similarity function is parameterized as:
        s(x,y) = (kappa + (d(x,y).^tau)).^(-alpha),
    where x and y are n-dimensional vectors. The heavy-tailed family is
    a generalization of the Student-t family.

    """

    def __init__(
            self, fit_tau=True, fit_kappa=True, fit_alpha=True,
            tau_initializer=None, kappa_initializer=None,
            alpha_initializer=None, **kwargs):
        """Initialize.

        Arguments:
            fit_tau (optional): Boolean indicating if variable is
                trainable.
            fit_kappa (optional): Boolean indicating if variable is
                trainable.
            fit_alpha (optional): Boolean indicating if variable is
                trainable.

        """
        super(HeavyTailedSimilarity, self).__init__(**kwargs)

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

        self.fit_kappa = fit_kappa
        if kappa_initializer is None:
            kappa_initializer = tf.random_uniform_initializer(1., 11.)
        self.kappa_initializer = tf.keras.initializers.get(kappa_initializer)
        kappa_trainable = self.trainable and self.fit_kappa
        with tf.name_scope(self.name):
            self.kappa = self.add_weight(
                shape=[], initializer=self.kappa_initializer,
                trainable=kappa_trainable, name="kappa", dtype=K.floatx(),
                constraint=pk_constraints.GreaterEqualThan(min_value=0.0)
            )

        self.fit_alpha = fit_alpha
        if alpha_initializer is None:
            alpha_initializer = tf.random_uniform_initializer(10., 60.)
        self.alpha_initializer = tf.keras.initializers.get(alpha_initializer)
        alpha_trainable = self.trainable and self.fit_alpha
        with tf.name_scope(self.name):
            self.alpha = self.add_weight(
                shape=[], initializer=self.alpha_initializer,
                trainable=alpha_trainable, name="alpha", dtype=K.floatx(),
                constraint=pk_constraints.GreaterEqualThan(min_value=0.0)
            )

    def call(self, inputs):
        """Call.

        Arguments:
            inputs: A tensor of distances.

        Returns:
            A tensor of similarities.

        """
        return tf.pow(
            self.kappa + tf.pow(inputs, self.tau), (tf.negative(self.alpha))
        )

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        config.update({
            'fit_tau': self.fit_tau,
            'fit_kappa': self.fit_kappa,
            'fit_alpha': self.fit_alpha,
            'tau_initializer': tf.keras.initializers.serialize(
                self.tau_initializer
            ),
            'kappa_initializer': tf.keras.initializers.serialize(
                self.kappa_initializer
            ),
            'alpha_initializer': tf.keras.initializers.serialize(
                self.alpha_initializer
            ),
        })
        return config
