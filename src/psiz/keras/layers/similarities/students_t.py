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
    StudentsTSimilarity: A parameterized Student's t-distribution
        similarity layer.

"""

import tensorflow as tf
from tensorflow.python.keras import backend as K

import psiz.keras.constraints as pk_constraints


@tf.keras.utils.register_keras_serializable(
    package='psiz.keras.layers', name='StudentsTSimilarity'
)
class StudentsTSimilarity(tf.keras.layers.Layer):
    """Student's t-distribution similarity function.

    The Student's t-distribution similarity function is parameterized
    as:
        s(x,y) = (1 + (((d(x,y)^tau)/alpha))^(-(alpha + 1)/2),
    where x and y are n-dimensional vectors. The original Student-t
    kernel proposed by van der Maaten [1] uses a L2 distance (which is
    governed by the distance kernel), tau=2, and alpha=n_dim-1. By
    default, all variables are fit to the data.

    References:
    [1] van der Maaten, L., & Weinberger, K. (2012, Sept). Stochastic
        triplet embedding. In Machine learning for signal processing
        (MLSP), 2012 IEEE international workshop on (p. 1-6).
        doi:10.1109/MLSP.2012.6349720

    """

    def __init__(
            self, fit_tau=True, fit_alpha=True,
            tau_initializer=None, alpha_initializer=None, **kwargs):
        """Initialize.

        Arguments:
            fit_tau (optional): Boolean indicating if variable is
                trainable.
            fit_alpha (optional): Boolean indicating if variable is
                trainable.

        """
        super(StudentsTSimilarity, self).__init__(**kwargs)

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

        self.fit_alpha = fit_alpha
        if alpha_initializer is None:
            alpha_initializer = tf.random_uniform_initializer(1., 10.)
        self.alpha_initializer = tf.keras.initializers.get(alpha_initializer)
        alpha_trainable = self.trainable and self.fit_alpha
        with tf.name_scope(self.name):
            self.alpha = self.add_weight(
                shape=[], initializer=self.alpha_initializer,
                trainable=alpha_trainable, name="alpha", dtype=K.floatx(),
                constraint=pk_constraints.GreaterEqualThan(min_value=0.000001)
            )

    def call(self, inputs):
        """Call.

        Arguments:
            inputs: A tensor of distances.

        Returns:
            A tensor of similarities.

        """
        return tf.pow(
            1 + (tf.pow(inputs, self.tau) / self.alpha),
            tf.negative(self.alpha + 1) / 2
        )

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        config.update({
            'fit_tau': self.fit_tau,
            'fit_alpha': self.fit_alpha,
            'tau_initializer': tf.keras.initializers.serialize(
                self.tau_initializer
            ),
            'alpha_initializer': tf.keras.initializers.serialize(
                self.alpha_initializer
            ),
        })
        return config
