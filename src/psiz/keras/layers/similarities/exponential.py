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
    ExponentialSimilarity: A parameterized exponential similarity
        layer.

"""

import tensorflow as tf
from tensorflow.python.keras import backend as K

import psiz.keras.constraints as pk_constraints


@tf.keras.utils.register_keras_serializable(
    package='psiz.keras.layers', name='ExponentialSimilarity'
)
class ExponentialSimilarity(tf.keras.layers.Layer):
    """Exponential family similarity function.

    This exponential-family similarity function is parameterized as:
        s(x,y) = exp(-beta .* d(x,y).^tau) + gamma,
    where x and y are n-dimensional vectors. The exponential family
    function is obtained by integrating across various psychological
    theories [1,2,3,4].

    By default beta=10. and is not trainable to prevent redundancy with
    trainable embeddings and to prevent short-circuiting any
    regularizers placed on the embeddings.

    References:
        [1] Jones, M., Love, B. C., & Maddox, W. T. (2006). Recency
            effects as a window to generalization: Separating
            decisional and perceptual sequential effects in category
            learning. Journal of Experimental Psychology: Learning,
            Memory, & Cognition, 32 , 316-332.
        [2] Jones, M., Maddox, W. T., & Love, B. C. (2006). The role of
            similarity in generalization. In Proceedings of the 28th
            annual meeting of the cognitive science society (pp. 405-
            410).
        [3] Nosofsky, R. M. (1986). Attention, similarity, and the
            identification-categorization relationship. Journal of
            Experimental Psychology: General, 115, 39-57.
        [4] Shepard, R. N. (1987). Toward a universal law of
            generalization for psychological science. Science, 237,
            1317-1323.

    """

    def __init__(
            self, fit_tau=True, fit_gamma=True, fit_beta=False,
            tau_initializer=None, gamma_initializer=None,
            beta_initializer=None, **kwargs):
        """Initialize.

        Arguments:
            fit_tau (optional): Boolean indicating if variable is
                trainable.
            fit_gamma (optional): Boolean indicating if variable is
                trainable.
            fit_beta (optional): Boolean indicating if variable is
                trainable.

        """
        super(ExponentialSimilarity, self).__init__(**kwargs)

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

        self.fit_gamma = fit_gamma
        if gamma_initializer is None:
            gamma_initializer = tf.random_uniform_initializer(0., .001)
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        gamma_trainable = self.trainable and self.fit_gamma
        with tf.name_scope(self.name):
            self.gamma = self.add_weight(
                shape=[], initializer=self.gamma_initializer,
                trainable=gamma_trainable, name="gamma", dtype=K.floatx(),
                constraint=pk_constraints.GreaterEqualThan(min_value=0.0)
            )

        self.fit_beta = fit_beta
        if beta_initializer is None:
            if fit_beta:
                beta_initializer = tf.random_uniform_initializer(1., 30.)
            else:
                beta_initializer = tf.keras.initializers.Constant(value=10.)
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        beta_trainable = self.trainable and self.fit_beta
        with tf.name_scope(self.name):
            self.beta = self.add_weight(
                shape=[], initializer=self.beta_initializer,
                trainable=beta_trainable, name="beta", dtype=K.floatx(),
                constraint=pk_constraints.GreaterEqualThan(min_value=1.0)
            )

    def call(self, inputs):
        """Call.

        Arguments:
            inputs: A tensor of distances.

        Returns:
            A tensor of similarities.

        """
        return tf.exp(
            tf.negative(self.beta) * tf.pow(inputs, self.tau)
        ) + self.gamma

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        config.update({
            'fit_tau': self.fit_tau,
            'fit_gamma': self.fit_gamma,
            'fit_beta': self.fit_beta,
            'tau_initializer': tf.keras.initializers.serialize(
                self.tau_initializer
            ),
            'gamma_initializer': tf.keras.initializers.serialize(
                self.gamma_initializer
            ),
            'beta_initializer': tf.keras.initializers.serialize(
                self.beta_initializer
            ),
        })
        return config
