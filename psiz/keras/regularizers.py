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
"""Module of custom TensorFlow regularizers.

Classes:
    StimulusNormedL1:
    AttentionEntropy:
    Squeeze:

"""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package='psiz.keras.regularizers')
class StimulusNormedL1(tf.keras.regularizers.Regularizer):
    """Stimulus-normed L1 regularization."""

    def __init__(self, l1=0.):
        """Initialize.

        Arguments:
            l1: Rate of L1 regularization.

        """
        self.l1 = l1

    def __call__(self, z):
        """Call."""
        return self.l1 * (tf.reduce_sum(tf.abs(z)) / z.shape[0])

    def get_config(self):
        """Return config."""
        return {'l1': float(self.l1)}


@tf.keras.utils.register_keras_serializable(package='psiz.keras.regularizers')
class AttentionEntropy(tf.keras.regularizers.Regularizer):
    """Entropy-based regularization to encourage sparsity."""

    def __init__(self, rate=0.):
        """Initialize.

        Arguments:
            rate: Rate at which regularization is applied.

        """
        self.rate = rate

    def __call__(self, w):
        """Call."""
        n_dim = tf.cast(tf.shape(w)[0], dtype=tf.keras.backend.floatx())
        # Scale weights to sum to one and add fudge factor. Here we assume
        # that weights sum to n_dim.
        w = w / n_dim + tf.keras.backend.epsilon()
        t = tf.negative(tf.math.reduce_sum(w * tf.math.log(w), axis=1))
        return self.rate * (tf.reduce_mean(
            t
        ))

    def get_config(self):
        """Return config."""
        return {'rate': float(self.rate)}


@tf.keras.utils.register_keras_serializable(package='psiz.keras.regularizers')
class Squeeze(tf.keras.regularizers.Regularizer):
    """Squeeze representation into a low number of dimensions."""

    def __init__(self, rate=0.):
        """Initialize.

        Arguments:
            rate: Rate at which regularization is applied.

        """
        self.rate = rate

    def __call__(self, z):
        """Call."""
        # Sum across stimuli, but within a dimension.
        dimension_usage = tf.reduce_max(tf.math.abs(z), axis=0)
        return self.rate * tf.reduce_sum(dimension_usage)

    def get_config(self):
        """Return config."""
        return {'rate': float(self.rate)}
