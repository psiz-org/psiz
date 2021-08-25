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
    AttentionEntropy:

"""

import tensorflow as tf


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
        n_dim = tf.cast(tf.shape(w)[0], tf.keras.backend.floatx())
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
