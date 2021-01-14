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
"""Module of custom TensorFlow constraints.

Classes:
    NonNegNorm:

"""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import constraints


@tf.keras.utils.register_keras_serializable(package='psiz.keras.constraints')
class NonNegNorm(constraints.Constraint):
    """Non-negative norm weight constraint.

    Constrains the weights incident to each hidden unit
    to have non-negative weights and a norm of the specified magnitude.

    """

    def __init__(self, scale=1.0, p=2.0, axis=0):
        """Initialize.

        Arguments:
            scale (optional): The scale (i.e., magnitude) of the norm.
            p (optional): Type of p-norm (must be  >=1).
            axis (optional): integer, axis along which to calculate
                weight norms.

        """
        self.scale = scale
        self.p = p
        self.axis = axis

    def __call__(self, w):
        """Call."""
        # Enforce nonnegative.
        w = w * tf.cast(tf.math.greater_equal(w, 0.), K.floatx())

        # Enforce norm.
        return self.scale * (
            w / (
                K.epsilon() + tf.pow(
                    tf.reduce_sum(w**self.p, axis=self.axis, keepdims=True),
                    tf.divide(1.0, self.p)
                )
            )
        )

    def get_config(self):
        """Return configuration."""
        return {'scale': self.scale, 'p': self.p, 'axis': self.axis}
