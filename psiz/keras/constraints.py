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
# ==============================================================================

"""Module of custom TensorFlow constraints.

Classes:
    GreaterThan:
    LessThan:
    GreaterEqualThan:
    LessEqualThan:
    MinMax:
    Center:
    ProjectAttention:

"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import backend as K
from tensorflow.keras import constraints


@tf.keras.utils.register_keras_serializable(package='psiz.keras.constraints')
class GreaterThan(constraints.Constraint):
    """Constrains the weights to be greater than a value."""

    def __init__(self, min_value=0.):
        """Initialize."""
        self.min_value = min_value

    def __call__(self, w):
        """Call."""
        w = w - self.min_value
        w = w * tf.cast(tf.math.greater(w, 0.), K.floatx())
        w = w + self.min_value
        return w

    def get_config(self):
        """Return configuration."""
        return {'min_value': self.min_value}


@tf.keras.utils.register_keras_serializable(package='psiz.keras.constraints')
class LessThan(constraints.Constraint):
    """Constrains the weights to be less than a value."""

    def __init__(self, max_value=0.):
        """Initialize."""
        self.max_value = max_value

    def __call__(self, w):
        """Call."""
        w = w - self.max_value
        w = w * tf.cast(tf.math.greater(0., w), K.floatx())
        w = w + self.max_value
        return w

    def get_config(self):
        """Return configuration."""
        return {'max_value': self.max_value}


@tf.keras.utils.register_keras_serializable(package='psiz.keras.constraints')
class GreaterEqualThan(constraints.Constraint):
    """Constrains the weights to be greater/equal than a value."""

    def __init__(self, min_value=0.):
        """Initialize."""
        self.min_value = min_value

    def __call__(self, w):
        """Call."""
        w = w - self.min_value
        w = w * tf.cast(tf.math.greater_equal(w, 0.), K.floatx())
        w = w + self.min_value
        return w

    def get_config(self):
        """Return configuration."""
        return {'min_value': self.min_value}


@tf.keras.utils.register_keras_serializable(package='psiz.keras.constraints')
class LessEqualThan(constraints.Constraint):
    """Constrains the weights to be greater/equal than a value."""

    def __init__(self, max_value=0.):
        """Initialize."""
        self.max_value = max_value

    def __call__(self, w):
        """Call."""
        w = w - self.max_value
        w = w * tf.cast(tf.math.greater_equal(0., w), K.floatx())
        w = w + self.max_value
        return w

    def get_config(self):
        """Return configuration."""
        return {'max_value': self.max_value}


@tf.keras.utils.register_keras_serializable(package='psiz.keras.constraints')
class MinMax(constraints.Constraint):
    """Constrains the weights to be between/equal values."""

    def __init__(self, min_value, max_value):
        """Initialize."""
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, w):
        """Call."""
        w = w - self.min_value
        w = w * tf.cast(tf.math.greater_equal(w, 0.), K.floatx())
        w = w + self.min_value

        w = w - self.max_value
        w = w * tf.cast(tf.math.greater_equal(0., w), K.floatx())
        w = w + self.max_value

        return w

    def get_config(self):
        """Return configuration."""
        return {'min_value': self.min_value, 'max_value': self.max_value}


@tf.keras.utils.register_keras_serializable(package='psiz.keras.constraints')
class Center(constraints.Constraint):
    """Constrains the weights to be zero-centered.

    This constraint can be used to improve the numerical stability of
    an embedding.

    """

    def __init__(self, axis=0):
        """Initialize."""
        self.axis = axis

    def __call__(self, w):
        """Call."""
        return w - tf.reduce_mean(w, axis=self.axis, keepdims=True)

    def get_config(self):
        """Return configuration."""
        return {'axis': self.axis}


@tf.keras.utils.register_keras_serializable(package='psiz.keras.constraints')
class ProjectAttention(constraints.Constraint):
    """Return projection of attention weights."""

    def __init__(self, n_dim=None):
        """Initialize."""
        self.n_dim = n_dim

    def __call__(self, attention_0):
        """Call."""
        # First, make sure weights are not negative.
        attention_0 = attention_0 * tf.cast(
            tf.math.greater_equal(attention_0, 0.), K.floatx()
        )

        attention_1 = tf.divide(
            tf.reduce_sum(attention_0, axis=1, keepdims=True), self.n_dim
        )
        attention_proj = tf.divide(
            attention_0, attention_1
        )
        return attention_proj

    def get_config(self):
        """Return configuration."""
        return {'n_dim': self.n_dim}


@tf.keras.utils.register_keras_serializable(package='psiz.keras.constraints')
class NonNegUnitNorm(constraints.Constraint):
    """NonNegUnitNorm weight constraint.

    Constrains the weights incident to each hidden unit
    to have non-negative weights and a unit norm.

    Arguments:
        p (optional): Type of p-norm (must be  >=1).
        axis (optional): integer, axis along which to calculate weight norms.

    """

    def __init__(self, p=2.0, axis=0):
        """Initialize."""
        self.p = p
        self.axis = axis

    def __call__(self, w):
        """Call."""
        # Enforce nonnegative.
        w = w * tf.cast(tf.math.greater_equal(w, 0.), K.floatx())

        # Enforce unit norm.
        return w / (
            K.epsilon() + tf.pow(
                tf.reduce_sum(w**self.p, axis=self.axis, keepdims=True),
                tf.divide(1, self.p)
            )
        )

    def get_config(self):
        """Return configuration."""
        return {'p': self.p, 'axis': self.axis}
