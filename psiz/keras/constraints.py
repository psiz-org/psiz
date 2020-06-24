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
    NonNegNorm:

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
        """Initialize.

        Arguments:
            min_value: The minimum allowed weight value.

        """
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
        """Initialize.

        Arguments:
            max_value: The maximum allowed weight value.

        """
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
        """Initialize.

        Arguments:
            min_value: The minimum allowed weight value.

        """
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
        """Initialize.

        Arguments:
            max_value: The maximum allowed weight value.

        """
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
        """Initialize.

        Arguments:
            min_value: The minimum allowed weight value.
            max_value: The maximum allowed weight value.

        """
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
        """Initialize.

        Arguments:
            axis (optional): integer, axis along which to reduce
                weights in order to compute mean.

        """
        self.axis = axis

    def __call__(self, w):
        """Call."""
        return w - tf.reduce_mean(w, axis=self.axis, keepdims=True)

    def get_config(self):
        """Return configuration."""
        return {'axis': self.axis}


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
                    tf.divide(1, self.p)
                )
            )
        )

    def get_config(self):
        """Return configuration."""
        return {'scale': self.scale, 'p': self.p, 'axis': self.axis}


@tf.keras.utils.register_keras_serializable(
    package='psiz.keras.constraints', name='SharedMean'
)
class SharedMean(constraints.Constraint):
    """Constrains all weights to equal the incoming mean."""

    def __init__(self, axis=None):
        """Initialize.

        Arguments:
            axis (optional): integer, axis along which to reduce
                weights in order to compute mean.

        """
        self.axis = axis  # TODO

    def __call__(self, w):
        """Call."""
        w_avg = tf.reduce_mean(w, axis=None, keepdims=True)
        return w - (w - w_avg)

    def get_config(self):
        """Return configuration."""
        return {'axis': self.axis}


@tf.keras.utils.register_keras_serializable(
    package='psiz.keras.constraints', name='SharedMedian'
)
class SharedMedian(constraints.Constraint):
    """Constrains all weights to equal the incoming mean."""

    def __init__(self, axis=None):
        """Initialize.

        Arguments:
            axis (optional): integer, axis along which to reduce
                weights in order to compute mean.

        """
        self.axis = axis  # TODO

    def __call__(self, w):
        """Call."""
        w_avg = tfp.stats.percentile(w, 50.0, interpolation='midpoint')
        return w - (w - w_avg)

    def get_config(self):
        """Return configuration."""
        return {'axis': self.axis}
