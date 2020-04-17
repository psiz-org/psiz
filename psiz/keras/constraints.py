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
    RandomScaleMVN:
    RandomAttention:


"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import backend as K
from tensorflow.keras import constraints


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


class ZeroCenterZ(constraints.Constraint):
    """Constrains the embedding to be zero-centered.

    This constraint is used to improve numerical stability.
    """

    def __init__(self, **kwargs):
        """Initialize."""
        # super().__init__(**kwargs)

    def __call__(self, z):
        """Call."""
        return z - tf.reduce_mean(z, axis=0, keepdims=True)


class ProjectAttention(constraints.Constraint):
    """Return projection of attention weights."""

    def __init__(self):
        """Initialize."""

    def __call__(self, tf_attention_0):
        """Call."""
        n_dim = tf.shape(tf_attention_0, out_type=K.floatx())[1]
        tf_attention_1 = tf.divide(
            tf.reduce_sum(tf_attention_0, axis=1, keepdims=True), n_dim
        )
        tf_attention_proj = tf.divide(
            tf_attention_0, tf_attention_1
        )
        return tf_attention_proj

