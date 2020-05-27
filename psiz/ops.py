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

"""Module of custom TensorFlow ops.

Functions:
    wpnorm: Weighted p-norm.

"""

import tensorflow as tf


@tf.custom_gradient
def wpnorm(x, w, p):
    """Weighted p-norm.

    ||x||_{w,p} = [sum_i w_i abs(x_i)^p]^(1/p)

    """
    abs_x = tf.abs(x)
    abs_x_p = tf.pow(abs_x, p)
    w_abs_x_p = tf.multiply(abs_x_p, w)
    sum_x = tf.reduce_sum(w_abs_x_p, axis=1, keepdims=True)
    y = tf.pow(sum_x, 1. / p)

    def grad(dy):
        # Gradients of coordinates `x`.
        dydx = dy * (
            (w * x * tf.math.divide_no_nan(abs_x_p, abs_x**2)) /
            (y**(p-1) + tf.keras.backend.epsilon())
        )

        # Gradients of weights `w`.
        dydw = dy * (
            abs_x_p / (p * y**(p-1) + tf.keras.backend.epsilon())
        )
        dydw = tf.reduce_sum(dydw, axis=[2, 3], keepdims=True)

        # Gradients of `p`.
        p_0 = (1. / p) * tf.math.divide_no_nan(y, sum_x) * tf.reduce_sum(
            w_abs_x_p * tf.math.log(abs_x + tf.keras.backend.epsilon()),
            axis=1, keepdims=True
        )
        p_1 = (1. / p**2) * y * tf.math.log(sum_x + tf.keras.backend.epsilon())
        dydp = dy * (
            p_0 - p_1
        )
        return dydx, dydw, dydp

    return y, grad
