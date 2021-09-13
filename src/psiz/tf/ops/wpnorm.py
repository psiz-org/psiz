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
"""Module of custom TensorFlow ops.

Functions:
    wpnorm: Weighted p-norm.

"""

import tensorflow as tf


@tf.custom_gradient
def wpnorm(x, w, p):
    """Weighted p-norm.

    ||x||_{w,p} = [sum_i w_i abs(x_i)^p]^(1/p)

    Arguments:
        x: A tf.Tensor indicating the vectors. The vector can have
            arbitrary shape, subject to the following constraint:
            shape=(batch_size, [n, m, ...] n_dim)
        w: A tf.Tensor indicating the dimension weights.
            shape=(batch_size, [n, m, ...] n_dim)
        p: A parameter controlling the weighted minkowski metric.
            shape=(batch_size, [n, m, ...])

    Returns:
        shape=(batch_size, [n, m, ...] 1)

    """
    abs_x = tf.abs(x)
    # Add singleton axis to end of p for `n_dim`.
    p_exp = tf.expand_dims(p, axis=-1)
    abs_x_p = tf.pow(abs_x, p_exp)
    w_abs_x_p = tf.multiply(abs_x_p, w)
    # Sum over last axis (i.e., `n_dim`).
    sum_x = tf.reduce_sum(w_abs_x_p, axis=-1, keepdims=True)
    y = tf.pow(sum_x, 1. / p_exp)

    def grad(dy):
        # Gradients of coordinates `x` with fudge factor to avoid problematic
        # gradients.
        # shape=(batch_size, [n, m, ...] n_dim)
        dydx = dy * (
            (w * x * tf.math.divide_no_nan(abs_x_p, abs_x**2))
            / (y**(p_exp - 1) + tf.keras.backend.epsilon())
        )

        # Gradients of weights `w` with fudge factor to avoid problematic
        # gradients.
        # shape=(batch_size, [n, m, ...] n_dim)
        dydw = dy * (
            abs_x_p / (p_exp * y**(p_exp - 1) + tf.keras.backend.epsilon())
        )

        # Gradients of `p` with fudge factor and to avoid problematic
        # gradients.
        # shape=(batch_size, [n, m, ...])
        p_0 = (1. / p_exp) * tf.math.divide_no_nan(y, sum_x) * tf.reduce_sum(
            w_abs_x_p * tf.math.log(abs_x + tf.keras.backend.epsilon()),
            axis=-1, keepdims=True
        )
        p_1 = (1. / p_exp**2) * y * tf.math.log(
            sum_x + tf.keras.backend.epsilon()
        )
        dydp = dy * (p_0 - p_1)
        dydp = tf.squeeze(dydp, [-1])
        return dydx, dydw, dydp

    return y, grad
