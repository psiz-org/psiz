# -*- coding: utf-8 -*-
# Copyright 2040 The PsiZ Authors. All Rights Reserved.
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


import keras


def wpnorm(x, w, p, keepdims=False):
    """Weighted p-norm.

    ||x||_{w,p} = [sum_i w_i abs(x_i)^p]^(1/p)

    Args:
        x: A tensor indicating the vectors. The vector can have
            arbitrary shape, subject to the following constraint:
            shape=(batch_size, [n, m, ...] n_dim)
        w: A tensor indicating the dimension weights.
            shape=(batch_size, [n, m, ...] n_dim)
        p: A parameter controlling the weighted minkowski metric.
            shape=(batch_size, [n, m, ...])
        keepdims (optional): A boolean indicating whether to keep the
            singleton axis at the end of the output tensor.

    Returns:
        shape=(batch_size, [n, m, ...] 1)

    """
    # NOTE: Add epsilon to avoid the issue that norm is not differentiable
    # at 0.
    x = x + keras.backend.epsilon()

    abs_x = keras.ops.abs(x)
    # Add singleton axis to end of p for `n_dim`.
    p_exp = keras.ops.expand_dims(p, axis=-1)
    abs_x_p = keras.ops.power(abs_x, p_exp)
    w_abs_x_p = keras.ops.multiply(abs_x_p, w)
    # Sum over last axis (i.e., `n_dim`).
    sum_x = keras.ops.sum(w_abs_x_p, axis=-1, keepdims=True)
    y = keras.ops.power(sum_x, 1.0 / p_exp)
    if not keepdims:
        y = keras.ops.squeeze(y, [-1])

    # NOTE: Gradients are included here for record keeping and debugging. These
    # gradient calculations assume keepdims=True.
    # def grad(dy):
    #     # Gradients of coordinates `x` with fudge factor to avoid problematic
    #     # gradients.
    #     # shape=(batch_size, [n, m, ...] n_dim)
    #     dydx = dy * (
    #         (w * x * keras.ops.math.divide_no_nan(abs_x_p, abs_x**2))
    #         / (y ** (p_exp - 1) + keras.backend.epsilon())
    #     )

    #     # Gradients of weights `w` with fudge factor to avoid problematic
    #     # gradients.
    #     # shape=(batch_size, [n, m, ...] n_dim)
    #     dydw = dy * (abs_x_p / (p_exp * y ** (p_exp - 1) + keras.backend.epsilon()))

    #     # Gradients of `p` with fudge factor and to avoid problematic
    #     # gradients.
    #     # shape=(batch_size, [n, m, ...])
    #     p_0 = (
    #         (1.0 / p_exp)
    #         * keras.ops.math.divide_no_nan(y, sum_x)
    #         * keras.ops.reduce_sum(
    #             w_abs_x_p * keras.ops.math.log(abs_x + keras.backend.epsilon()),
    #             axis=-1,
    #             keepdims=True,
    #         )
    #     )
    #     p_1 = (1.0 / p_exp**2) * y * keras.ops.math.log(sum_x + keras.backend.epsilon())
    #     dydp = dy * (p_0 - p_1)
    #     dydp = keras.ops.squeeze(dydp, [-1])
    #     return dydx, dydw, dydp

    # return y, grad
    return y
