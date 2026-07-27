# -*- coding: utf-8 -*-
# Copyright 2026 The PsiZ Authors. All Rights Reserved.
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
"""Gradient-scaling ops."""

import keras


def scale_gradient(x, scale):
    """Return `x` with its backward gradient scaled by `scale`.

    This implementation is backend agnostic (TensorFlow, JAX, PyTorch) because
    it only uses `keras.ops` primitives.

    Trick:
        y = scale * x + stop_gradient((1 - scale) * x)

    Forward pass:
        y == x

    Backward pass:
        dy/dx == scale
    """
    scale = keras.ops.cast(scale, x.dtype)

    # The stop_gradient term preserves the forward value while removing its
    # contribution to the derivative, resulting in gradient scaling by `scale`.
    return scale * x + keras.ops.stop_gradient((1.0 - scale) * x)