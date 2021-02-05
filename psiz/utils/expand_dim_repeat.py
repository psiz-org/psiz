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
"""Module of utility functions.

Functions:
    expand_dim_repeat: Repeat Tensor along a newly inserted axis.

"""

import tensorflow as tf


def expand_dim_repeat(x, n_repeat, axis=1):
    """Repeat Tensor along a newly inserted axis."""
    x = tf.expand_dims(x, axis=axis)
    return tf.repeat(x, n_repeat, axis=axis)
