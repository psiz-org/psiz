# -*- coding: utf-8 -*-
# Copyright 2021 The PsiZ Authors. All Rights Reserved.
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
"""Test constraints module."""

import numpy as np
import tensorflow as tf

from psiz.keras.initializers import Dirichlet


def test_all():
    """Test all methods."""
    # Initialize.
    concentration = [1.1, 1.1, 1.1]
    initializer = Dirichlet(concentration, seed=123)
    np.testing.assert_array_equal(
        initializer.concentration, concentration
    )
    assert initializer.scale == 1.0

    # Check `get_config`.
    config = initializer.get_config()
    assert config['concentration'] == [1.1, 1.1, 1.1]
    assert config['scale'] == 1.0
    assert config['seed'] == 123

    # Check call.
    tf_shape = tf.TensorShape([2, 4])
    sample = initializer(tf_shape)

    desired_sample = tf.constant(
        [
            [
                [0.4180948, 0.34049425, 0.24141102],
                [0.7538074, 0.20453003, 0.041662533],
                [0.29102674, 0.64605397, 0.06291929],
                [0.19294274, 0.38227272, 0.42478448]
            ],
            [
                [0.02725979, 0.7716456, 0.20109455],
                [0.63922215, 0.24183095, 0.11894692],
                [0.3857724, 0.33163887, 0.28258872],
                [0.004561754, 0.43929902, 0.5561392]
            ]
        ], dtype=tf.float32
    )
    tf.debugging.assert_equal(sample, desired_sample)
