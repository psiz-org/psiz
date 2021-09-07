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

from psiz.keras.initializers import RandomAttention


def test_all():
    """Test all methods."""
    # Initialize.
    concentration = [1.1, 1.1, 1.1]
    initializer = RandomAttention(concentration, seed=123)
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
                [0.12923722, 0.84517914, 0.02558367],
                [0.46458238, 0.19863836, 0.33677924],
                [0.37943587, 0.43351644, 0.1870478],
                [0.4106196, 0.0709199, 0.5184606]
            ],
            [
                [0.11999004, 0.16976392, 0.7102461],
                [0.3742919, 0.35068202, 0.27502614],
                [0.30005947, 0.30656993, 0.39337057],
                [0.00385686, 0.16461745, 0.8315257]
            ]
        ], dtype=tf.float32
    )
    tf.debugging.assert_equal(sample, desired_sample)
