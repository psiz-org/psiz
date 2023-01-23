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
from tensorflow.keras import backend as K

from psiz.keras.initializers import Dirichlet


def test_all():
    """Test all methods."""
    # Initialize.
    concentration = [1.1, 1.1, 1.1]
    initializer = Dirichlet(concentration, seed=[123, 252])
    np.testing.assert_array_equal(
        initializer.concentration, concentration
    )
    assert initializer.scale == 1.0

    # Check `get_config`.
    config = initializer.get_config()
    assert config['concentration'] == [1.1, 1.1, 1.1]
    assert config['scale'] == 1.0
    assert config['seed'][0] == 123
    assert config['seed'][1] == 252

    # Check call does not raise error.
    tf_shape = tf.TensorShape([2, 4])
    _ = initializer(tf_shape)

    _ = initializer(tf_shape, dtype=K.floatx())

    # TODO Solve RNG seed issue so that the following test works
    # locally and on CI servers.
    # desired_sample = tf.constant(
    #     [
    #         [
    #             [0.2211622, 0.45494598, 0.32389188],
    #             [0.74135166, 0.11674761, 0.14190075],
    #             [0.31588084, 0.523675, 0.16044414],
    #             [0.90197235, 0.021803739, 0.07622407]
    #         ],
    #         [
    #             [0.1747177, 0.6633501, 0.16193213],
    #             [0.45146146, 0.51521087, 0.033327676],
    #             [0.048038144, 0.14123419, 0.81072766],
    #             [0.06657295, 0.33051702, 0.60291004]
    #         ]
    #     ], dtype=tf.float32
    # )
    # tf.debugging.assert_equal(sample, desired_sample)
