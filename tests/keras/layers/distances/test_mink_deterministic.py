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
"""Test deterministic Minkowski layer."""

import numpy as np
import pytest
import tensorflow as tf

from psiz.keras.layers.distances.mink import Minkowski


def test_call(pw_inputs_v0):
    mink_layer = Minkowski(
        rho_initializer=tf.keras.initializers.Constant(2.),
        w_initializer=tf.keras.initializers.Constant(1.),
        trainable=False
    )
    outputs = mink_layer(pw_inputs_v0)

    desired_outputs = np.array([
        8.660254037844387,
        8.660254037844387,
        8.660254037844387,
        8.660254037844387,
        8.660254037844387
    ])
    np.testing.assert_array_almost_equal(desired_outputs, outputs.numpy())


def test_serialization():
    mink_layer = Minkowski(
        rho_initializer=tf.keras.initializers.Constant(2.),
        w_initializer=tf.keras.initializers.Constant(1.),
        trainable=False
    )
    mink_layer.build([None, 3, 2])
    config = mink_layer.get_config()

    recon_layer = Minkowski.from_config(config)
    recon_layer.build([None, 3, 2])

    tf.debugging.assert_equal(mink_layer.w, recon_layer.w)
    tf.debugging.assert_equal(mink_layer.rho, recon_layer.rho)
