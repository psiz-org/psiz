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


def test_call(paired_inputs_v0):
    mink_layer = Minkowski(
        rho_initializer=tf.keras.initializers.Constant(2.),
        w_initializer=tf.keras.initializers.Constant(1.),
        trainable=False
    )
    outputs = mink_layer(paired_inputs_v0)

    desired_outputs = np.array([
        8.660254037844387,
        8.660254037844387,
        8.660254037844387,
        8.660254037844387,
        8.660254037844387
    ])
    np.testing.assert_array_almost_equal(desired_outputs, outputs.numpy())


def test_output_shape(paired_inputs_v0):
    """Test output_shape method."""
    mink_layer = Minkowski()
    input_shape = [
        tf.TensorShape(tf.shape(paired_inputs_v0[0])),
        tf.TensorShape(tf.shape(paired_inputs_v0[1]))
    ]
    output_shape = mink_layer.compute_output_shape(input_shape)
    desired_output_shape = tf.TensorShape([5])
    tf.debugging.assert_equal(output_shape, desired_output_shape)


def test_serialization():
    """Test serialization."""
    mink_layer = Minkowski(
        rho_initializer=tf.keras.initializers.Constant(2.),
        w_initializer=tf.keras.initializers.Constant(1.),
        trainable=False
    )
    mink_layer.build([[None, 3], [None, 3]])
    tf.debugging.assert_equal(tf.shape(mink_layer.w)[0], tf.constant(3))
    config = mink_layer.get_config()

    recon_layer = Minkowski.from_config(config)
    recon_layer.build([[None, 3], [None, 3]])
    tf.debugging.assert_equal(tf.shape(recon_layer.w)[0], tf.constant(3))

    tf.debugging.assert_equal(mink_layer.w, recon_layer.w)
    tf.debugging.assert_equal(mink_layer.rho, recon_layer.rho)
