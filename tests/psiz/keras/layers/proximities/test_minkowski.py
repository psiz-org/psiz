# -*- coding: utf-8 -*-
# Copyright 2024 The PsiZ Authors. All Rights Reserved.
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


import keras
import numpy as np

from psiz.keras.layers.activations.exponential import ExponentialSimilarity
from psiz.keras.layers.proximities.minkowski import Minkowski


def test_call_default(paired_inputs_v0):
    """Test call with default (linear) activation."""
    mink_layer = Minkowski(
        rho_initializer=keras.initializers.Constant(2.0),
        w_initializer=keras.initializers.Constant(1.0),
        trainable=False,
    )
    outputs = mink_layer(paired_inputs_v0)

    desired_outputs = np.array(
        [
            8.660254037844387,
            8.660254037844387,
            8.660254037844387,
            8.660254037844387,
            8.660254037844387,
        ]
    )
    np.testing.assert_array_almost_equal(desired_outputs, outputs.numpy())


def test_call_exponential(paired_inputs_v0):
    """Test call with exponential activation function."""
    mink_layer = Minkowski(
        rho_initializer=keras.initializers.Constant(2.0),
        w_initializer=keras.initializers.Constant(1.0),
        activation=ExponentialSimilarity(
            beta_initializer=keras.initializers.Constant(0.1),
            tau_initializer=keras.initializers.Constant(1.0),
            gamma_initializer=keras.initializers.Constant(0.001),
            trainable=False,
        ),
        trainable=False,
    )
    outputs = mink_layer(paired_inputs_v0)

    desired_outputs = np.array(
        [
            0.4216200260541147,
            0.4216200260541147,
            0.4216200260541147,
            0.4216200260541147,
            0.4216200260541147,
        ]
    )
    np.testing.assert_array_almost_equal(desired_outputs, outputs.numpy())


def test_serialization_0():
    """Test serialization."""
    mink_layer = Minkowski(
        rho_initializer=keras.initializers.Constant(2.0),
        w_initializer=keras.initializers.Constant(1.0),
        activation=ExponentialSimilarity(
            beta_initializer=keras.initializers.Constant(0.1),
            tau_initializer=keras.initializers.Constant(1.0),
            gamma_initializer=keras.initializers.Constant(0.001),
            trainable=False,
        ),
        trainable=False,
    )
    mink_layer.build([[None, 3], [None, 3]])
    np.testing.assert_array_equal(
        keras.ops.convert_to_numpy(mink_layer.w),
        np.array([1.0, 1.0, 1.0], dtype="float32"),
    )
    np.testing.assert_array_equal(
        keras.ops.convert_to_numpy(mink_layer.activation.beta),
        np.array(0.1, dtype="float32"),
    )
    np.testing.assert_array_equal(
        keras.ops.convert_to_numpy(mink_layer.activation.tau),
        np.array(1.0, dtype="float32"),
    )
    np.testing.assert_array_equal(
        keras.ops.convert_to_numpy(mink_layer.activation.gamma),
        np.array(0.001, dtype="float32"),
    )
    config = mink_layer.get_config()

    recon_layer = Minkowski.from_config(config)
    recon_layer.build([[None, 3], [None, 3]])
    np.testing.assert_array_equal(
        keras.ops.convert_to_numpy(recon_layer.w),
        keras.ops.convert_to_numpy(mink_layer.w),
    )
    np.testing.assert_array_equal(
        keras.ops.convert_to_numpy(recon_layer.activation.beta),
        keras.ops.convert_to_numpy(mink_layer.activation.beta),
    )
    np.testing.assert_array_equal(
        keras.ops.convert_to_numpy(recon_layer.activation.tau),
        keras.ops.convert_to_numpy(mink_layer.activation.tau),
    )
    np.testing.assert_array_equal(
        keras.ops.convert_to_numpy(recon_layer.activation.gamma),
        keras.ops.convert_to_numpy(mink_layer.activation.gamma),
    )
    np.testing.assert_array_equal(
        keras.ops.convert_to_numpy(recon_layer.rho),
        keras.ops.convert_to_numpy(mink_layer.rho),
    )


def test_serialization_1(paired_inputs_v0):
    """Test serialization with weights."""
    mink_layer = Minkowski(
        rho_initializer=keras.initializers.Constant(2.0),
        w_initializer=keras.initializers.RandomUniform(),
        activation=ExponentialSimilarity(
            beta_initializer=keras.initializers.Constant(0.1),
            tau_initializer=keras.initializers.Constant(1.0),
            gamma_initializer=keras.initializers.Constant(0.001),
            trainable=False,
        ),
        trainable=False,
    )
    mink_layer.build([[None, 3], [None, 3]])
    np.testing.assert_array_equal(keras.ops.convert_to_numpy(mink_layer.w).shape[0], 3)
    np.testing.assert_array_equal(
        keras.ops.convert_to_numpy(mink_layer.activation.beta),
        np.array(0.1, dtype="float32"),
    )
    np.testing.assert_array_equal(
        keras.ops.convert_to_numpy(mink_layer.activation.tau),
        np.array(1.0, dtype="float32"),
    )
    np.testing.assert_array_equal(
        keras.ops.convert_to_numpy(mink_layer.activation.gamma),
        np.array(0.001, dtype="float32"),
    )
    config = mink_layer.get_config()
    weights = mink_layer.get_weights()

    recon_layer = Minkowski.from_config(config)
    # NOTE: Calling build is necessary for the weights to be set, otherwise an error is thrown.
    recon_layer.build([[None, 3], [None, 3]])
    recon_layer.set_weights(weights)
    np.testing.assert_array_equal(
        keras.ops.convert_to_numpy(mink_layer.w),
        keras.ops.convert_to_numpy(recon_layer.w),
    )
    np.testing.assert_array_equal(
        keras.ops.convert_to_numpy(mink_layer.activation.beta),
        keras.ops.convert_to_numpy(recon_layer.activation.beta),
    )
    np.testing.assert_array_equal(
        keras.ops.convert_to_numpy(mink_layer.activation.tau),
        keras.ops.convert_to_numpy(recon_layer.activation.tau),
    )
    np.testing.assert_array_equal(
        keras.ops.convert_to_numpy(mink_layer.activation.gamma),
        keras.ops.convert_to_numpy(recon_layer.activation.gamma),
    )
    np.testing.assert_array_equal(
        keras.ops.convert_to_numpy(mink_layer.rho),
        keras.ops.convert_to_numpy(recon_layer.rho),
    )


def test_serialization_2():
    """Test serialization with default activation layer."""
    mink_layer = Minkowski(
        rho_initializer=keras.initializers.Constant(2.0),
        w_initializer=keras.initializers.Constant(1.0),
        trainable=False,
    )
    mink_layer.build([[None, 3], [None, 3]])
    np.testing.assert_array_equal(
        keras.ops.convert_to_numpy(mink_layer.w),
        np.array([1.0, 1.0, 1.0], dtype="float32"),
    )

    assert mink_layer.activation.weights == []
    config = mink_layer.get_config()

    recon_layer = Minkowski.from_config(config)
    recon_layer.build([[None, 3], [None, 3]])
    np.testing.assert_array_equal(
        keras.ops.convert_to_numpy(mink_layer.w),
        keras.ops.convert_to_numpy(recon_layer.w),
    )
    assert recon_layer.activation.weights == []

    np.testing.assert_array_equal(
        keras.ops.convert_to_numpy(mink_layer.rho),
        keras.ops.convert_to_numpy(recon_layer.rho),
    )
