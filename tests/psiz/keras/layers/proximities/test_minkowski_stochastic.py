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
"""Test MinkowskiStochastic layer."""


import keras
import numpy as np

from psiz.keras.layers.activations.exponential import ExponentialSimilarity
from psiz.keras.layers.proximities.minkowski_stochastic import MinkowskiStochastic


def test_call_default(paired_inputs_v0):
    """Test call with default (linear) activation."""
    mink_layer = MinkowskiStochastic()
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
    np.testing.assert_array_almost_equal(desired_outputs, outputs.numpy(), decimal=4)


def test_call_exponential(paired_inputs_v0):
    """Test call with exponential activation function."""
    mink_layer = MinkowskiStochastic(
        activation=ExponentialSimilarity(
            beta_initializer=keras.initializers.Constant(0.1),
            tau_initializer=keras.initializers.Constant(1.0),
            gamma_initializer=keras.initializers.Constant(0.001),
            trainable=False,
        ),
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


def test_serialization():
    """Test serialization."""
    mink_layer = MinkowskiStochastic(
        rho_loc_initializer=keras.initializers.Constant(2.1),
        w_loc_initializer=keras.initializers.Constant(1.1),
        activation=ExponentialSimilarity(
            beta_initializer=keras.initializers.Constant(0.1),
            tau_initializer=keras.initializers.Constant(1.0),
            gamma_initializer=keras.initializers.Constant(0.001),
            trainable=False,
        ),
    )
    mink_layer.build([[None, 3], [None, 3]])
    np.testing.assert_array_equal(mink_layer.w.event_shape, [3])
    np.testing.assert_array_equal(
        keras.ops.convert_to_numpy(mink_layer.rho.mode()),
        np.array(2.1, dtype="float32"),
    )
    np.testing.assert_array_equal(
        keras.ops.convert_to_numpy(mink_layer.w.mode()),
        np.array([1.1, 1.1, 1.1], dtype="float32"),
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

    recon_layer = MinkowskiStochastic.from_config(config)
    recon_layer.build([[None, 3], [None, 3]])
    np.testing.assert_array_equal(recon_layer.w.event_shape, [3])
    np.testing.assert_array_equal(
        keras.ops.convert_to_numpy(recon_layer.rho.mode()),
        np.array(2.1, dtype="float32"),
    )
    np.testing.assert_array_equal(
        keras.ops.convert_to_numpy(mink_layer.w.mode()),
        keras.ops.convert_to_numpy(recon_layer.w.mode()),
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
