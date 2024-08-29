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
import tensorflow as tf

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
    mink_layer.build(
        [[None, 3], [None, 3]]
    )  # TODO I suspect I'm not calling build correctly
    tf.debugging.assert_equal(mink_layer.w.event_shape, tf.TensorShape([3]))
    tf.debugging.assert_equal(mink_layer.rho.mode(), tf.constant([2.1]))
    tf.debugging.assert_equal(mink_layer.w.mode(), tf.constant([1.1, 1.1, 1.1]))
    tf.debugging.assert_equal(mink_layer.activation.beta, tf.constant(0.1))
    tf.debugging.assert_equal(mink_layer.activation.tau, tf.constant(1.0))
    tf.debugging.assert_equal(mink_layer.activation.gamma, tf.constant(0.001))
    config = mink_layer.get_config()

    recon_layer = MinkowskiStochastic.from_config(config)
    recon_layer.build([[None, 3], [None, 3]])
    tf.debugging.assert_equal(recon_layer.w.event_shape, tf.TensorShape([3]))
    tf.debugging.assert_equal(recon_layer.rho.mode(), tf.constant([2.1]))
    tf.debugging.assert_equal(recon_layer.w.mode(), tf.constant([1.1, 1.1, 1.1]))
    tf.debugging.assert_equal(mink_layer.activation.beta, recon_layer.activation.beta)
    tf.debugging.assert_equal(mink_layer.activation.tau, recon_layer.activation.tau)
    tf.debugging.assert_equal(mink_layer.activation.gamma, recon_layer.activation.gamma)
