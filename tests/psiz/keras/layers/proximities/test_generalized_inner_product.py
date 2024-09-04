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
"""Test GeneralizedInnerProduct layer."""


import keras
import numpy as np

from psiz.keras.layers.proximities.experimental.generalized_inner_product import (
    GeneralizedInnerProduct,
)


def test_init_call_v0(paired_inputs_v0):
    """Test call with default (linear) activation."""
    proximity_layer = GeneralizedInnerProduct()
    outputs = proximity_layer(paired_inputs_v0)
    outputs = keras.ops.convert_to_numpy(outputs)

    inputs_0 = paired_inputs_v0[0]
    inputs_1 = paired_inputs_v0[1]
    desired_outputs = np.concatenate(
        [
            np.matmul(
                np.expand_dims(inputs_0[0], axis=0), np.expand_dims(inputs_1[0], axis=1)
            ),
            np.matmul(
                np.expand_dims(inputs_0[1], axis=0), np.expand_dims(inputs_1[1], axis=1)
            ),
            np.matmul(
                np.expand_dims(inputs_0[2], axis=0), np.expand_dims(inputs_1[2], axis=1)
            ),
            np.matmul(
                np.expand_dims(inputs_0[3], axis=0), np.expand_dims(inputs_1[3], axis=1)
            ),
            np.matmul(
                np.expand_dims(inputs_0[4], axis=0), np.expand_dims(inputs_1[4], axis=1)
            ),
        ],
        axis=0,
    ).squeeze()

    np.testing.assert_array_almost_equal(desired_outputs, outputs)


def test_init_call_v1(paired_inputs_v0):
    """Test call with weight matrix function."""
    covariance_matrix = np.array(
        [
            [2.0, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    proximity_layer = GeneralizedInnerProduct(
        w_initializer=keras.initializers.Constant(covariance_matrix),
    )
    outputs = proximity_layer(paired_inputs_v0)
    outputs = keras.ops.convert_to_numpy(outputs)

    desired_outputs = np.array(
        [1.295, 22.795, 51.295, 86.795, 129.295], dtype="float32"
    )
    np.testing.assert_array_equal(desired_outputs, outputs)


def test_init_call_v2(paired_inputs_v0):
    """Test call with weight matrix function."""
    covariance_matrix = np.array(
        [
            [2.0, 0.0, 0.1],
            [0.0, 0.5, 0.0],
            [0.1, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    proximity_layer = GeneralizedInnerProduct(
        w_initializer=keras.initializers.Constant(covariance_matrix)
    )
    outputs = proximity_layer(paired_inputs_v0)
    outputs = keras.ops.convert_to_numpy(outputs)

    desired_outputs = np.array(
        [1.395, 24.134998, 54.275, 91.815, 136.755], dtype="float32"
    )
    np.testing.assert_array_equal(desired_outputs, outputs)


def test_serialization():
    """Test serialization."""
    covariance_matrix = np.eye(3, dtype=np.float32)
    proximity_layer = GeneralizedInnerProduct(
        w_initializer=keras.initializers.Constant(covariance_matrix)
    )
    proximity_layer.build([[None, 3], [None, 3]])
    np.testing.assert_array_equal(
        keras.ops.convert_to_numpy(proximity_layer.w), covariance_matrix
    )
    config = proximity_layer.get_config()

    recon_layer = GeneralizedInnerProduct.from_config(config)
    recon_layer.build([[None, 3], [None, 3]])
    np.testing.assert_array_equal(
        keras.ops.convert_to_numpy(proximity_layer.w),
        keras.ops.convert_to_numpy(recon_layer.w),
    )
