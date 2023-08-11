# -*- coding: utf-8 -*-
# Copyright 2023 The PsiZ Authors. All Rights Reserved.
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
"""Test InnerProduct layer."""

import numpy as np
import tensorflow as tf

from psiz.keras.layers.proximities.experimental.inner_product import InnerProduct


def test_init_call_v0(paired_inputs_v0):
    """Test call with default (linear) activation."""
    proximity_layer = InnerProduct()
    outputs = proximity_layer(paired_inputs_v0)

    inputs_0 = paired_inputs_v0[0].numpy()
    inputs_1 = paired_inputs_v0[1].numpy()
    desired_outputs = np.concatenate(
        [
            np.matmul(np.expand_dims(inputs_0[0], axis=0), np.expand_dims(inputs_1[0], axis=1)),
            np.matmul(np.expand_dims(inputs_0[1], axis=0), np.expand_dims(inputs_1[1], axis=1)),
            np.matmul(np.expand_dims(inputs_0[2], axis=0), np.expand_dims(inputs_1[2], axis=1)),
            np.matmul(np.expand_dims(inputs_0[3], axis=0), np.expand_dims(inputs_1[3], axis=1)),
            np.matmul(np.expand_dims(inputs_0[4], axis=0), np.expand_dims(inputs_1[4], axis=1)),
        ], axis=0
    ).squeeze()

    np.testing.assert_array_almost_equal(desired_outputs, outputs.numpy())


def test_init_call_v1(paired_inputs_v0):
    """Test call with weight matrix function."""
    covariance_matrix = np.array(
        [
            [2.0, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32
    )
    proximity_layer = InnerProduct(
        w_tril_initializer=tf.initializers.Constant(
            tf.linalg.cholesky(covariance_matrix).numpy()
        ),
    )
    outputs = proximity_layer(paired_inputs_v0)

    desired_outputs = np.array(
        [1.295, 22.794998, 51.295, 86.795, 129.29498], dtype=np.float32
    )
    np.testing.assert_array_almost_equal(desired_outputs, outputs.numpy())


def test_init_call_v2(paired_inputs_v0):
    """Test call with weight matrix function."""
    covariance_matrix = np.array(
        [
            [2.0, 0.0, 0.1],
            [0.0, 0.5, 0.0],
            [0.1, 0.0, 1.0],
        ],
        dtype=np.float32
    )
    proximity_layer = InnerProduct(
        w_tril_initializer=tf.initializers.Constant(
            tf.linalg.cholesky(covariance_matrix).numpy()
        )
    )
    outputs = proximity_layer(paired_inputs_v0)

    desired_outputs = np.array(
        [1.3949997, 24.134998 , 54.274998, 91.814995, 136.75497],
        dtype=np.float32
    )
    np.testing.assert_array_almost_equal(desired_outputs, outputs.numpy())


def test_output_shape(paired_inputs_v0):
    """Test output_shape method."""
    covariance_matrix = np.eye(3, dtype=np.float32)
    proximity_layer = InnerProduct(
        w_tril_initializer=tf.initializers.Constant(
            tf.linalg.cholesky(covariance_matrix).numpy()
        )
    )
    input_shape = [
        tf.TensorShape(tf.shape(paired_inputs_v0[0])),
        tf.TensorShape(tf.shape(paired_inputs_v0[1])),
    ]
    output_shape = proximity_layer.compute_output_shape(input_shape)
    desired_output_shape = tf.TensorShape([5])
    tf.debugging.assert_equal(output_shape, desired_output_shape)


def test_serialization():
    """Test serialization."""
    covariance_matrix = np.eye(3, dtype=np.float32)
    proximity_layer = InnerProduct(
        w_tril_initializer=tf.initializers.Constant(
            tf.linalg.cholesky(covariance_matrix).numpy()
        )
    )
    proximity_layer.build([[None, 3], [None, 3]])
    tf.debugging.assert_equal(proximity_layer.w, tf.constant(covariance_matrix))
    config = proximity_layer.get_config()

    recon_layer = InnerProduct.from_config(config)
    recon_layer.build([[None, 3], [None, 3]])
    tf.debugging.assert_equal(proximity_layer.w, recon_layer.w)
