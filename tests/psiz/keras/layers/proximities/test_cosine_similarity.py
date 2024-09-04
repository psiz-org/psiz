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
"""Test CosineSimilarity layer."""


import keras
import numpy as np

from psiz.keras.layers.proximities.experimental.cosine_similarity import (
    CosineSimilarity,
)

from scipy.spatial.distance import cosine


def test_init_call_v0(paired_inputs_v0):
    """Test call with default (linear) activation."""
    proximity_layer = CosineSimilarity()
    outputs = proximity_layer(paired_inputs_v0)
    outputs = keras.ops.convert_to_numpy(outputs)

    inputs_0 = paired_inputs_v0[0]
    inputs_1 = paired_inputs_v0[1]
    desired_outputs = 1 - np.array(
        [
            cosine(inputs_0[0], inputs_1[0]),
            cosine(inputs_0[1], inputs_1[1]),
            cosine(inputs_0[2], inputs_1[2]),
            cosine(inputs_0[3], inputs_1[3]),
            cosine(inputs_0[4], inputs_1[4]),
        ]
    )

    np.testing.assert_array_almost_equal(desired_outputs, outputs)


def test_init_call_v1(paired_inputs_v0):
    """Test call with weight matrix function."""
    w = np.array([2.0, 0.5, 1.0], dtype="float32")
    proximity_layer = CosineSimilarity(
        w_initializer=keras.initializers.Constant(w),
    )
    outputs = proximity_layer(paired_inputs_v0)
    outputs = keras.ops.convert_to_numpy(outputs)

    # NOTE: If one wants to compute the desired value on the fly, the following
    # code block can be used. However, extra care will need to be taken in
    # handling differences due to float precision.
    # inputs_0 = paired_inputs_v0[0]
    # inputs_1 = paired_inputs_v0[1]
    # desired_outputs = 1 - np.array(
    #     [
    #         cosine(inputs_0[0], inputs_1[0], w=w),
    #         cosine(inputs_0[1], inputs_1[1], w=w),
    #         cosine(inputs_0[2], inputs_1[2], w=w),
    #         cosine(inputs_0[3], inputs_1[3], w=w),
    #         cosine(inputs_0[4], inputs_1[4], w=w),
    #     ], dtype="float32"
    # )

    desired_outputs = np.array(
        [0.64332986, 0.9977225, 0.9995489, 0.99984235, 0.9999291], dtype="float32"
    )

    np.testing.assert_array_equal(desired_outputs, outputs)


def test_serialization():
    """Test serialization."""
    w = np.ones([3], dtype="float32")
    proximity_layer = CosineSimilarity(w_initializer=keras.initializers.Constant(w))
    proximity_layer.build([[None, 3], [None, 3]])
    np.testing.assert_array_equal(keras.ops.convert_to_numpy(proximity_layer.w), w)
    config = proximity_layer.get_config()

    recon_layer = CosineSimilarity.from_config(config)
    recon_layer.build([[None, 3], [None, 3]])
    np.testing.assert_array_equal(
        keras.ops.convert_to_numpy(proximity_layer.w),
        keras.ops.convert_to_numpy(recon_layer.w),
    )
