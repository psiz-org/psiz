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
"""Test MinkowskiStochastic layer."""

import numpy as np
import pytest
import tensorflow as tf

from psiz.keras.layers.distances.mink_stochastic import MinkowskiStochastic


def test_call(pw_inputs_v0):
    # Test with defaults.
    mink_layer = MinkowskiStochastic()
    outputs = mink_layer(pw_inputs_v0)

    desired_outputs = np.array([
        8.660254037844387,
        8.660254037844387,
        8.660254037844387,
        8.660254037844387,
        8.660254037844387
    ])
    np.testing.assert_array_almost_equal(
        desired_outputs, outputs.numpy(), decimal=4
    )


def test_output_shape(pw_inputs_v0):
    """Test output_shape method."""
    mink_layer = MinkowskiStochastic()
    input_shape = tf.shape(pw_inputs_v0).numpy().tolist()
    output_shape = mink_layer.compute_output_shape(input_shape)
    desired_output_shape = tf.TensorShape([5])
    tf.debugging.assert_equal(output_shape, desired_output_shape)
