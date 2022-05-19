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
"""Test distance-based kernel."""

import numpy as np
import pytest
import tensorflow as tf

from psiz.keras.layers import DistanceBased
from psiz.keras.layers.distances.mink import Minkowski
from psiz.keras.layers.similarities.exponential import ExponentialSimilarity


@pytest.fixture
def kernel_db_static_v0():
    """Create a distance-based kernel."""
    # Default distance-based kernel.
    kernel = DistanceBased()
    return kernel


@pytest.fixture
def kernel_db_static_v1():
    """Create a distance-based kernel."""

    # Create "unweighted" Minkowski distance layer.
    distance = Minkowski(
        rho_initializer=tf.keras.initializers.Constant(2.),
        w_initializer=tf.keras.initializers.Constant(1.),
    )
    # Create exponential similarity function.
    similarity = ExponentialSimilarity(
        fit_beta=False,
        beta_initializer=tf.keras.initializers.Constant(1.),
        tau_initializer=tf.keras.initializers.Constant(1.),
        gamma_initializer=tf.keras.initializers.Constant(0.01),
    )
    # Package as distance-based kernel.
    kernel = DistanceBased(distance=distance, similarity=similarity)

    return kernel


def test_call_v0(paired_inputs_v1, groups_v0, kernel_db_static_v0):
    """Test call."""
    inputs_0 = paired_inputs_v1[0]
    inputs_1 = paired_inputs_v1[1]
    kernel = kernel_db_static_v0
    outputs = kernel([inputs_0, inputs_1, groups_v0])

    # Since weights are randomly initialized just check shapes.
    assert outputs.shape == tf.TensorShape([5])


def test_call_v1(paired_inputs_v1, groups_v0, kernel_db_static_v1):
    """Test call."""
    inputs_0 = paired_inputs_v1[0]
    inputs_1 = paired_inputs_v1[1]
    kernel = kernel_db_static_v1
    outputs = kernel([inputs_0, inputs_1, groups_v0])

    desired_outputs = np.array([
        0.18692121,
        0.15878458,
        0.13512264,
        0.13274848,
        1.01
    ])
    np.testing.assert_array_almost_equal(
        desired_outputs, outputs.numpy(), decimal=4
    )


def test_coompute_output_shape(
        paired_inputs_v0, groups_v0, kernel_db_static_v0):
    """Test compute_output_shape method."""
    inputs_0 = paired_inputs_v0[0]
    inputs_1 = paired_inputs_v0[1]
    kernel = kernel_db_static_v0

    input_shape_0 = tf.shape(inputs_0).numpy().tolist()
    input_shape_1 = tf.shape(inputs_1).numpy().tolist()
    group_shape = tf.shape(groups_v0).numpy().tolist()
    input_shape = [input_shape_0, input_shape_1, group_shape]

    output_shape = kernel.compute_output_shape(input_shape)
    desired_output_shape = tf.TensorShape([5])
    tf.debugging.assert_equal(output_shape, desired_output_shape)
