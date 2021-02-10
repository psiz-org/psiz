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
"""Test AttentionKernel layer."""

import numpy as np
import pytest
import tensorflow as tf

from psiz.keras.layers import AttentionKernel
from psiz.keras.layers import ExponentialSimilarity
from psiz.keras.layers import WeightedMinkowski
from psiz.keras.initializers import RandomAttention
from psiz.keras.constraints import NonNegNorm


@pytest.fixture
def kernel_ak_v0():
    n_dim = 3
    n_group = 3

    scale = n_dim
    alpha = np.ones((n_dim))
    kernel = AttentionKernel(
        group_level=1,
        attention=tf.keras.layers.Embedding(
            n_group, n_dim, mask_zero=False,
            embeddings_initializer=tf.keras.initializers.Constant(
                # np.array([
                #     [1.4, .8, .8],
                #     [.8, 1.4, .8],
                #     [.8, .8, 1.4],
                # ])
                np.ones([3, 3])
            ),
        ),
        distance=WeightedMinkowski(
            rho_initializer=tf.keras.initializers.Constant(2.),
            trainable=False
        ),
        similarity=ExponentialSimilarity(
            fit_tau=False, fit_gamma=False,
            tau_initializer=tf.keras.initializers.Constant(1.),
            gamma_initializer=tf.keras.initializers.Constant(0.0),
            beta_initializer=tf.keras.initializers.Constant(.1),
        )
    )
    return kernel


def test_call(paired_inputs_v0, group_v0, kernel_ak_v0):
    """Test call."""
    inputs_0 = paired_inputs_v0[0]
    inputs_1 = paired_inputs_v0[1]
    kernel = kernel_ak_v0
    outputs = kernel([inputs_0, inputs_1, group_v0])

    desired_outputs = np.array([
        0.4206200260541147,
        0.4206200260541147,
        0.4206200260541147,
        0.4206200260541147,
        0.4206200260541147
    ])
    np.testing.assert_array_almost_equal(
        desired_outputs, outputs.numpy(), decimal=4
    )


def test_output_shape(paired_inputs_v0, group_v0, kernel_ak_v0):
    """Test output_shape method."""
    inputs_0 = paired_inputs_v0[0]
    inputs_1 = paired_inputs_v0[1]
    kernel = kernel_ak_v0

    input_shape_0 = tf.shape(inputs_0).numpy().tolist()
    input_shape_1 = tf.shape(inputs_1).numpy().tolist()
    group_shape = tf.shape(group_v0).numpy().tolist()
    input_shape = [input_shape_0, input_shape_1, group_shape]

    output_shape = kernel.compute_output_shape(input_shape)
    desired_output_shape = tf.TensorShape([5])
    tf.debugging.assert_equal(output_shape, desired_output_shape)
