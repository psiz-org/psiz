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
"""Test GateMulti."""

import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.layers import Embedding
import tensorflow_probability as tfp

from psiz.keras.layers import DistanceBased
from psiz.keras.layers import EmbeddingNormalDiag
from psiz.keras.layers import ExponentialSimilarity
from psiz.keras.layers import GateMulti
from psiz.keras.layers import Minkowski
from psiz.keras.layers import MinkowskiStochastic
from psiz.keras.layers import MinkowskiVariational


def build_vi_kernel(similarity, n_dim, kl_weight):
    """Build kernel for single group."""
    mink_prior = MinkowskiStochastic(
        rho_loc_trainable=False, rho_scale_trainable=True,
        w_loc_trainable=False, w_scale_trainable=False,
        w_scale_initializer=tf.keras.initializers.Constant(.1)
    )

    mink_posterior = MinkowskiStochastic(
        rho_loc_trainable=False, rho_scale_trainable=True,
        w_loc_trainable=True, w_scale_trainable=True,
        w_scale_initializer=tf.keras.initializers.Constant(.1)
    )

    mink = MinkowskiVariational(
        prior=mink_prior, posterior=mink_posterior,
        kl_weight=kl_weight, kl_n_sample=30
    )

    kernel = DistanceBased(
        distance=mink,
        similarity=similarity
    )
    return kernel


@pytest.fixture
def kernel_subnets():
    """A list of subnets"""
    pw_0 = DistanceBased(
        distance=Minkowski(
            rho_initializer=tf.keras.initializers.Constant(2.),
            w_initializer=tf.keras.initializers.Constant(1.),
            trainable=False
        ),
        similarity=ExponentialSimilarity(
            fit_tau=False, fit_gamma=False,
            tau_initializer=tf.keras.initializers.Constant(1.),
            gamma_initializer=tf.keras.initializers.Constant(0.0),
            beta_initializer=tf.keras.initializers.Constant(.1),
        )
    )
    pw_1 = DistanceBased(
        distance=Minkowski(
            rho_initializer=tf.keras.initializers.Constant(2.),
            w_initializer=tf.keras.initializers.Constant(1.),
            trainable=False
        ),
        similarity=ExponentialSimilarity(
            fit_tau=False, fit_gamma=False,
            tau_initializer=tf.keras.initializers.Constant(1.),
            gamma_initializer=tf.keras.initializers.Constant(0.0),
            beta_initializer=tf.keras.initializers.Constant(.1),
        )
    )

    pw_2 = DistanceBased(
        distance=Minkowski(
            rho_initializer=tf.keras.initializers.Constant(2.),
            w_initializer=tf.keras.initializers.Constant(1.),
            trainable=False
        ),
        similarity=ExponentialSimilarity(
            fit_tau=False, fit_gamma=False,
            tau_initializer=tf.keras.initializers.Constant(1.),
            gamma_initializer=tf.keras.initializers.Constant(0.0),
            beta_initializer=tf.keras.initializers.Constant(.1),
        )
    )

    subnets = [pw_0, pw_1, pw_2]
    return subnets


def test_subnet_method(kernel_subnets):
    group_layer = GateMulti(subnets=kernel_subnets, group_col=0)
    group_layer.build([[None, 3], [None, 3], [None, 3]])

    subnet_0 = group_layer.subnets[0]
    subnet_1 = group_layer.subnets[1]
    subnet_2 = group_layer.subnets[2]

    tf.debugging.assert_equal(
        subnet_0.distance.rho, kernel_subnets[0].distance.rho
    )
    tf.debugging.assert_equal(
        subnet_0.distance.w, kernel_subnets[0].distance.w
    )

    tf.debugging.assert_equal(
        subnet_1.distance.rho, kernel_subnets[1].distance.rho
    )
    tf.debugging.assert_equal(
        subnet_1.distance.w, kernel_subnets[1].distance.w
    )

    tf.debugging.assert_equal(
        subnet_2.distance.rho, kernel_subnets[2].distance.rho
    )
    tf.debugging.assert_equal(
        subnet_2.distance.w, kernel_subnets[2].distance.w
    )


def test_kernel_call(kernel_subnets, paired_inputs_v0, group_v0):
    group_layer = GateMulti(subnets=kernel_subnets, group_col=0)
    outputs = group_layer(
        [paired_inputs_v0[0], paired_inputs_v0[1], group_v0]
    )

    # x = np.exp(-.1 * np.sqrt(3*(5**2)))
    desired_outputs = np.array([
        0.4206200260541147,
        0.4206200260541147,
        0.4206200260541147,
        0.4206200260541147,
        0.4206200260541147
    ])

    np.testing.assert_array_almost_equal(desired_outputs, outputs.numpy())


def test_call_kernel_empty_branch(paired_inputs_v0, group_3g_empty_v0):
    """Test call with empty branch."""
    n_dim = 3
    kl_weight = .1

    shared_similarity = ExponentialSimilarity(
        beta_initializer=tf.keras.initializers.Constant(10.),
        tau_initializer=tf.keras.initializers.Constant(1.),
        gamma_initializer=tf.keras.initializers.Constant(0.),
        trainable=False
    )

    # Define group-specific kernels.
    kernel_0 = build_vi_kernel(shared_similarity, n_dim, kl_weight)
    kernel_1 = build_vi_kernel(shared_similarity, n_dim, kl_weight)
    kernel_2 = build_vi_kernel(shared_similarity, n_dim, kl_weight)
    kernel_group = GateMulti(
        subnets=[kernel_0, kernel_1, kernel_2], group_col=0
    )

    outputs = kernel_group(
        [paired_inputs_v0[0], paired_inputs_v0[1], group_3g_empty_v0]
    )


def test_kernel_output_shape(kernel_subnets, paired_inputs_v0, group_v0):
    """Test output_shape method."""
    group_layer = GateMulti(subnets=kernel_subnets, group_col=0)

    input_shape = [
        tf.TensorShape(tf.shape(paired_inputs_v0[0])),
        tf.TensorShape(tf.shape(paired_inputs_v0[1])),
        tf.TensorShape(tf.shape(group_v0))
    ]
    output_shape = group_layer.compute_output_shape(input_shape)
    desired_output_shape = tf.TensorShape([5])
    tf.debugging.assert_equal(output_shape, desired_output_shape)
