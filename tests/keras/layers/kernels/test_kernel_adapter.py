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
"""Test MinkowskiVariational layer."""

import numpy as np
import pytest
import tensorflow as tf


from psiz.keras.layers import DistanceBased
from psiz.keras.layers import ExponentialSimilarity
from psiz.keras.layers import GroupSpecific
from psiz.keras.layers import KernelAdapter
from psiz.keras.layers import Minkowski
from psiz.keras.layers import MinkowskiStochastic
from psiz.keras.layers import MinkowskiVariational


@pytest.fixture
def db_kernel_v0():
    """A list of subnets"""
    kernel = DistanceBased(
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
    return kernel


def test_call(pw_inputs_v0, group_v0, db_kernel_v0):
    """Test call."""
    kl_weight = .1

    inputs_0, inputs_1 = tf.unstack(pw_inputs_v0, num=2, axis=-1)

    group_layer = GroupSpecific(subnets=[db_kernel_v0], group_col=0)
    kernel_adap = KernelAdapter(kernel=group_layer)
    outputs = kernel_adap([inputs_0, inputs_1, group_v0])

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


def test_output_shape(pw_inputs_v0, group_v0, db_kernel_v0):
    """Test output_shape method."""
    kl_weight = .1

    inputs_0, inputs_1 = tf.unstack(pw_inputs_v0, num=2, axis=-1)

    group_layer = GroupSpecific(subnets=[db_kernel_v0], group_col=0)
    kernel_adap = KernelAdapter(kernel=group_layer)

    input_shape_0 = tf.shape(inputs_0).numpy().tolist()
    input_shape_1 = tf.shape(inputs_1).numpy().tolist()
    group_shape = tf.shape(group_v0).numpy().tolist()
    input_shape = [input_shape_0, input_shape_1, group_shape]

    output_shape = kernel_adap.compute_output_shape(input_shape)
    desired_output_shape = tf.TensorShape([5])
    tf.debugging.assert_equal(output_shape, desired_output_shape)


# def test_serialization():
#     """Test serialization."""
#     kl_weight = .1

#     mink_posterior = MinkowskiStochastic()
#     mink_prior = MinkowskiStochastic()

#     mink_layer = MinkowskiVariational(
#         posterior=mink_posterior,
#         prior=mink_prior,
#         kl_weight=kl_weight, kl_n_sample=30
#     )
#     mink_layer.build([None, 3, 2])
#     config = mink_layer.get_config()

#     recon_layer = MinkowskiVariational.from_config(config)
#     recon_layer.build([None, 3, 2])

#     tf.debugging.assert_equal(
#         mink_layer.posterior.rho.mode(), recon_layer.posterior.rho.mode()
#     )
#     tf.debugging.assert_equal(
#         mink_layer.prior.rho.mode(), recon_layer.prior.rho.mode()
#     )
#     tf.debugging.assert_equal(
#         mink_layer.posterior.w.mode(), recon_layer.posterior.w.mode()
#     )
#     tf.debugging.assert_equal(
#         mink_layer.prior.w.mode(), recon_layer.prior.w.mode()
#     )
