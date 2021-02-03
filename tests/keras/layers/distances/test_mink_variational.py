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

from psiz.keras.layers.distances.mink_variational import MinkowskiVariational
from psiz.keras.layers.distances.mink_stochastic import MinkowskiStochastic


def test_call(pw_inputs_v0):
    """Test call."""
    kl_weight = .1

    mink_posterior = MinkowskiStochastic()
    mink_prior = MinkowskiStochastic()

    mink_layer = MinkowskiVariational(
        posterior=mink_posterior,
        prior=mink_prior,
        kl_weight=kl_weight, kl_n_sample=30
    )
    mink_layer.build([None, 3, 2])
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


def test_serialization():
    """Test serialization."""
    kl_weight = .1

    mink_posterior = MinkowskiStochastic()
    mink_prior = MinkowskiStochastic()

    mink_layer = MinkowskiVariational(
        posterior=mink_posterior,
        prior=mink_prior,
        kl_weight=kl_weight, kl_n_sample=30
    )
    mink_layer.build([None, 3, 2])
    config = mink_layer.get_config()

    recon_layer = MinkowskiVariational.from_config(config)
    recon_layer.build([None, 3, 2])

    tf.debugging.assert_equal(
        mink_layer.posterior.rho.mode(), recon_layer.posterior.rho.mode()
    )
    tf.debugging.assert_equal(
        mink_layer.prior.rho.mode(), recon_layer.prior.rho.mode()
    )
    tf.debugging.assert_equal(
        mink_layer.posterior.w.mode(), recon_layer.posterior.w.mode()
    )
    tf.debugging.assert_equal(
        mink_layer.prior.w.mode(), recon_layer.prior.w.mode()
    )
