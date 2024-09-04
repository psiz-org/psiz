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
"""Test MinkowskiVariational layer."""

import keras
import numpy as np
import pytest
import tensorflow_probability as tfp

from psiz.keras.layers.proximities.minkowski_variational import MinkowskiVariational
from psiz.keras.layers.proximities.minkowski_stochastic import MinkowskiStochastic


def test_call(paired_inputs_v0):
    """Test call."""
    kl_weight = 0.1

    mink_posterior = MinkowskiStochastic()
    mink_prior = MinkowskiStochastic()

    mink_layer = MinkowskiVariational(
        posterior=mink_posterior, prior=mink_prior, kl_weight=kl_weight, kl_n_sample=30
    )
    mink_layer.build([[None, 3], [None, 3]])
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


def test_serialization():
    """Test serialization."""
    kl_weight = 0.1

    mink_posterior = MinkowskiStochastic()
    mink_prior = MinkowskiStochastic()

    mink_layer = MinkowskiVariational(
        posterior=mink_posterior, prior=mink_prior, kl_weight=kl_weight, kl_n_sample=30
    )
    mink_layer.build([[None, 3], [None, 3]])
    config = mink_layer.get_config()

    recon_layer = MinkowskiVariational.from_config(config)

    np.testing.assert_array_equal(
        keras.ops.convert_to_numpy(mink_layer.posterior.rho.mode()),
        keras.ops.convert_to_numpy(recon_layer.posterior.rho.mode()),
    )
    np.testing.assert_array_equal(
        keras.ops.convert_to_numpy(mink_layer.prior.rho.mode()),
        keras.ops.convert_to_numpy(recon_layer.prior.rho.mode()),
    )
    np.testing.assert_array_equal(
        keras.ops.convert_to_numpy(mink_layer.posterior.w.mode()),
        keras.ops.convert_to_numpy(recon_layer.posterior.w.mode()),
    )
    np.testing.assert_array_equal(
        keras.ops.convert_to_numpy(mink_layer.prior.w.mode()),
        keras.ops.convert_to_numpy(recon_layer.prior.w.mode()),
    )


@pytest.mark.tfp
def test_properties():
    """Test properties."""
    kl_weight = 0.1

    mink_posterior = MinkowskiStochastic()
    mink_prior = MinkowskiStochastic()

    mink_layer = MinkowskiVariational(
        posterior=mink_posterior, prior=mink_prior, kl_weight=kl_weight, kl_n_sample=30
    )
    mink_layer.build([[None, 3], [None, 3]])

    # Test weight property.
    w = mink_layer.w
    assert isinstance(w, tfp.distributions.Distribution)
    assert w.event_shape == [3]

    # Test rho property.
    rho = mink_layer.rho
    assert isinstance(rho, tfp.distributions.Distribution)
    assert rho.event_shape == []
