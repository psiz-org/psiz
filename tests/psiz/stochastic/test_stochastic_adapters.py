# -*- coding: utf-8 -*-
# Copyright 2026 The PsiZ Authors. All Rights Reserved.
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
"""Tests for stochastic adapter abstractions."""

import keras
import numpy as np
import pytest

from psiz.stochastic import canonicalize_parameters
from psiz.stochastic import get_stochastic_adapter
from psiz.stochastic import kl_divergence
from psiz.stochastic import softplus_inverse


pytestmark = pytest.mark.adapter_surface


@pytest.mark.parametrize("backend", ["tensorflow", "torch", "jax"])
def test_stochastic_adapter_distribution_surface_parity(backend):
    """Each backend adapter exposes the same core distribution surface."""
    adapter = get_stochastic_adapter(backend_override=backend)

    dist = adapter.normal(
        loc=keras.ops.zeros([3, 2], dtype="float32"),
        scale=keras.ops.ones([3, 2], dtype="float32"),
    )
    wrapped = adapter.independent(
        dist,
        reinterpreted_batch_ndims=keras.ops.size(dist.batch_shape_tensor()),
    )

    for attr in [
        "sample",
        "log_prob",
        "mean",
        "variance",
        "mode",
        "batch_shape_tensor",
    ]:
        assert hasattr(wrapped, attr)

    assert hasattr(wrapped, "distribution")


def test_stochastic_adapter_parameter_alias_parity():
    """Canonical parameter aliases resolve to the same canonical keys."""
    canonical = canonicalize_parameters({"mean": 0.5, "stddev": 0.2, "beta": 3.0})
    assert canonical["loc"] == 0.5
    assert canonical["scale"] == 0.2
    assert canonical["rate"] == 3.0

    with pytest.raises(ValueError, match="Multiple aliases"):
        _ = canonicalize_parameters({"loc": 0.0, "mean": 1.0})


@pytest.mark.parametrize("backend", ["tensorflow", "torch", "jax"])
def test_stochastic_adapter_sample_shape_parity(backend):
    """Sampling contracts preserve leading sample dimensions for all adapters."""
    adapter = get_stochastic_adapter(backend_override=backend)
    dist = adapter.normal(
        loc=keras.ops.zeros([4, 2], dtype="float32"),
        scale=keras.ops.ones([4, 2], dtype="float32"),
    )
    sample = dist.sample(sample_shape=[3, 5])
    np.testing.assert_array_equal(np.shape(keras.ops.convert_to_numpy(sample)), [3, 5, 4, 2])


@pytest.mark.parametrize("backend", ["tensorflow", "torch", "jax"])
def test_stochastic_adapter_kl_parity(backend):
    """KL API yields parity-level scalar KL values across backends."""
    adapter = get_stochastic_adapter(backend_override=backend)
    posterior = adapter.normal(
        loc=keras.ops.ones([128], dtype="float32"),
        scale=keras.ops.ones([128], dtype="float32"),
    )
    prior = adapter.normal(
        loc=keras.ops.zeros([128], dtype="float32"),
        scale=keras.ops.ones([128], dtype="float32"),
    )

    kl = kl_divergence(
        posterior,
        prior,
        backend_override=backend,
        fallback="monte_carlo",
        n_sample=2048,
    )
    kl_value = float(np.mean(keras.ops.convert_to_numpy(kl)))
    assert abs(kl_value - 0.5) < 0.08


@pytest.mark.parametrize("backend", ["tensorflow", "torch", "jax"])
def test_stochastic_adapter_transform_parity(backend):
    """Inverse-softplus transform round-trips across adapters."""
    x = keras.ops.convert_to_tensor(np.array([0.3, 1.0, 2.5], dtype="float32"))
    y = softplus_inverse(x, backend_override=backend)
    x_roundtrip = keras.ops.softplus(y)
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(x),
        keras.ops.convert_to_numpy(x_roundtrip),
        atol=1e-5,
        rtol=1e-5,
    )
