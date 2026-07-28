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
"""Backend parity tests for migrated stochastic layers."""

import keras
import numpy as np
import pytest

import psiz
from psiz.keras.layers.variational import Variational


@pytest.mark.parametrize("backend", ["tensorflow", "torch", "jax"])
def test_variational_layer_backend_parity(monkeypatch, backend):
    """Variational KL path works across stochastic adapters."""
    monkeypatch.setattr("psiz.stochastic.adapters.resolve_backend", lambda _: backend)

    layer = Variational(
        posterior=keras.layers.Layer(),
        prior=keras.layers.Layer(),
        kl_weight=1.0,
        kl_use_exact=True,
        kl_n_sample=512,
    )
    layer.build([None, 2])

    adapter = psiz.stochastic.get_stochastic_adapter()
    posterior = adapter.independent(
        adapter.normal(
            loc=keras.ops.ones([4, 2], dtype="float32"),
            scale=keras.ops.ones([4, 2], dtype="float32"),
        ),
        reinterpreted_batch_ndims=2,
    )
    prior = adapter.independent(
        adapter.normal(
            loc=keras.ops.zeros([4, 2], dtype="float32"),
            scale=keras.ops.ones([4, 2], dtype="float32"),
        ),
        reinterpreted_batch_ndims=2,
    )
    layer.add_kl_loss(posterior, prior)

    assert len(layer.losses) == 1
    loss_value = float(keras.ops.convert_to_numpy(layer.losses[0]))
    assert np.isfinite(loss_value)


@pytest.mark.parametrize("backend", ["tensorflow", "torch", "jax"])
def test_embedding_variational_backend_parity(monkeypatch, backend):
    """EmbeddingVariational executes with parity-level behavior across adapters."""
    monkeypatch.setattr("psiz.stochastic.adapters.resolve_backend", lambda _: backend)

    n_stimuli = 9
    n_dim = 2
    inputs = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)

    posterior = psiz.keras.layers.EmbeddingNormalDiag(n_stimuli, n_dim, sample_shape=())
    prior = psiz.keras.layers.EmbeddingNormalDiag(n_stimuli, n_dim, sample_shape=())

    layer = psiz.keras.layers.EmbeddingVariational(
        posterior=posterior,
        prior=prior,
        kl_weight=0.1,
        kl_n_sample=128,
        kl_use_exact=False,
    )
    output = layer(inputs)

    np.testing.assert_array_equal(np.shape(keras.ops.convert_to_numpy(output)), [2, 3, 2])
    assert len(layer.losses) == 1


@pytest.mark.parametrize("backend", ["tensorflow", "torch", "jax"])
def test_minkowski_stochastic_backend_parity(monkeypatch, backend):
    """MinkowskiStochastic produces finite outputs across adapters."""
    monkeypatch.setattr("psiz.stochastic.adapters.resolve_backend", lambda _: backend)

    layer = psiz.keras.layers.MinkowskiStochastic()
    z_0 = np.array(
        [[1.0, 2.0, 3.0], [2.0, 0.0, 1.0], [0.5, 1.0, 1.5]], dtype=np.float32
    )
    z_1 = np.array(
        [[0.5, 1.5, 2.5], [1.0, 0.0, 0.0], [0.0, 0.5, 1.0]], dtype=np.float32
    )

    output = layer([z_0, z_1])
    output_np = keras.ops.convert_to_numpy(output)
    np.testing.assert_array_equal(output_np.shape, [3])
    assert np.all(np.isfinite(output_np))
