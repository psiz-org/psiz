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
"""Backend matrix runtime smoke tests for M6 gates."""

from __future__ import annotations

import os

import keras
import numpy as np
import pytest

import psiz
from psiz.backend import resolve_backend


pytestmark = pytest.mark.backend_runtime


@keras.saving.register_keras_serializable(
    package="psiz.keras.tests", name="RankWorkflowSmokeModel"
)
class RankWorkflowSmokeModel(keras.Model):
    """Tiny rank model used to smoke test backend runtime execution."""

    def __init__(self, n_stimuli=32, n_dim=2, **kwargs):
        super().__init__(**kwargs)
        self.percept = keras.layers.Embedding(n_stimuli + 1, n_dim, mask_zero=True)
        self.proximity = psiz.keras.layers.Minkowski(
            activation=psiz.keras.layers.ExponentialSimilarity(
                beta_trainable=False,
                tau_trainable=False,
                gamma_trainable=False,
            )
        )
        self.soft_rank = psiz.keras.layers.SoftRank(n_select=2, trainable=False)
        self.stimuli_axis = 1

    def call(self, inputs):
        z = self.percept(inputs)
        z_q, z_r = keras.ops.split(z, [1], axis=self.stimuli_axis)
        s = self.proximity([z_q, z_r])
        return self.soft_rank(s)

    def get_config(self):
        config = super().get_config()
        config.update({"n_stimuli": self.percept.input_dim - 1, "n_dim": self.percept.output_dim})
        return config


def _make_rank_batch(n_sample, n_stimuli, n_reference=8):
    rng = np.random.default_rng(seed=128)
    x = np.zeros((n_sample, n_reference + 1), dtype=np.int32)
    for i in range(n_sample):
        x[i] = rng.choice(np.arange(1, n_stimuli + 1), size=n_reference + 1, replace=False)
    return x


def _run_rank_workflow_smoke(n_epoch=1):
    keras.utils.set_random_seed(124)
    x = _make_rank_batch(n_sample=16, n_stimuli=32, n_reference=8)

    model = RankWorkflowSmokeModel(name="rank_smoke")
    y_pred0 = keras.ops.convert_to_numpy(model(x))
    y = np.zeros_like(y_pred0)
    y[:, 0] = 1.0

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.005),
        loss=keras.losses.CategoricalCrossentropy(),
        run_eagerly=True,
    )
    history = model.fit(x, y, batch_size=8, epochs=n_epoch, verbose=0)
    y_pred = keras.ops.convert_to_numpy(model(x))
    return model, history, y_pred


def _assert_backend_specific_runtime(runtime_backend, requested_backend):
    if runtime_backend != requested_backend:
        pytest.skip(
            f"This test validates '{requested_backend}' runtime and was invoked under "
            f"'{runtime_backend}'."
        )


def _assert_stochastic_runtime(runtime_backend, backend_tolerance, requested_backend):
    _assert_backend_specific_runtime(runtime_backend, requested_backend)
    adapter = psiz.stochastic.get_stochastic_adapter(backend_override=runtime_backend)

    posterior = adapter.normal(
        loc=keras.ops.ones([128], dtype="float32"),
        scale=keras.ops.ones([128], dtype="float32"),
    )
    prior = adapter.normal(
        loc=keras.ops.zeros([128], dtype="float32"),
        scale=keras.ops.ones([128], dtype="float32"),
    )
    sample = posterior.sample(sample_shape=[4])
    sample_np = keras.ops.convert_to_numpy(sample)
    assert np.all(np.isfinite(sample_np))

    kl = psiz.stochastic.kl_divergence(
        posterior,
        prior,
        backend_override=runtime_backend,
        fallback="monte_carlo",
        n_sample=1024,
    )
    kl_value = float(np.mean(keras.ops.convert_to_numpy(kl)))
    rtol, atol = backend_tolerance
    np.testing.assert_allclose(kl_value, 0.5, rtol=10 * rtol, atol=10 * atol)


@pytest.mark.backend_runtime
def test_runtime_backend_identity_assertion(runtime_backend):
    expected = os.environ.get("PSIZ_EXPECTED_BACKEND") or os.environ.get("KERAS_BACKEND")
    assert runtime_backend == expected
    assert keras.backend.backend() == expected


@pytest.mark.backend_runtime
def test_backend_matrix_core_smoke(runtime_backend):
    resolved = resolve_backend()
    assert resolved == runtime_backend

    tensor = keras.ops.convert_to_tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    mean_value = keras.ops.mean(tensor)
    np.testing.assert_allclose(float(keras.ops.convert_to_numpy(mean_value)), 2.5)


@pytest.mark.backend_slow
@pytest.mark.slow
def test_backend_matrix_slow_subset(runtime_backend):
    _assert_backend_specific_runtime(runtime_backend, runtime_backend)
    _, history, y_pred = _run_rank_workflow_smoke(n_epoch=2)

    assert np.all(np.isfinite(y_pred))
    losses = history.history.get("loss", [])
    assert len(losses) == 2
    assert all(np.isfinite(loss_value) for loss_value in losses)


@pytest.mark.docs_example
def test_docs_example_storage_roundtrip(tmp_path, runtime_backend, backend_tolerance):
    _assert_backend_specific_runtime(runtime_backend, runtime_backend)
    model, _, _ = _run_rank_workflow_smoke(n_epoch=1)
    x = _make_rank_batch(n_sample=4, n_stimuli=32, n_reference=8)

    y_expected = keras.ops.convert_to_numpy(model(x))
    artifact_dir = tmp_path / f"rank-smoke-{runtime_backend}.psiz"
    psiz.keras.save_psiz_model(model, artifact_dir, backend_override=runtime_backend)
    loaded = psiz.keras.load_psiz_model(artifact_dir, backend_override=runtime_backend)
    y_loaded = keras.ops.convert_to_numpy(loaded(x))

    rtol, atol = backend_tolerance
    np.testing.assert_allclose(y_loaded, y_expected, rtol=rtol, atol=atol)


@pytest.mark.backend_tensorflow
def test_example_rank_workflow_tensorflow(runtime_backend):
    _assert_backend_specific_runtime(runtime_backend, "tensorflow")
    _, _, y_pred = _run_rank_workflow_smoke(n_epoch=1)
    assert np.all(np.isfinite(y_pred))


@pytest.mark.backend_torch
def test_example_rank_workflow_pytorch(runtime_backend):
    _assert_backend_specific_runtime(runtime_backend, "torch")
    _, _, y_pred = _run_rank_workflow_smoke(n_epoch=1)
    assert np.all(np.isfinite(y_pred))


@pytest.mark.backend_jax
def test_example_rank_workflow_jax(runtime_backend):
    _assert_backend_specific_runtime(runtime_backend, "jax")
    _, _, y_pred = _run_rank_workflow_smoke(n_epoch=1)
    assert np.all(np.isfinite(y_pred))


@pytest.mark.backend_tensorflow
def test_stochastic_runtime_tensorflow_true_backend(runtime_backend, backend_tolerance):
    _assert_stochastic_runtime(runtime_backend, backend_tolerance, "tensorflow")


@pytest.mark.backend_torch
def test_stochastic_runtime_torch_true_backend(runtime_backend, backend_tolerance):
    _assert_stochastic_runtime(runtime_backend, backend_tolerance, "torch")


@pytest.mark.backend_jax
def test_stochastic_runtime_jax_true_backend(runtime_backend, backend_tolerance):
    _assert_stochastic_runtime(runtime_backend, backend_tolerance, "jax")
