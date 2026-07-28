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
"""Backend-runtime helpers and guards for matrix smoke tests."""

from __future__ import annotations

import importlib.util
import os

import keras
import pytest


_BACKEND_IMPORT_GUARDS = {
    "tensorflow": ("tensorflow", "tensorflow_probability"),
    "torch": ("torch",),
    "jax": ("jax", "jaxlib"),
}

_BACKEND_RTOL_ATOL = {
    "tensorflow": (1e-6, 1e-6),
    "torch": (1e-5, 1e-5),
    "jax": (2e-5, 2e-5),
}


def _expected_backend_from_env() -> str | None:
    expected = os.environ.get("PSIZ_EXPECTED_BACKEND", "").strip()
    if expected:
        return expected
    expected = os.environ.get("KERAS_BACKEND", "").strip()
    if expected:
        return expected
    return None


def _assert_backend_imports_available(backend: str):
    modules = _BACKEND_IMPORT_GUARDS.get(backend, ())
    missing = [name for name in modules if importlib.util.find_spec(name) is None]
    if missing:
        joined = ", ".join(missing)
        pytest.fail(
            f"Backend runtime job for '{backend}' is missing required dependencies: {joined}."
        )


@pytest.fixture
def runtime_backend() -> str:
    """Return active runtime backend after validating matrix-job assumptions."""
    expected = _expected_backend_from_env()
    if expected is None:
        pytest.skip("Backend runtime smoke tests require KERAS_BACKEND or PSIZ_EXPECTED_BACKEND.")

    active = keras.backend.backend()
    assert active == expected, (
        f"Active backend mismatch: expected '{expected}', got '{active}'. "
        "Set KERAS_BACKEND before importing Keras/PsiZ."
    )
    _assert_backend_imports_available(expected)
    return expected


@pytest.fixture
def backend_tolerance(runtime_backend: str) -> tuple[float, float]:
    """Return backend-specific (rtol, atol) tuple used in smoke checks."""
    return _BACKEND_RTOL_ATOL[runtime_backend]
