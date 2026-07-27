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
"""Tests for backend resolution and capability utilities."""

import keras
import pytest

from psiz.backend import resolve_backend
from psiz.backend import validate_backend_support


def test_backend_resolution_precedence(monkeypatch):
    """Explicit override takes precedence over active backend and default."""
    monkeypatch.setattr(keras.backend, "backend", lambda: "jax")

    resolved = resolve_backend(backend_override="tensorflow", default_backend="torch")

    assert resolved == "tensorflow"


def test_backend_resolution_explicit_override():
    """Explicit override resolves directly and normalizes aliases."""
    resolved = resolve_backend(backend_override="pytorch")

    assert resolved == "torch"


def test_backend_resolution_keras_fallback(monkeypatch):
    """Active Keras backend is used when no explicit override is provided."""
    monkeypatch.setattr(keras.backend, "backend", lambda: "jax")

    resolved = resolve_backend(default_backend="torch")

    assert resolved == "jax"


def test_backend_resolution_psiz_default(monkeypatch):
    """PsiZ default is used when no active Keras backend is available."""
    monkeypatch.setattr(keras.backend, "backend", lambda: "")

    resolved = resolve_backend(default_backend="torch")

    assert resolved == "torch"


def test_backend_resolution_invalid_backend_errors(monkeypatch):
    """Invalid backend values raise clear validation errors."""
    with pytest.raises(ValueError, match="backend override"):
        _ = resolve_backend(backend_override="mxnet")

    monkeypatch.setattr(keras.backend, "backend", lambda: "mxnet")
    with pytest.raises(ValueError, match="active backend"):
        _ = resolve_backend()


def test_capability_flag_blocks_unsupported_feature():
    """Enabled capability raises if the selected backend lacks support."""
    with pytest.raises(ValueError, match="exact_kl_divergence"):
        _ = validate_backend_support(
            "torch",
            feature_name="exact_kl_divergence",
            supported_backends=("tensorflow",),
            capability_enabled=True,
        )
