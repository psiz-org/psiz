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
"""Utilities for backend resolution and backend capability validation."""

from __future__ import annotations

from collections.abc import Iterable

import keras

PSIZ_DEFAULT_BACKEND = "torch"
SUPPORTED_BACKENDS = ("tensorflow", "torch", "jax")

_BACKEND_ALIASES = {
    "tf": "tensorflow",
    "tensorflow": "tensorflow",
    "torch": "torch",
    "pytorch": "torch",
    "jax": "jax",
}


def normalize_backend_name(backend: str) -> str:
    """Normalize a backend string and map common aliases."""
    if not isinstance(backend, str):
        raise ValueError("Backend must be provided as a string.")

    normalized = backend.strip().lower()
    if not normalized:
        raise ValueError("Backend cannot be empty.")
    return _BACKEND_ALIASES.get(normalized, normalized)


def validate_backend_name(backend: str, argument_name: str = "backend") -> str:
    """Validate that a backend is recognized and return normalized form."""
    normalized = normalize_backend_name(backend)
    if normalized not in SUPPORTED_BACKENDS:
        supported = ", ".join(SUPPORTED_BACKENDS)
        raise ValueError(
            f"Unsupported {argument_name} '{backend}'. Supported backends: {supported}."
        )
    return normalized


def resolve_backend(
    backend_override: str | None = None,
    *,
    default_backend: str = PSIZ_DEFAULT_BACKEND,
) -> str:
    """Resolve backend using PsiZ precedence policy.

    Precedence:
        1) Explicit user override.
        2) Active Keras backend.
        3) PsiZ default backend.
    """
    if backend_override is not None:
        return validate_backend_name(backend_override, argument_name="backend override")

    active_backend = _active_keras_backend()
    if active_backend is not None:
        return active_backend

    return validate_backend_name(default_backend, argument_name="default backend")


def validate_backend_support(
    backend: str,
    *,
    feature_name: str,
    supported_backends: Iterable[str],
    capability_enabled: bool = True,
) -> str:
    """Validate backend support for a feature when capability is enabled."""
    normalized_backend = validate_backend_name(backend)
    if not capability_enabled:
        return normalized_backend

    normalized_supported = tuple(
        validate_backend_name(value, argument_name="supported backend")
        for value in supported_backends
    )
    if normalized_backend not in normalized_supported:
        supported = ", ".join(normalized_supported)
        raise ValueError(
            f"Feature '{feature_name}' is not supported for backend "
            f"'{normalized_backend}'. Supported backends: {supported}."
        )
    return normalized_backend


def _active_keras_backend() -> str | None:
    """Return normalized active Keras backend if available."""
    try:
        backend_name = keras.backend.backend()
    except Exception:
        return None

    if backend_name is None:
        return None
    if not isinstance(backend_name, str):
        raise ValueError("Active Keras backend must be a string.")

    backend_name = backend_name.strip()
    if not backend_name:
        return None
    return validate_backend_name(backend_name, argument_name="active backend")
