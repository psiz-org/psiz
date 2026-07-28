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
"""Model index helpers for PsiZ .psiz artifacts."""

from __future__ import annotations

from typing import Any


def variable_identifier(variable: Any) -> str:
    """Return a stable variable identifier for storage mapping."""
    path = getattr(variable, "path", None)
    if isinstance(path, str) and path.strip():
        return _canonicalize_identifier(path)

    name = getattr(variable, "name", None)
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Encountered model weight without a valid name/path.")
    return _canonicalize_identifier(name.split(":")[0])


def _canonicalize_identifier(identifier: str) -> str:
    """Canonicalize weight identifiers across serialization boundaries.

    Keras may include a model-root prefix in one session and omit it in
    another. Strip only that leading scope when the identifier has at least
    three path components.
    """
    parts = [part for part in identifier.split("/") if part]
    if len(parts) >= 3:
        return "/".join(parts[1:])
    return "/".join(parts)


def build_model_index(model: Any, key_map: dict[str, str]) -> dict[str, Any]:
    """Build deterministic model_index.json payload."""
    weights: list[dict[str, Any]] = []
    for variable in model.weights:
        identifier = variable_identifier(variable)
        if identifier not in key_map:
            raise ValueError(f"Weight '{identifier}' is missing safetensors key mapping.")

        shape = [int(dim) for dim in variable.shape]
        weights.append(
            {
                "name": identifier,
                "key": key_map[identifier],
                "shape": shape,
                "dtype": str(variable.dtype),
            }
        )

    weights.sort(key=lambda entry: entry["key"])

    return {
        "weight_format": "safetensors",
        "weight_file": "model.safetensors",
        "weights": weights,
    }
