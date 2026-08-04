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
"""Deterministic safetensors weight IO for PsiZ artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import keras
import numpy as np
from safetensors.numpy import load_file
from safetensors.numpy import save_file

from psiz.storage.index import variable_identifier


def canonical_weight_key_map(model: Any) -> dict[str, str]:
    """Create deterministic mapping from weight names to safetensors keys."""
    identifiers = [variable_identifier(weight) for weight in model.weights]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(
            "Model contains duplicate weight identifiers; cannot create deterministic key map."
        )

    ordered_ids = sorted(identifiers)
    return {identifier: f"weight_{idx:05d}" for idx, identifier in enumerate(ordered_ids)}


def write_safetensors_weights(
    model: Any, path: str | Path, key_map: dict[str, str]
) -> set[str]:
    """Write model weights to safetensors using a deterministic key map."""
    tensors: dict[str, np.ndarray] = {}
    for variable in model.weights:
        identifier = variable_identifier(variable)
        key = key_map[identifier]
        tensors[key] = np.asarray(keras.ops.convert_to_numpy(variable))

    save_file(tensors, str(path))
    return set(tensors.keys())


def write_safetensors_tensors(
    tensors: dict[str, np.ndarray], path: str | Path
) -> set[str]:
    """Write a tensor mapping to safetensors."""
    normalized: dict[str, np.ndarray] = {}
    for key, value in tensors.items():
        normalized[key] = np.asarray(value)

    save_file(normalized, str(path))
    return set(normalized.keys())


def read_safetensors_weights(path: str | Path) -> dict[str, np.ndarray]:
    """Load safetensors weights into memory."""
    return load_file(str(path))
