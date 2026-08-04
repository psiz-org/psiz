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
"""PsiZ-controlled save/load APIs for .psiz artifacts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import keras

from psiz.backend import resolve_backend
from psiz.storage.index import build_model_index
from psiz.storage.index import variable_identifier
from psiz.storage.model_config_compaction import compact_model_config
from psiz.storage.model_config_compaction import restore_externalized_model_config
from psiz.storage.schema import ARTIFACT_TYPE
from psiz.storage.schema import ArtifactSpecError
from psiz.storage.schema import FORMAT_NAME
from psiz.storage.schema import FORMAT_VERSION
from psiz.storage.schema import validate_artifact_directory
from psiz.storage.schema import validate_model_index_weight_integrity
from psiz.storage.weights import canonical_weight_key_map
from psiz.storage.weights import read_safetensors_weights
from psiz.storage.weights import write_safetensors_tensors
from psiz.storage.weights import write_safetensors_weights


def save_psiz_model(
    model: keras.Model,
    path: str | Path,
    *,
    backend_override: str | None = None,
    readme_text: str | None = None,
    license_text: str = "Apache-2.0\n",
    license_name: str = "Apache-2.0",
    license_policy: str = "include",
    min_externalized_config_bytes: int = 64 * 1024,
) -> dict[str, Any]:
    """Save a Keras model to a PsiZ .psiz artifact directory."""
    artifact_dir = Path(path)
    _prepare_artifact_directory(artifact_dir)

    resolved_backend = resolve_backend(backend_override=backend_override)

    key_map = canonical_weight_key_map(model)
    model_index = build_model_index(model, key_map)
    tensor_keys = write_safetensors_weights(
        model,
        artifact_dir / "model.safetensors",
        key_map,
    )

    architecture = {
        "class_name": model.__class__.__name__,
        "module": model.__class__.__module__,
    }

    model_config = keras.saving.serialize_keras_object(model)
    model_config, config_blob_tensors, model_config_compaction = compact_model_config(
        model_config,
        min_externalized_bytes=min_externalized_config_bytes,
    )

    if model_config_compaction is not None:
        blob_file = model_config_compaction["blob_file"]
        write_safetensors_tensors(config_blob_tensors, artifact_dir / blob_file)

    config = {
        "artifact_type": ARTIFACT_TYPE,
        "format_name": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "backend": resolved_backend,
        "architecture": architecture,
        "license": {"name": license_name, "policy": license_policy},
        "model_config": model_config,
    }
    if model_config_compaction is not None:
        config["model_config_compaction"] = model_config_compaction

    metadata = {
        "artifact_type": ARTIFACT_TYPE,
        "format_name": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "backend": resolved_backend,
        "architecture": architecture,
        "license": {"name": license_name, "policy": license_policy},
        "storage": {
            "weight_format": "safetensors",
            "weight_file": "model.safetensors",
            "weight_count": len(tensor_keys),
        },
    }

    _write_json(artifact_dir / "config.json", config)
    _write_json(artifact_dir / "metadata.json", metadata)
    _write_json(artifact_dir / "model_index.json", model_index)

    if readme_text is None:
        readme_text = (
            "# PsiZ Artifact\n\n"
            f"- class_name: {architecture['class_name']}\n"
            f"- format_version: {FORMAT_VERSION}\n"
            f"- backend: {resolved_backend}\n"
        )
    (artifact_dir / "README.md").write_text(readme_text, encoding="utf-8")
    (artifact_dir / "LICENSE").write_text(license_text, encoding="utf-8")

    return validate_artifact_directory(artifact_dir, backend_override=backend_override)


def load_psiz_model(
    path: str | Path,
    *,
    backend_override: str | None = None,
    custom_objects: dict[str, Any] | None = None,
) -> keras.Model:
    """Load a model from a PsiZ .psiz artifact directory."""
    manifest = validate_artifact_directory(path, backend_override=backend_override)
    artifact_dir = Path(manifest["artifact_dir"])

    config = manifest["config"]
    model_config = config.get("model_config")
    if not isinstance(model_config, dict):
        raise ArtifactSpecError("config.model_config must be present for loading.")

    model_config_compaction = config.get("model_config_compaction")
    if isinstance(model_config_compaction, dict):
        blob_file = model_config_compaction["blob_file"]
        blob_tensors = read_safetensors_weights(artifact_dir / blob_file)
        try:
            model_config = restore_externalized_model_config(model_config, blob_tensors)
        except ValueError as exc:
            raise ArtifactSpecError(str(exc)) from exc

    model = keras.saving.deserialize_keras_object(
        model_config,
        custom_objects=custom_objects,
        safe_mode=False,
    )

    tensors = read_safetensors_weights(artifact_dir / "model.safetensors")
    validate_model_index_weight_integrity(manifest["model_index"], set(tensors.keys()))

    index_entries = manifest["model_index"]["weights"]
    entries_by_name = {entry["name"]: entry for entry in index_entries}

    if len(model.weights) != len(index_entries):
        raise ArtifactSpecError(
            "Weight count mismatch between loaded model and model_index.json. "
            f"Model has {len(model.weights)} weights, index has {len(index_entries)}."
        )

    normalized_entry_map: dict[str, list[dict[str, Any]]] = {}
    for entry in index_entries:
        normalized_name = _normalize_weight_name(entry["name"])
        normalized_entry_map.setdefault(normalized_name, []).append(entry)

    used_keys: set[str] = set()
    weight_values = []
    for variable in model.weights:
        identifier = variable_identifier(variable)
        normalized_identifier = _normalize_weight_name(identifier)
        variable_shape = tuple(int(v) for v in variable.shape)

        candidate_entry = _choose_entry_for_variable(
            identifier=identifier,
            normalized_identifier=normalized_identifier,
            variable_shape=variable_shape,
            entries_by_name=entries_by_name,
            normalized_entry_map=normalized_entry_map,
            used_keys=used_keys,
        )

        key = candidate_entry["key"]
        tensor = tensors[key]
        if tuple(tensor.shape) != variable_shape:
            raise ArtifactSpecError(
                "Loaded model weight shape does not match artifact tensor shape for "
                f"'{identifier}': expected {variable_shape}, "
                f"found {tuple(tensor.shape)}."
            )

        used_keys.add(key)
        weight_values.append(tensor)

    model.set_weights(weight_values)
    return model


def _prepare_artifact_directory(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise ArtifactSpecError(
            "Target artifact directory already exists and is not empty: " f"{path}"
        )
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _normalize_weight_name(name: str) -> str:
    """Normalize generated Keras names that can differ by auto-number suffix."""
    parts = [segment for segment in name.split("/") if segment]
    normalized = []
    for segment in parts:
        normalized.append(re.sub(r"_\d+$", "", segment))
    return "/".join(normalized)


def _choose_entry_for_variable(
    *,
    identifier: str,
    normalized_identifier: str,
    variable_shape: tuple[int, ...],
    entries_by_name: dict[str, dict[str, Any]],
    normalized_entry_map: dict[str, list[dict[str, Any]]],
    used_keys: set[str],
) -> dict[str, Any]:
    """Pick the best model_index entry for a given variable.

    Preference:
        1) Exact identifier match.
        2) Unique normalized identifier + shape match.
        3) Unique normalized suffix + shape match.
        4) Unique shape-only match among remaining entries.
    """
    exact = entries_by_name.get(identifier)
    if exact is not None and exact["key"] not in used_keys:
        if tuple(exact["shape"]) != variable_shape:
            raise ArtifactSpecError(
                "Weight/index integrity check failed; shape mismatch for exact-name "
                f"match '{identifier}'."
            )
        return exact

    candidates = [
        entry
        for entry in normalized_entry_map.get(normalized_identifier, [])
        if entry["key"] not in used_keys and tuple(entry["shape"]) == variable_shape
    ]
    if len(candidates) == 1:
        return candidates[0]

    suffix_candidates = [
        entry
        for entry in entries_by_name.values()
        if entry["key"] not in used_keys
        and tuple(entry["shape"]) == variable_shape
        and _matches_normalized_suffix(entry["name"], normalized_identifier)
    ]
    if len(suffix_candidates) == 1:
        return suffix_candidates[0]

    remaining_shape_matches = [
        entry
        for entry in entries_by_name.values()
        if entry["key"] not in used_keys and tuple(entry["shape"]) == variable_shape
    ]
    if len(remaining_shape_matches) == 1:
        return remaining_shape_matches[0]

    raise ArtifactSpecError(
        "Weight/index integrity check failed; could not uniquely map model "
        f"weight '{identifier}' with shape {variable_shape} to a model_index entry."
    )


def _matches_normalized_suffix(entry_name: str, normalized_identifier: str) -> bool:
    """Return true if normalized entry path ends with normalized identifier path."""
    entry_parts = _normalize_weight_name(entry_name).split("/")
    identifier_parts = normalized_identifier.split("/")
    if len(identifier_parts) > len(entry_parts):
        return False
    return entry_parts[-len(identifier_parts) :] == identifier_parts
