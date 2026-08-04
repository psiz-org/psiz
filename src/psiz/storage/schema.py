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
"""Schema utilities for PsiZ .psiz artifact directories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from psiz.backend import PSIZ_DEFAULT_BACKEND
from psiz.backend import resolve_backend
from psiz.backend import validate_backend_name

ARTIFACT_TYPE = "psiz_model"
FORMAT_NAME = "psiz"
FORMAT_VERSION = "1.0.0"
SUPPORTED_FORMAT_MAJOR_VERSION = 1

REQUIRED_ARTIFACT_FILES = [
    "README.md",
    "config.json",
    "model.safetensors",
    "model_index.json",
    "metadata.json",
    "LICENSE",
]


class ArtifactSpecError(ValueError):
    """Raised when an artifact directory violates the PsiZ storage contract."""


def validate_artifact_directory(
    path: str | Path,
    backend_override: str | None = None,
    default_backend: str = PSIZ_DEFAULT_BACKEND,
) -> dict[str, Any]:
    """Validate a PsiZ artifact directory against the v1.0.0 contract."""
    artifact_dir = Path(path)
    if not artifact_dir.exists() or not artifact_dir.is_dir():
        raise ArtifactSpecError(f"Artifact directory does not exist: {artifact_dir}")

    missing_files = [
        name for name in REQUIRED_ARTIFACT_FILES if not (artifact_dir / name).exists()
    ]
    if missing_files:
        raise ArtifactSpecError("Missing required files: " + ", ".join(missing_files))

    config = _load_json_file(artifact_dir / "config.json")
    metadata = _load_json_file(artifact_dir / "metadata.json")
    model_index = _load_json_file(artifact_dir / "model_index.json")

    _validate_config(config)
    _validate_metadata(metadata)
    _validate_model_index(model_index)

    model_config_compaction = config.get("model_config_compaction")
    if isinstance(model_config_compaction, dict):
        blob_file = model_config_compaction["blob_file"]
        if not (artifact_dir / blob_file).exists():
            raise ArtifactSpecError(
                "Missing required model_config compaction blob file: " f"{blob_file}"
            )

    try:
        resolved_backend = resolve_backend(
            backend_override=backend_override,
            default_backend=default_backend,
        )
    except ValueError as exc:
        raise ArtifactSpecError(str(exc)) from exc

    return {
        "artifact_dir": str(artifact_dir),
        "required_files": list(REQUIRED_ARTIFACT_FILES),
        "config": config,
        "metadata": metadata,
        "model_index": model_index,
        "resolved_backend": resolved_backend,
    }


def validate_model_index_weight_integrity(
    model_index: dict[str, Any], tensor_keys: set[str]
) -> None:
    """Validate that model index keys exactly match keys stored in weights file."""
    index_keys = {entry["key"] for entry in model_index["weights"]}
    missing = sorted(index_keys - tensor_keys)
    unexpected = sorted(tensor_keys - index_keys)

    if missing or unexpected:
        details = []
        if missing:
            details.append("missing keys: " + ", ".join(missing))
        if unexpected:
            details.append("unexpected keys: " + ", ".join(unexpected))
        raise ArtifactSpecError(
            "Weight/index integrity check failed; " + "; ".join(details)
        )


def _validate_metadata(metadata: dict[str, Any]) -> None:
    if metadata.get("artifact_type") != ARTIFACT_TYPE:
        raise ArtifactSpecError("metadata.artifact_type must be 'psiz_model'.")
    if metadata.get("format_name") != FORMAT_NAME:
        raise ArtifactSpecError("metadata.format_name must be 'psiz'.")

    _validate_format_version(metadata.get("format_version"), "metadata.format_version")

    backend = metadata.get("backend")
    if not isinstance(backend, str) or not backend.strip():
        raise ArtifactSpecError("metadata.backend must be a non-empty string.")
    try:
        validate_backend_name(backend, argument_name="metadata.backend")
    except ValueError as exc:
        raise ArtifactSpecError(str(exc)) from exc

    architecture = metadata.get("architecture")
    _validate_architecture(architecture, "metadata.architecture")

    license_payload = metadata.get("license")
    _validate_license(license_payload, "metadata.license")


def _validate_config(config: dict[str, Any]) -> None:
    if not isinstance(config, dict):
        raise ArtifactSpecError("config.json must contain a JSON object.")

    if config.get("artifact_type") != ARTIFACT_TYPE:
        raise ArtifactSpecError("config.artifact_type must be 'psiz_model'.")
    if config.get("format_name") != FORMAT_NAME:
        raise ArtifactSpecError("config.format_name must be 'psiz'.")

    _validate_format_version(config.get("format_version"), "config.format_version")

    backend = config.get("backend")
    if not isinstance(backend, str) or not backend.strip():
        raise ArtifactSpecError("config.backend must be a non-empty string.")
    try:
        validate_backend_name(backend, argument_name="config.backend")
    except ValueError as exc:
        raise ArtifactSpecError(str(exc)) from exc

    architecture = config.get("architecture")
    _validate_architecture(architecture, "config.architecture")

    license_payload = config.get("license")
    _validate_license(license_payload, "config.license")

    model_config = config.get("model_config")
    if not isinstance(model_config, dict):
        raise ArtifactSpecError("config.model_config must be an object.")

    model_config_compaction = config.get("model_config_compaction")
    if model_config_compaction is not None:
        _validate_model_config_compaction(model_config_compaction)


def _validate_model_config_compaction(compaction: Any) -> None:
    if not isinstance(compaction, dict):
        raise ArtifactSpecError("config.model_config_compaction must be an object.")

    blob_file = compaction.get("blob_file")
    if not isinstance(blob_file, str) or not blob_file.strip():
        raise ArtifactSpecError(
            "config.model_config_compaction.blob_file must be a non-empty string."
        )

    blob_count = compaction.get("blob_count")
    if not isinstance(blob_count, int) or blob_count <= 0:
        raise ArtifactSpecError(
            "config.model_config_compaction.blob_count must be a positive integer."
        )

    marker_schema_version = compaction.get("marker_schema_version")
    if marker_schema_version != 1:
        raise ArtifactSpecError(
            "config.model_config_compaction.marker_schema_version must be 1."
        )

    min_externalized_bytes = compaction.get("min_externalized_bytes")
    if not isinstance(min_externalized_bytes, int) or min_externalized_bytes <= 0:
        raise ArtifactSpecError(
            "config.model_config_compaction.min_externalized_bytes must be a positive integer."
        )

    externalized_tensor_bytes = compaction.get("externalized_tensor_bytes")
    if not isinstance(externalized_tensor_bytes, int) or externalized_tensor_bytes <= 0:
        raise ArtifactSpecError(
            "config.model_config_compaction.externalized_tensor_bytes must be a positive integer."
        )

    externalized_json_estimate_bytes = compaction.get("externalized_json_estimate_bytes")
    if (
        not isinstance(externalized_json_estimate_bytes, int)
        or externalized_json_estimate_bytes <= 0
    ):
        raise ArtifactSpecError(
            "config.model_config_compaction.externalized_json_estimate_bytes must be a positive integer."
        )


def _validate_model_index(model_index: dict[str, Any]) -> None:
    if not isinstance(model_index, dict):
        raise ArtifactSpecError("model_index.json must contain a JSON object.")

    weight_file = model_index.get("weight_file")
    if weight_file != "model.safetensors":
        raise ArtifactSpecError("model_index.weight_file must be 'model.safetensors'.")

    weights = model_index.get("weights")
    if not isinstance(weights, list):
        raise ArtifactSpecError("model_index.weights must be a list.")

    seen_names: set[str] = set()
    seen_keys: set[str] = set()
    for entry in weights:
        if not isinstance(entry, dict):
            raise ArtifactSpecError("Each weight entry must be an object.")

        name = entry.get("name")
        key = entry.get("key")
        shape = entry.get("shape")
        dtype = entry.get("dtype")

        if not isinstance(name, str) or not name.strip():
            raise ArtifactSpecError("Each weight entry must define a non-empty name.")
        if not isinstance(key, str) or not key.strip():
            raise ArtifactSpecError("Each weight entry must define a non-empty key.")
        if not isinstance(shape, list) or not all(isinstance(v, int) for v in shape):
            raise ArtifactSpecError("Each weight entry must define an integer shape list.")
        if not isinstance(dtype, str) or not dtype.strip():
            raise ArtifactSpecError("Each weight entry must define a non-empty dtype.")

        if name in seen_names:
            raise ArtifactSpecError(f"Duplicate weight name in index: {name}")
        if key in seen_keys:
            raise ArtifactSpecError(f"Duplicate safetensors key in index: {key}")

        seen_names.add(name)
        seen_keys.add(key)


def _validate_format_version(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ArtifactSpecError(f"{field_name} must be a non-empty string.")

    parts = value.split(".")
    if len(parts) != 3:
        raise ArtifactSpecError(f"{field_name} must follow semver x.y.z.")

    try:
        major = int(parts[0])
        int(parts[1])
        int(parts[2])
    except ValueError as exc:
        raise ArtifactSpecError(f"{field_name} must follow semver x.y.z.") from exc

    if major != SUPPORTED_FORMAT_MAJOR_VERSION:
        raise ArtifactSpecError(
            "Unsupported artifact format major version "
            f"{major}; expected {SUPPORTED_FORMAT_MAJOR_VERSION}."
        )


def _validate_architecture(architecture: Any, field_name: str) -> None:
    if not isinstance(architecture, dict):
        raise ArtifactSpecError(f"{field_name} must be an object.")

    class_name = architecture.get("class_name")
    if not isinstance(class_name, str) or not class_name.strip():
        raise ArtifactSpecError(f"{field_name}.class_name must be a non-empty string.")


def _validate_license(license_payload: Any, field_name: str) -> None:
    if not isinstance(license_payload, dict):
        raise ArtifactSpecError(f"{field_name} must be an object.")

    name = license_payload.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ArtifactSpecError(f"{field_name}.name must be a non-empty string.")

    policy = license_payload.get("policy")
    if policy not in {"include", "omit", "custom"}:
        raise ArtifactSpecError(
            f"{field_name}.policy must be one of include/omit/custom."
        )


def _load_json_file(path: Path) -> dict[str, Any]:
    try:
        contents = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ArtifactSpecError(f"Missing required file: {path.name}") from exc

    try:
        parsed = json.loads(contents)
    except json.JSONDecodeError as exc:
        raise ArtifactSpecError(f"Invalid JSON in {path.name}") from exc

    if not isinstance(parsed, dict):
        raise ArtifactSpecError(f"{path.name} must contain a JSON object.")
    return parsed
