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
"""Utilities for validating PsiZ .psiz artifact directories.

The implementation intentionally stays lightweight and dependency-free so it can
be used as a stable contract for future save/load work.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ArtifactSpecError(ValueError):
    """Raised when an artifact directory violates the PsiZ v1.0.0 spec."""


REQUIRED_ARTIFACT_FILES = [
    "README.md",
    "config.json",
    "model.safetensors",
    "LICENSE",
]

SUPPORTED_FORMAT_MAJOR_VERSION = 1


def validate_artifact_directory(path: str | Path) -> dict[str, Any]:
    """Validate a PsiZ artifact directory against the v1.0.0 contract.

    Parameters
    ----------
    path:
        Path to the artifact directory.

    Returns
    -------
    dict
        A manifest containing the validated metadata and parsed JSON content.
    """

    artifact_dir = Path(path)
    if not artifact_dir.exists() or not artifact_dir.is_dir():
        raise ArtifactSpecError(f"Artifact directory does not exist: {artifact_dir}")

    missing_files = [name for name in REQUIRED_ARTIFACT_FILES if not (artifact_dir / name).exists()]
    if missing_files:
        raise ArtifactSpecError(
            "Missing required files: " + ", ".join(missing_files)
        )

    config = _load_json_file(artifact_dir / "config.json")
    metadata = _load_optional_json_file(artifact_dir / "metadata.json")
    model_index = _load_optional_json_file(artifact_dir / "model_index.json")

    _validate_config(config)
    if metadata is not None:
        _validate_metadata(metadata)
    if model_index is not None:
        _validate_model_index(model_index)

    return {
        "artifact_dir": str(artifact_dir),
        "required_files": REQUIRED_ARTIFACT_FILES,
        "config": config,
        "metadata": metadata,
        "model_index": model_index,
    }


def _validate_metadata(metadata: dict[str, Any]) -> None:
    if metadata.get("artifact_type") != "psiz_model":
        raise ArtifactSpecError("metadata.artifact_type must be 'psiz_model'.")
    if metadata.get("format_name") != "psiz":
        raise ArtifactSpecError("metadata.format_name must be 'psiz'.")

    version = metadata.get("format_version")
    if not isinstance(version, str):
        raise ArtifactSpecError("metadata.format_version must be a string.")

    try:
        major, minor, patch = (int(part) for part in version.split("."))
    except ValueError as exc:
        raise ArtifactSpecError("metadata.format_version must follow semver x.y.z.") from exc

    if major != SUPPORTED_FORMAT_MAJOR_VERSION:
        raise ArtifactSpecError(
            f"Unsupported artifact format major version {major}; expected {SUPPORTED_FORMAT_MAJOR_VERSION}."
        )

    if not isinstance(metadata.get("backend"), str) or not metadata["backend"].strip():
        raise ArtifactSpecError("metadata.backend must be a non-empty string.")

    architecture = metadata.get("architecture")
    if not isinstance(architecture, dict):
        raise ArtifactSpecError("metadata.architecture must be an object.")
    if not isinstance(architecture.get("class_name"), str) or not architecture["class_name"].strip():
        raise ArtifactSpecError("metadata.architecture.class_name must be a non-empty string.")

    license = metadata.get("license")
    if not isinstance(license, dict):
        raise ArtifactSpecError("metadata.license must be an object.")
    if not isinstance(license.get("name"), str) or not license["name"].strip():
        raise ArtifactSpecError("metadata.license.name must be a non-empty string.")
    if license.get("policy") not in {"include", "omit", "custom"}:
        raise ArtifactSpecError("metadata.license.policy must be one of include/omit/custom.")


def _validate_config(config: dict[str, Any]) -> None:
    if not isinstance(config, dict):
        raise ArtifactSpecError("config.json must contain a JSON object.")
    if not isinstance(config.get("artifact_type"), str) or not config["artifact_type"].strip():
        raise ArtifactSpecError("config.artifact_type must be a non-empty string.")
    if not isinstance(config.get("format_name"), str) or not config["format_name"].strip():
        raise ArtifactSpecError("config.format_name must be a non-empty string.")
    version = config.get("format_version")
    if not isinstance(version, str) or not version.strip():
        raise ArtifactSpecError("config.format_version must be a non-empty string.")
    try:
        major, minor, patch = (int(part) for part in version.split("."))
    except ValueError as exc:
        raise ArtifactSpecError("config.format_version must follow semver x.y.z.") from exc
    if major != SUPPORTED_FORMAT_MAJOR_VERSION:
        raise ArtifactSpecError(
            f"Unsupported artifact format major version {major}; expected {SUPPORTED_FORMAT_MAJOR_VERSION}."
        )
    if not isinstance(config.get("backend"), str) or not config["backend"].strip():
        raise ArtifactSpecError("config.backend must be a non-empty string.")
    architecture = config.get("architecture")
    if not isinstance(architecture, dict):
        raise ArtifactSpecError("config.architecture must be an object.")
    if not isinstance(architecture.get("class_name"), str) or not architecture["class_name"].strip():
        raise ArtifactSpecError("config.architecture.class_name must be a non-empty string.")
    license = config.get("license")
    if not isinstance(license, dict):
        raise ArtifactSpecError("config.license must be an object.")
    if not isinstance(license.get("name"), str) or not license["name"].strip():
        raise ArtifactSpecError("config.license.name must be a non-empty string.")
    if license.get("policy") not in {"include", "omit", "custom"}:
        raise ArtifactSpecError("config.license.policy must be one of include/omit/custom.")


def _validate_model_index(model_index: dict[str, Any]) -> None:
    if not isinstance(model_index, dict):
        raise ArtifactSpecError("model_index.json must contain a JSON object.")

    weights = model_index.get("weights")
    if not isinstance(weights, list):
        raise ArtifactSpecError("model_index.weights must be a list.")
    for entry in weights:
        if not isinstance(entry, dict):
            raise ArtifactSpecError("Each weight entry must be an object.")
        if not isinstance(entry.get("name"), str) or not entry["name"].strip():
            raise ArtifactSpecError("Each weight entry must define a non-empty name.")
        if not isinstance(entry.get("shape"), list):
            raise ArtifactSpecError("Each weight entry must define a shape list.")



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


def _load_optional_json_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return _load_json_file(path)
