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
"""Tests for the PsiZ .psiz artifact specification helpers."""

import json
from pathlib import Path

import pytest

from psiz.keras.artifact_spec import ArtifactSpecError
from psiz.keras.artifact_spec import validate_artifact_directory


def _write_json(path: Path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_minimal_artifact(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    _write_json(
        path / "config.json",
        {
            "artifact_type": "psiz_model",
            "format_name": "psiz",
            "format_version": "1.0.0",
            "backend": "torch",
            "architecture": {"class_name": "ExampleModel", "module": "example"},
            "license": {"name": "Apache-2.0", "policy": "include"},
        },
    )
    (path / "README.md").write_text("# Example\n", encoding="utf-8")
    (path / "LICENSE").write_text("Apache-2.0\n", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"placeholder")


def test_validate_minimal_artifact_directory(tmp_path):
    artifact_dir = tmp_path / "minimal-model.psiz"
    _write_minimal_artifact(artifact_dir)

    manifest = validate_artifact_directory(artifact_dir, backend_override="jax")

    assert manifest["config"]["format_version"] == "1.0.0"
    assert manifest["resolved_backend"] == "jax"
    assert manifest["required_files"] == [
        "README.md",
        "config.json",
        "model.safetensors",
        "LICENSE",
    ]


def test_validate_hierarchical_artifact_directory(tmp_path):
    artifact_dir = tmp_path / "hierarchical-model.psiz"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        artifact_dir / "config.json",
        {
            "artifact_type": "psiz_model",
            "format_name": "psiz",
            "format_version": "1.0.0",
            "backend": "torch",
            "architecture": {"class_name": "HierarchicalVIModel", "module": "example"},
            "license": {"name": "Apache-2.0", "policy": "include"},
        },
    )
    _write_json(
        artifact_dir / "model_index.json",
        {
            "model_type": "hierarchical_vi",
            "weights": [
                {"name": "global/loc", "shape": [2]},
                {"name": "global/scale", "shape": [2]},
                {"name": "intermediate/loc", "shape": [4, 2]},
                {"name": "intermediate/scale", "shape": [4, 2]},
                {"name": "leaf/loc", "shape": [8, 2]},
                {"name": "leaf/scale", "shape": [8, 2]},
            ],
        },
    )
    (artifact_dir / "README.md").write_text("# Example\n", encoding="utf-8")
    (artifact_dir / "LICENSE").write_text("Apache-2.0\n", encoding="utf-8")
    (artifact_dir / "model.safetensors").write_bytes(b"placeholder")

    manifest = validate_artifact_directory(artifact_dir)

    assert [entry["name"] for entry in manifest["model_index"]["weights"]] == [
        "global/loc",
        "global/scale",
        "intermediate/loc",
        "intermediate/scale",
        "leaf/loc",
        "leaf/scale",
    ]


def test_validate_artifact_directory_rejects_missing_files_and_bad_versions(tmp_path):
    artifact_dir = tmp_path / "broken-model.psiz"
    _write_minimal_artifact(artifact_dir)
    (artifact_dir / "LICENSE").unlink()

    with pytest.raises(ArtifactSpecError, match="Missing required files"):
        validate_artifact_directory(artifact_dir)

    _write_minimal_artifact(artifact_dir)
    config = json.loads((artifact_dir / "config.json").read_text(encoding="utf-8"))
    config["format_version"] = "2.0.0"
    _write_json(artifact_dir / "config.json", config)

    with pytest.raises(ArtifactSpecError, match="major version"):
        validate_artifact_directory(artifact_dir)
