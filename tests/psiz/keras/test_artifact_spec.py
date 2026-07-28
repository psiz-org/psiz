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
    architecture = {"class_name": "ExampleModel", "module": "example"}
    license_info = {"name": "Apache-2.0", "policy": "include"}
    _write_json(
        path / "config.json",
        {
            "artifact_type": "psiz_model",
            "format_name": "psiz",
            "format_version": "1.0.0",
            "backend": "torch",
            "architecture": architecture,
            "license": license_info,
            "model_config": {
                "module": "keras.src.models.functional",
                "class_name": "Functional",
                "config": {},
                "registered_name": "Functional",
            },
        },
    )
    _write_json(
        path / "metadata.json",
        {
            "artifact_type": "psiz_model",
            "format_name": "psiz",
            "format_version": "1.0.0",
            "backend": "torch",
            "architecture": architecture,
            "license": license_info,
            "storage": {
                "weight_format": "safetensors",
                "weight_file": "model.safetensors",
                "weight_count": 1,
            },
        },
    )
    _write_json(
        path / "model_index.json",
        {
            "weight_format": "safetensors",
            "weight_file": "model.safetensors",
            "weights": [{"name": "model/dense/kernel", "key": "weight_00000", "shape": [2, 2], "dtype": "float32"}],
        },
    )
    (path / "README.md").write_text("# Example\n", encoding="utf-8")
    (path / "LICENSE").write_text("Apache-2.0\n", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"FAKE")


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
        "model_index.json",
        "metadata.json",
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
            "model_config": {
                "module": "keras.src.models.functional",
                "class_name": "Functional",
                "config": {},
                "registered_name": "Functional",
            },
        },
    )
    _write_json(
        artifact_dir / "metadata.json",
        {
            "artifact_type": "psiz_model",
            "format_name": "psiz",
            "format_version": "1.0.0",
            "backend": "torch",
            "architecture": {"class_name": "HierarchicalVIModel", "module": "example"},
            "license": {"name": "Apache-2.0", "policy": "include"},
            "storage": {
                "weight_format": "safetensors",
                "weight_file": "model.safetensors",
                "weight_count": 6,
            },
        },
    )
    _write_json(
        artifact_dir / "model_index.json",
        {
            "model_type": "hierarchical_vi",
            "weight_file": "model.safetensors",
            "weights": [
                {"name": "global/loc", "key": "weight_00000", "shape": [2], "dtype": "float32"},
                {"name": "global/scale", "key": "weight_00001", "shape": [2], "dtype": "float32"},
                {"name": "intermediate/loc", "key": "weight_00002", "shape": [4, 2], "dtype": "float32"},
                {"name": "intermediate/scale", "key": "weight_00003", "shape": [4, 2], "dtype": "float32"},
                {"name": "leaf/loc", "key": "weight_00004", "shape": [8, 2], "dtype": "float32"},
                {"name": "leaf/scale", "key": "weight_00005", "shape": [8, 2], "dtype": "float32"},
            ],
        },
    )
    (artifact_dir / "README.md").write_text("# Example\n", encoding="utf-8")
    (artifact_dir / "LICENSE").write_text("Apache-2.0\n", encoding="utf-8")
    (artifact_dir / "model.safetensors").write_bytes(b"FAKE")

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
