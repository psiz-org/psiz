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
"""Schema utilities for PsiZ dataset artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

DATASET_FORMAT = "psiz-dataset"
DATASET_FORMAT_VERSION = "1.0.0"
SUPPORTED_DATASET_FORMAT_MAJOR_VERSION = 1

REQUIRED_TOP_LEVEL_KEYS = {
    "format",
    "format_version",
    "dataset_id",
    "created_at",
    "license",
    "tables",
    "split_config",
    "runtime_contract",
    "semantic_contract",
}

REQUIRED_TABLE_KEYS = {
    "name",
    "path",
    "kind",
    "primary_key",
    "columns",
    "sha256",
    "row_count",
}

REQUIRED_SPLIT_CONFIG_KEYS = {
    "split_assignment_table",
    "active_split_set_id",
    "allowed_split_labels",
}

REQUIRED_RUNTIME_KEYS = {
    "observation_table",
    "x_features",
    "y_features",
    "w_features",
    "batch_axis",
    "timestep",
}

REQUIRED_TIMESTEP_KEYS = {"mode", "sequence_id_column", "timestep_index_column"}

REQUIRED_SEMANTIC_CONTRACT_KEYS = {
    "schema_version",
    "dataset_class",
    "components",
    "load_policy",
}

REQUIRED_LOAD_POLICY_KEYS = {
    "require_semantic_contract",
    "allow_runtime_fallback",
}


class DatasetArtifactSpecError(ValueError):
    """Raised when dataset artifact schema rules are violated."""


def compute_file_sha256(path: str | Path) -> str:
    """Compute SHA256 hash for file contents."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fp:
        while True:
            chunk = fp.read(8192)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load and parse manifest JSON file."""
    manifest_path = Path(path)
    payload = manifest_path.read_text(encoding="utf-8")
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise DatasetArtifactSpecError("manifest.json must contain a JSON object.")
    return data


def validate_manifest_schema(manifest: dict[str, Any]) -> None:
    """Validate required v1 manifest keys and selected field contracts."""
    _require_keys(manifest, REQUIRED_TOP_LEVEL_KEYS, "manifest")

    if manifest["format"] != DATASET_FORMAT:
        raise DatasetArtifactSpecError(
            f"manifest.format must be '{DATASET_FORMAT}'."
        )

    _validate_format_version(manifest["format_version"], "manifest.format_version")

    _validate_non_empty_str(manifest["dataset_id"], "manifest.dataset_id")
    _validate_non_empty_str(manifest["created_at"], "manifest.created_at")
    _validate_non_empty_str(manifest["license"], "manifest.license")

    tables = manifest["tables"]
    if not isinstance(tables, list) or len(tables) == 0:
        raise DatasetArtifactSpecError("manifest.tables must be a non-empty list.")

    table_names = set()
    for idx, table in enumerate(tables):
        location = f"manifest.tables[{idx}]"
        if not isinstance(table, dict):
            raise DatasetArtifactSpecError(f"{location} must be an object.")
        _require_keys(table, REQUIRED_TABLE_KEYS, location)
        _validate_non_empty_str(table["name"], f"{location}.name")
        _validate_non_empty_str(table["path"], f"{location}.path")
        _validate_non_empty_str(table["kind"], f"{location}.kind")

        if table["name"] in table_names:
            raise DatasetArtifactSpecError(
                f"Duplicate table name in manifest: {table['name']}"
            )
        table_names.add(table["name"])

        if table["kind"] not in {"fact", "dimension", "bridge"}:
            raise DatasetArtifactSpecError(
                f"{location}.kind must be one of fact/dimension/bridge."
            )

        if not isinstance(table["primary_key"], list) or len(table["primary_key"]) == 0:
            raise DatasetArtifactSpecError(
                f"{location}.primary_key must be a non-empty list."
            )
        if not isinstance(table["columns"], list) or len(table["columns"]) == 0:
            raise DatasetArtifactSpecError(
                f"{location}.columns must be a non-empty list."
            )
        if not isinstance(table["row_count"], int) or table["row_count"] < 0:
            raise DatasetArtifactSpecError(
                f"{location}.row_count must be a non-negative integer."
            )
        _validate_non_empty_str(table["sha256"], f"{location}.sha256")

    split_config = manifest["split_config"]
    if not isinstance(split_config, dict):
        raise DatasetArtifactSpecError("manifest.split_config must be an object.")
    _require_keys(split_config, REQUIRED_SPLIT_CONFIG_KEYS, "manifest.split_config")
    _validate_non_empty_str(
        split_config["split_assignment_table"],
        "manifest.split_config.split_assignment_table",
    )
    _validate_non_empty_str(
        split_config["active_split_set_id"],
        "manifest.split_config.active_split_set_id",
    )

    labels = split_config["allowed_split_labels"]
    if not isinstance(labels, list) or len(labels) == 0:
        raise DatasetArtifactSpecError(
            "manifest.split_config.allowed_split_labels must be a non-empty list."
        )
    for label in labels:
        _validate_non_empty_str(label, "manifest.split_config.allowed_split_labels[]")

    runtime = manifest["runtime_contract"]
    if not isinstance(runtime, dict):
        raise DatasetArtifactSpecError("manifest.runtime_contract must be an object.")
    _require_keys(runtime, REQUIRED_RUNTIME_KEYS, "manifest.runtime_contract")

    _validate_non_empty_str(
        runtime["observation_table"], "manifest.runtime_contract.observation_table"
    )
    if not isinstance(runtime["batch_axis"], int):
        raise DatasetArtifactSpecError(
            "manifest.runtime_contract.batch_axis must be an integer."
        )

    for key in ["x_features", "y_features", "w_features"]:
        mapping = runtime[key]
        if not isinstance(mapping, dict):
            raise DatasetArtifactSpecError(
                f"manifest.runtime_contract.{key} must be an object."
            )

    timestep = runtime["timestep"]
    if not isinstance(timestep, dict):
        raise DatasetArtifactSpecError(
            "manifest.runtime_contract.timestep must be an object."
        )
    _require_keys(timestep, REQUIRED_TIMESTEP_KEYS, "manifest.runtime_contract.timestep")
    if timestep["mode"] not in {"with_timestep", "without_timestep", "either"}:
        raise DatasetArtifactSpecError(
            "manifest.runtime_contract.timestep.mode must be one of "
            "with_timestep/without_timestep/either."
        )
    _validate_non_empty_str(
        timestep["sequence_id_column"],
        "manifest.runtime_contract.timestep.sequence_id_column",
    )
    _validate_non_empty_str(
        timestep["timestep_index_column"],
        "manifest.runtime_contract.timestep.timestep_index_column",
    )

    semantic_contract = manifest["semantic_contract"]
    if not isinstance(semantic_contract, dict):
        raise DatasetArtifactSpecError("manifest.semantic_contract must be an object.")
    _require_keys(
        semantic_contract,
        REQUIRED_SEMANTIC_CONTRACT_KEYS,
        "manifest.semantic_contract",
    )
    _validate_non_empty_str(
        semantic_contract["schema_version"],
        "manifest.semantic_contract.schema_version",
    )
    _validate_non_empty_str(
        semantic_contract["dataset_class"],
        "manifest.semantic_contract.dataset_class",
    )
    if not isinstance(semantic_contract["components"], list):
        raise DatasetArtifactSpecError(
            "manifest.semantic_contract.components must be a list."
        )

    load_policy = semantic_contract["load_policy"]
    if not isinstance(load_policy, dict):
        raise DatasetArtifactSpecError(
            "manifest.semantic_contract.load_policy must be an object."
        )
    _require_keys(
        load_policy,
        REQUIRED_LOAD_POLICY_KEYS,
        "manifest.semantic_contract.load_policy",
    )
    if load_policy["require_semantic_contract"] is not True:
        raise DatasetArtifactSpecError(
            "manifest.semantic_contract.load_policy.require_semantic_contract "
            "must be true."
        )
    if load_policy["allow_runtime_fallback"] is not False:
        raise DatasetArtifactSpecError(
            "manifest.semantic_contract.load_policy.allow_runtime_fallback "
            "must be false."
        )


def validate_dataset_artifact_directory(path: str | Path) -> dict[str, Any]:
    """Validate a dataset artifact directory and file integrity."""
    artifact_dir = Path(path)
    if not artifact_dir.exists() or not artifact_dir.is_dir():
        raise DatasetArtifactSpecError(
            f"Dataset artifact directory does not exist: {artifact_dir}"
        )

    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.exists():
        raise DatasetArtifactSpecError("Missing required file: manifest.json")

    manifest = load_manifest(manifest_path)
    validate_manifest_schema(manifest)

    table_by_name = {entry["name"]: entry for entry in manifest["tables"]}
    _validate_split_contract(manifest, table_by_name)

    for table in manifest["tables"]:
        table_path = artifact_dir / table["path"]
        if not table_path.exists():
            raise DatasetArtifactSpecError(
                f"Missing declared table file: {table['path']}"
            )

        actual_sha256 = compute_file_sha256(table_path)
        if actual_sha256 != table["sha256"]:
            raise DatasetArtifactSpecError(
                f"SHA256 mismatch for table {table['name']}"
            )

        frame = pd.read_parquet(table_path)
        if len(frame) != table["row_count"]:
            raise DatasetArtifactSpecError(
                f"Row count mismatch for table {table['name']}"
            )

        declared_cols = {c["name"] for c in table["columns"]}
        missing_cols = sorted(declared_cols - set(frame.columns))
        if missing_cols:
            raise DatasetArtifactSpecError(
                f"Table {table['name']} is missing declared columns: "
                + ", ".join(missing_cols)
            )

        _validate_primary_key(frame, table)

    return {
        "artifact_dir": str(artifact_dir),
        "manifest": manifest,
    }


def _validate_split_contract(
    manifest: dict[str, Any], table_by_name: dict[str, dict[str, Any]]
) -> None:
    split_config = manifest["split_config"]
    split_table_name = split_config["split_assignment_table"]
    if split_table_name not in table_by_name:
        raise DatasetArtifactSpecError(
            "split_config.split_assignment_table must reference an existing table."
        )

    split_table = table_by_name[split_table_name]
    split_cols = {c["name"] for c in split_table["columns"]}
    required_cols = {"observation_id", "split", "split_set_id"}
    missing = sorted(required_cols - split_cols)
    if missing:
        raise DatasetArtifactSpecError(
            "split assignment table is missing required columns: "
            + ", ".join(missing)
        )


def _validate_primary_key(frame: pd.DataFrame, table: dict[str, Any]) -> None:
    pk_cols = table["primary_key"]
    if any(col not in frame.columns for col in pk_cols):
        raise DatasetArtifactSpecError(
            f"Primary key columns missing for table {table['name']}"
        )

    if frame[pk_cols].isnull().any().any():
        raise DatasetArtifactSpecError(
            f"Primary key columns contain null values for table {table['name']}"
        )

    if frame.duplicated(subset=pk_cols).any():
        raise DatasetArtifactSpecError(
            f"Primary key rows are not unique for table {table['name']}"
        )


def _validate_non_empty_str(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise DatasetArtifactSpecError(f"{field_name} must be a non-empty string.")


def _validate_format_version(value: Any, field_name: str) -> None:
    _validate_non_empty_str(value, field_name)
    parts = value.split(".")
    if len(parts) != 3:
        raise DatasetArtifactSpecError(f"{field_name} must follow semver x.y.z.")

    try:
        major = int(parts[0])
        int(parts[1])
        int(parts[2])
    except ValueError as exc:
        raise DatasetArtifactSpecError(
            f"{field_name} must follow semver x.y.z."
        ) from exc

    if major != SUPPORTED_DATASET_FORMAT_MAJOR_VERSION:
        raise DatasetArtifactSpecError(
            "Unsupported dataset format major version "
            f"{major}; expected {SUPPORTED_DATASET_FORMAT_MAJOR_VERSION}."
        )


def _require_keys(payload: dict[str, Any], required: set[str], location: str) -> None:
    missing = sorted(required - set(payload.keys()))
    if missing:
        raise DatasetArtifactSpecError(
            f"{location} is missing required keys: {', '.join(missing)}"
        )
