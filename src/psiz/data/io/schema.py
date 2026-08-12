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
DATASET_DEFAULT_LICENSE = "cc-by-4.0"

REQUIRED_TOP_LEVEL_KEYS = {
    "format",
    "format_version",
    "dataset_id",
    "dataset_version",
    "created_at",
    "license",
    "tables",
    "split_config",
    "runtime_contract",
    "semantic_contract",
}

REQUIRED_SOURCE_KEYS = {
    "id",
    "role",
    "official_name",
    "version",
    "split",
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

STIMULI_TABLE_NAME = "stimuli"
OBSERVATION_STIMULI_TABLE_NAME = "observation_stimuli"
OBSERVATIONS_TABLE_NAME = "observations"
PARTICIPANTS_TABLE_NAME = "participants"

RUNTIME_OPTIONAL_STIMULI_TABLE_KEY = "stimuli_table"
RUNTIME_OPTIONAL_STIMULUS_FEATURES_KEY = "stimulus_id_features"

MANIFEST_KEY_ORDER = (
    "dataset_id",
    "dataset_version",
    "description",
    "created_at",
    "license",
    "format",
    "format_version",
    "sources",
    "runtime_contract",
    "semantic_contract",
    "tables",
    "split_config",
)


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


def order_manifest_keys(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of `manifest` with keys in human-friendly display order."""
    ordered = {key: manifest[key] for key in MANIFEST_KEY_ORDER if key in manifest}
    ordered.update(
        (key, value) for key, value in manifest.items() if key not in ordered
    )
    return ordered


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
    _validate_non_empty_str(manifest["dataset_version"], "manifest.dataset_version")
    if "description" in manifest:
        _validate_non_empty_str(manifest["description"], "manifest.description")
    _validate_non_empty_str(manifest["created_at"], "manifest.created_at")
    _validate_non_empty_str(manifest["license"], "manifest.license")

    tables = manifest["tables"]
    if not isinstance(tables, list) or len(tables) == 0:
        raise DatasetArtifactSpecError("manifest.tables must be a non-empty list.")

    table_names = set()
    table_by_name: dict[str, dict[str, Any]] = {}
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
        table_by_name[table["name"]] = table

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

    _validate_optional_stimuli_tables_schema(table_by_name)
    _validate_optional_participants_table_schema(table_by_name)

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

    _validate_optional_runtime_contract(runtime, table_by_name)

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

    _validate_optional_sources_array(manifest)





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
    frame_by_name: dict[str, pd.DataFrame] = {}

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
        frame_by_name[table["name"]] = frame
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

    _validate_optional_stimuli_tables_data(manifest, table_by_name, frame_by_name)
    _validate_optional_participants_table_data(frame_by_name)

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


def _validate_optional_sources_array(manifest: dict[str, Any]) -> None:
    if "sources" not in manifest:
        return

    sources = manifest["sources"]
    if not isinstance(sources, list) or len(sources) == 0:
        raise DatasetArtifactSpecError(
            "manifest.sources must be a non-empty array."
        )
    
    for idx, source in enumerate(sources):
        location = f"manifest.sources[{idx}]"
        if not isinstance(source, dict):
            raise DatasetArtifactSpecError(f"{location} must be an object.")
        _require_keys(source, REQUIRED_SOURCE_KEYS, location)
        for key in REQUIRED_SOURCE_KEYS:
            _validate_non_empty_str(source[key], f"{location}.{key}")


def _validate_optional_stimuli_tables_schema(
    table_by_name: dict[str, dict[str, Any]]
) -> None:
    stimuli_table = table_by_name.get(STIMULI_TABLE_NAME)
    bridge_table = table_by_name.get(OBSERVATION_STIMULI_TABLE_NAME)

    if stimuli_table is not None:
        if stimuli_table["kind"] != "dimension":
            raise DatasetArtifactSpecError(
                "manifest table 'stimuli' must have kind='dimension'."
            )
        if stimuli_table["primary_key"] != ["stimulus_id"]:
            raise DatasetArtifactSpecError(
                "manifest table 'stimuli' must use primary_key=['stimulus_id']."
            )
        _validate_declared_column_requirements(
            stimuli_table,
            {
                "stimulus_id": {"nullable": False, "dtype": "int"},
                "filepath": {"nullable": False, "dtype": "string"},
            },
            "manifest.tables[stimuli]",
        )

    if bridge_table is not None:
        if bridge_table["kind"] != "bridge":
            raise DatasetArtifactSpecError(
                "manifest table 'observation_stimuli' must have kind='bridge'."
            )
        expected_pk = ["observation_id", "x_feature_name", "position"]
        if bridge_table["primary_key"] != expected_pk:
            raise DatasetArtifactSpecError(
                "manifest table 'observation_stimuli' must use primary_key="
                "['observation_id', 'x_feature_name', 'position']."
            )
        _validate_declared_column_requirements(
            bridge_table,
            {
                "observation_id": {"nullable": False, "dtype": "int"},
                "x_feature_name": {"nullable": False, "dtype": "string"},
                "position": {"nullable": False, "dtype": "int"},
                "stimulus_id": {"nullable": False, "dtype": "int"},
            },
            "manifest.tables[observation_stimuli]",
        )

        if stimuli_table is None:
            raise DatasetArtifactSpecError(
                "manifest table 'observation_stimuli' requires a 'stimuli' table."
            )


def _validate_optional_participants_table_schema(
    table_by_name: dict[str, dict[str, Any]]
) -> None:
    participants_table = table_by_name.get(PARTICIPANTS_TABLE_NAME)
    if participants_table is None:
        return

    if participants_table["kind"] != "dimension":
        raise DatasetArtifactSpecError(
            "manifest table 'participants' must have kind='dimension'."
        )
    if participants_table["primary_key"] != ["participant_id"]:
        raise DatasetArtifactSpecError(
            "manifest table 'participants' must use primary_key=['participant_id']."
        )
    _validate_declared_column_requirements(
        participants_table,
        {
            "participant_id": {"nullable": False, "dtype": "int64"},
            "n_sequence": {"nullable": False, "dtype": "int64"},
            "n_trial": {"nullable": False, "dtype": "int64"},
        },
        "manifest.tables[participants]",
    )


def _validate_optional_runtime_contract(
    runtime: dict[str, Any], table_by_name: dict[str, dict[str, Any]]
) -> None:
    if RUNTIME_OPTIONAL_STIMULI_TABLE_KEY in runtime:
        stimuli_table_name = runtime[RUNTIME_OPTIONAL_STIMULI_TABLE_KEY]
        _validate_non_empty_str(
            stimuli_table_name,
            f"manifest.runtime_contract.{RUNTIME_OPTIONAL_STIMULI_TABLE_KEY}",
        )
        if stimuli_table_name != STIMULI_TABLE_NAME:
            raise DatasetArtifactSpecError(
                "manifest.runtime_contract.stimuli_table must reference 'stimuli'."
            )
        if stimuli_table_name not in table_by_name:
            raise DatasetArtifactSpecError(
                "manifest.runtime_contract.stimuli_table references a missing table."
            )

    if RUNTIME_OPTIONAL_STIMULUS_FEATURES_KEY in runtime:
        features = runtime[RUNTIME_OPTIONAL_STIMULUS_FEATURES_KEY]
        if not isinstance(features, list):
            raise DatasetArtifactSpecError(
                "manifest.runtime_contract.stimulus_id_features must be a list."
            )
        x_feature_keys = set(runtime["x_features"].keys())
        for idx, feature_name in enumerate(features):
            _validate_non_empty_str(
                feature_name,
                f"manifest.runtime_contract.stimulus_id_features[{idx}]",
            )
            if feature_name not in x_feature_keys:
                raise DatasetArtifactSpecError(
                    "manifest.runtime_contract.stimulus_id_features includes "
                    f"unknown x feature '{feature_name}'."
                )


def _validate_optional_stimuli_tables_data(
    manifest: dict[str, Any],
    table_by_name: dict[str, dict[str, Any]],
    frame_by_name: dict[str, pd.DataFrame],
) -> None:
    runtime = manifest["runtime_contract"]
    stimuli_frame = frame_by_name.get(STIMULI_TABLE_NAME)
    bridge_frame = frame_by_name.get(OBSERVATION_STIMULI_TABLE_NAME)

    if stimuli_frame is not None:
        if stimuli_frame["stimulus_id"].isnull().any():
            raise DatasetArtifactSpecError(
                "Table stimuli has null values in required column 'stimulus_id'."
            )
        if stimuli_frame["filepath"].isnull().any():
            raise DatasetArtifactSpecError(
                "Table stimuli has null values in required column 'filepath'."
            )
        if stimuli_frame.duplicated(subset=["stimulus_id"]).any():
            raise DatasetArtifactSpecError(
                "Table stimuli has duplicate values in primary key column 'stimulus_id'."
            )

        invalid_paths = [
            value
            for value in stimuli_frame["filepath"].tolist()
            if not _is_dataset_relative_path(value)
        ]
        if invalid_paths:
            raise DatasetArtifactSpecError(
                "Table stimuli.filepath must contain non-empty dataset-root-relative "
                "paths."
            )

    if bridge_frame is not None:
        required_cols = ["observation_id", "x_feature_name", "position", "stimulus_id"]
        if bridge_frame[required_cols].isnull().any().any():
            raise DatasetArtifactSpecError(
                "Table observation_stimuli has null values in required columns."
            )

        if stimuli_frame is not None:
            known_stimulus_ids = set(stimuli_frame["stimulus_id"].tolist())
            unknown_stimulus_ids = sorted(
                set(bridge_frame["stimulus_id"].tolist()) - known_stimulus_ids
            )
            if unknown_stimulus_ids:
                raise DatasetArtifactSpecError(
                    "Table observation_stimuli references unknown stimulus_id values."
                )

        observations_frame = frame_by_name.get(OBSERVATIONS_TABLE_NAME)
        if observations_frame is not None:
            known_observation_ids = set(observations_frame["observation_id"].tolist())
            unknown_observation_ids = sorted(
                set(bridge_frame["observation_id"].tolist()) - known_observation_ids
            )
            if unknown_observation_ids:
                raise DatasetArtifactSpecError(
                    "Table observation_stimuli references unknown observation_id values."
                )

        if RUNTIME_OPTIONAL_STIMULUS_FEATURES_KEY in runtime:
            allowed_features = set(runtime[RUNTIME_OPTIONAL_STIMULUS_FEATURES_KEY])
            observed_features = set(bridge_frame["x_feature_name"].tolist())
            invalid_features = sorted(observed_features - allowed_features)
            if invalid_features:
                raise DatasetArtifactSpecError(
                    "Table observation_stimuli references x_feature_name values not "
                    "declared in runtime_contract.stimulus_id_features."
                )


def _validate_optional_participants_table_data(
        frame_by_name: dict[str, pd.DataFrame],
    ) -> None:
        participants_frame = frame_by_name.get(PARTICIPANTS_TABLE_NAME)
        if participants_frame is None:
            return

        participant_ids = participants_frame["participant_id"]
        if not pd.api.types.is_integer_dtype(participant_ids.dtype):
            raise DatasetArtifactSpecError(
                "Table participants.participant_id must use an integer dtype."
            )
        if str(participant_ids.dtype) != "int64":
            raise DatasetArtifactSpecError(
                "Table participants.participant_id must use dtype int64."
            )
        if participant_ids.isnull().any():
            raise DatasetArtifactSpecError(
                "Table participants has null values in required column 'participant_id'."
            )
        if participants_frame.duplicated(subset=["participant_id"]).any():
            raise DatasetArtifactSpecError(
                "Table participants has duplicate values in primary key column "
                "'participant_id'."
            )

        for column in ["n_sequence", "n_trial"]:
            values = participants_frame[column]
            if values.isnull().any():
                raise DatasetArtifactSpecError(
                    f"Table participants has null values in required column '{column}'."
                )
            if not pd.api.types.is_integer_dtype(values.dtype):
                raise DatasetArtifactSpecError(
                    f"Table participants.{column} must use an integer dtype."
                )
            if (values < 0).any():
                raise DatasetArtifactSpecError(
                    f"Table participants.{column} must contain non-negative values."
                )

        observations_frame = frame_by_name.get(OBSERVATIONS_TABLE_NAME)
        if observations_frame is None or "participant_id" not in observations_frame:
            raise DatasetArtifactSpecError(
                "Table participants requires observations.participant_id."
            )

        known_participant_ids = set(participant_ids.tolist())
        observation_participant_ids = set(
            observations_frame["participant_id"].dropna().tolist()
        )
        unknown_participant_ids = sorted(observation_participant_ids - known_participant_ids)
        if unknown_participant_ids:
            raise DatasetArtifactSpecError(
                "Table observations references unknown participant_id values."
            )

        if "sequence_id" not in observations_frame:
            raise DatasetArtifactSpecError(
                "Table participants requires observations.sequence_id to validate n_sequence."
            )
        if "timestep_index" not in observations_frame:
            raise DatasetArtifactSpecError(
                "Table participants requires observations.timestep_index to validate n_trial."
            )

        unique_trials = (
            observations_frame[["participant_id", "sequence_id", "timestep_index"]]
            .drop_duplicates()
            .groupby("participant_id", dropna=False)
            .size()
        )

        expected = observations_frame.groupby("participant_id", dropna=False).agg(
            n_sequence=("sequence_id", "nunique"),
        )
        expected["n_trial"] = unique_trials
        actual = participants_frame.set_index("participant_id")[["n_sequence", "n_trial"]]
        expected = expected.reindex(actual.index, fill_value=0)
        if not actual[["n_sequence", "n_trial"]].equals(
            expected[["n_sequence", "n_trial"]].astype("int64")
        ):
            raise DatasetArtifactSpecError(
                "Table participants n_sequence/n_trial values do not match observations."
            )


def _validate_declared_column_requirements(
    table: dict[str, Any],
    requirements: dict[str, dict[str, Any]],
    location: str,
) -> None:
    column_by_name = {}
    for idx, column in enumerate(table["columns"]):
        if not isinstance(column, dict):
            raise DatasetArtifactSpecError(
                f"{location}.columns[{idx}] must be an object."
            )
        name = column.get("name")
        if isinstance(name, str):
            column_by_name[name] = column

    for col_name, spec in requirements.items():
        column = column_by_name.get(col_name)
        if column is None:
            raise DatasetArtifactSpecError(
                f"{location} is missing required column '{col_name}'."
            )

        nullable = column.get("nullable")
        if spec.get("nullable") is False and nullable is not False:
            raise DatasetArtifactSpecError(
                f"{location}.{col_name} must declare nullable=false."
            )

        dtype = str(column.get("dtype", ""))
        expected_kind = spec.get("dtype")
        if expected_kind == "int" and not _declared_dtype_is_integer(dtype):
            raise DatasetArtifactSpecError(
                f"{location}.{col_name} must declare an integer dtype."
            )
        if expected_kind == "string" and not _declared_dtype_is_string(dtype):
            raise DatasetArtifactSpecError(
                f"{location}.{col_name} must declare a string dtype."
            )


def _declared_dtype_is_integer(dtype: str) -> bool:
    lowered = dtype.lower()
    return "int" in lowered or "uint" in lowered


def _declared_dtype_is_string(dtype: str) -> bool:
    lowered = dtype.lower()
    return "str" in lowered or "string" in lowered or lowered == "object"


def _is_dataset_relative_path(value: Any) -> bool:
    if not isinstance(value, str):
        return False

    path_value = value.strip()
    if not path_value:
        return False
    if path_value.startswith("/") or path_value.startswith("\\"):
        return False
    if "://" in path_value:
        return False
    if len(path_value) >= 2 and path_value[1] == ":" and path_value[0].isalpha():
        return False

    normalized = path_value.replace("\\", "/")
    parts = [part for part in normalized.split("/") if part not in {"", "."}]
    if ".." in parts:
        return False
    return True


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
