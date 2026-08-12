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
"""Parquet artifact read/write utilities for PsiZ datasets."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import keras
import numpy as np
import pandas as pd

from psiz.data.io.schema import DATASET_DEFAULT_LICENSE
from psiz.data.io.schema import DATASET_FORMAT
from psiz.data.io.schema import DATASET_FORMAT_VERSION
from psiz.data.io.schema import DatasetArtifactSpecError
from psiz.data.io.schema import compute_file_sha256
from psiz.data.io.schema import order_manifest_keys
from psiz.data.io.schema import validate_dataset_artifact_directory

OBSERVATIONS_TABLE_NAME = "observations"
SPLIT_ASSIGNMENTS_TABLE_NAME = "split_assignments"


def write_dataset_artifact_from_samples(
    samples: list[Any],
    output_dir: str | Path,
    *,
    dataset_id: str,
    split_set_id: str = "split_set_v1",
    split_label: str = "train",
    split_version: int = 1,
    license_name: str = DATASET_DEFAULT_LICENSE,
    dataset_version: str = "0.1.0",
    description: str = "",
    sources: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Create a PsiZ dataset artifact from normalized sample payloads."""
    output_path = Path(output_dir)
    if output_path.exists() and any(output_path.iterdir()):
        raise DatasetArtifactSpecError(
            f"Target output directory already exists and is not empty: {output_path}"
        )

    tables_dir = output_path / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    observations = _samples_to_observations(samples)
    if observations.empty:
        raise DatasetArtifactSpecError("No samples provided for dataset artifact write.")

    split_assignments = pd.DataFrame(
        {
            "observation_id": observations["observation_id"].astype("int64"),
            "split": split_label,
            "split_set_id": split_set_id,
            "split_version": int(split_version),
        }
    )

    observations_path = tables_dir / "observations.parquet"
    split_path = tables_dir / "split_assignments.parquet"

    observations.to_parquet(observations_path, index=False)
    split_assignments.to_parquet(split_path, index=False)

    x_features, y_features, w_features = _feature_maps_from_observations(observations)

    manifest = {
        "format": DATASET_FORMAT,
        "format_version": DATASET_FORMAT_VERSION,
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "license": license_name,
        "tables": [
            _build_table_entry(
                name=OBSERVATIONS_TABLE_NAME,
                rel_path="tables/observations.parquet",
                frame=observations,
                kind="fact",
                primary_key=["observation_id"],
            ),
            _build_table_entry(
                name=SPLIT_ASSIGNMENTS_TABLE_NAME,
                rel_path="tables/split_assignments.parquet",
                frame=split_assignments,
                kind="bridge",
                primary_key=["observation_id", "split_set_id"],
                foreign_keys=[
                    {
                        "columns": ["observation_id"],
                        "ref_table": OBSERVATIONS_TABLE_NAME,
                        "ref_columns": ["observation_id"],
                        "on_missing": "error",
                    }
                ],
            ),
        ],
        "split_config": {
            "split_assignment_table": SPLIT_ASSIGNMENTS_TABLE_NAME,
            "active_split_set_id": split_set_id,
            "allowed_split_labels": [split_label],
        },
        "runtime_contract": {
            "observation_table": OBSERVATIONS_TABLE_NAME,
            "x_features": x_features,
            "y_features": y_features,
            "w_features": w_features,
            "batch_axis": 0,
            "timestep": {
                "mode": "either",
                "sequence_id_column": "sequence_id",
                "timestep_index_column": "timestep_index",
            },
        },
        "semantic_contract": {
            "schema_version": "1.0.0",
            "dataset_class": "psiz.data.Dataset",
            "components": [],
            "load_policy": {
                "require_semantic_contract": True,
                "allow_runtime_fallback": False,
            },
        },
    }
    if description and description.strip():
        manifest["description"] = description
    if sources is not None:
        manifest["sources"] = sources

    manifest = _materialize_table_hashes(output_path, manifest)
    _write_manifest(output_path / "manifest.json", manifest)

    validated = validate_dataset_artifact_directory(output_path)
    return validated["manifest"]


def read_dataset_artifact(
    path: str | Path,
    *,
    split_set_id: str | None = None,
    split_labels: list[str] | None = None,
) -> dict[str, Any]:
    """Read a validated dataset artifact and materialize observations."""
    validated = validate_dataset_artifact_directory(path)
    artifact_dir = Path(validated["artifact_dir"])
    manifest = validated["manifest"]

    split_config = manifest["split_config"]
    selected_split_set_id = split_set_id or split_config["active_split_set_id"]

    table_lookup = {entry["name"]: entry for entry in manifest["tables"]}
    obs_table = table_lookup[manifest["runtime_contract"]["observation_table"]]
    split_table = table_lookup[split_config["split_assignment_table"]]

    observations = pd.read_parquet(artifact_dir / obs_table["path"])
    split_assignments = pd.read_parquet(artifact_dir / split_table["path"])

    split_rows = split_assignments[
        split_assignments["split_set_id"] == selected_split_set_id
    ]
    if split_rows.empty:
        raise DatasetArtifactSpecError(
            "No split assignments found for selected split_set_id="
            f"{selected_split_set_id}"
        )

    if split_labels is not None:
        split_rows = split_rows[split_rows["split"].isin(split_labels)]

    merged = observations.merge(
        split_rows[["observation_id", "split"]],
        on="observation_id",
        how="inner",
    )

    if merged.empty:
        raise DatasetArtifactSpecError("Selected split filters yielded no observations.")

    sort_columns = ["split"]
    for col in ["participant_id", "sequence_id", "timestep_index", "observation_id"]:
        if col in merged.columns and col not in sort_columns:
            sort_columns.append(col)

    merged = merged.sort_values(sort_columns).reset_index(drop=True)

    return {
        "manifest": manifest,
        "observations": merged,
        "selected_split_set_id": selected_split_set_id,
    }


def decode_observations_to_xyw(
    observations: pd.DataFrame,
    runtime_contract: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Decode serialized observation columns to x/y/w arrays."""
    x = _decode_feature_block(observations, runtime_contract["x_features"])
    y = _decode_feature_block(observations, runtime_contract["y_features"])
    w = _decode_feature_block(observations, runtime_contract["w_features"])
    return x, y, w


def _samples_to_observations(samples: list[Any]) -> pd.DataFrame:
    rows = []
    for idx, sample in enumerate(samples):
        x, y, w = _normalize_sample(sample)
        row = {
            "observation_id": int(idx),
            "sequence_id": str(idx),
            "timestep_index": int(0),
        }
        row.update(_encode_feature_block("x", x))
        row.update(_encode_feature_block("y", y))
        row.update(_encode_feature_block("w", w))
        rows.append(row)

    return pd.DataFrame(rows)


def _normalize_sample(sample: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if isinstance(sample, tuple):
        if len(sample) == 3:
            x, y, w = sample
        elif len(sample) == 2:
            x, y = sample
            w = {}
        else:
            raise DatasetArtifactSpecError(
                "Tuple sample payloads must be (x, y, w) or (x, y)."
            )
    else:
        x = sample
        y = {}
        w = {}

    x = _as_named_mapping(x, default_key="x")
    y = _as_named_mapping(y, default_key="y")
    w = _as_named_mapping(w, default_key="w")
    return x, y, w


def _as_named_mapping(payload: Any, *, default_key: str) -> dict[str, Any]:
    if payload is None:
        return {}
    if isinstance(payload, dict):
        return payload
    return {default_key: payload}


def _encode_feature_block(prefix: str, block: dict[str, Any]) -> dict[str, str]:
    encoded = {}
    for name, value in block.items():
        array_value = keras.ops.convert_to_numpy(value)
        json_ready = _json_compatible(np.asarray(array_value).tolist())
        encoded[f"{prefix}::{name}"] = json.dumps(json_ready)
    return encoded


def _json_compatible(value: Any) -> Any:
    """Recursively convert NumPy/bytes values to JSON-compatible objects."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.decode("utf-8")
    if isinstance(value, list):
        return [_json_compatible(v) for v in value]
    if isinstance(value, tuple):
        return [_json_compatible(v) for v in value]
    if isinstance(value, dict):
        return {k: _json_compatible(v) for k, v in value.items()}
    return value


def _decode_feature_block(
    observations: pd.DataFrame, mapping: dict[str, str]
) -> dict[str, np.ndarray]:
    decoded = {}
    for feature_name, column_name in mapping.items():
        if column_name not in observations.columns:
            raise DatasetArtifactSpecError(
                f"Mapped column '{column_name}' not found in observations table."
            )

        values = [json.loads(v) for v in observations[column_name].tolist()]
        decoded[feature_name] = np.asarray(values)
    return decoded


def _feature_maps_from_observations(
    observations: pd.DataFrame,
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    x_features = {}
    y_features = {}
    w_features = {}
    for col in observations.columns:
        if col.startswith("x::"):
            x_features[col[3:]] = col
        elif col.startswith("y::"):
            y_features[col[3:]] = col
        elif col.startswith("w::"):
            w_features[col[3:]] = col
    return x_features, y_features, w_features


def _build_table_entry(
    *,
    name: str,
    rel_path: str,
    frame: pd.DataFrame,
    kind: str,
    primary_key: list[str],
    foreign_keys: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    columns = []
    for column in frame.columns:
        series = frame[column]
        columns.append(
            {
                "name": column,
                "dtype": str(series.dtype),
                "nullable": bool(series.isnull().any()),
            }
        )

    # Caller writes relative to artifact root tables dir, then resolves hash later.
    # Here we only keep relative path and metadata.
    entry = {
        "name": name,
        "path": rel_path,
        "kind": kind,
        "primary_key": primary_key,
        "columns": columns,
        "sha256": "",
        "row_count": int(len(frame)),
    }
    if foreign_keys is not None:
        entry["foreign_keys"] = foreign_keys

    return entry


def refresh_manifest_integrity_hashes(artifact_dir: str | Path) -> dict[str, Any]:
    """Populate sha256 fields for all tables and rewrite manifest in-place."""
    artifact_path = Path(artifact_dir)
    manifest_path = artifact_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = _materialize_table_hashes(artifact_path, manifest)
    _write_manifest(manifest_path, manifest)
    return manifest


def _materialize_table_hashes(
    artifact_path: Path, manifest: dict[str, Any]
) -> dict[str, Any]:
    for table in manifest["tables"]:
        table_path = artifact_path / table["path"]
        table["sha256"] = compute_file_sha256(table_path)
    return manifest


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    ordered = order_manifest_keys(manifest)
    path.write_text(json.dumps(ordered, indent=2) + "\n", encoding="utf-8")
