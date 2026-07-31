# -*- coding: utf-8 -*-
"""Migration API for TensorFlow dataset workflows to PsiZ dataset artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import keras
import numpy as np

from psiz.data.io import decode_observations_to_xyw
from psiz.data.io import read_dataset_artifact
from psiz.data.io import refresh_manifest_integrity_hashes
from psiz.data.io import write_dataset_artifact_from_samples

from .validators import MigrationReportValidationError
from .validators import validate_dataset_migration_report_schema


def migrate_dataset_from_tfds(
    source: Any,
    output_dir: str | Path,
    *,
    split_set_id: str | None = None,
    with_timestep_axis: bool | None = None,
    validate: bool = True,
    dataset_id: str = "migrated_dataset",
) -> dict[str, Any]:
    """Migrate a TensorFlow dataset-style source to PsiZ dataset artifacts."""
    del with_timestep_axis

    samples = _extract_samples(source)
    if len(samples) == 0:
        raise MigrationReportValidationError(
            "Cannot migrate empty dataset source.",
            code="dataset_migration_empty_source",
        )

    selected_split_set = split_set_id or "split_set_v1"
    manifest = write_dataset_artifact_from_samples(
        samples,
        output_dir,
        dataset_id=dataset_id,
        split_set_id=selected_split_set,
    )
    manifest = refresh_manifest_integrity_hashes(output_dir)

    parity = {
        "enabled": bool(validate),
        "validated": False,
        "passed": None,
        "max_abs_error": None,
    }

    if validate:
        parity = _validate_migration_parity(output_dir, samples)

    report = {
        "status": "success",
        "source": {
            "type": type(source).__name__,
            "sample_count": len(samples),
        },
        "destination": {
            "path": str(Path(output_dir)),
            "format": "psiz-dataset",
            "dataset_id": manifest["dataset_id"],
        },
        "split": {
            "split_set_id": selected_split_set,
            "labels": manifest["split_config"]["allowed_split_labels"],
        },
        "tables": [
            {
                "name": table["name"],
                "row_count": table["row_count"],
                "sha256": table["sha256"],
            }
            for table in manifest["tables"]
        ],
        "parity": parity,
        "diagnostics": {
            "warnings": [],
            "errors": [],
        },
    }
    validate_dataset_migration_report_schema(report)
    return report


def _extract_samples(source: Any) -> list[Any]:
    if hasattr(source, "as_numpy_iterator"):
        return list(source.as_numpy_iterator())

    if isinstance(source, list):
        return source

    if isinstance(source, tuple):
        return [source]

    if hasattr(source, "__iter__"):
        return list(source)

    raise MigrationReportValidationError(
        "Unsupported dataset source type for migration.",
        code="dataset_migration_unsupported_source",
    )


def _validate_migration_parity(output_dir: str | Path, source_samples: list[Any]) -> dict[str, Any]:
    payload = read_dataset_artifact(output_dir)
    migrated_x, migrated_y, migrated_w = decode_observations_to_xyw(
        payload["observations"], payload["manifest"]["runtime_contract"]
    )

    source_x, source_y, source_w = _stack_source_samples(source_samples)

    max_abs_error = max(
        _max_block_delta(source_x, migrated_x),
        _max_block_delta(source_y, migrated_y),
        _max_block_delta(source_w, migrated_w),
    )

    passed = bool(np.isfinite(max_abs_error) and max_abs_error <= 1e-6)
    return {
        "enabled": True,
        "validated": True,
        "passed": passed,
        "max_abs_error": float(max_abs_error),
    }


def _split_sample(sample: Any):
    if isinstance(sample, tuple):
        if len(sample) == 3:
            x, y, w = sample
            return x, y, w
        if len(sample) == 2:
            return sample[0], sample[1], {}
    return sample, {}, {}


def _stack_source_samples(samples: list[Any]):
    x_rows = {}
    y_rows = {}
    w_rows = {}

    for sample in samples:
        x, y, w = _split_sample(sample)
        _append_rows(x_rows, x)
        _append_rows(y_rows, y)
        _append_rows(w_rows, w)

    return _stack_rows(x_rows), _stack_rows(y_rows), _stack_rows(w_rows)


def _append_rows(store: dict[str, list[np.ndarray]], payload: Any) -> None:
    if payload is None:
        return
    if not isinstance(payload, dict):
        payload = {"value": payload}

    for key, value in payload.items():
        arr = keras.ops.convert_to_numpy(value)
        store.setdefault(key, []).append(np.asarray(arr))


def _stack_rows(store: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    stacked = {}
    for key, rows in store.items():
        stacked[key] = np.asarray(rows)
    return stacked




def _max_block_delta(a: Any, b: Any) -> float:
    if isinstance(a, dict):
        if not isinstance(b, dict):
            if len(a) != 1:
                return float("inf")
            a = next(iter(a.values()))
            return _max_block_delta(a, b)
        keys = set(a.keys())
        if keys != set(b.keys()):
            return float("inf")
        values = [_max_block_delta(a[k], b[k]) for k in keys]
        return max(values) if values else 0.0

    if isinstance(b, dict):
        if len(b) != 1:
            return float("inf")
        b = next(iter(b.values()))

    aa = keras.ops.convert_to_numpy(a)
    bb = keras.ops.convert_to_numpy(b)
    if aa.shape != bb.shape:
        return float("inf")
    if aa.size == 0:
        return 0.0
    return float((abs(aa - bb)).max())
