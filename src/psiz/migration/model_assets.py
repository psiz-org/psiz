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
"""Migration APIs for legacy model artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import keras
import numpy as np

from psiz.storage import load_psiz_model
from psiz.storage import save_psiz_model

from .validators import LegacyModelLoadError
from .validators import ParityValidationError
from .validators import validate_destination_path
from .validators import validate_legacy_asset_path
from .validators import validate_migration_report_schema


def migrate_model_from_keras(
    source_path: str | Path,
    destination_path: str | Path,
    *,
    backend_override: str | None = None,
    custom_objects: dict[str, Any] | None = None,
    validate_parity: bool = False,
    parity_inputs: Any | None = None,
    rtol: float = 1e-6,
    atol: float = 1e-6,
    readme_text: str | None = None,
    license_text: str = "Apache-2.0\n",
    license_name: str = "Apache-2.0",
    license_policy: str = "include",
) -> dict[str, Any]:
    """Migrate a legacy .keras model asset into a PsiZ .psiz artifact.

    Args:
        source_path: Path to an existing legacy .keras model file.
        destination_path: Destination .psiz artifact directory path.
        backend_override: Optional backend resolution override.
        custom_objects: Optional custom Keras object map for loading.
        validate_parity: If True, run optional prediction parity checks.
        parity_inputs: Inputs used for optional parity validation.
        rtol: Relative tolerance for parity validation.
        atol: Absolute tolerance for parity validation.
        readme_text: Optional README.md artifact payload.
        license_text: Artifact LICENSE file payload.
        license_name: License metadata name.
        license_policy: License metadata policy.

    Returns:
        A migration report with validation diagnostics and metadata.
    """
    source_path = validate_legacy_asset_path(source_path)
    destination_path = validate_destination_path(destination_path)

    legacy_model = _load_legacy_keras_model(source_path, custom_objects=custom_objects)
    intermediate = _to_intermediate_representation(legacy_model)

    manifest = save_psiz_model(
        legacy_model,
        destination_path,
        backend_override=backend_override,
        readme_text=readme_text,
        license_text=license_text,
        license_name=license_name,
        license_policy=license_policy,
    )

    parity_payload = _build_default_parity_payload(
        enabled=validate_parity,
        rtol=rtol,
        atol=atol,
    )
    if validate_parity:
        parity_payload = _run_parity_check(
            legacy_model=legacy_model,
            artifact_path=destination_path,
            parity_inputs=parity_inputs,
            backend_override=backend_override,
            custom_objects=custom_objects,
            rtol=rtol,
            atol=atol,
        )

    report = {
        "status": "success",
        "source": {
            "path": str(source_path),
            "format": "keras",
        },
        "destination": {
            "path": str(destination_path),
            "format": "psiz",
        },
        "resolved_backend": manifest["resolved_backend"],
        "model": {
            "class_name": legacy_model.__class__.__name__,
            "module": legacy_model.__class__.__module__,
            "weight_count": len(legacy_model.weights),
        },
        "intermediate": intermediate,
        "parity": parity_payload,
        "diagnostics": {
            "warnings": [],
            "errors": [],
            "storage_compaction": _build_storage_compaction_payload(destination_path),
        },
    }
    validate_migration_report_schema(report)
    return report


def _build_storage_compaction_payload(artifact_path: Path) -> dict[str, Any]:
    """Summarize config compaction diagnostics for migration reporting."""
    config_path = artifact_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    payload: dict[str, Any] = {
        "enabled": False,
        "blob_count": 0,
        "bytes_moved_estimate": 0,
        "blob_bytes": 0,
        "config_json_bytes": int(config_path.stat().st_size),
    }

    compaction = config.get("model_config_compaction")
    if not isinstance(compaction, dict):
        return payload

    blob_file = compaction["blob_file"]
    blob_path = artifact_path / blob_file
    payload.update(
        {
            "enabled": True,
            "blob_file": blob_file,
            "blob_count": int(compaction["blob_count"]),
            "bytes_moved_estimate": int(compaction["externalized_json_estimate_bytes"]),
            "blob_bytes": int(blob_path.stat().st_size),
        }
    )
    return payload


def _load_legacy_keras_model(
    source_path: Path,
    *,
    custom_objects: dict[str, Any] | None,
) -> keras.Model:
    """Load a legacy Keras model with guarded error translation."""
    try:
        model = keras.saving.load_model(
            source_path,
            custom_objects=custom_objects,
            compile=False,
            safe_mode=False,
        )
    except Exception as exc:
        raise LegacyModelLoadError(
            f"Unable to load legacy .keras model at {source_path}: {exc}",
            code="legacy_model_load_failed",
        ) from exc
    return model


def _to_intermediate_representation(model: keras.Model) -> dict[str, Any]:
    """Create a deterministic intermediate representation for diagnostics."""
    model_config = keras.saving.serialize_keras_object(model)
    return {
        "representation": "keras_serialized_object",
        "model_config_keys": sorted(model_config.keys()),
        "weight_count": len(model.weights),
    }


def _build_default_parity_payload(
    *,
    enabled: bool,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    return {
        "enabled": bool(enabled),
        "validated": False,
        "passed": None,
        "rtol": float(rtol),
        "atol": float(atol),
        "max_abs_error": None,
    }


def _run_parity_check(
    *,
    legacy_model: keras.Model,
    artifact_path: Path,
    parity_inputs: Any,
    backend_override: str | None,
    custom_objects: dict[str, Any] | None,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    if parity_inputs is None:
        raise ParityValidationError(
            "When validate_parity=True, parity_inputs must be provided.",
            code="parity_inputs_required",
        )

    migrated_model = load_psiz_model(
        artifact_path,
        backend_override=backend_override,
        custom_objects=custom_objects,
    )

    legacy_outputs = keras.ops.convert_to_numpy(legacy_model(parity_inputs, training=False))
    migrated_outputs = keras.ops.convert_to_numpy(
        migrated_model(parity_inputs, training=False)
    )

    if legacy_outputs.shape != migrated_outputs.shape:
        raise ParityValidationError(
            "Parity validation failed because output shapes differ: "
            f"legacy={legacy_outputs.shape}, migrated={migrated_outputs.shape}.",
            code="parity_shape_mismatch",
        )

    deltas = np.abs(migrated_outputs - legacy_outputs)
    max_abs_error = float(np.max(deltas)) if deltas.size > 0 else 0.0
    passed = bool(np.allclose(migrated_outputs, legacy_outputs, rtol=rtol, atol=atol))
    if not passed:
        raise ParityValidationError(
            "Parity validation failed because migrated predictions are outside "
            "tolerance bounds.",
            code="parity_value_mismatch",
        )

    return {
        "enabled": True,
        "validated": True,
        "passed": True,
        "rtol": float(rtol),
        "atol": float(atol),
        "max_abs_error": max_abs_error,
    }
