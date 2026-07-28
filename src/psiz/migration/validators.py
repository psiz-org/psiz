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
"""Validation helpers for legacy asset migration."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class MigrationError(ValueError):
    """Base migration error with a machine-readable code."""

    def __init__(self, message: str, code: str):
        super().__init__(message)
        self.code = code


class LegacyAssetValidationError(MigrationError):
    """Raised when legacy asset input is invalid."""


class UnsupportedLegacyFormatError(MigrationError):
    """Raised when an unsupported legacy format is requested."""


class LegacyModelLoadError(MigrationError):
    """Raised when a legacy asset cannot be loaded safely."""


class ParityValidationError(MigrationError):
    """Raised when optional migration parity validation fails."""


class MigrationReportValidationError(MigrationError):
    """Raised when migration report payload is malformed."""


def detect_legacy_asset_format(path: str | Path) -> str:
    """Detect supported legacy asset format from file suffix."""
    suffix = Path(path).suffix.lower()
    if suffix == ".keras":
        return "keras"
    if suffix in {".h5", ".hdf5"}:
        return "h5"
    return "unknown"


def validate_legacy_asset_path(path: str | Path) -> Path:
    """Validate and return a normalized legacy source path."""
    source_path = Path(path)
    if not source_path.exists():
        raise LegacyAssetValidationError(
            f"Legacy model asset does not exist: {source_path}",
            code="legacy_asset_not_found",
        )
    if source_path.is_dir():
        raise LegacyAssetValidationError(
            "Legacy model asset path must point to a .keras file, not a directory: "
            f"{source_path}",
            code="legacy_asset_not_file",
        )

    detected_format = detect_legacy_asset_format(source_path)
    if detected_format == "h5":
        raise UnsupportedLegacyFormatError(
            "Legacy .h5/.hdf5 migration is out of scope for v0.14. "
            "Provide a .keras source asset.",
            code="unsupported_legacy_format_h5",
        )
    if detected_format != "keras":
        raise UnsupportedLegacyFormatError(
            "Unsupported legacy model format. Expected a .keras file.",
            code="unsupported_legacy_format",
        )
    return source_path


def validate_destination_path(path: str | Path) -> Path:
    """Validate and return a normalized destination artifact path."""
    destination_path = Path(path)
    if destination_path.suffix.lower() != ".psiz":
        raise LegacyAssetValidationError(
            "Destination path must end with '.psiz' to create a PsiZ artifact "
            f"directory. Received: {destination_path}",
            code="invalid_destination_suffix",
        )
    return destination_path


def validate_migration_report_schema(report: dict[str, Any]) -> None:
    """Validate minimal report schema emitted by migration API."""
    if not isinstance(report, dict):
        raise MigrationReportValidationError(
            "Migration report must be a dictionary payload.",
            code="invalid_report_type",
        )

    required_top_level = {
        "status",
        "source",
        "destination",
        "resolved_backend",
        "model",
        "intermediate",
        "parity",
        "diagnostics",
    }
    missing_top_level = sorted(required_top_level - set(report.keys()))
    if missing_top_level:
        raise MigrationReportValidationError(
            "Migration report is missing required keys: " + ", ".join(missing_top_level),
            code="missing_report_keys",
        )

    if report.get("status") != "success":
        raise MigrationReportValidationError(
            "Migration report status must be 'success'.",
            code="invalid_report_status",
        )

    if not isinstance(report["source"], dict) or "path" not in report["source"]:
        raise MigrationReportValidationError(
            "Migration report source payload is invalid.",
            code="invalid_report_source",
        )
    if not isinstance(report["destination"], dict) or "path" not in report["destination"]:
        raise MigrationReportValidationError(
            "Migration report destination payload is invalid.",
            code="invalid_report_destination",
        )

    parity = report["parity"]
    if not isinstance(parity, dict):
        raise MigrationReportValidationError(
            "Migration report parity payload must be a dictionary.",
            code="invalid_report_parity",
        )
    for key in ["enabled", "validated", "passed", "rtol", "atol", "max_abs_error"]:
        if key not in parity:
            raise MigrationReportValidationError(
                f"Migration report parity payload is missing '{key}'.",
                code="invalid_report_parity",
            )

    diagnostics = report["diagnostics"]
    if not isinstance(diagnostics, dict):
        raise MigrationReportValidationError(
            "Migration report diagnostics payload must be a dictionary.",
            code="invalid_report_diagnostics",
        )
    if "warnings" not in diagnostics or "errors" not in diagnostics:
        raise MigrationReportValidationError(
            "Migration report diagnostics payload must define warnings and errors.",
            code="invalid_report_diagnostics",
        )
