# -*- coding: utf-8 -*-
"""Tests for TensorFlow dataset migration into PsiZ dataset artifacts."""

from __future__ import annotations

import numpy as np
import pytest

from psiz.data.io import validate_dataset_artifact_directory
from psiz.migration import migrate_dataset_from_tfds
from psiz.migration import validate_dataset_migration_report_schema


def _build_tf_source_dataset():
    tf = pytest.importorskip("tensorflow")

    x = {
        "stimulus_set": np.array(
            [[1, 2, 3], [3, 4, 5], [5, 6, 7]],
            dtype=np.int32,
        )
    }
    y = {
        "outcome": np.array(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
            dtype=np.float32,
        )
    }
    w = {
        "outcome": np.array(
            [[1.0], [0.5], [1.0]],
            dtype=np.float32,
        )
    }
    return tf.data.Dataset.from_tensor_slices((x, y, w))


def test_dataset_migration_from_tfds_smoke(tmp_path):
    ds = _build_tf_source_dataset()
    out_dir = tmp_path / "migrated_dataset.psiz"

    report = migrate_dataset_from_tfds(
        ds,
        out_dir,
        split_set_id="split_set_v1",
        validate=False,
        dataset_id="test_dataset_migration_from_tfds_smoke",
    )

    assert report["status"] == "success"
    assert report["destination"]["format"] == "psiz-dataset"


def test_dataset_migration_manifest_validity(tmp_path):
    ds = _build_tf_source_dataset()
    out_dir = tmp_path / "migrated_dataset.psiz"

    _ = migrate_dataset_from_tfds(
        ds,
        out_dir,
        split_set_id="split_set_v1",
        validate=False,
        dataset_id="test_dataset_migration_manifest_validity",
    )

    validated = validate_dataset_artifact_directory(out_dir)
    assert validated["manifest"]["dataset_id"] == "test_dataset_migration_manifest_validity"


def test_dataset_migration_xyw_parity_fixed_fixture(tmp_path):
    ds = _build_tf_source_dataset()
    out_dir = tmp_path / "migrated_dataset.psiz"

    report = migrate_dataset_from_tfds(
        ds,
        out_dir,
        split_set_id="split_set_v1",
        validate=True,
        dataset_id="test_dataset_migration_xyw_parity_fixed_fixture",
    )

    assert report["parity"]["enabled"] is True
    assert report["parity"]["validated"] is True
    assert report["parity"]["passed"] is True
    assert report["parity"]["max_abs_error"] <= 1e-6


def test_dataset_migration_report_schema(tmp_path):
    ds = _build_tf_source_dataset()
    out_dir = tmp_path / "migrated_dataset.psiz"

    report = migrate_dataset_from_tfds(
        ds,
        out_dir,
        split_set_id="split_set_v1",
        validate=False,
        dataset_id="test_dataset_migration_report_schema",
    )

    validate_dataset_migration_report_schema(report)
    assert sorted(report["diagnostics"].keys()) == ["errors", "warnings"]
