# -*- coding: utf-8 -*-
"""Tests for PsiZ dataset artifacts and runtime ingestion."""

from __future__ import annotations

import json

import numpy as np
import pytest
import psiz

from psiz.data.io import read_dataset_artifact
from psiz.data.io import validate_dataset_artifact_directory
from psiz.data.io import validate_manifest_schema
from psiz.data.io import write_dataset_artifact_from_samples


def _build_samples(with_targets: bool = True):
    x0 = {"stimulus_set": np.array([1, 2, 3], dtype=np.int32)}
    x1 = {"stimulus_set": np.array([3, 4, 5], dtype=np.int32)}
    if not with_targets:
        return [x0, x1]

    y0 = {"outcome": np.array([1.0, 0.0], dtype=np.float32)}
    y1 = {"outcome": np.array([0.0, 1.0], dtype=np.float32)}
    w0 = {"outcome": np.array([1.0], dtype=np.float32)}
    w1 = {"outcome": np.array([0.5], dtype=np.float32)}
    return [(x0, y0, w0), (x1, y1, w1)]


def test_manifest_schema_roundtrip(tmp_path):
    artifact_dir = tmp_path / "dataset.psiz"
    write_dataset_artifact_from_samples(
        _build_samples(),
        artifact_dir,
        dataset_id="test_manifest_schema_roundtrip",
    )

    validated = validate_dataset_artifact_directory(artifact_dir)
    manifest = validated["manifest"]
    validate_manifest_schema(manifest)

    assert manifest["format"] == "psiz-dataset"
    assert manifest["runtime_contract"]["observation_table"] == "observations"
    timestep = manifest["runtime_contract"]["timestep"]
    assert timestep["sequence_id_column"] == "sequence_id"
    assert timestep["timestep_index_column"] == "timestep_index"
    semantic_contract = manifest["semantic_contract"]
    assert semantic_contract["dataset_class"] == "psiz.data.Dataset"
    assert semantic_contract["schema_version"] == "1.0.0"
    assert semantic_contract["load_policy"]["require_semantic_contract"] is True
    assert semantic_contract["load_policy"]["allow_runtime_fallback"] is False


def test_manifest_major_version_guard(tmp_path):
    artifact_dir = tmp_path / "dataset.psiz"
    write_dataset_artifact_from_samples(
        _build_samples(),
        artifact_dir,
        dataset_id="test_manifest_major_version_guard",
    )

    manifest_path = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["format_version"] = "2.0.0"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    try:
        validate_dataset_artifact_directory(artifact_dir)
        assert False, "Expected major-version guard to fail validation."
    except ValueError as exc:
        assert "Unsupported dataset format major version" in str(exc)


def test_manifest_requires_semantic_contract(tmp_path):
    artifact_dir = tmp_path / "dataset.psiz"
    write_dataset_artifact_from_samples(
        _build_samples(),
        artifact_dir,
        dataset_id="test_manifest_requires_semantic_contract",
    )

    manifest_path = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("semantic_contract")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="manifest is missing required keys: semantic_contract"):
        validate_dataset_artifact_directory(artifact_dir)


def test_parquet_fact_only_artifact_roundtrip(tmp_path):
    artifact_dir = tmp_path / "dataset.psiz"
    write_dataset_artifact_from_samples(
        _build_samples(with_targets=False),
        artifact_dir,
        dataset_id="test_parquet_fact_only_artifact_roundtrip",
    )

    payload = read_dataset_artifact(artifact_dir)
    assert len(payload["observations"]) == 2
    assert "x::stimulus_set" in payload["observations"].columns
    assert "sequence_id" in payload["observations"].columns
    assert "timestep_index" in payload["observations"].columns
    assert "sequence_length" not in payload["observations"].columns


def test_parquet_relational_artifact_roundtrip(tmp_path):
    artifact_dir = tmp_path / "dataset.psiz"
    write_dataset_artifact_from_samples(
        _build_samples(with_targets=True),
        artifact_dir,
        dataset_id="test_parquet_relational_artifact_roundtrip",
        split_set_id="split_set_v1",
    )

    payload = read_dataset_artifact(artifact_dir, split_set_id="split_set_v1")
    assert len(payload["observations"]) == 2
    assert "y::outcome" in payload["observations"].columns
    assert "w::outcome" in payload["observations"].columns


def test_relational_join_determinism(tmp_path):
    artifact_dir = tmp_path / "dataset.psiz"
    write_dataset_artifact_from_samples(
        _build_samples(),
        artifact_dir,
        dataset_id="test_relational_join_determinism",
    )

    payload = read_dataset_artifact(artifact_dir)
    obs = payload["observations"]
    assert list(obs["observation_id"].values) == sorted(obs["observation_id"].values)


def test_runtime_pydataset_shape_contract(tmp_path):
    artifact_dir = tmp_path / "dataset.psiz"
    write_dataset_artifact_from_samples(
        _build_samples(),
        artifact_dir,
        dataset_id="test_runtime_pydataset_shape_contract",
    )

    ds = psiz.data.load(artifact_dir)
    batch = ds[0]
    x, y, w = batch

    assert x["stimulus_set"].shape == (2, 3)
    assert y.shape == (2, 2)
    assert w.shape == (2, 1)


def test_tier_a_tensorflow_adapter_smoke(tmp_path):
    tf = pytest.importorskip("tensorflow")
    artifact_dir = tmp_path / "dataset.psiz"
    write_dataset_artifact_from_samples(
        _build_samples(),
        artifact_dir,
        dataset_id="test_tier_a_tensorflow_adapter_smoke",
    )

    ds = psiz.data.load(artifact_dir)
    tf_dataset = ds.tensorflow()
    first = next(iter(tf_dataset))
    assert len(first) == 3
    assert isinstance(first[0]["stimulus_set"], tf.Tensor)


def test_tier_a_torch_adapter_smoke(tmp_path):
    torch = pytest.importorskip("torch")
    artifact_dir = tmp_path / "dataset.psiz"
    write_dataset_artifact_from_samples(
        _build_samples(),
        artifact_dir,
        dataset_id="test_tier_a_torch_adapter_smoke",
    )

    ds = psiz.data.load(artifact_dir)
    loader = ds.torch()
    first = loader[0]
    assert len(first) == 3
    assert isinstance(first[0]["stimulus_set"], torch.Tensor)


def test_tier_a_numpy_adapter_smoke(tmp_path):
    artifact_dir = tmp_path / "dataset.psiz"
    write_dataset_artifact_from_samples(
        _build_samples(),
        artifact_dir,
        dataset_id="test_tier_a_numpy_adapter_smoke",
    )

    ds = psiz.data.load(artifact_dir)
    payload = ds.numpy()
    assert len(payload) == 3
    assert payload[0]["stimulus_set"].shape == (2, 3)


def test_tier_a_arrow_adapter_smoke(tmp_path):
    pa = pytest.importorskip("pyarrow")
    artifact_dir = tmp_path / "dataset.psiz"
    write_dataset_artifact_from_samples(
        _build_samples(),
        artifact_dir,
        dataset_id="test_tier_a_arrow_adapter_smoke",
    )

    ds = psiz.data.load(artifact_dir)
    table = ds.arrow()
    assert isinstance(table, pa.Table)
    assert "x::stimulus_set" in table.column_names
    assert table.num_rows == 2


def test_adapter_seed_and_shuffle_consistency(tmp_path):
    artifact_dir = tmp_path / "dataset.psiz"
    samples = []
    for i in range(8):
        x = {"stimulus_set": np.array([i, i + 1, i + 2], dtype=np.int32)}
        y = {"outcome": np.array([float(i % 2), float((i + 1) % 2)], dtype=np.float32)}
        w = {"outcome": np.array([1.0], dtype=np.float32)}
        samples.append((x, y, w))

    write_dataset_artifact_from_samples(
        samples,
        artifact_dir,
        dataset_id="test_adapter_seed_and_shuffle_consistency",
    )

    ds0 = psiz.data.load(artifact_dir)
    ds1 = psiz.data.load(artifact_dir)
    ds0.batch_size = 2
    ds1.batch_size = 2
    ds0.shuffle = True
    ds1.shuffle = True
    ds0.seed = 123
    ds1.seed = 123
    ds0.on_epoch_end()
    ds1.on_epoch_end()

    a0 = ds0[0][0]["stimulus_set"]
    a1 = ds1[0][0]["stimulus_set"]
    np.testing.assert_array_equal(a0, a1)


def test_tf_dataset_transition_behavior(tmp_path):
    pytest.importorskip("tensorflow")
    artifact_dir = tmp_path / "dataset.psiz"
    write_dataset_artifact_from_samples(
        _build_samples(),
        artifact_dir,
        dataset_id="test_tf_dataset_transition_behavior",
    )

    ds = psiz.data.load(artifact_dir)
    tf_dataset = ds.tensorflow()
    a = next(iter(tf_dataset))
    np.testing.assert_equal(a[0]["stimulus_set"].shape[0], 3)
