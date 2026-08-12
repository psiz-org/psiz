# -*- coding: utf-8 -*-
"""Tests for PsiZ dataset artifacts and runtime ingestion."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import psiz

from psiz.data.io import read_dataset_artifact
from psiz.data.io import refresh_manifest_integrity_hashes
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


def _table_entry_from_frame(
    *, name: str, rel_path: str, kind: str, primary_key: list[str], frame: pd.DataFrame
):
    return {
        "name": name,
        "path": rel_path,
        "kind": kind,
        "primary_key": primary_key,
        "columns": [
            {
                "name": col,
                "dtype": str(frame[col].dtype),
                "nullable": bool(frame[col].isnull().any()),
            }
            for col in frame.columns
        ],
        "sha256": "",
        "row_count": int(len(frame)),
    }


def _augment_with_stimuli_tables(
    artifact_dir,
    *,
    stimuli_frame: pd.DataFrame | None = None,
    bridge_frame: pd.DataFrame | None = None,
    stimulus_id_features: list[str] | None = None,
    stimuli_table_ref: str = "stimuli",
):
    tables_dir = artifact_dir / "tables"
    manifest_path = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if stimuli_frame is None:
        stimuli_frame = pd.DataFrame(
            {
                "stimulus_id": [1, 2, 3, 4, 5],
                "filepath": [
                    "images/a.jpg",
                    "images/b.jpg",
                    "images/c.jpg",
                    "images/d.jpg",
                    "images/e.jpg",
                ],
                "leaf_level": ["a", "b", "c", "d", "e"],
            }
        )

    stimuli_frame.to_parquet(tables_dir / "stimuli.parquet", index=False)
    manifest["tables"].append(
        _table_entry_from_frame(
            name="stimuli",
            rel_path="tables/stimuli.parquet",
            kind="dimension",
            primary_key=["stimulus_id"],
            frame=stimuli_frame,
        )
    )

    if bridge_frame is not None:
        bridge_frame.to_parquet(tables_dir / "observation_stimuli.parquet", index=False)
        manifest["tables"].append(
            _table_entry_from_frame(
                name="observation_stimuli",
                rel_path="tables/observation_stimuli.parquet",
                kind="bridge",
                primary_key=["observation_id", "x_feature_name", "position"],
                frame=bridge_frame,
            )
        )

    manifest["runtime_contract"]["stimuli_table"] = stimuli_table_ref
    if stimulus_id_features is not None:
        manifest["runtime_contract"]["stimulus_id_features"] = stimulus_id_features

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    refresh_manifest_integrity_hashes(artifact_dir)


def _augment_with_participants_table(
    artifact_dir,
    *,
    participants_frame: pd.DataFrame | None = None,
):
    tables_dir = artifact_dir / "tables"
    manifest_path = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if participants_frame is None:
        participants_frame = pd.DataFrame(
            {
                "participant_id": pd.Series([10, 20], dtype="int64"),
                "n_sequence": pd.Series([1, 1], dtype="int64"),
                "n_trial": pd.Series([1, 1], dtype="int64"),
                "external_participant_id": ["worker-a", "worker-b"],
            }
        )

    participants_frame.to_parquet(tables_dir / "participants.parquet", index=False)
    manifest["tables"].append(
        _table_entry_from_frame(
            name="participants",
            rel_path="tables/participants.parquet",
            kind="dimension",
            primary_key=["participant_id"],
            frame=participants_frame,
        )
    )
    manifest["tables"][-1]["sensitive_columns"] = ["external_participant_id"]

    observations_path = tables_dir / "observations.parquet"
    observations = pd.read_parquet(observations_path)
    observations["participant_id"] = pd.Series([10, 20], dtype="int64")
    observations.to_parquet(observations_path, index=False)
    for table in manifest["tables"]:
        if table["name"] == "observations":
            table["columns"].append(
                {"name": "participant_id", "dtype": "int64", "nullable": False}
            )

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    refresh_manifest_integrity_hashes(artifact_dir)


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


def test_optional_stimuli_table_schema_valid(tmp_path):
    artifact_dir = tmp_path / "dataset.psiz"
    write_dataset_artifact_from_samples(
        _build_samples(),
        artifact_dir,
        dataset_id="test_optional_stimuli_table_schema_valid",
    )

    _augment_with_stimuli_tables(artifact_dir)
    validated = validate_dataset_artifact_directory(artifact_dir)
    names = {table["name"] for table in validated["manifest"]["tables"]}
    assert "stimuli" in names


def test_optional_stimuli_table_rejects_absolute_filepath(tmp_path):
    artifact_dir = tmp_path / "dataset.psiz"
    write_dataset_artifact_from_samples(
        _build_samples(),
        artifact_dir,
        dataset_id="test_optional_stimuli_table_rejects_absolute_filepath",
    )

    stimuli = pd.DataFrame(
        {
            "stimulus_id": [1, 2],
            "filepath": ["/abs/path/a.jpg", "images/b.jpg"],
        }
    )
    _augment_with_stimuli_tables(artifact_dir, stimuli_frame=stimuli)

    with pytest.raises(ValueError, match="dataset-root-relative"):
        validate_dataset_artifact_directory(artifact_dir)


def test_optional_stimuli_table_rejects_duplicate_stimulus_id(tmp_path):
    artifact_dir = tmp_path / "dataset.psiz"
    write_dataset_artifact_from_samples(
        _build_samples(),
        artifact_dir,
        dataset_id="test_optional_stimuli_table_rejects_duplicate_stimulus_id",
    )

    stimuli = pd.DataFrame(
        {
            "stimulus_id": [1, 1],
            "filepath": ["images/a.jpg", "images/b.jpg"],
        }
    )
    _augment_with_stimuli_tables(artifact_dir, stimuli_frame=stimuli)

    with pytest.raises(ValueError, match="Primary key rows are not unique"):
        validate_dataset_artifact_directory(artifact_dir)


def test_optional_observation_stimuli_bridge_schema_valid(tmp_path):
    artifact_dir = tmp_path / "dataset.psiz"
    write_dataset_artifact_from_samples(
        _build_samples(),
        artifact_dir,
        dataset_id="test_optional_observation_stimuli_bridge_schema_valid",
    )

    bridge = pd.DataFrame(
        {
            "observation_id": [0, 0, 1, 1],
            "x_feature_name": ["stimulus_set", "stimulus_set", "stimulus_set", "stimulus_set"],
            "position": [0, 1, 0, 1],
            "stimulus_id": [1, 2, 3, 4],
        }
    )
    _augment_with_stimuli_tables(
        artifact_dir,
        bridge_frame=bridge,
        stimulus_id_features=["stimulus_set"],
    )

    validated = validate_dataset_artifact_directory(artifact_dir)
    names = {table["name"] for table in validated["manifest"]["tables"]}
    assert "observation_stimuli" in names


def test_optional_observation_stimuli_rejects_unknown_stimulus_id(tmp_path):
    artifact_dir = tmp_path / "dataset.psiz"
    write_dataset_artifact_from_samples(
        _build_samples(),
        artifact_dir,
        dataset_id="test_optional_observation_stimuli_rejects_unknown_stimulus_id",
    )

    bridge = pd.DataFrame(
        {
            "observation_id": [0],
            "x_feature_name": ["stimulus_set"],
            "position": [0],
            "stimulus_id": [999],
        }
    )
    _augment_with_stimuli_tables(
        artifact_dir,
        bridge_frame=bridge,
        stimulus_id_features=["stimulus_set"],
    )

    with pytest.raises(ValueError, match="unknown stimulus_id"):
        validate_dataset_artifact_directory(artifact_dir)


def test_optional_observation_stimuli_rejects_unknown_observation_id(tmp_path):
    artifact_dir = tmp_path / "dataset.psiz"
    write_dataset_artifact_from_samples(
        _build_samples(),
        artifact_dir,
        dataset_id="test_optional_observation_stimuli_rejects_unknown_observation_id",
    )

    bridge = pd.DataFrame(
        {
            "observation_id": [99],
            "x_feature_name": ["stimulus_set"],
            "position": [0],
            "stimulus_id": [1],
        }
    )
    _augment_with_stimuli_tables(
        artifact_dir,
        bridge_frame=bridge,
        stimulus_id_features=["stimulus_set"],
    )

    with pytest.raises(ValueError, match="unknown observation_id"):
        validate_dataset_artifact_directory(artifact_dir)


def test_runtime_contract_stimulus_id_features_subset_of_x_features(tmp_path):
    artifact_dir = tmp_path / "dataset.psiz"
    write_dataset_artifact_from_samples(
        _build_samples(),
        artifact_dir,
        dataset_id="test_runtime_contract_stimulus_id_features_subset_of_x_features",
    )

    _augment_with_stimuli_tables(
        artifact_dir,
        stimulus_id_features=["not_a_feature"],
    )

    with pytest.raises(ValueError, match="unknown x feature"):
        validate_dataset_artifact_directory(artifact_dir)


def test_runtime_contract_stimuli_table_reference_valid(tmp_path):
    artifact_dir = tmp_path / "dataset.psiz"
    write_dataset_artifact_from_samples(
        _build_samples(),
        artifact_dir,
        dataset_id="test_runtime_contract_stimuli_table_reference_valid",
    )

    _augment_with_stimuli_tables(
        artifact_dir,
        stimuli_table_ref="wrong_name",
    )

    with pytest.raises(ValueError, match="must reference 'stimuli'"):
        validate_dataset_artifact_directory(artifact_dir)


def test_optional_participants_table_schema_valid(tmp_path):
    artifact_dir = tmp_path / "dataset.psiz"
    write_dataset_artifact_from_samples(
        _build_samples(), artifact_dir, dataset_id="participants_valid"
    )
    _augment_with_participants_table(artifact_dir)

    validated = validate_dataset_artifact_directory(artifact_dir)
    participants = next(
        table for table in validated["manifest"]["tables"] if table["name"] == "participants"
    )
    assert participants["sensitive_columns"] == ["external_participant_id"]


def test_optional_participants_table_rejects_non_numeric_participant_id(tmp_path):
    artifact_dir = tmp_path / "dataset.psiz"
    write_dataset_artifact_from_samples(
        _build_samples(), artifact_dir, dataset_id="participants_non_numeric"
    )
    participants = pd.DataFrame(
        {"participant_id": ["worker-a", "worker-b"], "n_sequence": [1, 1], "n_trial": [1, 1]}
    )
    _augment_with_participants_table(artifact_dir, participants_frame=participants)

    with pytest.raises(ValueError, match="participant_id must use an integer dtype"):
        validate_dataset_artifact_directory(artifact_dir)


def test_optional_participants_table_rejects_duplicate_participant_id(tmp_path):
    artifact_dir = tmp_path / "dataset.psiz"
    write_dataset_artifact_from_samples(
        _build_samples(), artifact_dir, dataset_id="participants_duplicate"
    )
    participants = pd.DataFrame(
        {"participant_id": [10, 10], "n_sequence": [1, 1], "n_trial": [1, 1]}
    )
    _augment_with_participants_table(artifact_dir, participants_frame=participants)

    with pytest.raises(ValueError, match="Primary key rows are not unique"):
        validate_dataset_artifact_directory(artifact_dir)


def test_optional_participants_table_rejects_n_sequence_mismatch(tmp_path):
    artifact_dir = tmp_path / "dataset.psiz"
    write_dataset_artifact_from_samples(
        _build_samples(), artifact_dir, dataset_id="participants_sequence_mismatch"
    )
    participants = pd.DataFrame(
        {"participant_id": [10, 20], "n_sequence": [2, 1], "n_trial": [1, 1]}
    )
    _augment_with_participants_table(artifact_dir, participants_frame=participants)

    with pytest.raises(ValueError, match="n_sequence/n_trial"):
        validate_dataset_artifact_directory(artifact_dir)


def test_optional_participants_table_rejects_n_trial_mismatch(tmp_path):
    artifact_dir = tmp_path / "dataset.psiz"
    write_dataset_artifact_from_samples(
        _build_samples(), artifact_dir, dataset_id="participants_trial_mismatch"
    )
    participants = pd.DataFrame(
        {"participant_id": [10, 20], "n_sequence": [1, 1], "n_trial": [2, 1]}
    )
    _augment_with_participants_table(artifact_dir, participants_frame=participants)

    with pytest.raises(ValueError, match="n_sequence/n_trial"):
        validate_dataset_artifact_directory(artifact_dir)


def test_optional_participants_table_n_trial_uses_unique_sequence_timestep_pairs(tmp_path):
    artifact_dir = tmp_path / "dataset.psiz"
    write_dataset_artifact_from_samples(
        _build_samples(), artifact_dir, dataset_id="participants_trial_unique_timesteps"
    )
    _augment_with_participants_table(artifact_dir)

    observations_path = artifact_dir / "tables" / "observations.parquet"
    observations = pd.read_parquet(observations_path)
    duplicated = observations.iloc[[0]].copy()
    duplicated["observation_id"] = observations["observation_id"].max() + 1
    observations = pd.concat([observations, duplicated], ignore_index=True)
    observations.to_parquet(observations_path, index=False)

    manifest_path = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for table in manifest["tables"]:
        if table["name"] == "observations":
            table["row_count"] = int(len(observations))
            break
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    refresh_manifest_integrity_hashes(artifact_dir)

    # Same participant/sequence/timestep appears twice but should count as one trial.
    validate_dataset_artifact_directory(artifact_dir)


def test_optional_participants_table_rejects_unknown_participant_id(tmp_path):
    artifact_dir = tmp_path / "dataset.psiz"
    write_dataset_artifact_from_samples(
        _build_samples(), artifact_dir, dataset_id="participants_unknown"
    )
    _augment_with_participants_table(artifact_dir)
    observations_path = artifact_dir / "tables" / "observations.parquet"
    observations = pd.read_parquet(observations_path)
    observations.loc[0, "participant_id"] = 99
    observations.to_parquet(observations_path, index=False)
    refresh_manifest_integrity_hashes(artifact_dir)

    with pytest.raises(ValueError, match="unknown participant_id"):
        validate_dataset_artifact_directory(artifact_dir)


def test_optional_participants_table_preserves_optional_passthrough_columns(tmp_path):
    artifact_dir = tmp_path / "dataset.psiz"
    write_dataset_artifact_from_samples(
        _build_samples(), artifact_dir, dataset_id="participants_passthrough"
    )
    _augment_with_participants_table(artifact_dir)

    participants = pd.read_parquet(artifact_dir / "tables" / "participants.parquet")
    assert list(participants["external_participant_id"]) == ["worker-a", "worker-b"]


def test_optional_participants_table_sensitive_columns_metadata_valid(tmp_path):
    artifact_dir = tmp_path / "dataset.psiz"
    write_dataset_artifact_from_samples(
        _build_samples(), artifact_dir, dataset_id="participants_sensitive_metadata"
    )
    _augment_with_participants_table(artifact_dir)

    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    participants = next(table for table in manifest["tables"] if table["name"] == "participants")
    assert participants["sensitive_columns"] == ["external_participant_id"]


def test_participant_id_uses_numeric_surrogate_not_legacy_shim_string(tmp_path):
    artifact_dir = tmp_path / "dataset.psiz"
    write_dataset_artifact_from_samples(
        _build_samples(), artifact_dir, dataset_id="participants_numeric_writer"
    )
    _augment_with_participants_table(artifact_dir)

    participants = pd.read_parquet(artifact_dir / "tables" / "participants.parquet")
    assert str(participants["participant_id"].dtype) == "int64"
    assert not participants["participant_id"].map(lambda value: isinstance(value, str)).any()
