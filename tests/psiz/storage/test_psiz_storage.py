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
"""Tests for PsiZ .psiz storage save/load APIs."""

from __future__ import annotations

import json

import keras
import numpy as np
import pytest
from safetensors.numpy import load_file
from safetensors.numpy import save_file

from psiz.keras.layers.hierarchical_specs import HierarchyLevelSpec
from psiz.keras.layers.hierarchical_specs import HierarchySpec
from psiz.keras.layers.hierarchical_specs import MembershipInput
from psiz.keras.layers.hierarchical_vi_builder import build_hierarchical_vi_embedding
from psiz.keras.layers.posterior_factory import NonCenteredPosteriorFactory
from psiz.storage import load_psiz_model
from psiz.storage import save_psiz_model
from psiz.storage import validate_artifact_directory
from psiz.storage.schema import ArtifactSpecError


@keras.saving.register_keras_serializable(package="psiz.keras.tests", name="SimplePsizModel")
class SimplePsizModel(keras.Model):
    """Simple deterministic model for storage round-trip tests."""

    def __init__(self, n_hidden=4, **kwargs):
        super().__init__(**kwargs)
        self.n_hidden = int(n_hidden)
        self.hidden = keras.layers.Dense(n_hidden, activation="relu", name="hidden")
        self.out = keras.layers.Dense(1, use_bias=False, name="out")

    def call(self, inputs):
        hidden = self.hidden(inputs)
        return self.out(hidden)

    def get_config(self):
        config = super().get_config()
        config.update({"n_hidden": self.n_hidden})
        return config


@keras.saving.register_keras_serializable(
    package="psiz.keras.tests", name="SharedWeightIdentityModel"
)
class SharedWeightIdentityModel(keras.Model):
    """Model intentionally reusing one Dense instance on two branches."""

    def __init__(self, shared_dense=None, **kwargs):
        super().__init__(**kwargs)
        if shared_dense is None:
            shared_dense = keras.layers.Dense(3, activation=None, name="shared_dense")
        self.shared_dense_0 = shared_dense
        self.shared_dense_1 = shared_dense

    def call(self, inputs):
        x0, x1 = inputs
        return self.shared_dense_0(x0) + self.shared_dense_1(x1)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "shared_dense": keras.saving.serialize_keras_object(self.shared_dense_0),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config["shared_dense"] = keras.saving.deserialize_keras_object(
            config["shared_dense"]
        )
        return cls(**config)


@keras.saving.register_keras_serializable(
    package="psiz.keras.tests", name="SimpleVIAccessContractModelPsiz"
)
class SimpleVIAccessContractModelPsiz(keras.Model):
    """Small serializable wrapper used to freeze simple VI access continuity."""

    def __init__(self, percept=None, **kwargs):
        super().__init__(**kwargs)
        self.percept = percept
        self._build_input_shape = None

    def call(self, inputs):
        return self.percept(inputs)

    def build(self, input_shape):
        self._build_input_shape = input_shape
        if (
            self.percept is not None
            and hasattr(self.percept, "build")
            and not self.percept.built
        ):
            self.percept.build(input_shape)
        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "percept": keras.saving.serialize_keras_object(self.percept),
            }
        )
        return config

    def get_build_config(self):
        return {"input_shape": self._build_input_shape}

    def build_from_config(self, config):
        input_shape = config.get("input_shape", None)
        if input_shape is not None:
            self.build(input_shape)

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config["percept"] = keras.saving.deserialize_keras_object(config["percept"])
        return cls(**config)


@keras.saving.register_keras_serializable(
    package="psiz.keras.tests", name="HierarchicalVIAccessContractModelPsiz"
)
class HierarchicalVIAccessContractModelPsiz(keras.Model):
    """Small serializable wrapper used to freeze hierarchical VI continuity."""

    def __init__(self, percept=None, **kwargs):
        super().__init__(**kwargs)
        self.percept = percept
        self._build_input_shape = None

    def call(self, inputs):
        return self.percept(inputs)

    def build(self, input_shape):
        self._build_input_shape = input_shape
        if (
            self.percept is not None
            and hasattr(self.percept, "build")
            and not self.percept.built
        ):
            self.percept.build(input_shape)
        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "percept": keras.saving.serialize_keras_object(self.percept),
            }
        )
        return config

    def get_build_config(self):
        return {"input_shape": self._build_input_shape}

    def build_from_config(self, config):
        input_shape = config.get("input_shape", None)
        if input_shape is not None:
            self.build(input_shape)

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config["percept"] = keras.saving.deserialize_keras_object(config["percept"])
        return cls(**config)


def _build_simple_model():
    model = SimplePsizModel(n_hidden=4, name="simple_psiz")
    _ = model(np.zeros((2, 3), dtype=np.float32))
    model.set_weights(
        [
            np.full((3, 4), 0.15, dtype=np.float32),
            np.full((4,), -0.1, dtype=np.float32),
            np.full((4, 1), 0.2, dtype=np.float32),
        ]
    )
    return model


def _save_load_roundtrip(tmp_path, backend_name):
    model = _build_simple_model()
    x = np.array([[0.1, 0.2, 0.3], [0.7, 0.5, 0.4]], dtype=np.float32)
    y_expected = keras.ops.convert_to_numpy(model(x))

    artifact_dir = tmp_path / f"simple-{backend_name}.psiz"
    save_psiz_model(model, artifact_dir, backend_override=backend_name)

    loaded = load_psiz_model(artifact_dir, backend_override=backend_name)
    y_loaded = keras.ops.convert_to_numpy(loaded(x))

    np.testing.assert_allclose(y_loaded, y_expected, rtol=1e-6, atol=1e-6)


@pytest.mark.backend_tensorflow
def test_psiz_save_load_roundtrip_tensorflow(tmp_path):
    _save_load_roundtrip(tmp_path, "tensorflow")


@pytest.mark.backend_torch
def test_psiz_save_load_roundtrip_pytorch(tmp_path):
    _save_load_roundtrip(tmp_path, "torch")


@pytest.mark.backend_jax
def test_psiz_save_load_roundtrip_jax(tmp_path):
    _save_load_roundtrip(tmp_path, "jax")


@pytest.mark.backend_tensorflow
def test_psiz_save_load_simple_vi_structure(tmp_path):
    memberships = np.array([[0, 10], [0, 10], [0, 11], [0, 12]], dtype="int32")
    hierarchy = HierarchySpec(
        levels=[
            HierarchyLevelSpec(role="global", membership_key=None),
            HierarchyLevelSpec(role="leaf", membership_key="leaf_id"),
        ],
        mask_zero=True,
    )
    percept = build_hierarchical_vi_embedding(
        n_stimuli=4,
        n_dim=2,
        hierarchy=hierarchy,
        membership=MembershipInput(memberships=memberships),
        posterior_factory=NonCenteredPosteriorFactory(),
        n_sample_train=100,
    )
    model = SimpleVIAccessContractModelPsiz(percept=percept)

    inputs = np.array([1, 2, 3], dtype=np.int32)
    original_outputs = keras.ops.convert_to_numpy(model(inputs))
    original_loc = keras.ops.convert_to_numpy(model.percept.prior.embeddings.distribution.loc)

    artifact_dir = tmp_path / "hierarchical-vi.psiz"
    save_psiz_model(model, artifact_dir, backend_override="tensorflow")
    loaded = load_psiz_model(artifact_dir, backend_override="tensorflow")

    loaded_outputs = keras.ops.convert_to_numpy(loaded(inputs))
    loaded_loc = keras.ops.convert_to_numpy(
        loaded.percept.prior.embeddings.distribution.loc
    )

    np.testing.assert_equal(original_outputs.shape, loaded_outputs.shape)
    np.testing.assert_allclose(original_loc, loaded_loc)


@pytest.mark.backend_tensorflow
def test_psiz_save_load_hierarchical_vi_structure(tmp_path):
    memberships = np.array(
        [
            [0, 10, 100],
            [0, 10, 101],
            [0, 11, 110],
            [0, 11, 111],
            [0, 12, 120],
            [0, 12, 121],
        ],
        dtype="int32",
    )
    hierarchy = HierarchySpec(
        levels=[
            HierarchyLevelSpec(role="global", membership_key=None),
            HierarchyLevelSpec(role="intermediate", membership_key="branch_id"),
            HierarchyLevelSpec(role="leaf", membership_key="leaf_id"),
        ],
        mask_zero=True,
    )
    percept = build_hierarchical_vi_embedding(
        n_stimuli=6,
        n_dim=2,
        hierarchy=hierarchy,
        membership=MembershipInput(memberships=memberships),
        posterior_factory=NonCenteredPosteriorFactory(),
        n_sample_train=100,
    )
    model = HierarchicalVIAccessContractModelPsiz(percept=percept)

    inputs = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    original_outputs = keras.ops.convert_to_numpy(model(inputs))
    original_loc = keras.ops.convert_to_numpy(
        model.percept.prior.embeddings.distribution.loc
    )

    artifact_dir = tmp_path / "hierarchical-3level-vi.psiz"
    save_psiz_model(model, artifact_dir, backend_override="tensorflow")
    loaded = load_psiz_model(artifact_dir, backend_override="tensorflow")

    loaded_outputs = keras.ops.convert_to_numpy(loaded(inputs))
    loaded_loc = keras.ops.convert_to_numpy(
        loaded.percept.prior.embeddings.distribution.loc
    )

    np.testing.assert_equal(original_outputs.shape, loaded_outputs.shape)
    assert np.all(np.isfinite(loaded_outputs))
    np.testing.assert_allclose(original_loc, loaded_loc)


@pytest.mark.backend_tensorflow
def test_psiz_save_load_shared_weight_identity(tmp_path):
    model = SharedWeightIdentityModel()
    x0 = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
    x1 = np.array([[0.5, 0.6, 0.7]], dtype=np.float32)
    _ = model([x0, x1])

    artifact_dir = tmp_path / "shared-weight.psiz"
    save_psiz_model(model, artifact_dir, backend_override="tensorflow")
    loaded = load_psiz_model(artifact_dir, backend_override="tensorflow")

    assert loaded.shared_dense_0 is loaded.shared_dense_1
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(model([x0, x1])),
        keras.ops.convert_to_numpy(loaded([x0, x1])),
    )


def test_psiz_schema_version_gate(tmp_path):
    model = _build_simple_model()
    artifact_dir = tmp_path / "version-gate.psiz"
    save_psiz_model(model, artifact_dir)

    config_path = artifact_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["format_version"] = "2.0.0"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ArtifactSpecError, match="major version"):
        _ = load_psiz_model(artifact_dir)


def test_psiz_missing_file_errors(tmp_path):
    model = _build_simple_model()
    artifact_dir = tmp_path / "missing-file.psiz"
    save_psiz_model(model, artifact_dir)

    (artifact_dir / "model.safetensors").unlink()

    with pytest.raises(ArtifactSpecError, match="Missing required files"):
        _ = validate_artifact_directory(artifact_dir)


def test_psiz_index_weight_key_integrity(tmp_path):
    model = _build_simple_model()
    artifact_dir = tmp_path / "weight-integrity.psiz"
    save_psiz_model(model, artifact_dir)

    index_path = artifact_dir / "model_index.json"
    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    index_payload["weights"][0]["key"] = "not_a_real_key"
    index_path.write_text(
        json.dumps(index_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ArtifactSpecError, match="integrity"):
        _ = load_psiz_model(artifact_dir)


@pytest.mark.backend_tensorflow
def test_psiz_externalizes_hierarchical_membership_payloads(tmp_path):
    memberships = np.array(
        [
            [0, 10, 100],
            [0, 10, 101],
            [0, 11, 110],
            [0, 11, 111],
            [0, 12, 120],
            [0, 12, 121],
        ],
        dtype="int32",
    )
    hierarchy = HierarchySpec(
        levels=[
            HierarchyLevelSpec(role="global", membership_key=None),
            HierarchyLevelSpec(role="intermediate", membership_key="branch_id"),
            HierarchyLevelSpec(role="leaf", membership_key="leaf_id"),
        ],
        mask_zero=True,
    )
    percept = build_hierarchical_vi_embedding(
        n_stimuli=6,
        n_dim=2,
        hierarchy=hierarchy,
        membership=MembershipInput(memberships=memberships),
        posterior_factory=NonCenteredPosteriorFactory(),
        n_sample_train=100,
    )
    model = HierarchicalVIAccessContractModelPsiz(percept=percept)

    artifact_dir = tmp_path / "compaction-hierarchical.psiz"
    save_psiz_model(model, artifact_dir, backend_override="tensorflow")

    config_path = artifact_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    compaction = config.get("model_config_compaction")
    assert isinstance(compaction, dict)
    assert compaction["blob_count"] > 0

    blob_path = artifact_dir / compaction["blob_file"]
    assert blob_path.exists()

    config_text = config_path.read_text(encoding="utf-8")
    assert "__psiz_external_blob__" in config_text


@pytest.mark.backend_tensorflow
def test_psiz_missing_compaction_blob_fails_validation(tmp_path):
    memberships = np.array(
        [
            [0, 10],
            [0, 11],
            [0, 12],
            [0, 13],
        ],
        dtype="int32",
    )
    hierarchy = HierarchySpec(
        levels=[
            HierarchyLevelSpec(role="global", membership_key=None),
            HierarchyLevelSpec(role="leaf", membership_key="leaf_id"),
        ],
        mask_zero=True,
    )
    percept = build_hierarchical_vi_embedding(
        n_stimuli=4,
        n_dim=2,
        hierarchy=hierarchy,
        membership=MembershipInput(memberships=memberships),
        posterior_factory=NonCenteredPosteriorFactory(),
        n_sample_train=100,
    )
    model = SimpleVIAccessContractModelPsiz(percept=percept)

    artifact_dir = tmp_path / "missing-compaction-blob.psiz"
    save_psiz_model(model, artifact_dir, backend_override="tensorflow")

    config = json.loads((artifact_dir / "config.json").read_text(encoding="utf-8"))
    blob_file = config["model_config_compaction"]["blob_file"]
    (artifact_dir / blob_file).unlink()

    with pytest.raises(ArtifactSpecError, match="compaction blob file"):
        _ = validate_artifact_directory(artifact_dir)


@pytest.mark.backend_tensorflow
def test_psiz_externalizes_large_constant_initializer_payloads(tmp_path):
    constant_value = np.full((128, 128), 0.125, dtype=np.float32)
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(128,), name="x"),
            keras.layers.Dense(
                128,
                kernel_initializer=keras.initializers.Constant(constant_value),
                bias_initializer="zeros",
                name="dense_constant",
            ),
        ],
        name="constant_initializer_model",
    )
    _ = model(np.zeros((2, 128), dtype=np.float32))

    artifact_dir = tmp_path / "constant-init-compaction.psiz"
    save_psiz_model(model, artifact_dir, backend_override="tensorflow")

    config_path = artifact_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    compaction = config.get("model_config_compaction")
    assert isinstance(compaction, dict)
    assert compaction["blob_count"] > 0
    assert compaction["externalized_json_estimate_bytes"] > 0

    blob_path = artifact_dir / compaction["blob_file"]
    assert blob_path.exists()

    loaded = load_psiz_model(artifact_dir, backend_override="tensorflow")
    x = np.ones((1, 128), dtype=np.float32)
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(model(x)),
        keras.ops.convert_to_numpy(loaded(x)),
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.backend_tensorflow
def test_psiz_externalized_blob_dtype_mismatch_fails_load(tmp_path):
    memberships = np.array(
        [
            [0, 10],
            [0, 11],
            [0, 12],
            [0, 13],
        ],
        dtype="int32",
    )
    hierarchy = HierarchySpec(
        levels=[
            HierarchyLevelSpec(role="global", membership_key=None),
            HierarchyLevelSpec(role="leaf", membership_key="leaf_id"),
        ],
        mask_zero=True,
    )
    percept = build_hierarchical_vi_embedding(
        n_stimuli=4,
        n_dim=2,
        hierarchy=hierarchy,
        membership=MembershipInput(memberships=memberships),
        posterior_factory=NonCenteredPosteriorFactory(),
        n_sample_train=100,
    )
    model = SimpleVIAccessContractModelPsiz(percept=percept)

    artifact_dir = tmp_path / "dtype-mismatch.psiz"
    save_psiz_model(model, artifact_dir, backend_override="tensorflow")

    config = json.loads((artifact_dir / "config.json").read_text(encoding="utf-8"))
    blob_file = config["model_config_compaction"]["blob_file"]
    blob_path = artifact_dir / blob_file

    tensors = load_file(str(blob_path))
    first_key = sorted(tensors.keys())[0]
    tensors[first_key] = tensors[first_key].astype(np.float32)
    save_file(tensors, str(blob_path))

    with pytest.raises(ArtifactSpecError, match="dtype mismatch"):
        _ = load_psiz_model(artifact_dir, backend_override="tensorflow")
