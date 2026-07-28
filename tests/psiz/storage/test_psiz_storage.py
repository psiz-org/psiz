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

    def call(self, inputs):
        return self.percept(inputs)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "percept": keras.saving.serialize_keras_object(self.percept),
            }
        )
        return config

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

    def call(self, inputs):
        return self.percept(inputs)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "percept": keras.saving.serialize_keras_object(self.percept),
            }
        )
        return config

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


def test_psiz_save_load_roundtrip_tensorflow(tmp_path):
    _save_load_roundtrip(tmp_path, "tensorflow")


def test_psiz_save_load_roundtrip_pytorch(tmp_path):
    _save_load_roundtrip(tmp_path, "torch")


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
