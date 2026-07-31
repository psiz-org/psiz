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
"""Tests for legacy .keras to .psiz migration APIs."""

from __future__ import annotations

import keras
import numpy as np
import pytest

from psiz.keras.layers.hierarchical_specs import HierarchyLevelSpec
from psiz.keras.layers.hierarchical_specs import HierarchySpec
from psiz.keras.layers.hierarchical_specs import MembershipInput
from psiz.keras.layers.hierarchical_vi_builder import build_hierarchical_vi_embedding
from psiz.keras.layers.posterior_factory import NonCenteredPosteriorFactory
from psiz.migration import UnsupportedLegacyFormatError
from psiz.migration import migrate_model_from_keras
from psiz.migration import validate_migration_report_schema
from psiz.storage import load_psiz_model


def _build_simple_dense_model():
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(3,), name="x"),
            keras.layers.Dense(4, activation="relu", name="hidden"),
            keras.layers.Dense(1, name="out"),
        ],
        name="simple_dense_model",
    )
    _ = model(np.zeros((2, 3), dtype=np.float32))
    model.set_weights(
        [
            np.full((3, 4), 0.2, dtype=np.float32),
            np.full((4,), 0.1, dtype=np.float32),
            np.full((4, 1), -0.3, dtype=np.float32),
            np.full((1,), 0.05, dtype=np.float32),
        ]
    )
    return model


def _build_shared_weight_model():
    x0 = keras.Input(shape=(3,), name="x0")
    x1 = keras.Input(shape=(3,), name="x1")
    shared = keras.layers.Dense(2, use_bias=False, name="shared")
    y = shared(x0) + shared(x1)
    model = keras.Model(inputs=[x0, x1], outputs=y, name="shared_weight_model")
    _ = model([np.zeros((1, 3), dtype=np.float32), np.zeros((1, 3), dtype=np.float32)])
    model.set_weights([np.full((3, 2), 0.25, dtype=np.float32)])
    return model


def _build_hierarchical_vi_model():
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
        n_sample_train=10,
    )

    x = keras.Input(shape=(), dtype="int32", name="stimulus")
    y = percept(x)
    model = keras.Model(inputs=x, outputs=y, name="hierarchical_vi_model")
    _ = model(np.array([1, 2, 3], dtype=np.int32))
    return model


def _save_legacy_keras_model(model, path):
    model.save(path)


@pytest.mark.backend_tensorflow
def test_migrate_model_from_keras_tensorflow(tmp_path):
    legacy_path = tmp_path / "legacy_tf.keras"
    destination_path = tmp_path / "migrated_tf.psiz"
    _save_legacy_keras_model(_build_simple_dense_model(), legacy_path)

    report = migrate_model_from_keras(
        legacy_path,
        destination_path,
        backend_override="tensorflow",
    )

    assert report["status"] == "success"
    assert report["resolved_backend"] == "tensorflow"


@pytest.mark.backend_torch
def test_migrate_model_from_keras_pytorch(tmp_path):
    legacy_path = tmp_path / "legacy_torch.keras"
    destination_path = tmp_path / "migrated_torch.psiz"
    _save_legacy_keras_model(_build_simple_dense_model(), legacy_path)

    report = migrate_model_from_keras(
        legacy_path,
        destination_path,
        backend_override="torch",
    )

    assert report["status"] == "success"
    assert report["resolved_backend"] == "torch"


@pytest.mark.backend_jax
def test_migrate_model_from_keras_jax(tmp_path):
    legacy_path = tmp_path / "legacy_jax.keras"
    destination_path = tmp_path / "migrated_jax.psiz"
    _save_legacy_keras_model(_build_simple_dense_model(), legacy_path)

    report = migrate_model_from_keras(
        legacy_path,
        destination_path,
        backend_override="jax",
    )

    assert report["status"] == "success"
    assert report["resolved_backend"] == "jax"


@pytest.mark.backend_tensorflow
def test_migrate_hierarchical_vi_model(tmp_path):
    legacy_path = tmp_path / "legacy_hierarchical.keras"
    destination_path = tmp_path / "migrated_hierarchical.psiz"
    model = _build_hierarchical_vi_model()
    x = np.array([1, 2, 3], dtype=np.int32)
    reference_outputs = keras.ops.convert_to_numpy(model(x))
    reference_loc = keras.ops.convert_to_numpy(model.layers[-1].prior.embeddings.distribution.loc)
    _save_legacy_keras_model(model, legacy_path)

    report = migrate_model_from_keras(
        legacy_path,
        destination_path,
        backend_override="tensorflow",
    )

    assert report["status"] == "success"
    assert report["model"]["class_name"] == "Functional"
    migrated = load_psiz_model(destination_path, backend_override="tensorflow")
    migrated_outputs = keras.ops.convert_to_numpy(migrated(x))
    migrated_loc = keras.ops.convert_to_numpy(
        migrated.layers[-1].prior.embeddings.distribution.loc
    )
    np.testing.assert_equal(reference_outputs.shape, migrated_outputs.shape)
    np.testing.assert_allclose(reference_loc, migrated_loc)
    assert np.all(np.isfinite(reference_loc))


@pytest.mark.backend_tensorflow
def test_migrate_shared_weight_graph(tmp_path):
    legacy_path = tmp_path / "legacy_shared.keras"
    destination_path = tmp_path / "migrated_shared.psiz"
    model = _build_shared_weight_model()
    _save_legacy_keras_model(model, legacy_path)

    report = migrate_model_from_keras(
        legacy_path,
        destination_path,
        backend_override="tensorflow",
    )

    assert report["status"] == "success"
    migrated = load_psiz_model(destination_path, backend_override="tensorflow")
    assert len(migrated.weights) == 1
    x0 = np.array([[0.3, 0.4, 0.5]], dtype=np.float32)
    x1 = np.array([[0.5, 0.6, 0.7]], dtype=np.float32)
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(model([x0, x1])),
        keras.ops.convert_to_numpy(migrated([x0, x1])),
    )


def test_migrate_invalid_input_format_errors(tmp_path):
    unsupported_path = tmp_path / "legacy_model.h5"
    unsupported_path.write_bytes(b"not-a-real-h5")

    with pytest.raises(UnsupportedLegacyFormatError, match="out of scope"):
        migrate_model_from_keras(unsupported_path, tmp_path / "bad_output.psiz")


@pytest.mark.backend_tensorflow
def test_migrate_validation_report_schema(tmp_path):
    legacy_path = tmp_path / "legacy_report.keras"
    destination_path = tmp_path / "migrated_report.psiz"
    _save_legacy_keras_model(_build_simple_dense_model(), legacy_path)

    report = migrate_model_from_keras(
        legacy_path,
        destination_path,
        backend_override="tensorflow",
    )
    validate_migration_report_schema(report)

    assert sorted(report["diagnostics"].keys()) == ["errors", "warnings"]


@pytest.mark.backend_tensorflow
def test_migrate_parity_with_fixed_seed(tmp_path):
    keras.utils.set_random_seed(411)
    legacy_path = tmp_path / "legacy_parity.keras"
    destination_path = tmp_path / "migrated_parity.psiz"
    model = _build_simple_dense_model()
    x = np.array([[0.1, 0.2, 0.3], [0.8, 0.4, 0.5]], dtype=np.float32)
    _save_legacy_keras_model(model, legacy_path)

    report = migrate_model_from_keras(
        legacy_path,
        destination_path,
        backend_override="tensorflow",
        validate_parity=True,
        parity_inputs=x,
        rtol=1e-6,
        atol=1e-6,
    )

    assert report["parity"]["enabled"] is True
    assert report["parity"]["validated"] is True
    assert report["parity"]["passed"] is True
