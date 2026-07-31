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
"""Tests for hierarchical VI builder APIs."""

import keras
import numpy as np
import pytest

import psiz
from psiz.stochastic.transforms import softplus_inverse

from psiz.keras.layers.embeddings.embedding_shared import EmbeddingShared
from psiz.keras.layers.embeddings.embedding_take import EmbeddingTake
from psiz.keras.layers.embeddings.embedding_variational import EmbeddingVariational
from psiz.keras.layers.embeddings.normal_diag import EmbeddingNormalDiag
from psiz.keras.layers.hierarchical_specs import HierarchyLevelSpec
from psiz.keras.layers.hierarchical_specs import HierarchySpec
from psiz.keras.layers.hierarchical_specs import KLWeightingPolicy
from psiz.keras.layers.hierarchical_specs import MembershipInput
from psiz.keras.layers.hierarchical_specs import MembershipSourcePolicy
from psiz.keras.layers.hierarchical_specs import ParentMapPolicy
from psiz.keras.layers.hierarchical_specs import ScaleInitializationPolicy
from psiz.keras.layers.hierarchical_specs import LevelInitializationMode
from psiz.keras.layers.hierarchical_vi_builder import build_hierarchical_vi_embedding
from psiz.keras.layers.hierarchical_vi_builder import (
    AdvancedHierarchicalVIEmbeddingBuilder,
)
from psiz.keras.layers.hierarchical_vi_builder import HierarchicalVIEmbeddingBuilder
from psiz.keras.layers.hierarchical_pretrained_hooks import (
    PretrainedNonCenteredFactoryHooks,
)
from psiz.keras.layers.posterior_factory import NonCenteredPosteriorFactory


@keras.saving.register_keras_serializable(
    package="psiz.keras.tests", name="HierarchyAccessContractModel"
)
class HierarchyAccessContractModel(keras.Model):
    """Small serializable wrapper used to freeze access-path continuity."""

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


def _build_hierarchy_spec():
    return HierarchySpec(
        levels=[
            HierarchyLevelSpec(role="global", membership_key=None),
            HierarchyLevelSpec(role="leaf", membership_key="leaf_id"),
        ],
        mask_zero=True,
    )


def test_locked_enum_values():
    """Ensure locked enum values are stable."""
    assert KLWeightingPolicy.PER_SAMPLE_PER_BRANCH.value == "per_sample_per_branch"
    assert KLWeightingPolicy.PER_SAMPLE.value == "per_sample"
    assert KLWeightingPolicy.PER_SAMPLE_PER_LEVEL.value == "per_sample_per_level"
    assert KLWeightingPolicy.CUSTOM.value == "custom"

    assert ScaleInitializationPolicy.GEOMETRIC_DECAY.value == "geometric_decay"
    assert ScaleInitializationPolicy.CONSTANT.value == "constant"
    assert ScaleInitializationPolicy.CUSTOM.value == "custom"

    assert MembershipSourcePolicy.PRECOMPUTED.value == "precomputed"
    assert MembershipSourcePolicy.DATAFRAME_RESOLVER.value == "dataframe_resolver"
    assert MembershipSourcePolicy.CUSTOM.value == "custom"
    assert (
        MembershipSourcePolicy.STRICT_PRECOMPUTED_OVERRIDES_RESOLVER.value
        == "strict_precomputed_overrides_resolver"
    )

    assert ParentMapPolicy.MINIMAL_FIRST_OCCURRENCE.value == "minimal_first_occurrence"
    assert ParentMapPolicy.FULL_IDENTITY.value == "full_identity"
    assert ParentMapPolicy.CUSTOM.value == "custom"


@pytest.mark.backend_tensorflow
def test_function_builder_builds_and_calls():
    """Test simple function wrapper creates callable layer stack."""
    memberships = np.array(
        [
            [0, 10],
            [0, 10],
            [0, 11],
            [0, 12],
        ],
        dtype="int32",
    )

    model_layer = build_hierarchical_vi_embedding(
        n_stimuli=4,
        n_dim=2,
        hierarchy=_build_hierarchy_spec(),
        membership=MembershipInput(memberships=memberships),
        posterior_factory=NonCenteredPosteriorFactory(),
        n_sample_train=100,
    )

    outputs = model_layer(np.array([1, 2, 3], dtype=np.int32))
    assert outputs.shape == (3, 2)
    assert len(model_layer.losses) >= 1


@pytest.mark.backend_tensorflow
def test_class_builder_serialization_roundtrip():
    """Test builder config round-trip."""
    builder = HierarchicalVIEmbeddingBuilder(
        hierarchy=_build_hierarchy_spec(),
        posterior_factory=NonCenteredPosteriorFactory(),
        kl_policy=KLWeightingPolicy.PER_SAMPLE_PER_BRANCH,
    )

    config = builder.get_config()
    reconstructed = HierarchicalVIEmbeddingBuilder.from_config(config)

    assert reconstructed.kl_policy == KLWeightingPolicy.PER_SAMPLE_PER_BRANCH
    assert reconstructed.scale_policy == ScaleInitializationPolicy.GEOMETRIC_DECAY
    assert (
        reconstructed.membership_source_policy
        == MembershipSourcePolicy.PRECOMPUTED
    )


def test_precomputed_policy_requires_memberships():
    """Ensure precomputed policy enforces required membership matrix."""
    builder = HierarchicalVIEmbeddingBuilder(
        hierarchy=_build_hierarchy_spec(),
        posterior_factory=NonCenteredPosteriorFactory(),
        membership_source_policy=MembershipSourcePolicy.PRECOMPUTED,
    )

    with pytest.raises(ValueError):
        builder.build(
            n_stimuli=4,
            n_dim=2,
            membership=MembershipInput(memberships=None),
            n_sample_train=100,
        )


def test_custom_policy_requires_callback():
    """Ensure custom policy requires custom callback."""
    with pytest.raises(ValueError):
        _ = HierarchicalVIEmbeddingBuilder(
            hierarchy=_build_hierarchy_spec(),
            posterior_factory=NonCenteredPosteriorFactory(),
            membership_source_policy=MembershipSourcePolicy.CUSTOM,
            custom_membership_resolver=None,
        )


@pytest.mark.backend_tensorflow
def test_default_builder_pretrained_mode_requires_hook():
    """Ensure default builder still protects pretrained modes."""
    hierarchy = HierarchySpec(
        levels=[
            HierarchyLevelSpec(role="global", membership_key=None),
            HierarchyLevelSpec(
                role="leaf",
                membership_key="leaf_id",
                initialization=LevelInitializationMode.PRETRAINED,
                metadata={
                    "pretrained_epsilon_loc": np.zeros((3, 2), dtype="float32"),
                    "pretrained_epsilon_scale": np.ones((3, 2), dtype="float32"),
                },
            ),
        ],
        mask_zero=True,
    )

    memberships = np.array(
        [
            [0, 10],
            [0, 10],
            [0, 11],
            [0, 12],
        ],
        dtype="int32",
    )

    builder = HierarchicalVIEmbeddingBuilder(
        hierarchy=hierarchy,
        posterior_factory=NonCenteredPosteriorFactory(),
    )

    with pytest.raises(NotImplementedError):
        builder.build(
            n_stimuli=4,
            n_dim=2,
            membership=MembershipInput(memberships=memberships),
            n_sample_train=100,
        )


@pytest.mark.backend_tensorflow
def test_advanced_builder_pretrained_hook_builds_and_calls():
    """Ensure pretrained and point-estimate modes are enabled by hooks."""
    memberships = np.array(
        [
            [0, 10],
            [0, 10],
            [0, 11],
            [0, 12],
        ],
        dtype="int32",
    )

    hierarchy_pretrained = HierarchySpec(
        levels=[
            HierarchyLevelSpec(role="global", membership_key=None),
            HierarchyLevelSpec(
                role="leaf",
                membership_key="leaf_id",
                initialization=LevelInitializationMode.PRETRAINED,
                metadata={
                    "pretrained_epsilon_loc": np.array(
                        [
                            [0.2, -0.1],
                            [0.0, 0.0],
                            [-0.2, 0.1],
                        ],
                        dtype="float32",
                    ),
                    "pretrained_epsilon_scale": np.ones((3, 2), dtype="float32"),
                },
            ),
        ],
        mask_zero=True,
    )

    hierarchy_point = HierarchySpec(
        levels=[
            HierarchyLevelSpec(role="global", membership_key=None),
            HierarchyLevelSpec(
                role="leaf",
                membership_key="leaf_id",
                initialization=LevelInitializationMode.PRETRAINED_POINT_ESTIMATE,
                metadata={
                    "pretrained_epsilon_loc": np.array(
                        [
                            [0.1, -0.3],
                            [0.3, 0.1],
                            [-0.1, 0.4],
                        ],
                        dtype="float32",
                    )
                },
            ),
        ],
        mask_zero=True,
    )

    hooks = PretrainedNonCenteredFactoryHooks()

    for hierarchy in [hierarchy_pretrained, hierarchy_point]:
        builder = AdvancedHierarchicalVIEmbeddingBuilder(
            hierarchy=hierarchy,
            posterior_factory=NonCenteredPosteriorFactory(),
            hooks=hooks,
        )
        model_layer = builder.build(
            n_stimuli=4,
            n_dim=2,
            membership=MembershipInput(memberships=memberships),
            n_sample_train=100,
        )
        outputs = model_layer(np.array([1, 2, 3], dtype=np.int32))
        assert outputs.shape == (3, 2)


@pytest.mark.backend_tensorflow
def test_hierarchical_contract_preserves_nested_prior_chain():
    """Freeze the nested layer access chain used by downstream code."""
    memberships = np.array(
        [
            [0, 10],
            [0, 10],
            [0, 11],
            [0, 12],
        ],
        dtype="int32",
    )

    model_layer = build_hierarchical_vi_embedding(
        n_stimuli=4,
        n_dim=2,
        hierarchy=_build_hierarchy_spec(),
        membership=MembershipInput(memberships=memberships),
        posterior_factory=NonCenteredPosteriorFactory(),
        n_sample_train=100,
    )

    assert isinstance(model_layer.prior, EmbeddingTake)

    loc = keras.ops.convert_to_numpy(model_layer.prior.embeddings.distribution.loc)
    scale = keras.ops.convert_to_numpy(
        model_layer.prior.embeddings.distribution.scale
    )

    np.testing.assert_equal(loc.shape, (5, 2))
    np.testing.assert_allclose(loc, np.zeros((5, 2), dtype="float32"))
    np.testing.assert_equal(scale.shape, (5, 2))
    np.testing.assert_array_less(np.zeros_like(scale), scale)

    outputs = keras.ops.convert_to_numpy(
        model_layer(np.array([1, 2, 3], dtype=np.int32))
    )
    np.testing.assert_equal(outputs.shape, (3, 2))


@pytest.mark.backend_tensorflow
def test_variational_contract_preserves_distribution_chain():
    """Freeze the public distribution access pattern used in examples."""
    n_stimuli = 10
    n_dim = 3
    prior_scale = 0.2

    posterior = EmbeddingNormalDiag(
        n_stimuli,
        n_dim,
        mask_zero=False,
        scale_initializer=keras.initializers.Constant(
            keras.ops.convert_to_numpy(softplus_inverse(prior_scale))
        ),
    )
    prior = EmbeddingShared(
        n_stimuli,
        n_dim,
        mask_zero=False,
        embedding=EmbeddingNormalDiag(
            1,
            1,
            loc_initializer=keras.initializers.Constant(0.0),
            scale_initializer=keras.initializers.Constant(
                keras.ops.convert_to_numpy(softplus_inverse(prior_scale))
            ),
            loc_trainable=False,
        ),
    )
    layer = EmbeddingVariational(
        posterior=posterior,
        prior=prior,
        kl_weight=0.1,
        kl_n_sample=30,
    )

    layer.build([None, n_dim])

    dist = layer.prior.embeddings.distribution
    scale = keras.ops.convert_to_numpy(dist.distribution.distribution.scale)

    np.testing.assert_equal(scale.shape, (1, 1))
    np.testing.assert_array_less(np.zeros_like(scale), scale)

    outputs = keras.ops.convert_to_numpy(
        layer(np.array([0, 1, 2], dtype=np.int32))
    )
    np.testing.assert_equal(outputs.shape, (3, n_dim))


@pytest.mark.backend_tensorflow
def test_contract_keras_save_load_access_continuity(tmp_path):
    """Freeze Keras save/load continuity for the hierarchical access chain."""
    memberships = np.array(
        [
            [0, 10],
            [0, 10],
            [0, 11],
            [0, 12],
        ],
        dtype="int32",
    )
    percept = build_hierarchical_vi_embedding(
        n_stimuli=4,
        n_dim=2,
        hierarchy=_build_hierarchy_spec(),
        membership=MembershipInput(memberships=memberships),
        posterior_factory=NonCenteredPosteriorFactory(),
        n_sample_train=100,
    )
    model = HierarchyAccessContractModel(percept=percept)

    inputs = np.array([1, 2, 3], dtype=np.int32)
    original_outputs = keras.ops.convert_to_numpy(model(inputs))
    original_loc = keras.ops.convert_to_numpy(
        model.percept.prior.embeddings.distribution.loc
    )

    fp_model = tmp_path / "hierarchy_access_contract.keras"
    model.save(fp_model)

    loaded = keras.models.load_model(
        fp_model,
        custom_objects={"HierarchyAccessContractModel": HierarchyAccessContractModel},
    )

    loaded_outputs = keras.ops.convert_to_numpy(loaded(inputs))
    loaded_loc = keras.ops.convert_to_numpy(
        loaded.percept.prior.embeddings.distribution.loc
    )

    np.testing.assert_equal(original_outputs.shape, loaded_outputs.shape)
    np.testing.assert_allclose(original_loc, loaded_loc)


@pytest.mark.backend_tensorflow
def test_contract_psiz_save_load_access_continuity(tmp_path):
    """Freeze save/load continuity for the hierarchical access chain."""
    memberships = np.array(
        [
            [0, 10],
            [0, 10],
            [0, 11],
            [0, 12],
        ],
        dtype="int32",
    )
    percept = build_hierarchical_vi_embedding(
        n_stimuli=4,
        n_dim=2,
        hierarchy=_build_hierarchy_spec(),
        membership=MembershipInput(memberships=memberships),
        posterior_factory=NonCenteredPosteriorFactory(),
        n_sample_train=100,
    )
    model = HierarchyAccessContractModel(percept=percept)

    inputs = np.array([1, 2, 3], dtype=np.int32)
    original_outputs = keras.ops.convert_to_numpy(model(inputs))
    original_loc = keras.ops.convert_to_numpy(model.percept.prior.embeddings.distribution.loc)

    fp_model = tmp_path / "hierarchy_access_contract.psiz"
    psiz.keras.save_psiz_model(model, fp_model)

    loaded = psiz.keras.load_psiz_model(
        fp_model,
        custom_objects={"HierarchyAccessContractModel": HierarchyAccessContractModel},
    )

    loaded_outputs = keras.ops.convert_to_numpy(loaded(inputs))
    loaded_loc = keras.ops.convert_to_numpy(
        loaded.percept.prior.embeddings.distribution.loc
    )

    np.testing.assert_equal(original_outputs.shape, loaded_outputs.shape)
    np.testing.assert_allclose(original_loc, loaded_loc)


@pytest.mark.backend_tensorflow
def test_contract_stochastic_sample_shape():
    """Freeze stochastic sample shape semantics for distribution embeddings."""
    embedding = EmbeddingNormalDiag(
        10,
        2,
        mask_zero=False,
        sample_shape=(2, 4),
    )

    outputs = keras.ops.convert_to_numpy(embedding(np.array([0, 1, 2], dtype=np.int32)))

    np.testing.assert_equal(outputs.shape, (2, 4, 3, 2))


@pytest.mark.backend_tensorflow
def test_contract_invalid_path_errors_are_clear():
    """Freeze the failure mode for unsupported hierarchical hops."""
    memberships = np.array(
        [
            [0, 10],
            [0, 10],
            [0, 11],
            [0, 12],
        ],
        dtype="int32",
    )

    model_layer = build_hierarchical_vi_embedding(
        n_stimuli=4,
        n_dim=2,
        hierarchy=_build_hierarchy_spec(),
        membership=MembershipInput(memberships=memberships),
        posterior_factory=NonCenteredPosteriorFactory(),
        n_sample_train=100,
    )

    with pytest.raises(AttributeError, match="prior"):
        _ = model_layer.prior.prior
