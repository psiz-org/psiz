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

import numpy as np
import pytest

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


@pytest.mark.tfp
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


@pytest.mark.tfp
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


@pytest.mark.tfp
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


@pytest.mark.tfp
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
