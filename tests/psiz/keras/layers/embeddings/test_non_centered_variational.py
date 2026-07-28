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
"""Tests for EmbeddingNonCenteredVariational."""

import numpy as np
import pytest

from psiz.keras.layers.embeddings.embedding_take import EmbeddingTake
from psiz.keras.layers.embeddings.non_centered_variational import (
    EmbeddingNonCenteredVariational,
)
from psiz.keras.layers.embeddings.normal_diag import EmbeddingNormalDiag
from psiz.keras.layers.posterior_factory import NonCenteredPosteriorFactory


@pytest.mark.backend_tensorflow
def test_roundtrip_preserves_factory_configuration():
    """Ensure config round-trip reconstructs posterior factory."""
    prior_core = EmbeddingNormalDiag(
        input_dim=3,
        output_dim=2,
        mask_zero=False,
    )
    prior_full = EmbeddingTake(
        embedding=prior_core,
        input_map=np.array([0, 0, 1, 2], dtype="int32"),
    )

    posterior_factory = NonCenteredPosteriorFactory(
        epsilon_loc_gradient_scale=0.25,
        epsilon_scale_gradient_scale=0.5,
        epsilon_loc_trainable=True,
        epsilon_scale_trainable=False,
    )

    layer = EmbeddingNonCenteredVariational(
        prior_full=prior_full,
        membership_parent=np.array([0, 0, 1, 2], dtype="int32"),
        membership_current=np.array([0, 1, 2, 3], dtype="int32"),
        posterior_factory=posterior_factory,
        kl_weight=0.3,
        kl_use_exact=True,
        kl_n_sample=11,
    )

    config = layer.get_config()
    layer_reconstructed = EmbeddingNonCenteredVariational.from_config(config)

    assert isinstance(
        layer_reconstructed.posterior_factory, NonCenteredPosteriorFactory
    )
    assert np.array_equal(
        layer_reconstructed.membership_parent,
        np.array([0, 0, 1, 2], dtype="int32"),
    )
    assert np.array_equal(
        layer_reconstructed.membership_current,
        np.array([0, 1, 2, 3], dtype="int32"),
    )
    assert layer_reconstructed.kl_weight == 0.3
    assert layer_reconstructed.kl_use_exact is True
    assert layer_reconstructed.kl_n_sample == 11

    reconstructed_factory = layer_reconstructed.posterior_factory
    assert reconstructed_factory.epsilon_loc_gradient_scale == 0.25
    assert reconstructed_factory.epsilon_scale_gradient_scale == 0.5
    assert reconstructed_factory.epsilon_loc_trainable is True
    assert reconstructed_factory.epsilon_scale_trainable is False


@pytest.mark.backend_tensorflow
def test_posterior_full_map_recovers_destination_ids_all_levels():
    """Ensure remap recovers destination IDs for non-monotonic levels."""
    base_memberships = np.array(
        [
            [0, 0, 1, 19, 233],
            [0, 0, 1, 19, 233],
            [0, 1, 5, 7, 929],
            [0, 0, 1, 19, 366],
            [0, 0, 1, 19, 237],
            [0, 0, 1, 19, 237],
            [0, 1, 0, 34, 411],
            [0, 0, 1, 4, 84],
            [0, 0, 1, 19, 155],
            [0, 0, 1, 19, 155],
        ],
        dtype=np.int32,
    )
    memberships = np.tile(base_memberships, (3, 1))
    leaf_ids = np.arange(memberships.shape[0], dtype=np.int32).reshape(-1, 1)
    memberships = np.concatenate([memberships, leaf_ids], axis=1)

    layer = object.__new__(EmbeddingNonCenteredVariational)

    for i_level in range(1, memberships.shape[1]):
        membership_current = memberships[:, i_level]
        posterior_map_full = layer._build_posterior_full_map(membership_current)

        _, first_indices = np.unique(membership_current, return_index=True)
        classes_in_minimal_order = membership_current[np.sort(first_indices)]
        recovered_classes = classes_in_minimal_order[posterior_map_full]
        mismatch = int(np.sum(recovered_classes != membership_current))

        assert mismatch == 0


@pytest.mark.backend_tensorflow
def test_call_registers_kl_loss():
    """Ensure call adds non-centered KL loss term."""
    prior_core = EmbeddingNormalDiag(
        input_dim=3,
        output_dim=2,
        mask_zero=False,
    )
    prior_full = EmbeddingTake(
        embedding=prior_core,
        input_map=np.array([0, 0, 1, 2], dtype="int32"),
    )

    layer = EmbeddingNonCenteredVariational(
        prior_full=prior_full,
        membership_parent=np.array([0, 0, 1, 2], dtype="int32"),
        membership_current=np.array([0, 1, 2, 3], dtype="int32"),
        posterior_factory=NonCenteredPosteriorFactory(
            epsilon_loc_gradient_scale=1.0,
            epsilon_scale_gradient_scale=1.0,
        ),
        kl_weight=0.2,
        kl_use_exact=False,
        kl_n_sample=5,
    )

    outputs = layer(np.array([0, 1, 2], dtype=np.int32))

    assert outputs.shape == (3, 2)
    assert len(layer.losses) >= 1
