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
"""Tests for EmbeddingNonCenteredNormalDiag."""

import keras
import numpy as np
import pytest

import psiz


@pytest.mark.backend_tensorflow
def test_call_shape_and_take_distribution():
    """Test non-centered call shape and take distribution shape."""
    base_embedding = psiz.keras.layers.EmbeddingNormalDiag(
        5,
        3,
        mask_zero=False,
    )
    emb = psiz.keras.layers.EmbeddingNonCenteredNormalDiag(
        base_embedding,
        epsilon_loc_gradient_scale=0.5,
        epsilon_scale_gradient_scale=0.25,
    )

    inputs = np.array([0, 1, 2], dtype=np.int32)
    outputs = emb(inputs)

    np.testing.assert_array_equal(np.shape(outputs), [3, 3])

    dist = emb.take(np.array([0, 1], dtype=np.int32))
    dist_mean = keras.ops.convert_to_numpy(dist.mean())
    np.testing.assert_array_equal(np.shape(dist_mean), [2, 3])


@pytest.mark.backend_tensorflow
def test_serialization_preserves_non_centered_config():
    """Test serialization round-trip preserves key non-centered fields."""
    base_embedding = psiz.keras.layers.EmbeddingNormalDiag(
        6,
        2,
        mask_zero=False,
    )

    emb = psiz.keras.layers.EmbeddingNonCenteredNormalDiag(
        base_embedding,
        epsilon_loc_trainable=True,
        epsilon_scale_trainable=False,
        epsilon_loc_gradient_scale=0.3,
        epsilon_scale_gradient_scale=0.7,
    )

    _ = emb(np.array([0, 1], dtype=np.int32))
    config = emb.get_config()

    recon_emb = psiz.keras.layers.EmbeddingNonCenteredNormalDiag.from_config(config)
    _ = recon_emb(np.array([0, 1], dtype=np.int32))

    assert recon_emb.epsilon_loc_trainable is True
    assert recon_emb.epsilon_scale_trainable is False
    assert recon_emb.epsilon_loc_gradient_scale == 0.3
    assert recon_emb.epsilon_scale_gradient_scale == 0.7

    mean_shape = keras.ops.convert_to_numpy(recon_emb.embeddings.mean()).shape
    np.testing.assert_array_equal(mean_shape, [6, 2])
