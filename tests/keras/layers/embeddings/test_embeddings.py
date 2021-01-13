# -*- coding: utf-8 -*-
# Copyright 2020 The PsiZ Authors. All Rights Reserved.
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
"""Module for testing embeddings."""

import pytest

import numpy as np
import tensorflow as tf

import psiz


@pytest.mark.parametrize("n_sample", [(None), (1), (3), (10)])
@pytest.mark.parametrize(
    'embedding_class', [
        psiz.keras.layers.EmbeddingDeterministic,
        psiz.keras.layers.EmbeddingGammaDiag,
        psiz.keras.layers.EmbeddingLaplaceDiag,
        psiz.keras.layers.EmbeddingLogNormalDiag,
        psiz.keras.layers.EmbeddingLogitNormalDiag,
        psiz.keras.layers.EmbeddingNormalDiag,
        psiz.keras.layers.EmbeddingTruncatedNormalDiag
    ]
)
def test_call_return_shape(n_sample, embedding_class):
    """Test call() return shape.

    Returned shape must include a `n_sample` dimension." Using
    tf.keras.layers.Embedding will fail because it does not include
    a sample dimension.

    """
    n_stimuli = 3
    n_dim = 2

    if n_sample is None:
        # Test default.
        n_sample = 1

        embedding = psiz.keras.layers.EmbeddingDeterministic(
            n_stimuli+1, n_dim, mask_zero=True
        )
    else:
        embedding = psiz.keras.layers.EmbeddingDeterministic(
            n_stimuli+1, n_dim, mask_zero=True, n_sample=n_sample
        )

    stimuli = psiz.keras.layers.Stimuli(embedding=embedding)

    batch_size = 3
    input_batch = [
        tf.constant(np.array([0, 1, 2])),
        tf.constant(np.array([[0], [0], [0]])),
    ]
    output = stimuli(input_batch)
    np.testing.assert_array_equal(
        np.shape(output.numpy()),
        np.array([n_sample, batch_size, n_dim])
    )
