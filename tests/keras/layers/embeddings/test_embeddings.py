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


@pytest.fixture
def emb_input_1d():
    """Create a 1D input."""
    batch = tf.constant(np.array([0, 1, 2]))
    return batch


@pytest.fixture
def emb_input_2d():
    """Create a 2D input."""
    batch = tf.constant(np.array([[0, 1], [2, 3], [4, 5]]))
    return batch


@pytest.mark.parametrize("n_sample", [(None), (1), (3), (10)])
@pytest.mark.parametrize(
    'embedding_class', [
        psiz.keras.layers.EmbeddingGammaDiag,
        psiz.keras.layers.EmbeddingLaplaceDiag,
        psiz.keras.layers.EmbeddingLogNormalDiag,
        psiz.keras.layers.EmbeddingLogitNormalDiag,
        psiz.keras.layers.EmbeddingNormalDiag,
        psiz.keras.layers.EmbeddingTruncatedNormalDiag
    ]
)
def test_call_1d_input(emb_input_1d, n_sample, embedding_class):
    """Test call() return shape.

    Returned shape must include a `n_sample` dimension." Using
    tf.keras.layers.Embedding will fail because it does not include
    a sample dimension.

    """
    input = emb_input_1d
    batch_shape_0 = 3
    n_stimuli = 10
    n_dim = 2

    if n_sample is None:
        # Test default.
        embedding = embedding_class(
            n_stimuli, n_dim, mask_zero=True
        )
        desired_output_shape = np.array([batch_shape_0, n_dim])
    else:
        embedding = embedding_class(
            n_stimuli, n_dim, mask_zero=True, n_sample=n_sample
        )
        desired_output_shape = np.array([n_sample, batch_shape_0, n_dim])

    output = embedding(input)
    np.testing.assert_array_equal(
        np.shape(output.numpy()),
        desired_output_shape
    )


@pytest.mark.parametrize("n_sample", [(None), (1), (3), (10)])
@pytest.mark.parametrize(
    'embedding_class', [
        psiz.keras.layers.EmbeddingGammaDiag,
        psiz.keras.layers.EmbeddingLaplaceDiag,
        psiz.keras.layers.EmbeddingLogNormalDiag,
        psiz.keras.layers.EmbeddingLogitNormalDiag,
        psiz.keras.layers.EmbeddingNormalDiag,
        psiz.keras.layers.EmbeddingTruncatedNormalDiag
    ]
)
def test_call_2d_input(emb_input_2d, n_sample, embedding_class):
    """Test call() return shape.

    Returned shape must include a `n_sample` dimension." Using
    tf.keras.layers.Embedding will fail because it does not include
    a sample dimension.

    """
    input = emb_input_2d
    batch_shape_0 = 3
    batch_shape_1 = 2
    n_stimuli = 10
    n_dim = 2

    if n_sample is None:
        # Test default.
        embedding = embedding_class(
            n_stimuli, n_dim, mask_zero=True
        )
        desired_output_shape = np.array([
            batch_shape_0, batch_shape_1, n_dim
        ])
    else:
        embedding = embedding_class(
            n_stimuli, n_dim, mask_zero=True, n_sample=n_sample
        )
        desired_output_shape = np.array([
            n_sample, batch_shape_0, batch_shape_1, n_dim
        ])

    output = embedding(input)
    np.testing.assert_array_equal(
        np.shape(output.numpy()),
        desired_output_shape
    )
