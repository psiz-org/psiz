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
"""Module for testing stochastic embeddings."""

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


def test_init_with_shape(emb_input_1d):
    """Test initialization with `input_shape`."""
    input = emb_input_1d

    n_stimuli = 10
    n_dim = 3
    mask_zero = False
    input_shape = (3,)
    sample_shape = ()
    embedding = psiz.keras.layers.EmbeddingNormalDiag(
        n_stimuli, n_dim, mask_zero=mask_zero, input_shape=input_shape
    )

    output = embedding(input)

    desired_output_shape = np.hstack([sample_shape, input_shape, n_dim]).astype(int)
    np.testing.assert_array_equal(desired_output_shape, np.shape(output.numpy()))


@pytest.mark.parametrize("is_eager", [True, False])  # TODO uncomment
@pytest.mark.parametrize("mask_zero", [True, False])
@pytest.mark.parametrize("sample_shape", [None, (), 1, 10, [2, 4]])
@pytest.mark.parametrize(
    "embedding_class",
    [
        psiz.keras.layers.EmbeddingGammaDiag,
        psiz.keras.layers.EmbeddingLaplaceDiag,
        psiz.keras.layers.EmbeddingLogNormalDiag,
        psiz.keras.layers.EmbeddingLogitNormalDiag,
        psiz.keras.layers.EmbeddingNormalDiag,
        psiz.keras.layers.EmbeddingTruncatedNormalDiag,
    ],
)
def test_call_1d_input(
    emb_input_1d, sample_shape, embedding_class, mask_zero, is_eager
):
    """Test call() return shape.

    Returned shape must include appropriate `sample_shape`."

    """
    tf.config.run_functions_eagerly(is_eager)
    input = emb_input_1d
    input_shape = [3]
    n_stimuli = 10
    n_dim = 2

    if sample_shape is None:
        # Test default `sample_shape`.
        sample_shape = ()
        embedding = embedding_class(n_stimuli, n_dim, mask_zero=mask_zero)
    else:
        embedding = embedding_class(
            n_stimuli, n_dim, mask_zero=mask_zero, sample_shape=sample_shape
        )

    desired_input_length = 1
    desired_output_shape = np.hstack([sample_shape, input_shape, n_dim]).astype(int)

    # Test call
    output = embedding(input)
    np.testing.assert_array_equal(desired_output_shape, np.shape(output.numpy()))

    assert embedding.mask_zero == mask_zero

    # Verify `get_config` dictionary.
    config = embedding.get_config()
    assert config["input_dim"] == n_stimuli
    assert config["output_dim"] == n_dim
    assert config["mask_zero"] == mask_zero
    assert config["input_length"] == desired_input_length
    assert config["sample_shape"] == sample_shape

    # Reconstruct embedding and test.
    recon_emb = embedding_class.from_config(
        config
    )  # TODO YOU ARE HERE something not right
    recon_output = recon_emb(input)
    np.testing.assert_array_equal(desired_output_shape, np.shape(recon_output.numpy()))


@pytest.mark.parametrize(
    "embedding_class",
    [
        psiz.keras.layers.EmbeddingGammaDiag,
        psiz.keras.layers.EmbeddingLaplaceDiag,
        psiz.keras.layers.EmbeddingLogNormalDiag,
        pytest.param(
            psiz.keras.layers.EmbeddingLogitNormalDiag,
            marks=pytest.mark.xfail(reason="Mode not implemented."),
        ),
        psiz.keras.layers.EmbeddingNormalDiag,
        psiz.keras.layers.EmbeddingTruncatedNormalDiag,
    ],
)
def test_mode(emb_input_1d, embedding_class):
    """Test mode() call and return shape."""
    input = emb_input_1d
    n_stimuli = 10
    n_dim = 2

    embedding = embedding_class(n_stimuli, n_dim, mask_zero=False)

    # Make call to ensure weight are built.
    _ = embedding(input)

    # Test `mode` method to make sure implemented, then check shape.
    emb_mode = embedding.embeddings.mode()
    tf.debugging.assert_equal(emb_mode.shape, tf.TensorShape([n_stimuli, n_dim]))


@pytest.mark.parametrize("mask_zero", [True, False])
@pytest.mark.parametrize("sample_shape", [None, (), 1, 10, [2, 4]])
@pytest.mark.parametrize(
    "embedding_class",
    [
        psiz.keras.layers.EmbeddingGammaDiag,
        psiz.keras.layers.EmbeddingLaplaceDiag,
        psiz.keras.layers.EmbeddingLogNormalDiag,
        psiz.keras.layers.EmbeddingLogitNormalDiag,
        psiz.keras.layers.EmbeddingNormalDiag,
        psiz.keras.layers.EmbeddingTruncatedNormalDiag,
    ],
)
def test_call_2d_input(emb_input_2d, sample_shape, embedding_class, mask_zero):
    """Test call() return shape.

    Returned shape must include appropriate `sample_shape`."

    """
    input = emb_input_2d
    input_shape = [3, 2]
    n_stimuli = 10
    n_dim = 2

    if sample_shape is None:
        # Test default `sample_shape`.
        sample_shape = ()
        embedding = embedding_class(n_stimuli, n_dim, mask_zero=mask_zero)
    else:
        embedding = embedding_class(
            n_stimuli, n_dim, mask_zero=mask_zero, sample_shape=sample_shape
        )

    desired_output_shape = np.hstack([sample_shape, input_shape, n_dim]).astype(int)

    output = embedding(input)
    np.testing.assert_array_equal(desired_output_shape, np.shape(output.numpy()))

    assert embedding.mask_zero == mask_zero
