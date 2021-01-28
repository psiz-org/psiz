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
"""Test EmbeddingND."""

import pytest

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding

from psiz.keras.layers import EmbeddingND


def test_init():
    phys_emb = Embedding(
        input_dim=12, output_dim=3
    )

    # Raise no error, since shapes are compatible.
    nd_emb = EmbeddingND(
        embedding=phys_emb
    )
    nd_emb = EmbeddingND(
        embedding=phys_emb, input_dims=[12]
    )
    nd_emb = EmbeddingND(
        embedding=phys_emb, input_dims=[2, 6]
    )
    nd_emb = EmbeddingND(
        embedding=phys_emb, input_dims=[2, 2, 3]
    )

    # Raise ValueError for incompatible shapes.
    with pytest.raises(Exception) as e_info:
        nd_emb = EmbeddingND(
            embedding=phys_emb, input_dims=[3, 5]
        )
        assert str(e_info.value) == (
            'The provided `input_dims` and `embedding` are not shape '
            'compatible. The provided embedding has input_dim=12, which '
            'cannot be reshaped to ([3, 5]).'
        )


def test_standard_2d_call(flat_embeddings):
    """Test 2D input embedding call."""
    phys_emb = Embedding(
        input_dim=12, output_dim=3,
        embeddings_initializer=tf.keras.initializers.Constant(
            flat_embeddings
        )
    )

    nd_emb = EmbeddingND(
        embedding=phys_emb, input_dims=[6, 2]
    )

    reshaped_embeddings = np.reshape(flat_embeddings, [6, 2, 3])

    # Test reshape.
    inputs = [
        tf.constant(np.array(2, dtype=np.int32)),
        tf.constant(np.array(1, dtype=np.int32))
    ]
    multi_index = tf.stack(inputs, axis=0)
    output = nd_emb(multi_index).numpy()
    desired_output = reshaped_embeddings[2, 1]
    np.testing.assert_array_almost_equal(output, desired_output)

    inputs = [
        tf.constant(np.array(5, dtype=np.int32)),
        tf.constant(np.array(1, dtype=np.int32))
    ]
    multi_index = tf.stack(inputs, axis=0)
    output = nd_emb(multi_index).numpy()
    desired_output = reshaped_embeddings[5, 1]
    np.testing.assert_array_almost_equal(output, desired_output)

    inputs_0 = tf.constant(
        np.array([
            [1, 2],
            [1, 2]
        ], dtype=np.int32)
    )
    inputs_1 = tf.constant(
        np.array([
            [0, 0],
            [1, 1]
        ], dtype=np.int32)
    )
    inputs = [inputs_0, inputs_1]
    multi_index = tf.stack(inputs, axis=0)
    output = nd_emb(multi_index).numpy()

    desired_output = np.array([
        [[2.0, 2.1, 2.2], [4.0, 4.1, 4.2]],
        [[3.0, 3.1, 3.2], [5.0, 5.1, 5.2]],
    ])
    # Assert almost equal because of TF 32 bit casting.
    np.testing.assert_array_almost_equal(output, desired_output)


def test_standard_3d_call(flat_embeddings):
    """Test 3D input Embedding call."""
    phys_emb = Embedding(
        input_dim=12, output_dim=3,
        embeddings_initializer=tf.keras.initializers.Constant(
            flat_embeddings
        )
    )

    nd_emb = EmbeddingND(
        embedding=phys_emb, input_dims=[3, 2, 2]
    )

    reshaped_embeddings = np.reshape(flat_embeddings, [3, 2, 2, 3])

    # Test reshape.
    inputs = [
        tf.constant(np.array(0, dtype=np.int32)),
        tf.constant(np.array(0, dtype=np.int32)),
        tf.constant(np.array(0, dtype=np.int32))
    ]
    multi_index = tf.stack(inputs, axis=0)
    output = nd_emb(multi_index).numpy()
    desired_output = reshaped_embeddings[0, 0, 0]
    np.testing.assert_array_almost_equal(output, desired_output)

    inputs = [
        tf.constant(np.array(2, dtype=np.int32)),
        tf.constant(np.array(1, dtype=np.int32)),
        tf.constant(np.array(1, dtype=np.int32))
    ]
    multi_index = tf.stack(inputs, axis=0)
    output = nd_emb(multi_index).numpy()
    desired_output = reshaped_embeddings[2, 1, 1]
    np.testing.assert_array_almost_equal(output, desired_output)

    inputs = [
        tf.constant(np.array(1, dtype=np.int32)),
        tf.constant(np.array(1, dtype=np.int32)),
        tf.constant(np.array(0, dtype=np.int32))
    ]
    multi_index = tf.stack(inputs, axis=0)
    output = nd_emb(multi_index).numpy()
    desired_output = reshaped_embeddings[1, 1, 0]
    np.testing.assert_array_almost_equal(output, desired_output)

    inputs_0 = tf.constant(
        np.array([
            [1, 2],
            [1, 2]
        ], dtype=np.int32)
    )
    inputs_1 = tf.constant(
        np.array([
            [0, 0],
            [1, 1]
        ], dtype=np.int32)
    )
    inputs_2 = tf.constant(
        np.array([
            [0, 1],
            [0, 1]
        ], dtype=np.int32)
    )
    inputs = (inputs_0, inputs_1, inputs_2)
    multi_index = tf.stack(inputs, axis=0)
    output = nd_emb(multi_index).numpy()

    desired_output = np.array([
        [[4.0, 4.1, 4.2], [9.0, 9.1, 9.2]],
        [[6.0, 6.1, 6.2], [11.0, 11.1, 11.2]],
    ])
    np.testing.assert_array_almost_equal(output, desired_output)


def test_serialization(flat_embeddings):
    # Build ND embedding.
    phys_emb = Embedding(
        input_dim=12, output_dim=3,
        embeddings_initializer=tf.keras.initializers.Constant(
            flat_embeddings
        )
    )
    nd_emb = EmbeddingND(
        embedding=phys_emb, input_dims=[3, 2, 2]
    )

    # Get configuration.
    config = nd_emb.get_config()

    # Reconstruct layer from configuration.
    nd_emb_reconstructed = EmbeddingND.from_config(config)

    # Assert reconstructed attributes are the same as original.
    np.testing.assert_array_equal(
        nd_emb.input_dims,
        nd_emb_reconstructed.input_dims
    )
    assert nd_emb.output_dim == nd_emb_reconstructed.output_dim

    # Assert calls are equal (given constant initializer).
    inputs = [
        tf.constant(np.array(1, dtype=np.int32)),
        tf.constant(np.array(1, dtype=np.int32)),
        tf.constant(np.array(0, dtype=np.int32))
    ]
    multi_index = tf.stack(inputs, axis=0)
    output_orig = nd_emb(multi_index).numpy()
    output_recon = nd_emb_reconstructed(inputs).numpy()
    np.testing.assert_array_almost_equal(output_orig, output_recon)
