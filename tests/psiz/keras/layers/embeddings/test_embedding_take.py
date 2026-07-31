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
"""Test EmbeddingTake layer."""

import keras
import numpy as np

from psiz.keras.layers import EmbeddingLaplaceDiag, EmbeddingNormalDiag
from psiz.keras.layers import EmbeddingTake
from psiz.stochastic.transforms import softplus_inverse


def test_deterministic_0():
    """Test using (deterministic) Keras Embedding layer.

    Globally shared, n_dim=1.

    """
    n_stimuli = 4
    input_map = np.zeros([n_stimuli])
    n_dim = 1

    # Create core embedding composed of one 1D point.
    embedding_core = keras.layers.Embedding(
        1,
        n_dim,
        embeddings_initializer=keras.initializers.Constant(0.1),
    )
    # Map one point to four points.
    embedding_take = EmbeddingTake(embedding=embedding_core, input_map=input_map)

    # Test call with 1D input.
    x = np.array([0, 1, 2, 3], dtype="int32")
    z = embedding_take(x)
    z_desired = np.array([[0.1], [0.1], [0.1], [0.1]], dtype="float32")
    np.testing.assert_array_equal(z_desired, keras.ops.convert_to_numpy(z))

    # Test call with 2D input.
    x = np.array([[0, 1], [2, 3], [0, 1], [2, 3]], dtype="int32")
    z = embedding_take(x)
    z_desired = np.array(
        [[[0.1], [0.1]], [[0.1], [0.1]], [[0.1], [0.1]], [[0.1], [0.1]]],
        dtype="float32",
    )
    np.testing.assert_array_equal(z_desired, keras.ops.convert_to_numpy(z))

    # Test embedding properties.
    embeddings_desired = np.array([[0.1], [0.1], [0.1], [0.1]], dtype="float32")
    embeddings = embedding_take.embeddings
    np.testing.assert_array_equal(
        embeddings_desired, keras.ops.convert_to_numpy(embeddings)
    )

    mask_zero_desired = False
    assert mask_zero_desired == embedding_take.mask_zero

    input_dim_desired = n_stimuli
    assert input_dim_desired == embedding_take.input_dim

    output_dim_desired = n_dim
    assert output_dim_desired == embedding_take.output_dim


def test_deterministic_1():
    """Test using (deterministic) Keras Embedding layer.

    Globally shared, n_dim=2.

    """
    n_stimuli = 4
    input_map = np.zeros([n_stimuli])
    n_dim = 2

    # Create core embedding composed of one 2D point.
    embedding_core = keras.layers.Embedding(
        1,
        n_dim,
        embeddings_initializer=keras.initializers.Constant([0.1, 0.2]),
    )
    # Map points.
    embedding_take = EmbeddingTake(embedding=embedding_core, input_map=input_map)

    # Test call with 1D input.
    x = np.array([0, 1, 2, 3], dtype="int32")
    z = embedding_take(x)
    z_desired = np.array(
        [[0.1, 0.2], [0.1, 0.2], [0.1, 0.2], [0.1, 0.2]], dtype="float32"
    )
    np.testing.assert_array_equal(z_desired, keras.ops.convert_to_numpy(z))

    # Test call with 2d input.
    x = np.array([[0, 1], [2, 3], [0, 1], [2, 3]], dtype="int32")
    z = embedding_take(x)
    z_desired = np.array(
        [
            [[0.1, 0.2], [0.1, 0.2]],
            [[0.1, 0.2], [0.1, 0.2]],
            [[0.1, 0.2], [0.1, 0.2]],
            [[0.1, 0.2], [0.1, 0.2]],
        ],
        dtype="float32",
    )
    np.testing.assert_array_equal(z_desired, keras.ops.convert_to_numpy(z))

    # Test embedding properties.
    embeddings_desired = np.array(
        [[0.1, 0.2], [0.1, 0.2], [0.1, 0.2], [0.1, 0.2]], dtype="float32"
    )
    embeddings = embedding_take.embeddings
    np.testing.assert_array_equal(
        embeddings_desired, keras.ops.convert_to_numpy(embeddings)
    )

    mask_zero_desired = False
    assert mask_zero_desired == embedding_take.mask_zero

    input_dim_desired = n_stimuli
    assert input_dim_desired == embedding_take.input_dim

    output_dim_desired = n_dim
    assert output_dim_desired == embedding_take.output_dim


def test_deterministic_2():
    """Test using (deterministic) Keras Embedding layer.

    Hierarchically shared, n_dim=2.

    """
    input_map = np.array([0, 0, 1, 1, 2, 2])
    n_stimuli = len(input_map)
    n_dim = 2

    # Create core embedding composed of one 1D point.
    embedding_core = keras.layers.Embedding(
        3,
        n_dim,
        embeddings_initializer=keras.initializers.Constant(
            np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        ),
    )
    # Map points.
    embedding_take = EmbeddingTake(embedding=embedding_core, input_map=input_map)

    # Test call with 1D input.
    x = np.array([0, 1, 2, 3, 4, 5], dtype="int32")
    z = embedding_take(x)
    z_desired = np.array(
        [[0.1, 0.2], [0.1, 0.2], [0.3, 0.4], [0.3, 0.4], [0.5, 0.6], [0.5, 0.6]],
        dtype="float32",
    )
    np.testing.assert_array_equal(z_desired, keras.ops.convert_to_numpy(z))

    # Test embedding properties.
    embeddings_desired = np.array(
        [[0.1, 0.2], [0.1, 0.2], [0.3, 0.4], [0.3, 0.4], [0.5, 0.6], [0.5, 0.6]],
        dtype="float32",
    )
    embeddings = embedding_take.embeddings
    np.testing.assert_array_equal(
        embeddings_desired, keras.ops.convert_to_numpy(embeddings)
    )

    mask_zero_desired = False
    assert mask_zero_desired == embedding_take.mask_zero

    input_dim_desired = n_stimuli
    assert input_dim_desired == embedding_take.input_dim

    output_dim_desired = n_dim
    assert output_dim_desired == embedding_take.output_dim


def test_stochastic_0():
    """Test using stochastic embedding.

    Globally shared, n_dim=1.
    """
    n_stimuli = 4
    input_map = np.zeros([n_stimuli])
    n_dim = 1

    # Create core embedding composed of one 1D point.
    embedding_core = EmbeddingNormalDiag(
        1,
        1,
        loc_initializer=keras.initializers.Constant(0.1),
        scale_initializer=keras.initializers.Constant(
            keras.ops.convert_to_numpy(softplus_inverse(0.01))
        ),
        loc_trainable=False,
    )
    # Map points.
    embedding_take = EmbeddingTake(embedding=embedding_core, input_map=input_map)

    # Test call with 1D input.
    x = np.array([0, 1, 2, 3], dtype="int32")
    z = keras.ops.convert_to_numpy(embedding_take(x))
    assert z.shape[0] == 4
    assert z.shape[1] == n_dim

    # Test call with 2D input.
    x = np.array([[0, 1], [2, 3], [0, 1], [2, 3]], dtype="int32")
    z = keras.ops.convert_to_numpy(embedding_take(x))
    assert z.shape[0] == 4
    assert z.shape[1] == 2
    assert z.shape[2] == n_dim

    # Test embedding properties.
    embeddings_loc_desired = np.array([[0.1], [0.1], [0.1], [0.1]], dtype="float32")
    embeddings_scale_desired = np.array(
        [[0.0100001], [0.0100001], [0.0100001], [0.0100001]], dtype="float32"
    )
    embeddings = embedding_take.embeddings
    np.testing.assert_array_equal(
        embeddings_loc_desired, keras.ops.convert_to_numpy(embeddings.distribution.loc)
    )
    np.testing.assert_array_almost_equal(
        embeddings_scale_desired,
        keras.ops.convert_to_numpy(embeddings.distribution.scale),
    )

    mask_zero_desired = False
    assert mask_zero_desired == embedding_take.mask_zero

    input_dim_desired = 4
    assert input_dim_desired == embedding_take.input_dim

    output_dim_desired = 1
    assert output_dim_desired == embedding_take.output_dim


def test_stochastic_1():
    """Test using stochastic embedding.

    Globally shared, n_dim=2.
    """
    n_stimuli = 4
    input_map = np.zeros([n_stimuli])
    n_dim = 2

    # Create core embedding composed of one 1D point.
    embedding_core = EmbeddingNormalDiag(
        1,
        2,
        loc_initializer=keras.initializers.Constant(np.array([0.1, 0.2])),
        scale_initializer=keras.initializers.Constant(
            keras.ops.convert_to_numpy(softplus_inverse(np.array([0.01, 0.02])))
        ),
        loc_trainable=False,
    )
    # Map points.
    embedding_take = EmbeddingTake(embedding=embedding_core, input_map=input_map)

    # Test call with 1D input.
    x = np.array([0, 1, 2, 3], dtype="int32")
    z = keras.ops.convert_to_numpy(embedding_take(x))
    assert z.shape[0] == 4
    assert z.shape[1] == n_dim

    # Test call with 2D input.
    x = np.array([[0, 1], [2, 3], [0, 1], [2, 3]], dtype="int32")
    z = keras.ops.convert_to_numpy(embedding_take(x))
    assert z.shape[0] == 4
    assert z.shape[1] == 2
    assert z.shape[2] == n_dim

    # Test embedding properties.
    embeddings_loc_desired = np.array(
        [[0.1, 0.2], [0.1, 0.2], [0.1, 0.2], [0.1, 0.2]], dtype="float32"
    )
    embeddings_scale_desired = np.array(
        [
            [0.0100001, 0.0200001],
            [0.0100001, 0.0200001],
            [0.0100001, 0.0200001],
            [0.0100001, 0.0200001],
        ],
        dtype="float32",
    )
    embeddings = embedding_take.embeddings
    np.testing.assert_array_equal(
        embeddings_loc_desired, keras.ops.convert_to_numpy(embeddings.distribution.loc)
    )
    np.testing.assert_array_almost_equal(
        embeddings_scale_desired,
        keras.ops.convert_to_numpy(embeddings.distribution.scale),
    )

    mask_zero_desired = False
    assert mask_zero_desired == embedding_take.mask_zero

    input_dim_desired = n_stimuli
    assert input_dim_desired == embedding_take.input_dim

    output_dim_desired = n_dim
    assert output_dim_desired == embedding_take.output_dim


def test_stochastic_2a():
    """Test stochastic, hierarchically shared."""
    input_map = np.array([0, 0, 1, 2, 1, 2])
    n_stimuli = len(input_map)
    n_dim = 2

    # Create core embedding composed of one 1D point.
    embedding_core = EmbeddingNormalDiag(
        3,
        n_dim,
        loc_initializer=keras.initializers.Constant(
            np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        ),
        scale_initializer=keras.initializers.Constant(
            keras.ops.convert_to_numpy(softplus_inverse(
                np.array([[0.01, 0.02], [0.03, 0.04], [0.05, 0.06]])
            ))
        ),
        loc_trainable=False,
    )
    # Map points.
    embedding_take = EmbeddingTake(embedding=embedding_core, input_map=input_map)

    # Test call with 1D input.
    x = np.array([0, 1, 2, 3, 4, 5], dtype="int32")
    z = keras.ops.convert_to_numpy(embedding_take(x))
    assert z.shape[0] == 6
    assert z.shape[1] == n_dim

    # Test embedding properties.
    embeddings_loc_desired = np.array(
        [[0.1, 0.2], [0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.3, 0.4], [0.5, 0.6]],
        dtype="float32",
    )
    embeddings_scale_desired = np.array(
        [
            [0.0100001, 0.0200001],
            [0.0100001, 0.0200001],
            [0.0300001, 0.0400001],
            [0.05000011, 0.0600001],
            [0.0300001, 0.0400001],
            [0.05000011, 0.0600001],
        ],
        dtype="float32",
    )
    embeddings = embedding_take.embeddings
    np.testing.assert_array_equal(
        embeddings_loc_desired, keras.ops.convert_to_numpy(embeddings.distribution.loc)
    )
    np.testing.assert_array_almost_equal(
        embeddings_scale_desired,
        keras.ops.convert_to_numpy(embeddings.distribution.scale),
    )

    mask_zero_desired = False
    assert mask_zero_desired == embedding_take.mask_zero

    input_dim_desired = n_stimuli
    assert input_dim_desired == embedding_take.input_dim

    output_dim_desired = n_dim
    assert output_dim_desired == embedding_take.output_dim


def test_stochastic_2b():
    """Test stochastic, hierarchically shared."""
    input_map = np.array([0, 0, 1, 2, 1, 2])
    n_stimuli = len(input_map)
    n_dim = 2

    # Create core embedding composed of one 1D point.
    embedding_core = EmbeddingLaplaceDiag(
        3,
        n_dim,
        loc_initializer=keras.initializers.Constant(
            np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        ),
        scale_initializer=keras.initializers.Constant(
            keras.ops.convert_to_numpy(softplus_inverse(
                np.array([[0.01, 0.02], [0.03, 0.04], [0.05, 0.06]])
            ))
        ),
        loc_trainable=False,
    )
    # Map points.
    embedding_take = EmbeddingTake(embedding=embedding_core, input_map=input_map)

    # Test call with 1D input.
    x = np.array([0, 1, 2, 3, 4, 5], dtype="int32")
    z = keras.ops.convert_to_numpy(embedding_take(x))
    assert z.shape[0] == n_stimuli
    assert z.shape[1] == n_dim

    # Test embedding properties.
    embeddings_loc_desired = np.array(
        [[0.1, 0.2], [0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.3, 0.4], [0.5, 0.6]],
        dtype="float32",
    )
    embeddings_scale_desired = np.array(
        [
            [0.0100001, 0.0200001],
            [0.0100001, 0.0200001],
            [0.0300001, 0.0400001],
            [0.05000011, 0.0600001],
            [0.0300001, 0.0400001],
            [0.05000011, 0.0600001],
        ],
        dtype="float32",
    )
    embeddings = embedding_take.embeddings
    np.testing.assert_array_equal(
        embeddings_loc_desired, keras.ops.convert_to_numpy(embeddings.distribution.loc)
    )
    np.testing.assert_array_almost_equal(
        embeddings_scale_desired,
        keras.ops.convert_to_numpy(embeddings.distribution.scale),
    )

    mask_zero_desired = False
    assert mask_zero_desired == embedding_take.mask_zero

    input_dim_desired = n_stimuli
    assert input_dim_desired == embedding_take.input_dim

    output_dim_desired = n_dim
    assert output_dim_desired == embedding_take.output_dim
