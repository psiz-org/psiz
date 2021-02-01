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
"""Test EmbeddingGroup."""

import pytest

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding

from psiz.keras.layers import EmbeddingGroup


@pytest.mark.xfail
def test_call(flat_embeddings):
    """Test call."""
    phys_emb = Embedding(
        input_dim=12, output_dim=3,
        embeddings_initializer=tf.keras.initializers.Constant(
            flat_embeddings
        )
    )

    emb_group = EmbeddingGroup(
        embedding=phys_emb, input_dims=[6, 2]
    )

    # Scalars, matching shapes.
    # inputs = [
    #     tf.constant(np.array(2, dtype=np.int32)),
    #     tf.constant(np.array(1, dtype=np.int32))
    # ]
    # outputs = emb_group(inputs).numpy()

    # # Tensors matching shapes.
    # inputs = [
    #     tf.constant(
    #         np.array([
    #             [1, 2],
    #             [1, 2]
    #         ], dtype=np.int32)
    #     ),
    #     tf.constant(
    #         np.array([
    #             [0, 0],
    #             [1, 1]
    #         ], dtype=np.int32)
    #     )
    # ]
    # outputs = emb_group(inputs).numpy()

    # Tensors mmis-matching shapes.
    inputs = [
        tf.constant(
            np.array([
                [0, 1],
                [2, 3],
                [4, 5]
            ], dtype=np.int32)
        ),
        tf.constant(
            np.array([
                0, 1, 0
            ], dtype=np.int32)
        )
    ]
    outputs = emb_group(inputs).numpy()

    assert False
