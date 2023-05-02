# -*- coding: utf-8 -*-
# Copyright 2022 The PsiZ Authors. All Rights Reserved.
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
"""Test ALCOVECell."""

import tensorflow as tf

from psiz.keras.layers import ALCOVECell, ExponentialSimilarity


def test_serialization():
    """Test serialization."""
    units = 3
    similarity = ExponentialSimilarity(
        beta_initializer=tf.keras.initializers.Constant(3.2),
        tau_initializer=tf.keras.initializers.Constant(1.0),
        gamma_initializer=tf.keras.initializers.Constant(0.0),
        trainable=False,
    )
    embedding = tf.keras.layers.Embedding(
        11,
        4,
        mask_zero=True,
        trainable=False,
    )

    # Default options.
    layer = ALCOVECell(
        units, similarity=similarity, percept=embedding, name="alcove_cell"
    )
    cfg = layer.get_config()
    # Verify.
    assert cfg["name"] == "alcove_cell"
    assert cfg["trainable"] is True
    assert cfg["dtype"] == "float32"
    assert cfg["units"] == units
    assert cfg["data_scope"] == "categorize"

    recon_layer = ALCOVECell.from_config(cfg)
    tf.debugging.assert_equal(recon_layer.similarity.beta, 3.2)
    tf.debugging.assert_equal(recon_layer.similarity.tau, 1.0)
    assert recon_layer.units == units
    assert recon_layer.data_scope == "categorize"

    # Non-default options.
    layer = ALCOVECell(
        units, similarity=similarity, percept=embedding, data_scope="abc"
    )
    cfg = layer.get_config()
    # Verify.
    # Verify.
    assert cfg["name"] == "alcove_cell"
    assert cfg["trainable"] is True
    assert cfg["dtype"] == "float32"
    assert cfg["units"] == units
    assert cfg["data_scope"] == "abc"

    recon_layer = ALCOVECell.from_config(cfg)
    tf.debugging.assert_equal(recon_layer.similarity.beta, 3.2)
    tf.debugging.assert_equal(recon_layer.similarity.tau, 1.0)
    assert recon_layer.units == units
    assert recon_layer.data_scope == "abc"
