# -*- coding: utf-8 -*-
# Copyright 2024 The PsiZ Authors. All Rights Reserved.
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


import keras
import pytest
import tensorflow as tf

from psiz.keras.layers import ALCOVECell, ExponentialSimilarity


def test_call(category_learning_inputs_v0):
    """Test naked call (without RNN layer)."""
    n_stimuli = 20
    n_dim = 4
    n_output = 3

    percept = keras.layers.Embedding(
        n_stimuli + 1,
        n_dim,
        mask_zero=True,
        trainable=False,
    )
    similarity = ExponentialSimilarity(
        beta_initializer=keras.initializers.Constant(3.0),
        tau_initializer=keras.initializers.Constant(1.0),
        gamma_initializer=keras.initializers.Constant(0.0),
        trainable=False,
    )
    cell = ALCOVECell(
        n_output,
        percept=percept,
        similarity=similarity,
        rho_initializer=keras.initializers.Constant(2.0),
        temperature_initializer=keras.initializers.Constant(1.0),
        lr_attention_initializer=keras.initializers.Constant(0.03),
        lr_association_initializer=keras.initializers.Constant(0.03),
        trainable=False,
    )
    state_t0 = cell.get_initial_state(batch_size=4)
    output_t1, state_t1 = cell(category_learning_inputs_v0, state_t0)
    tf.debugging.assert_equal(tf.shape(output_t1), tf.TensorShape([4, 3]))


@pytest.mark.xfail(reason="Keras v3 RNN requires single input tensor.")
def test_call_v1(category_learning_inputs_v1):
    """Test wrapped call (with RNN layer)."""
    n_stimuli = 20
    n_dim = 4
    n_output = 3

    percept = keras.layers.Embedding(
        n_stimuli + 1,
        n_dim,
        mask_zero=True,
        trainable=False,
    )
    similarity = ExponentialSimilarity(
        beta_initializer=keras.initializers.Constant(3.0),
        tau_initializer=keras.initializers.Constant(1.0),
        gamma_initializer=keras.initializers.Constant(0.0),
        trainable=False,
    )
    cell = ALCOVECell(
        n_output,
        percept=percept,
        similarity=similarity,
        rho_initializer=keras.initializers.Constant(2.0),
        temperature_initializer=keras.initializers.Constant(1.0),
        lr_attention_initializer=keras.initializers.Constant(0.03),
        lr_association_initializer=keras.initializers.Constant(0.03),
        trainable=False,
    )
    rnn = keras.layers.RNN(cell, return_sequences=True, stateful=False)
    # TODO: Fix inputs
    output = rnn(category_learning_inputs_v1)
    # TODO add assertions


@pytest.mark.xfail(
    reason="Reconstructed RNN layer must be built first, but RNN requires single input tensor."
)
def test_serialization(category_learning_inputs_v1):
    """Test serialization."""
    units = 3
    similarity = ExponentialSimilarity(
        beta_initializer=keras.initializers.Constant(3.2),
        tau_initializer=keras.initializers.Constant(1.0),
        gamma_initializer=keras.initializers.Constant(0.0),
        trainable=False,
    )
    embedding = keras.layers.Embedding(
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
    # recon_layer.build(TODO)
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
