# -*- coding: utf-8 -*-
# Copyright 2023 The PsiZ Authors. All Rights Reserved.
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
"""Module for testing models."""

import tensorflow as tf
import tensorflow_probability as tfp

import psiz


def percept_static_v0():
    """Static percept layer."""
    n_stimuli = 20
    n_dim = 3
    percept = tf.keras.layers.Embedding(n_stimuli + 1, n_dim, mask_zero=True)
    return percept


def percept_stochastic_v0():
    """Stochastic percept layer."""
    n_stimuli = 20
    n_dim = 3
    prior_scale = 0.2
    percept = psiz.keras.layers.EmbeddingNormalDiag(
        n_stimuli + 1,
        n_dim,
        mask_zero=True,
        scale_initializer=tf.keras.initializers.Constant(
            tfp.math.softplus_inverse(prior_scale).numpy()
        ),
    )
    return percept


def kernel_v0():
    """A kernel layer."""
    kernel = psiz.keras.layers.Minkowski(
        rho_initializer=tf.keras.initializers.Constant(2.0),
        w_initializer=tf.keras.initializers.Constant(1.0),
        activation=psiz.keras.layers.ExponentialSimilarity(
            beta_initializer=tf.keras.initializers.Constant(10.0),
            tau_initializer=tf.keras.initializers.Constant(1.0),
            gamma_initializer=tf.keras.initializers.Constant(0.001),
            trainable=False,
        ),
        trainable=False,
    )
    return kernel


# TODO add test calls analogous to `test_rank_similarity_cell.py`


def test_serialization():
    """Test serialization."""
    percept = percept_static_v0()
    kernel = kernel_v0()

    # Default settings.
    rate_cell = psiz.keras.layers.RateSimilarityCell(
        percept=percept, kernel=kernel, name="rs1"
    )
    cfg = rate_cell.get_config()
    # Verify.
    assert cfg["name"] == "rs1"
    assert cfg["trainable"] is True
    assert cfg["dtype"] == "float32"
    assert cfg["lower_trainable"] is False
    assert cfg["upper_trainable"] is False
    assert cfg["midpoint_trainable"] is True
    assert cfg["rate_trainable"] is True
    assert cfg["data_scope"] == "rate2"

    rate_cell2 = psiz.keras.layers.RateSimilarityCell.from_config(cfg)
    assert rate_cell2.name == "rs1"
    assert rate_cell2.trainable is True
    assert rate_cell2.dtype == "float32"
    assert rate_cell2.lower_trainable is False
    assert rate_cell2.upper_trainable is False
    assert rate_cell2.midpoint_trainable is True
    assert rate_cell2.rate_trainable is True
    assert rate_cell2.data_scope == "rate2"

    # All optional settings.
    rate_cell = psiz.keras.layers.RateSimilarityCell(
        percept=percept,
        kernel=kernel,
        data_scope="abc",
        lower_trainable=True,
        upper_trainable=True,
        midpoint_trainable=False,
        rate_trainable=False,
        name="rs2",
    )
    cfg = rate_cell.get_config()
    # Verify.
    assert cfg["name"] == "rs2"
    assert cfg["trainable"] is True
    assert cfg["dtype"] == "float32"
    assert cfg["lower_trainable"] is True
    assert cfg["upper_trainable"] is True
    assert cfg["midpoint_trainable"] is False
    assert cfg["rate_trainable"] is False
    assert cfg["data_scope"] == "abc"

    rate_cell2 = psiz.keras.layers.RateSimilarityCell.from_config(cfg)
    assert rate_cell2.name == "rs2"
    assert rate_cell2.trainable is True
    assert rate_cell2.dtype == "float32"
    assert rate_cell2.lower_trainable is True
    assert rate_cell2.upper_trainable is True
    assert rate_cell2.midpoint_trainable is False
    assert rate_cell2.rate_trainable is False
    assert rate_cell2.data_scope == "abc"
