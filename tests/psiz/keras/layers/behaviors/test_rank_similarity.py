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

import psiz
import tensorflow as tf


def test_outcome_probability_v0():
    """Test outcome probabilty computation."""
    n_stimuli = 4
    n_dim = 2
    beta = 10.0

    percept = tf.keras.layers.Embedding(
        n_stimuli + 1,
        n_dim,
        mask_zero=True,
        embeddings_initializer=tf.initializers.Constant(
            tf.constant([[0.0, 0.0], [0.1, 0.1], [0.2, 0.1], [0.3, 0.1], [0.4, 0.1]])
        ),
    )
    kernel = psiz.keras.layers.Minkowski(
        rho_initializer=tf.keras.initializers.Constant(2.0),
        w_initializer=tf.keras.initializers.Constant(1.0),
        activation=psiz.keras.layers.ExponentialSimilarity(
            beta_initializer=tf.keras.initializers.Constant(beta),
            tau_initializer=tf.keras.initializers.Constant(1.0),
            gamma_initializer=tf.keras.initializers.Constant(0.0),
            trainable=False,
        ),
        trainable=False,
    )
    rank = psiz.keras.layers.RankSimilarity(
        n_reference=2, n_select=1, percept=percept, kernel=kernel
    )

    stimulus_set = tf.constant(
        [
            [1, 2, 3],
            [2, 3, 4],
            [2, 1, 4],
            [4, 3, 1],
        ]
    )
    x = {
        "given2rank1_stimulus_set": stimulus_set,
    }
    outcome_prob = rank(x)

    # Desired outcome.
    coords_x = 0.1 * tf.cast(stimulus_set, tf.float32)
    z_q = tf.gather(coords_x, indices=tf.constant([0]), axis=1)
    z_r = tf.gather(coords_x, indices=tf.constant([1, 2]), axis=1)
    d_qr = tf.abs(z_q - z_r)
    s_qr = tf.exp(-tf.constant(beta) * d_qr)
    total_s = tf.reduce_sum(s_qr, axis=1, keepdims=True)
    outcome_prob_desired = s_qr / total_s

    tf.debugging.assert_near(outcome_prob, outcome_prob_desired)


def test_outcome_probability_v1():
    """Test outcome probabilty computation."""
    n_stimuli = 4
    n_dim = 2
    beta = 10.0

    percept = tf.keras.layers.Embedding(
        n_stimuli + 1,
        n_dim,
        mask_zero=True,
        embeddings_initializer=tf.initializers.Constant(
            tf.constant([[0.0, 0.0], [0.1, 0.1], [0.2, 0.1], [0.3, 0.1], [0.4, 0.1]])
        ),
    )
    kernel = psiz.keras.layers.Minkowski(
        rho_initializer=tf.keras.initializers.Constant(2.0),
        w_initializer=tf.keras.initializers.Constant(1.0),
        activation=psiz.keras.layers.ExponentialSimilarity(
            beta_initializer=tf.keras.initializers.Constant(beta),
            tau_initializer=tf.keras.initializers.Constant(1.0),
            gamma_initializer=tf.keras.initializers.Constant(0.0),
            trainable=False,
        ),
        trainable=False,
    )
    rank = psiz.keras.layers.RankSimilarity(
        n_reference=2,
        n_select=1,
        percept=percept,
        kernel=kernel,
        temperature_initializer=tf.keras.initializers.Constant(value=0.001),
    )

    stimulus_set = tf.constant(
        [
            [1, 2, 3],
            [2, 3, 4],
            [2, 1, 4],
            [4, 3, 1],
        ]
    )
    x = {
        "given2rank1_stimulus_set": stimulus_set,
    }
    outcome_prob = rank(x)

    # Desired outcome.
    outcome_prob_desired = tf.constant(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=tf.float32,
    )
    tf.debugging.assert_near(outcome_prob, outcome_prob_desired)


def test_serialization_v0():
    """Test serialization."""
    n_stimuli = 4
    n_dim = 2
    beta = 10.0

    percept = tf.keras.layers.Embedding(
        n_stimuli + 1,
        n_dim,
        mask_zero=True,
        embeddings_initializer=tf.initializers.Constant(
            tf.constant([[0.0, 0.0], [0.1, 0.1], [0.2, 0.1], [0.3, 0.1], [0.4, 0.1]])
        ),
    )
    kernel = psiz.keras.layers.Minkowski(
        rho_initializer=tf.keras.initializers.Constant(2.0),
        w_initializer=tf.keras.initializers.Constant(1.0),
        activation=psiz.keras.layers.ExponentialSimilarity(
            beta_initializer=tf.keras.initializers.Constant(beta),
            tau_initializer=tf.keras.initializers.Constant(1.0),
            gamma_initializer=tf.keras.initializers.Constant(0.0),
            trainable=False,
        ),
        trainable=False,
    )

    # Default settings.
    rank = psiz.keras.layers.RankSimilarity(
        n_reference=2, n_select=1, percept=percept, kernel=kernel, name="rs1"
    )
    cfg = rank.get_config()
    # Verify.
    assert cfg["name"] == "rs1"
    assert cfg["trainable"] is True
    assert cfg["dtype"] == "float32"
    assert cfg["n_reference"] == 2
    assert cfg["n_select"] == 1
    assert cfg["data_scope"] == "given2rank1"
    assert cfg["temperature_initializer"]["class_name"] == "Constant"
    assert cfg["temperature_initializer"]["config"]["value"] == 1.0

    rank2 = psiz.keras.layers.RankSimilarity.from_config(cfg)
    assert rank2.name == "rs1"
    assert rank2.trainable is True
    assert rank2.dtype == "float32"
    assert rank2.n_reference == 2
    assert rank2.n_select == 1
    assert rank2.data_scope == "given2rank1"
    assert not rank2.fit_temperature
    assert rank2.temperature_initializer.value == 1.0

    # All optional settings.
    rank = psiz.keras.layers.RankSimilarity(
        n_reference=8,
        n_select=2,
        percept=percept,
        kernel=kernel,
        data_scope="abc",
        name="rs2",
        fit_temperature=True,
        temperature_initializer=tf.keras.initializers.Constant(0.01),
    )
    cfg = rank.get_config()
    # Verify.
    assert cfg["name"] == "rs2"
    assert cfg["trainable"] is True
    assert cfg["dtype"] == "float32"
    assert cfg["n_reference"] == 8
    assert cfg["n_select"] == 2
    assert cfg["data_scope"] == "abc"
    assert cfg["fit_temperature"]
    assert cfg["temperature_initializer"]["class_name"] == "Constant"
    assert cfg["temperature_initializer"]["config"]["value"] == 0.01

    rank2 = psiz.keras.layers.RankSimilarity.from_config(cfg)
    assert rank2.name == "rs2"
    assert rank2.trainable is True
    assert rank2.dtype == "float32"
    assert rank2.n_reference == 8
    assert rank2.n_select == 2
    assert rank2.data_scope == "abc"
    assert rank2.fit_temperature
    rank2.temperature_initializer.value == 0.01
