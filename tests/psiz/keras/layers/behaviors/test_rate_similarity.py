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
    rate = psiz.keras.layers.RateSimilarity(percept=percept, kernel=kernel)
    stimulus_set = tf.constant(
        [
            [1, 2],
            [2, 3],
            [3, 4],
            [1, 4],
        ]
    )
    x = {
        "rate2_stimulus_set": stimulus_set,
    }
    rating = rate(x)

    # Desired outcome.
    coords_x = 0.1 * tf.cast(stimulus_set, tf.float32)
    z_q = tf.gather(coords_x, indices=tf.constant([0]), axis=1)
    z_r = tf.gather(coords_x, indices=tf.constant([1]), axis=1)
    d_qr = tf.abs(z_q - z_r)
    s_qr = tf.exp(-tf.constant(beta) * d_qr)

    lower = tf.constant(0.0)
    upper = tf.constant(1.0)
    midpoint = tf.constant(0.5)
    rate = tf.constant(5.0)
    rating_desired = lower + tf.math.divide(
        upper - lower, 1 + tf.math.exp(-rate * (s_qr - midpoint))
    )

    tf.debugging.assert_near(rating, rating_desired)


def test_serialization():
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
    rate = psiz.keras.layers.RateSimilarity(percept=percept, kernel=kernel, name="rs1")
    cfg = rate.get_config()
    # Verify.
    assert cfg["name"] == "rs1"
    assert cfg["trainable"] is True
    assert cfg["dtype"] == "float32"
    assert cfg["lower_trainable"] is False
    assert cfg["upper_trainable"] is False
    assert cfg["midpoint_trainable"] is True
    assert cfg["rate_trainable"] is True
    assert cfg["data_scope"] == "rate2"

    rate2 = psiz.keras.layers.RateSimilarity.from_config(cfg)
    assert rate2.name == "rs1"
    assert rate2.trainable is True
    assert rate2.dtype == "float32"
    assert rate2.lower_trainable is False
    assert rate2.upper_trainable is False
    assert rate2.midpoint_trainable is True
    assert rate2.rate_trainable is True
    assert rate2.data_scope == "rate2"

    # All optional settings.
    rate = psiz.keras.layers.RateSimilarity(
        percept=percept,
        kernel=kernel,
        data_scope="abc",
        lower_trainable=True,
        upper_trainable=True,
        midpoint_trainable=False,
        rate_trainable=False,
        name="rs2",
    )
    cfg = rate.get_config()
    # Verify.
    assert cfg["name"] == "rs2"
    assert cfg["trainable"] is True
    assert cfg["dtype"] == "float32"
    assert cfg["lower_trainable"] is True
    assert cfg["upper_trainable"] is True
    assert cfg["midpoint_trainable"] is False
    assert cfg["rate_trainable"] is False
    assert cfg["data_scope"] == "abc"

    rate2 = psiz.keras.layers.RateSimilarity.from_config(cfg)
    assert rate2.name == "rs2"
    assert rate2.trainable is True
    assert rate2.dtype == "float32"
    assert rate2.lower_trainable is True
    assert rate2.upper_trainable is True
    assert rate2.midpoint_trainable is False
    assert rate2.rate_trainable is False
    assert rate2.data_scope == "abc"
