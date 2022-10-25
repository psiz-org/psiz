# -*- coding: utf-8 -*-
# Copyright 2021 The PsiZ Authors. All Rights Reserved.
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
"""Test similarities module."""

import tensorflow as tf

from psiz.keras.layers import ExponentialSimilarity


def test_init_default():
    """Test default initialization."""
    similarity = ExponentialSimilarity()

    assert similarity.fit_tau
    assert similarity.fit_gamma
    assert similarity.fit_beta


def test_init_options_0():
    """Test initialization with optional arguments."""
    similarity = ExponentialSimilarity(
        fit_tau=False, fit_gamma=False, fit_beta=False,
        tau_initializer=tf.keras.initializers.Constant(1.),
        gamma_initializer=tf.keras.initializers.Constant(0.01),
        beta_initializer=tf.keras.initializers.Constant(10.),
    )

    assert not similarity.fit_tau
    assert not similarity.fit_gamma
    assert not similarity.fit_beta


def test_init_options_1():
    """Test initialization with optional arguments."""
    similarity = ExponentialSimilarity(fit_beta=False)

    assert similarity.fit_tau
    assert similarity.fit_gamma
    assert not similarity.fit_beta


def test_call():
    """Test call."""
    similarity = ExponentialSimilarity(
        tau_initializer=tf.keras.initializers.Constant(2.1),
        gamma_initializer=tf.keras.initializers.Constant(0.001),
        beta_initializer=tf.keras.initializers.Constant(1.11)
    )

    dist = tf.constant(
        [
            [0.68166146, 1.394038],
            [0.81919687, 1.25966185]
        ], dtype=tf.float32
    )
    s_actual = similarity(dist)

    s_desired = tf.constant(
        [
            [0.6097281, 0.10853133],
            [0.48281538, 0.16589916]
        ], dtype=tf.float32
    )
    tf.debugging.assert_near(s_actual, s_desired)


def test_get_config():
    similarity = ExponentialSimilarity()
    config = similarity.get_config()

    assert config['fit_tau']
    assert config['fit_gamma']
    assert config['fit_beta']
