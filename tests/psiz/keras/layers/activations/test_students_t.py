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

from psiz.keras.layers import StudentsTSimilarity


def test_init_default():
    """Test default initialization."""
    similarity = StudentsTSimilarity()

    assert similarity.fit_tau
    assert similarity.fit_alpha


def test_init_options_0():
    """Test initialization with optional arguments."""
    similarity = StudentsTSimilarity(
        fit_tau=False,
        fit_alpha=False,
        tau_initializer=tf.keras.initializers.Constant(1.0),
        alpha_initializer=tf.keras.initializers.Constant(1.2),
    )

    assert not similarity.fit_tau
    assert not similarity.fit_alpha


def test_call():
    """Test call."""
    similarity = StudentsTSimilarity(
        tau_initializer=tf.keras.initializers.Constant(2.0),
        alpha_initializer=tf.keras.initializers.Constant(1.0),
    )

    dist = tf.constant(
        [[0.68166146, 1.394038], [0.81919687, 1.25966185]], dtype=tf.float32
    )
    s_actual = similarity(dist)

    s_desired = tf.constant(
        [[0.68275124, 0.33974987], [0.5984142, 0.3865858]], dtype=tf.float32
    )
    tf.debugging.assert_near(s_actual, s_desired)


def test_get_config():
    similarity = StudentsTSimilarity()
    config = similarity.get_config()

    assert config["fit_tau"]
    assert config["fit_alpha"]
