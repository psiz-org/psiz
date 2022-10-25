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
"""Test constraints module."""

import tensorflow as tf

from psiz.keras.constraints import NonNegNorm


def test_init_config():
    """Test all methods."""
    # Initialize.
    con = NonNegNorm(scale=1.0, p=2.0, axis=1)
    assert con.scale == 1.0
    assert con.p == 2.0
    assert con.axis == 1

    # Check get_config.
    config = con.get_config()
    assert config['scale'] == 1.0
    assert config['p'] == 2.0
    assert config['axis'] == 1


def test_call_l1():
    """Test call method.

    Using L1 norm.

    """
    # Initialize.
    con = NonNegNorm(scale=1.0, p=1.0, axis=1)

    # Check call.
    w0 = tf.constant(
        [
            [1.3, -0.35, 1.3],
            [1.40, 0.41, 1.6]
        ], dtype=tf.float32
    )
    w1 = con(w0)
    w_desired = tf.constant(
        [
            [0.5, 0., 0.5],
            [0.4105572, 0.12023461, 0.46920824]
        ], dtype=tf.float32
    )
    tf.debugging.assert_near(w_desired, w1)


def test_call_l2():
    """Test call method.

    Using L2 norm.

    """
    # Initialize.
    con = NonNegNorm(scale=1.0, p=2.0, axis=1)

    # Check call.
    w0 = tf.constant(
        [
            [1.3, -0.35, 1.3],
            [1.40, 0.41, 1.6]
        ], dtype=tf.float32
    )
    w1 = con(w0)
    w_desired = tf.constant(
        [
            [0.70710677, 0., 0.70710677],
            [0.6465909, 0.18935876, 0.73896104]
        ], dtype=tf.float32
    )
    tf.debugging.assert_near(w_desired, w1)
