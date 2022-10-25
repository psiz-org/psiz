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

from psiz.keras.constraints import GreaterThan


def test_all():
    """Test all methods."""
    # Initialize.
    con = GreaterThan(min_value=0.1)
    assert con.min_value == 0.1

    # Check get_config.
    config = con.get_config()
    assert config['min_value'] == 0.1

    # Check call.
    w0 = tf.constant(
        [
            [1.36, -0.35],
            [1.40, -0.41]
        ], dtype=tf.float32
    )
    w1 = con(w0)
    w_desired = tf.constant(
        [
            [1.36, 0.1],
            [1.40, 0.1]
        ], dtype=tf.float32
    )
    tf.debugging.assert_near(w_desired, w1)
