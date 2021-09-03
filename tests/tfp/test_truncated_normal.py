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
"""Module for testing `truncated_normal.py`."""

import pytest
import tensorflow as tf

from psiz.tfp.distributions import TruncatedNormal
import tensorflow_probability as tfp


def test_psiz_quantile_0():
    """Test psiz quantile method.

    In certain regimes, truncated adjustment is substantial and the
    distribution behaves differently from Normal distribution.

    """
    loc = 0.5
    scale = .2
    low = 1.0
    high = 9999.0

    d1 = TruncatedNormal(loc, scale, low, high)
    q1_25 = d1.quantile(0.25)
    q1_50 = d1.quantile(0.50)
    q1_75 = d1.quantile(0.75)

    d2 = tfp.distributions.Normal(loc, scale)
    q2_25 = d2.quantile(0.25)
    q2_50 = d2.quantile(0.50)
    q2_75 = d2.quantile(0.75)

    tf.debugging.assert_greater(q1_25, q2_25)
    tf.debugging.assert_greater(q1_50, q2_50)
    tf.debugging.assert_greater(q1_75, q2_75)

    tf.debugging.assert_equal(q1_25, tf.constant(1.0200577, dtype=tf.float32))
    tf.debugging.assert_equal(q1_50, tf.constant(1.0473006, dtype=tf.float32))
    tf.debugging.assert_equal(q1_75, tf.constant(1.0914333, dtype=tf.float32))


def test_psiz_quantile_1():
    """Test psiz quantile method.

    In certain regimes, truncated adjustment is negligable and the
    distribution behaves as a Normal distribution.

    """
    loc = 10.
    scale = .2
    low = 1.0
    high = 9999.0

    d1 = TruncatedNormal(loc, scale, low, high)
    q1_25 = d1.quantile(0.25)
    q1_50 = d1.quantile(0.50)
    q1_75 = d1.quantile(0.75)

    d2 = tfp.distributions.Normal(loc, scale)
    q2_25 = d2.quantile(0.25)
    q2_50 = d2.quantile(0.50)
    q2_75 = d2.quantile(0.75)

    tf.debugging.assert_equal(q1_25, q2_25)
    tf.debugging.assert_equal(q1_50, q2_50)
    tf.debugging.assert_equal(q1_75, q2_75)


@pytest.mark.xfail(
    reason="tfp does not implement quantile yet", raises=NotImplementedError
)
def test_tfp_quantile():
    """Test psiz quantile method."""
    loc = 0.5
    scale = .2
    low = 1.0
    high = 9999.0

    d2 = tfp.distributions.TruncatedNormal(
        loc, scale, low, high
    )
    q2_25 = d2.quantile(0.25)
    q2_50 = d2.quantile(0.50)
    q2_75 = d2.quantile(0.75)

    tf.debugging.assert_equal(q2_25, tf.constant(1.0200577, dtype=tf.float32))
    tf.debugging.assert_equal(q2_50, tf.constant(1.0473006, dtype=tf.float32))
    tf.debugging.assert_equal(q2_75, tf.constant(1.0914333, dtype=tf.float32))
