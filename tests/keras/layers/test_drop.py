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
"""Test SubnetGate."""

import numpy as np
import pytest
import tensorflow as tf

from psiz.keras.layers import Drop


class Dummy(tf.keras.layers.Layer):
    """A simple layer that passes inputs as outputs."""

    def __init__(self, **kwargs):
        """Initialize."""
        super(Dummy, self).__init__(**kwargs)

    def call(self, inputs):
        """Call."""
        return inputs


@pytest.fixture
def inputs_v0():
    """A minibatch inputs."""
    # Create a simple batch (batch_size=5).
    inputs = tf.constant(
        np.array(
            [
                [[0, 1, 2], [3, 4, 5]],
                [[0, 1, 2], [3, 4, 5]],
                [[0, 1, 2], [3, 4, 5]],
                [[6, 7, 8], [9, 10, 11]],
                [[6, 7, 8], [9, 10, 11]]
            ], dtype=np.int32
        )
    )
    return inputs


@pytest.fixture
def inputs_v1():
    """A minibatch inputs."""
    # Create a simple batch (batch_size=5).
    inputs = tf.constant(
        np.array(
            [
                [[0.1, 1.1, 2.1], [3.1, 4.1, 5.1]],
                [[0.1, 1.1, 2.1], [3.1, 4.1, 5.1]],
                [[0.1, 1.1, 2.1], [3.1, 4.1, 5.1]],
                [[6.1, 7.1, 8.1], [9.1, 10.1, 11.1]],
                [[6.1, 7.1, 8.1], [9.1, 10.1, 11.1]]
            ], dtype=np.int32
        )
    )
    return inputs


@pytest.fixture
def inputs_v2():
    """A minibatch inputs."""
    # Create a simple batch (batch_size=5).
    inputs = tf.constant(
        np.array(
            [
                [0.2, 0.2, 0.2],
                [0.2, 1.2, 0.2],
                [0.2, 2.2, 0.2],
                [0.2, 1.2, 1.2],
                [0.2, 2.2, 1.2]
            ], dtype=np.int32
        )
    )
    return inputs


def test_call_default(
        inputs_v0, inputs_v1, inputs_v2):
    """Test call that does not require an internal reshape."""
    subnet = Dummy()
    layer = Drop(subnet=subnet)

    outputs = layer([inputs_v0, inputs_v1, inputs_v2])

    assert len(outputs) == 2
    tf.debugging.assert_equal(
        inputs_v0, outputs[0]
    )
    tf.debugging.assert_equal(
        inputs_v1, outputs[1]
    )


def test_call_idx0(
        inputs_v0, inputs_v1, inputs_v2):
    """Test call that does not require an internal reshape."""
    subnet = Dummy()
    layer = Drop(subnet=subnet, drop_index=0)

    outputs = layer([inputs_v0, inputs_v1, inputs_v2])

    assert len(outputs) == 2
    tf.debugging.assert_equal(
        inputs_v1, outputs[0]
    )
    tf.debugging.assert_equal(
        inputs_v2, outputs[1]
    )


def test_call_idx1(
        inputs_v0, inputs_v1, inputs_v2):
    """Test call that does not require an internal reshape."""
    subnet = Dummy()
    layer = Drop(subnet=subnet, drop_index=1)

    outputs = layer([inputs_v0, inputs_v1, inputs_v2])

    assert len(outputs) == 2
    tf.debugging.assert_equal(
        inputs_v0, outputs[0]
    )
    tf.debugging.assert_equal(
        inputs_v2, outputs[1]
    )


def test_call_idx2(
        inputs_v0, inputs_v1, inputs_v2):
    """Test call that does not require an internal reshape."""
    subnet = Dummy()
    layer = Drop(subnet=subnet, drop_index=2)

    outputs = layer([inputs_v0, inputs_v1, inputs_v2])

    assert len(outputs) == 2
    tf.debugging.assert_equal(
        inputs_v0, outputs[0]
    )
    tf.debugging.assert_equal(
        inputs_v1, outputs[1]
    )
