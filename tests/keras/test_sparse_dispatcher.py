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

import numpy as np
import pytest
import tensorflow as tf

from psiz.keras.sparse_dispatcher import SparseDispatcher


class SimpleAdd(tf.keras.layers.Layer):
    """A simple Add layer."""

    def __init__(self, v, **kwargs):
        """Initialize."""
        super(SimpleAdd, self).__init__(**kwargs)
        self.v = tf.constant(v)

    def call(self, inputs):
        """Call."""
        return inputs + self.v


@pytest.fixture
def gates_v0():
    """A minibatch of gates."""
    # Create a simple batch (batch_size=5).
    gates = tf.constant(
        np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, .3, .7],
                [.5, 0, 0]
            ], dtype=np.float32
        )
    )
    return gates


@pytest.fixture
def inputs_v0():
    """A minibatch of non-gate inupts."""
    # Create a simple batch (batch_size=5).
    inputs = tf.constant(
        np.array(
            [
                [0.0, 0.1, 0.2],
                [1.0, 1.1, 1.2],
                [2.0, 2.1, 2.2],
                [3.0, 3.1, 3.2],
                [4.0, 4.1, 4.2]
            ], dtype=np.float32
        )
    )
    return inputs


def test_basic(gates_v0,  inputs_v0):
    """Test basic usage."""
    # gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    # inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    # experts: a list of length `num_experts` containing sub-networks.

    n_expert = 3
    expert_0 = SimpleAdd(0.00)
    expert_1 = SimpleAdd(0.01)
    expert_2 = SimpleAdd(0.02)

    experts = [
        expert_0, expert_1, expert_2
    ]

    # Initialize and run inputs through dispatcher.
    dispatcher = SparseDispatcher(n_expert, gates_v0)
    expert_inputs = dispatcher.dispatch_single(inputs_v0)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(n_expert)]

    # Test `multiply_by_gates=True`.
    outputs = dispatcher.combine(expert_outputs, multiply_by_gates=True)
    desired_outputs = np.array([
        [0.0, 0.1, 0.2],
        [1.01, 1.11, 1.21],
        [2.02, 2.12, 2.22],
        [3.017, 3.117, 3.217],
        [2.0, 2.05, 2.1]
    ])
    np.testing.assert_array_almost_equal(outputs.numpy(), desired_outputs)

    # Test `multiply_by_gates=False`. This does not weight by gates, but still
    # sums up all expert outputs.
    outputs = dispatcher.combine(expert_outputs, multiply_by_gates=False)
    desired_outputs = np.array([
        [0.0, 0.1, 0.2],
        [1.01, 1.11, 1.21],
        [2.02, 2.12, 2.22],
        [6.03, 6.23, 6.43],
        [4.0, 4.1, 4.2]
    ])
    np.testing.assert_array_almost_equal(outputs.numpy(), desired_outputs)
