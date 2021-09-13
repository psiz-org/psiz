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


class Increment(tf.keras.layers.Layer):
    """A simple layer that increments input by a value."""

    def __init__(self, v, **kwargs):
        """Initialize."""
        super(Increment, self).__init__(**kwargs)
        self.v = tf.constant(v)

    def call(self, inputs):
        """Call."""
        return inputs + self.v


class IncrementPairs(tf.keras.layers.Layer):
    """A simple layer that increments input by a value."""

    def __init__(self, v, **kwargs):
        """Initialize."""
        super(IncrementPairs, self).__init__(**kwargs)
        self.v = tf.constant(v)

    def call(self, inputs):
        """Call."""
        return inputs[0] + inputs[1] + self.v


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
def inputs_single():
    """A minibatch of inputs."""
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


@pytest.fixture
def inputs_multi():
    """A minibatch of a list of inputs."""
    # Create a simple batch (batch_size=5).
    inputs_0 = tf.constant(
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

    inputs_1 = tf.constant(
        np.array(
            [
                [10.0, 10.1, 10.2],
                [11.0, 11.1, 11.2],
                [12.0, 12.1, 12.2],
                [13.0, 13.1, 13.2],
                [14.0, 14.1, 14.2]
            ], dtype=np.float32
        )
    )

    inputs = [inputs_0, inputs_1]
    return inputs


def test_single_dispatch(gates_v0, inputs_single):
    """Test single-input dispatch."""
    # gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    # inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    # experts: a list of length `num_experts` containing sub-networks.

    n_expert = 3
    expert_0 = Increment(0.00)
    expert_1 = Increment(0.01)
    expert_2 = Increment(0.02)

    experts = [expert_0, expert_1, expert_2]

    # Initialize.
    dispatcher = SparseDispatcher(n_expert, gates_v0)

    # Test `expert_to_gates`. These values are determined by `gates_v0`, and
    # reflect the weight of each expert, collapsing across batch size.
    weights = dispatcher.expert_to_gates()
    desired_weight_0 = tf.constant([1., 0.5], dtype=tf.float32)
    desired_weight_1 = tf.constant([1., 0.3], dtype=tf.float32)
    desired_weight_2 = tf.constant([1., 0.7], dtype=tf.float32)
    tf.debugging.assert_equal(weights[0], desired_weight_0)
    tf.debugging.assert_equal(weights[1], desired_weight_1)
    tf.debugging.assert_equal(weights[2], desired_weight_2)

    # Test `expert_to_batch_indices`. These are the corresponding indices or
    # the weights returned by `expert_to_gates`.
    idx = dispatcher.expert_to_batch_indices()
    desired_idx_0 = tf.constant([0, 4], dtype=tf.int32)
    desired_idx_1 = tf.constant([1, 3], dtype=tf.int32)
    desired_idx_2 = tf.constant([2, 3], dtype=tf.int32)
    tf.debugging.assert_equal(idx[0], desired_idx_0)
    tf.debugging.assert_equal(idx[1], desired_idx_1)
    tf.debugging.assert_equal(idx[2], desired_idx_2)

    # Test `part_sizes`. The number of weights for each expert.
    part_sizes = dispatcher.part_sizes
    part_sizes_desired = tf.constant([2, 2, 2], dtype=tf.int32)
    tf.debugging.assert_equal(part_sizes, part_sizes_desired)

    # Run inputs through dispatcher.
    expert_inputs = dispatcher.dispatch_single(inputs_single)
    # Mimic calls to individual experts (i.e., subnets).
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(n_expert)]

    # Test `multiply_by_gates=True`.
    outputs = dispatcher.combine(expert_outputs, multiply_by_gates=True)
    desired_outputs = tf.constant(
        [
            [0.0, 0.1, 0.2],
            [1.01, 1.11, 1.21],
            [2.02, 2.12, 2.22],
            [3.017, 3.117, 3.217],
            [2.0, 2.05, 2.1]
        ], dtype=tf.float32
    )
    tf.debugging.assert_near(outputs, desired_outputs)

    # Test `multiply_by_gates=False`. This does not weight by gates, but still
    # sums up all expert outputs.
    outputs = dispatcher.combine(expert_outputs, multiply_by_gates=False)
    desired_outputs = tf.constant(
        [
            [0.0, 0.1, 0.2],
            [1.01, 1.11, 1.21],
            [2.02, 2.12, 2.22],
            [6.03, 6.23, 6.43],
            [4.0, 4.1, 4.2]
        ], dtype=tf.float32
    )
    tf.debugging.assert_near(outputs, desired_outputs)


def test_multi_dispatch(gates_v0, inputs_multi):
    """Test multi-input dispatch.

    Note that initialization behavior is the same as single-input
    dispatcher. The only thing that changes is the handling of inputs.

    """
    # gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    # inputs: A list of float32 `Tensor` with shape `[batch_size, input_size]`
    # experts: a list of length `num_experts` containing sub-networks.

    n_expert = 3
    expert_0 = IncrementPairs(0.00)
    expert_1 = IncrementPairs(0.01)
    expert_2 = IncrementPairs(0.02)

    experts = [expert_0, expert_1, expert_2]

    # Initialize.
    dispatcher = SparseDispatcher(n_expert, gates_v0)

    # Test `expert_to_gates`. These values are determined by `gates_v0`, and
    # reflect the weight of each expert, collapsing across batch size.
    weights = dispatcher.expert_to_gates()
    desired_weight_0 = tf.constant([1., 0.5], dtype=tf.float32)
    desired_weight_1 = tf.constant([1., 0.3], dtype=tf.float32)
    desired_weight_2 = tf.constant([1., 0.7], dtype=tf.float32)
    tf.debugging.assert_equal(weights[0], desired_weight_0)
    tf.debugging.assert_equal(weights[1], desired_weight_1)
    tf.debugging.assert_equal(weights[2], desired_weight_2)

    # Test `expert_to_batch_indices`. These are the corresponding indices or
    # the weights returned by `expert_to_gates`.
    idx = dispatcher.expert_to_batch_indices()
    desired_idx_0 = tf.constant([0, 4], dtype=tf.int32)
    desired_idx_1 = tf.constant([1, 3], dtype=tf.int32)
    desired_idx_2 = tf.constant([2, 3], dtype=tf.int32)
    tf.debugging.assert_equal(idx[0], desired_idx_0)
    tf.debugging.assert_equal(idx[1], desired_idx_1)
    tf.debugging.assert_equal(idx[2], desired_idx_2)

    # Test `part_sizes`. The number of weights for each expert.
    part_sizes = dispatcher.part_sizes
    part_sizes_desired = tf.constant([2, 2, 2], dtype=tf.int32)
    tf.debugging.assert_equal(part_sizes, part_sizes_desired)

    # Run inputs through dispatcher.
    expert_inputs = dispatcher.dispatch_multi(inputs_multi)
    # Mimic calls to individual experts (i.e., subnets).
    # Note that this simple version only works because there is only one
    # non-batch dimension. You would need to use MultiGate's functionality to
    # handle more complicated case.
    expert_outputs = []
    for idx, expert in enumerate(experts):
        out = expert(expert_inputs[idx])
        expert_outputs.append(out)

    # Test `multiply_by_gates=True`.
    outputs = dispatcher.combine(expert_outputs, multiply_by_gates=True)

    # NOTE: To parse these outputs, note that the hundreds place indicates
    # the expert subnet (i.e., +0.00, +0.01, +0.02) for cases where weight
    # is 1 and there is only one non-zero weight.
    desired_outputs = tf.constant(
        [
            [10.0, 10.200001, 10.4],
            [12.01, 12.210001, 12.41],
            [14.02, 14.220001, 14.42],
            [16.017, 16.217001, 16.417],
            [9.0, 9.1, 9.2]
        ], dtype=tf.float32
    )
    tf.debugging.assert_near(outputs, desired_outputs)

    # Test `multiply_by_gates=False`. This does not weight by gates, but still
    # sums up all expert outputs.
    outputs = dispatcher.combine(expert_outputs, multiply_by_gates=False)
    desired_outputs = tf.constant(
        [
            [10.0, 10.200001, 10.4],
            [12.01, 12.210001, 12.41],
            [14.02, 14.220001, 14.42],
            [32.03, 32.43, 32.83],  # 2 * [16.0, 16.2, 16.4] + .03
            [18.0, 18.2, 18.4]
        ], dtype=tf.float32
    )
    tf.debugging.assert_near(outputs, desired_outputs)
