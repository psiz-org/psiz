# -*- coding: utf-8 -*-
# Copyright 2024 The PsiZ Authors. All Rights Reserved.
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
"""Test Splitter."""


import keras
import tensorflow as tf

from psiz.keras.layers.splitter import Splitter
from psiz.keras.layers import Combiner


class Increment(keras.layers.Layer):
    """A simple layer that increments input by a value."""

    def __init__(self, v, **kwargs):
        """Initialize."""
        super(Increment, self).__init__(**kwargs)
        self.v = tf.constant(v)

    def call(self, inputs):
        """Call."""
        return inputs + self.v


class AddPairs(keras.layers.Layer):
    """A simple layer that increments input by a value."""

    def __init__(self, v, **kwargs):
        """Initialize."""
        super(AddPairs, self).__init__(**kwargs)
        self.v = tf.constant(v)

    def call(self, inputs):
        """Call."""
        return inputs[0] + inputs[1] + self.v


class AddPairsDict(keras.layers.Layer):
    """A simple layer that increments input by a value."""

    def __init__(self, v, **kwargs):
        """Initialize."""
        super(AddPairsDict, self).__init__(**kwargs)
        self.v = tf.constant(v)

    def call(self, inputs):
        """Call."""
        return inputs["inputs_0"] + inputs["inputs_1"] + self.v


def test_single_dispatch(gates_v0, inputs_single):
    """Test single-input split."""
    # gates: a float32 `Tensor` with shape `[batch_size, n_channel]`
    # inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    # experts: a list of length `n_channel` containing sub-networks.

    n_expert = 3
    expert_0 = Increment(0.00)
    expert_1 = Increment(0.01)
    expert_2 = Increment(0.02)

    experts = [expert_0, expert_1, expert_2]

    # Initialize.
    splitter = Splitter(n_expert)

    # Run inputs through splitter.
    expert_inputs = splitter(inputs_single)

    # Mimic calls to individual experts (i.e., subnets).
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(n_expert)]

    # Combine (i.e., mix) expert outputs.
    combiner = Combiner()
    outputs = combiner([gates_v0] + expert_outputs)
    desired_outputs = tf.constant(
        [
            [0.0, 0.1, 0.2],
            [1.01, 1.11, 1.21],
            [2.02, 2.12, 2.22],
            [3.017, 3.117, 3.217],
            [2.0, 2.05, 2.1],
        ],
        dtype=tf.float32,
    )
    tf.debugging.assert_near(outputs, desired_outputs)


def test_list_dispatch(gates_v0, inputs_list):
    """Test list input split.

    Note that initialization behavior is the same as single-input
    splitter. The only thing that changes is the handling of inputs.

    """
    # gates: a float32 `Tensor` with shape `[batch_size, n_channel]`
    # inputs: A list of float32 `Tensor` with shape `[batch_size, input_size]`
    # experts: a list of length `n_channel` containing sub-networks.

    n_expert = 3
    expert_0 = AddPairs(0.00)
    expert_1 = AddPairs(0.01)
    expert_2 = AddPairs(0.02)

    experts = [expert_0, expert_1, expert_2]

    # Initialize.
    splitter = Splitter(n_expert)

    # Run inputs through splitter.
    expert_inputs = splitter(inputs_list)

    # Mimic calls to individual experts (i.e., subnets).
    expert_outputs = []
    for idx, expert in enumerate(experts):
        out = expert(expert_inputs[idx])
        expert_outputs.append(out)

    # Combine (i.e., mix) expert outputs.
    combiner = Combiner()
    outputs = combiner([gates_v0] + expert_outputs)

    # NOTE: To parse these outputs, note that the hundreds place indicates
    # the expert subnet (i.e., +0.00, +0.01, +0.02) for cases where weight
    # is 1 and there is only one non-zero weight.
    desired_outputs = tf.constant(
        [
            [10.0, 10.200001, 10.4],
            [12.01, 12.210001, 12.41],
            [14.02, 14.220001, 14.42],
            [16.017, 16.217001, 16.417],
            [9.0, 9.1, 9.2],
        ],
        dtype=tf.float32,
    )
    tf.debugging.assert_near(outputs, desired_outputs)


def test_list_dispatch_timestep(gates_v0_timestep, inputs_list_timestep):
    """Test list split with timestep axis.

    Note that initialization behavior is the same as single-input
    splitter. The only thing that changes is the handling of inputs.

    Args:
        gates: a float32 `Tensor`
        shape=(batch_size, sequence_length, n_channel)
        inputs: A list of float32 `Tensor`
            shape=(batch_size, sequence_length, input_size)
        experts: A list of length `n_channel` containing sub-networks.

    """

    n_expert = 3
    expert_0 = AddPairs(0.00)
    expert_1 = AddPairs(0.01)
    expert_2 = AddPairs(0.02)

    experts = [expert_0, expert_1, expert_2]

    # Initialize.
    splitter = Splitter(n_expert, has_timestep_axis=True)

    # Run inputs through splitter.
    expert_inputs = splitter(inputs_list_timestep)

    # Mimic calls to individual experts (i.e., subnets).
    expert_outputs = []
    for idx, expert in enumerate(experts):
        out = expert(expert_inputs[idx])
        expert_outputs.append(out)

    # Combine (i.e., mix) expert outputs.
    combiner = Combiner(has_timestep_axis=True)
    outputs = combiner([gates_v0_timestep] + expert_outputs)
    desired_outputs = tf.constant(
        [
            [[10.0, 10.200001, 10.4], [10.02, 10.219999, 10.42]],
            [[12.01, 12.210001, 12.41], [12.030001, 12.23, 12.43]],
            [[14.02, 14.220001, 14.42], [14.040001, 14.24, 14.440001]],
            [[16.017, 16.217001, 16.417], [16.037, 16.237, 16.437]],
            [[9.0, 9.1, 9.2], [9.015, 9.115, 9.215]],
        ],
        dtype=tf.float32,
    )
    tf.debugging.assert_near(outputs, desired_outputs)


def test_dict_dispatch(gates_v0, inputs_dict):
    """Test dictionary input split."""
    # gates: a float32 `Tensor` with shape `[batch_size, n_channel]`
    # inputs: A list of float32 `Tensor` with shape `[batch_size, input_size]`
    # experts: a list of length `n_channel` containing sub-networks.

    n_expert = 3
    expert_0 = AddPairsDict(0.00)
    expert_1 = AddPairsDict(0.01)
    expert_2 = AddPairsDict(0.02)

    experts = [expert_0, expert_1, expert_2]

    # Initialize.
    splitter = Splitter(n_expert)

    # Run inputs through splitter.
    expert_inputs = splitter(inputs_dict)

    # Mimic calls to individual experts (i.e., subnets).
    expert_outputs = []
    for idx, expert in enumerate(experts):
        out = expert(expert_inputs[idx])
        expert_outputs.append(out)

    # Combine (i.e., mix) expert outputs.
    combiner = Combiner()
    outputs = combiner([gates_v0] + expert_outputs)

    # NOTE: To parse these outputs, note that the hundreds place indicates
    # the expert subnet (i.e., +0.00, +0.01, +0.02) for cases where weight
    # is 1 and there is only one non-zero weight.
    desired_outputs = tf.constant(
        [
            [10.0, 10.200001, 10.4],
            [12.01, 12.210001, 12.41],
            [14.02, 14.220001, 14.42],
            [16.017, 16.217001, 16.417],
            [9.0, 9.1, 9.2],
        ],
        dtype=tf.float32,
    )
    tf.debugging.assert_near(outputs, desired_outputs)


def test_dict_dispatch_timestep(gates_v0_timestep, inputs_dict_timestep):
    """Test dictionary split with timestep axis.

    Note that initialization behavior is the same as single-input
    splitter. The only thing that changes is the handling of inputs.

    Args:
        gates: a float32 `Tensor`
        shape=(batch_size, sequence_length, n_channel)
        inputs: A list of float32 `Tensor`
            shape=(batch_size, sequence_length, input_size)
        experts: A list of length `n_channel` containing sub-networks.

    """
    n_expert = 3
    expert_0 = AddPairsDict(0.00)
    expert_1 = AddPairsDict(0.01)
    expert_2 = AddPairsDict(0.02)

    experts = [expert_0, expert_1, expert_2]

    # Initialize.
    splitter = Splitter(n_expert, has_timestep_axis=True)

    # Run inputs through splitter.
    expert_inputs = splitter(inputs_dict_timestep)

    # Mimic calls to individual experts (i.e., subnets).
    expert_outputs = []
    for idx, expert in enumerate(experts):
        out = expert(expert_inputs[idx])
        expert_outputs.append(out)

    # Combine (i.e., mix) expert outputs.
    combiner = Combiner(has_timestep_axis=True)
    outputs = combiner([gates_v0_timestep] + expert_outputs)
    desired_outputs = tf.constant(
        [
            [[10.0, 10.200001, 10.4], [10.02, 10.219999, 10.42]],
            [[12.01, 12.210001, 12.41], [12.030001, 12.23, 12.43]],
            [[14.02, 14.220001, 14.42], [14.040001, 14.24, 14.440001]],
            [[16.017, 16.217001, 16.417], [16.037, 16.237, 16.437]],
            [[9.0, 9.1, 9.2], [9.015, 9.115, 9.215]],
        ],
        dtype=tf.float32,
    )
    tf.debugging.assert_near(outputs, desired_outputs)
