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
"""Test GateAdapter.

A key part of these tests is showing that using the same "outer layer"
architecture, the "inner layer" can have no gate, a single gate, or
nested gates. In other words, demonstrating that `GateAdapter`
satisfies it's role as an adapter.

"""

import pytest
import tensorflow as tf

from psiz.keras.layers import GateAdapter, BraidGate


class OuterPassesTuple(tf.keras.layers.Layer):
    """A layer that calls inner layer with tuple-formated inputs."""

    def __init__(self, inner=None, inner_gating_keys=None, **kwargs):
        """Initialize."""
        super(OuterPassesTuple, self).__init__(**kwargs)
        self.inner_layer = inner
        self.gate_adapter = GateAdapter(
            gating_keys=inner_gating_keys,
            format_inputs_as_tuple=True
        )
        self.gate_adapter.input_keys = ['x0', 'x1']

    def call(self, inputs):
        """Call."""
        inputs = self.gate_adapter(inputs)
        outputs = self.inner_layer(inputs)
        return outputs


class OuterPassesDict(tf.keras.layers.Layer):
    """A layer that calls inner layer with dictionary-formated inputs."""

    def __init__(self, inner=None, inner_gating_keys=None, **kwargs):
        """Initialize."""
        super(OuterPassesDict, self).__init__(**kwargs)
        self.inner_layer = inner
        self.gate_adapter = GateAdapter(
            gating_keys=inner_gating_keys,
            format_inputs_as_tuple=False
        )
        self.gate_adapter.input_keys = ['x0', 'x1']

    def call(self, inputs):
        """Call."""
        inputs = self.gate_adapter(inputs)
        outputs = self.inner_layer(inputs)
        return outputs


class InnerNeedsDict(tf.keras.layers.Layer):
    """A simple layer that passes inputs as outputs."""

    def __init__(self, factor=None, **kwargs):
        """Initialize."""
        super(InnerNeedsDict, self).__init__(**kwargs)
        self.factor = tf.constant(factor)

    def call(self, inputs):
        """Call."""
        return self.factor * (inputs['x0'] + inputs['x1'])


class InnerNeedsTuple(tf.keras.layers.Layer):
    """A simple layer that passes inputs as outputs."""

    def __init__(self, factor=None, **kwargs):
        """Initialize."""
        super(InnerNeedsTuple, self).__init__(**kwargs)
        self.factor = tf.constant(factor)

    def call(self, inputs):
        """Call."""
        return self.factor * (inputs[0] + inputs[1])


@pytest.fixture
def inputs_x0():
    """A minibatch inputs."""
    # Create a simple batch (batch_size=5).
    inputs = tf.constant(
        [
            [[0., 1., 2.], [3., 4., 5.]],
            [[0., 1., 2.], [3., 4., 5.]],
            [[0., 1., 2.], [3., 4., 5.]],
            [[6., 7., 8.], [9., 10., 11.]],
            [[6., 7., 8.], [9., 10., 11.]]
        ], dtype=tf.float32
    )
    return inputs


@pytest.fixture
def inputs_x1():
    """A minibatch inputs."""
    # Create a simple batch (batch_size=5).
    inputs = tf.constant(
        [
            [[0.1, 1.1, 2.1], [3.1, 4.1, 5.1]],
            [[0.1, 1.1, 2.1], [3.1, 4.1, 5.1]],
            [[0.1, 1.1, 2.1], [3.1, 4.1, 5.1]],
            [[6.1, 7.1, 8.1], [9.1, 10.1, 11.1]],
            [[6.1, 7.1, 8.1], [9.1, 10.1, 11.1]]
        ], dtype=tf.float32
    )
    return inputs


@pytest.fixture
def inputs_groups0():
    """A minibatch of group indices."""
    # Create a simple batch (batch_size=5).
    groups0 = tf.constant(
        [
            [0],
            [0],
            [1],
            [0],
            [1]
        ], dtype=tf.int32
    )
    return groups0


@pytest.fixture
def inputs_groups1():
    """A minibatch of group indices."""
    # Create a simple batch (batch_size=5).
    groups1 = tf.constant(
        [
            [0],
            [1],
            [0],
            [0],
            [0]
        ], dtype=tf.int32
    )
    return groups1


@pytest.fixture
def outputs_desired_v0():
    """Desired outputs."""
    outputs_desired = tf.constant(
        [
            [[0.1, 2.1, 4.1], [6.1, 8.1, 10.1]],
            [[0.1, 2.1, 4.1], [6.1, 8.1, 10.1]],
            [[0.1, 2.1, 4.1], [6.1, 8.1, 10.1]],
            [[12.1, 14.1, 16.1], [18.1, 20.1, 22.1]],
            [[12.1, 14.1, 16.1], [18.1, 20.1, 22.1]]
        ], dtype=tf.float32
    )
    return outputs_desired


@pytest.fixture
def outputs_desired_v1():
    # Just the addition yields:
    # outputs_desired = tf.constant(
    #     [
    #         [[0.1, 2.1, 4.1], [6.1, 8.1, 10.1]],
    #         [[0.1, 2.1, 4.1], [6.1, 8.1, 10.1]],
    #         [[0.1, 2.1, 4.1], [6.1, 8.1, 10.1]],
    #         [[12.1, 14.1, 16.1], [18.1, 20.1, 22.1]],
    #         [[12.1, 14.1, 16.1], [18.1, 20.1, 22.1]]
    #     ], dtype=tf.float32
    # )
    # Addition followed by weighting: [0 0 1 0 1] -> [1. 1. 2. 1. 2.]:
    outputs_desired = tf.constant(
        [
            [[0.1, 2.1, 4.1], [6.1, 8.1, 10.1]],
            [[0.1, 2.1, 4.1], [6.1, 8.1, 10.1]],
            [[0.2, 4.2, 8.2], [12.2, 16.2, 20.2]],
            [[12.1, 14.1, 16.1], [18.1, 20.1, 22.1]],
            [[24.2, 28.2, 32.2], [36.2, 40.2, 44.2]]
        ], dtype=tf.float32
    )
    return outputs_desired


@pytest.fixture
def outputs_desired_v2():
    # Addition followed by weighting: [0 0 1 0 1] -> [1. 1. 2. 1. 2.]:
    # assuming everything goes to branch_a_b
    # groups0 = [0], [0],[1],[0],[1]
    # outputs_desired = tf.constant(
    #     [
    #         [[0.1, 2.1, 4.1], [6.1, 8.1, 10.1]],
    #         [[0.1, 2.1, 4.1], [6.1, 8.1, 10.1]],
    #         [[0.2, 4.2, 8.2], [12.2, 16.2, 20.2]],
    #         [[12.1, 14.1, 16.1], [18.1, 20.1, 22.1]],
    #         [[24.2, 28.2, 32.2], [36.2, 40.2, 44.2]]
    #     ], dtype=tf.float32
    # )

    # c branch on impacts second example, with a x3 factor.
    # groups1 [0] [1], [0], [0], [0]
    outputs_desired = tf.constant(
        [
            [[0.1, 2.1, 4.1], [6.1, 8.1, 10.1]],
            [[0.3, 6.3, 12.3], [18.3, 24.3, 30.3]],
            [[0.2, 4.2, 8.2], [12.2, 16.2, 20.2]],
            [[12.1, 14.1, 16.1], [18.1, 20.1, 22.1]],
            [[24.2, 28.2, 32.2], [36.2, 40.2, 44.2]]
        ], dtype=tf.float32
    )
    return outputs_desired


class TestInnerNeedsTuple:
    """Test when inner layer needs tuple-formated inputs."""

    def test_no_gate(self, inputs_x0, inputs_x1, outputs_desired_v0):
        """Test using a subnetwork that does not have a gate."""
        outputs_desired = outputs_desired_v0
        inputs = {
            'x0': inputs_x0,
            'x1': inputs_x1
        }

        # Assemble and call module.
        inner = InnerNeedsTuple(factor=1., name="inner_a")
        outer = OuterPassesTuple(inner=inner, name="outer")
        outputs = outer(inputs)

        # Verify outputs.
        tf.debugging.assert_near(outputs, outputs_desired)

    def test_with_braid_gate(
            self, inputs_x0, inputs_x1, inputs_groups0, outputs_desired_v1):
        """Test using a subnetwork that has a braid gate."""
        outputs_desired = outputs_desired_v1
        inputs = {
            'x0': inputs_x0,
            'x1': inputs_x1,
            'groups0': inputs_groups0,
        }

        # Assemble and call module.
        inner_a = InnerNeedsTuple(factor=1., name="inner_a")
        inner_b = InnerNeedsTuple(factor=2., name="inner_b")
        branch_a_b = BraidGate(
            subnets=[inner_a, inner_b], gating_index=-1
        )
        outer = OuterPassesTuple(
            inner=branch_a_b,
            inner_gating_keys=['groups0'],
            name="outer"
        )
        outputs = outer(inputs)

        tf.debugging.assert_near(outputs, outputs_desired)

    def test_with_nested_braid_gate(
            self, inputs_x0, inputs_x1, inputs_groups0, inputs_groups1,
            outputs_desired_v2):
        """Test using a subnetwork that has nested braid gates."""
        outputs_desired = outputs_desired_v2
        inputs = {
            'x0': inputs_x0,
            'x1': inputs_x1,
            'groups0': inputs_groups0,
            'groups1': inputs_groups1,
        }

        # Assemble and call module.
        inner_a = InnerNeedsTuple(factor=1., name="inner_a")
        inner_b = InnerNeedsTuple(factor=2., name="inner_b")
        inner_c = InnerNeedsTuple(factor=3., name="inner_b")
        branch_a_b = BraidGate(
            subnets=[inner_a, inner_b], gating_index=-1
        )
        branch_ab_c = BraidGate(
            subnets=[branch_a_b, inner_c], gating_index=-1
        )
        outer = OuterPassesTuple(
            inner=branch_ab_c, inner_gating_keys=['groups0', 'groups1'],
            name="outer"
        )
        outputs = outer(inputs)
        tf.debugging.assert_near(outputs, outputs_desired)


class TestInnerNeedsDict:
    """Test when inner layer needs tuple-formated inputs."""

    def test_no_gate(self, inputs_x0, inputs_x1, outputs_desired_v0):
        """Test using a subnetwork that does not have a gate."""
        outputs_desired = outputs_desired_v0
        inputs = {
            'x0': inputs_x0,
            'x1': inputs_x1
        }

        inner = InnerNeedsDict(factor=1., name="inner_a")
        outer = OuterPassesDict(inner=inner, name="outer")
        outputs = outer(inputs)

        tf.debugging.assert_near(outputs, outputs_desired)

    def test_with_braid_gate(
            self, inputs_x0, inputs_x1, inputs_groups0, outputs_desired_v1):
        """Test using a subnetwork that has a braid gate."""
        outputs_desired = outputs_desired_v1
        inputs = {
            'x0': inputs_x0,
            'x1': inputs_x1,
            'groups0': inputs_groups0,
        }

        inner_a = InnerNeedsDict(factor=1., name="inner_a")
        inner_b = InnerNeedsDict(factor=2., name="inner_b")
        inner = BraidGate(
            subnets=[inner_a, inner_b], gating_key='groups0'
        )
        outer = OuterPassesDict(
            inner=inner, inner_gating_keys=['groups0'], name="outer"
        )
        outputs = outer(inputs)

        tf.debugging.assert_near(outputs, outputs_desired)

    def test_with_nested_braid_gate(
            self, inputs_x0, inputs_x1, inputs_groups0, inputs_groups1,
            outputs_desired_v2):
        """Test using a subnetwork that has nested braid gates."""
        outputs_desired = outputs_desired_v2
        inputs = {
            'x0': inputs_x0,
            'x1': inputs_x1,
            'groups0': inputs_groups0,
            'groups1': inputs_groups1,
        }

        # Assemble and call module.
        inner_a = InnerNeedsDict(factor=1., name="inner_a")
        inner_b = InnerNeedsDict(factor=2., name="inner_b")
        inner_c = InnerNeedsDict(factor=3., name="inner_b")
        branch_a_b = BraidGate(
            subnets=[inner_a, inner_b], gating_key='groups0'
        )
        branch_ab_c = BraidGate(
            subnets=[branch_a_b, inner_c], gating_key='groups1'
        )
        outer = OuterPassesDict(
            inner=branch_ab_c,
            inner_gating_keys=['groups0', 'groups1'],
            name="outer"
        )
        outputs = outer(inputs)
        tf.debugging.assert_near(outputs, outputs_desired)
