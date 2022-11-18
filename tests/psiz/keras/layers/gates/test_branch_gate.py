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
"""Test BranchGate."""

import numpy as np
import pytest
import tensorflow as tf

from psiz.keras.layers.gates.branch_gate import BranchGate


# Copied from test_sparse_dispatcher:Increment.
class Increment(tf.keras.layers.Layer):
    """A simple layer that increments input by a value."""

    def __init__(self, v, **kwargs):
        """Initialize."""
        super(Increment, self).__init__(**kwargs)
        self.v = tf.constant(v)

    def call(self, inputs):
        """Call."""
        return inputs + self.v


# Copied from test_sparse_dispatcher:AddPairs.
class AddPairs(tf.keras.layers.Layer):
    """A simple layer that increments input by a value."""

    def __init__(self, v, **kwargs):
        """Initialize."""
        super(AddPairs, self).__init__(**kwargs)
        self.v = tf.constant(v)

    def call(self, inputs):
        """Call."""
        return inputs[0] + inputs[1] + self.v


class Select(tf.keras.layers.Layer):
    """A simple layer that selects inputs of a specified index."""

    def __init__(self, index, **kwargs):
        """Initialize."""
        super(Select, self).__init__(**kwargs)
        self.index = index

    def call(self, inputs):
        """Call.

        Assumes inputs are at least rank-2.
        """
        return inputs[:, self.index]


class IncrementDict(tf.keras.layers.Layer):
    """A simple layer that increments input by a value."""

    def __init__(self, v, **kwargs):
        """Initialize."""
        super(IncrementDict, self).__init__(**kwargs)
        self.v = tf.constant(v)

    def call(self, inputs):
        """Call."""
        return inputs['inputs_0'] + self.v


# Copied from test_sparse_dispatcher:AddPairsDict.
class AddPairsDict(tf.keras.layers.Layer):
    """A simple layer that increments input by a value."""

    def __init__(self, v, **kwargs):
        """Initialize."""
        super(AddPairsDict, self).__init__(**kwargs)
        self.v = tf.constant(v)

    def call(self, inputs, mask=None):
        """Call."""
        return inputs['inputs_0'] + inputs['inputs_1'] + self.v


@pytest.fixture
def inputs_5x1_v0():
    """A minibatch of non-gate inupts."""
    # Create a simple batch (batch_size=5).
    inputs_0 = tf.constant(
        np.array(
            [
                [0.0],
                [1.0],
                [2.0],
                [3.0],
                [4.0]
            ], dtype=np.float32
        )
    )
    return inputs_0


@pytest.fixture
def inputs_5x3_v0():
    """A minibatch of non-gate inupts."""
    # Create a simple batch (batch_size=5).
    inputs_0 = tf.constant(
        [
            [0.0, 0.1, 0.2],
            [1.0, 1.1, 1.2],
            [2.0, 2.1, 2.2],
            [3.0, 3.1, 3.2],
            [4.0, 4.1, 4.2]
        ], dtype=np.float32
    )
    return inputs_0


@pytest.fixture
def inputs_5x3_v1():
    inputs_1 = tf.constant(
        np.array(
            [
                [5.0, 5.1, 5.2],
                [6.0, 6.1, 6.2],
                [7.0, 7.1, 7.2],
                [8.0, 8.1, 8.2],
                [9.0, 9.1, 9.2]
            ], dtype=np.float32
        )
    )
    return inputs_1


@pytest.fixture
def inputs_5x3x2_v0():
    """A minibatch of non-gate inupts."""
    # Create a simple batch (batch_size=5).
    inputs_0 = tf.constant(
        np.array(
            [
                [[0.0, 0.1], [0.2, 0.3], [0.4, 0.5]],
                [[1.0, 1.1], [1.2, 1.3], [1.4, 1.5]],
                [[2.0, 2.1], [2.2, 2.3], [2.4, 2.5]],
                [[3.0, 3.1], [3.2, 3.3], [3.4, 3.5]],
                [[4.0, 4.1], [4.2, 4.3], [4.4, 4.5]]
            ], dtype=np.float32
        )
    )
    return inputs_0


@pytest.fixture
def groups_v0_0():
    """A minibatch of group indices."""
    # Create a simple batch (batch_size=5).
    groups = tf.constant(
        [
            [0],
            [0],
            [0],
            [0],
            [0]
        ], dtype=tf.int32
    )
    return groups


@pytest.fixture
def groups_v0_1():
    """A minibatch of group indices."""
    # Create a simple batch (batch_size=5).
    groups = tf.constant(
        [
            [0],
            [1],
            [2],
            [1],
            [2]
        ], dtype=tf.int32
    )
    return groups


@pytest.fixture
def groups_v0_2():
    """A minibatch of group indices."""
    # Create a simple batch (batch_size=5).
    groups = tf.constant(
        [
            [0],
            [0],
            [0],
            [1],
            [1]
        ], dtype=tf.int32
    )
    return groups


@pytest.fixture
def groups_v1_12():
    """A minibatch of group indices.

    Disjoint gate weights.

    """
    # Create a simple batch (batch_size=5).
    groups = tf.constant(
        [
            [1, 0],
            [1, 0],
            [1, 0],
            [0, 1],
            [0, 1]
        ], dtype=np.int32
    )
    return groups


@pytest.fixture
def groups_v2_12():
    """A minibatch of group indices.

    Overlapping gate weights.

    """
    # Create a simple batch (batch_size=5).
    groups = tf.constant(
        [
            [1, 0],
            [1, 0],
            [1, 1],
            [0, 1],
            [0, 1]
        ], dtype=np.int32
    )
    return groups


@pytest.fixture
def groups_5x3x3_index_v0_2():
    """A minibatch of group indices.

    * 5 batches
    * index "gate weight" columns
    * 3 timesteps

    """
    # Create a simple batch (batch_size=5).
    groups = tf.constant(
        [
            [[0], [0], [0]],
            [[1], [1], [1]],
            [[0], [0], [0]],
            # Last two batches intentionally have different groups for
            # each timestep.
            [[1], [1], [0]],
            [[1], [0], [1]]
        ], dtype=tf.int32
    )
    return groups


def test_init_options():
    """Test init options."""
    branch = BranchGate(
        subnets=[Increment(-0.1), Increment(0.1)], gating_index=-1,
        name='branch5x1'
    )
    assert branch.gating_index == -1


def test_bad_instantiation_tuple(inputs_5x1_v0, groups_v0_2):
    """Test bad instantiation."""
    inputs_v0 = inputs_5x1_v0
    inputs = [inputs_v0, groups_v0_2]

    # Test bad instantiation that is missing `gating_index`.
    branch = BranchGate(
        subnets=[Increment(-0.1), Increment(0.1)], name='branch5x1'
    )

    with pytest.raises(Exception) as e_info:
        _ = branch(inputs)
    assert e_info.type == ValueError


def test_call_2g_5x1_disjoint_viaindex(inputs_5x1_v0, groups_v0_2):
    """Test call."""
    inputs_v0 = inputs_5x1_v0

    holder = tf.constant([0.]) - tf.constant(0.1)
    incremented = inputs_v0 - tf.constant(0.1)
    desired_output_br0 = tf.stack(
        [incremented[0], incremented[1], incremented[2], holder, holder],
        axis=0
    )
    holder = tf.constant([0.]) + tf.constant(0.1)
    incremented = inputs_v0 + tf.constant(0.1)
    desired_output_br1 = tf.stack(
        [holder, holder, holder, incremented[3], incremented[4]],
        axis=0
    )

    inputs = [inputs_v0, groups_v0_2]

    # Test default behavior when `output_names` is not provided.
    branch = BranchGate(
        subnets=[Increment(-0.1), Increment(0.1)], gating_index=-1,
        name='branch5x1'
    )
    outputs = branch(inputs)
    tf.debugging.assert_equal(outputs['branch5x1_0'], desired_output_br0)
    tf.debugging.assert_equal(outputs['branch5x1_1'], desired_output_br1)

    # Test behavior when `output_names` is provided.
    branch = BranchGate(
        subnets=[Increment(-0.1), Increment(0.1)], gating_index=-1,
        name='branch5x1', output_names=['br_a', 'br_b']
    )
    outputs = branch(inputs)
    tf.debugging.assert_equal(outputs['br_a'], desired_output_br0)
    tf.debugging.assert_equal(outputs['br_b'], desired_output_br1)


def test_call_2g_5x1_disjoint(inputs_5x1_v0, groups_v1_12):
    """Test call."""
    inputs_v0 = inputs_5x1_v0

    holder = tf.constant([0.]) - tf.constant(0.1)
    incremented = inputs_v0 - tf.constant(0.1)
    desired_output_br0 = tf.stack(
        [incremented[0], incremented[1], incremented[2], holder, holder],
        axis=0
    )
    holder = tf.constant([0.]) + tf.constant(0.1)
    incremented = inputs_v0 + tf.constant(0.1)
    desired_output_br1 = tf.stack(
        [holder, holder, holder, incremented[3], incremented[4]],
        axis=0
    )

    inputs = [inputs_v0, groups_v1_12]

    # Test default behavior when `output_names` is not provided.
    branch = BranchGate(
        subnets=[Increment(-0.1), Increment(0.1)], gating_index=-1,
        name='branch5x1'
    )
    outputs = branch(inputs)
    tf.debugging.assert_equal(outputs['branch5x1_0'], desired_output_br0)
    tf.debugging.assert_equal(outputs['branch5x1_1'], desired_output_br1)

    # Test behavior when `output_names` is provided.
    branch = BranchGate(
        subnets=[Increment(-0.1), Increment(0.1)], gating_index=-1,
        name='branch5x1', output_names=['br_a', 'br_b']
    )
    outputs = branch(inputs)
    tf.debugging.assert_equal(outputs['br_a'], desired_output_br0)
    tf.debugging.assert_equal(outputs['br_b'], desired_output_br1)


def test_call_2g_5x1_intersecting(inputs_5x1_v0, groups_v2_12):
    """Test call."""
    inputs_v0 = inputs_5x1_v0

    holder = tf.constant([0.]) - tf.constant(0.1)
    incremented = inputs_v0 - tf.constant(0.1)
    desired_output_br0 = tf.stack(
        [incremented[0], incremented[1], incremented[2], holder, holder],
        axis=0
    )
    holder = tf.constant([0.]) + tf.constant(0.1)
    incremented = inputs_v0 + tf.constant(0.1)
    desired_output_br1 = tf.stack(
        [holder, holder, incremented[2], incremented[3], incremented[4]],
        axis=0
    )

    inputs = [inputs_v0, groups_v2_12]

    # Test default behavior when `output_names` is not provided.
    branch = BranchGate(
        subnets=[Increment(-0.1), Increment(0.1)], gating_index=-1,
        name='branch5x1'
    )
    outputs = branch(inputs)
    tf.debugging.assert_equal(outputs['branch5x1_0'], desired_output_br0)
    tf.debugging.assert_equal(outputs['branch5x1_1'], desired_output_br1)

    # Test behavior when `output_names` is provided.
    branch = BranchGate(
        subnets=[Increment(-0.1), Increment(0.1)], gating_index=-1,
        name='branch5x1', output_names=['br_a', 'br_b']
    )
    outputs = branch(inputs)
    tf.debugging.assert_equal(outputs['br_a'], desired_output_br0)
    tf.debugging.assert_equal(outputs['br_b'], desired_output_br1)


def test_call_2g_5x3(inputs_5x3_v0, inputs_5x3_v1, groups_v0_2):
    """Test call."""
    inputs_v0 = inputs_5x3_v0
    inputs_v1 = inputs_5x3_v1

    holder = tf.constant([0., 0., 0.]) - tf.constant(0.1)
    added = inputs_v0 + inputs_v1 - tf.constant(0.1)
    desired_output_br0 = tf.stack(
        [added[0], added[1], added[2], holder, holder],
        axis=0
    )
    holder = tf.constant([0., 0., 0.]) + tf.constant(0.1)
    added = inputs_v0 + inputs_v1 + tf.constant(0.1)
    desired_output_br1 = tf.stack(
        [holder, holder, holder, added[3], added[4]],
        axis=0
    )

    inputs = [inputs_v0, inputs_v1, groups_v0_2]

    # Test default behavior when `output_names` is not provided.
    branch = BranchGate(
        subnets=[AddPairs(-0.1), AddPairs(0.1)], gating_index=-1,
        name='branch5x3'
    )
    outputs = branch(inputs)
    tf.debugging.assert_equal(outputs['branch5x3_0'], desired_output_br0)
    tf.debugging.assert_equal(outputs['branch5x3_1'], desired_output_br1)

    # Test behavior when `output_names` is provided.
    branch = BranchGate(
        subnets=[AddPairs(-0.1), AddPairs(0.1)], gating_index=-1,
        name='branch5x3', output_names=['br_a', 'br_b']
    )
    outputs = branch(inputs)
    tf.debugging.assert_equal(outputs['br_a'], desired_output_br0)
    tf.debugging.assert_equal(outputs['br_b'], desired_output_br1)


def test_call_3g_5x3x2(inputs_5x3x2_v0, groups_v0_1):
    """Test call.

    Each subnetwork simply selects a subset of the input based on the
    second input dimension.

    """
    inputs_v0 = inputs_5x3x2_v0

    # NOTE: The Select layers are provided indices in the order [1, 0, 2]
    # just to mix things up. Thus inputs_v0[x, y] where y refeflects that
    # order; x is just the batch index.
    holder = tf.zeros([2], tf.float32)
    desired_output_br0 = tf.stack(
        [inputs_v0[0, 1], holder, holder, holder, holder],
        axis=0
    )
    desired_output_br1 = tf.stack(
        [holder, inputs_v0[1, 0], holder, inputs_v0[3, 0], holder],
        axis=0
    )
    desired_output_br2 = tf.stack(
        [holder, holder, inputs_v0[2, 2], holder, inputs_v0[4, 2]],
        axis=0
    )

    inputs = [inputs_v0, groups_v0_1]

    # Test default behavior when `output_names` is not provided.
    branch = BranchGate(
        subnets=[Select(1), Select(0), Select(2)], gating_index=-1,
        name='branch5x3x2'
    )
    outputs = branch(inputs)
    tf.debugging.assert_equal(outputs['branch5x3x2_0'], desired_output_br0)
    tf.debugging.assert_equal(outputs['branch5x3x2_1'], desired_output_br1)
    tf.debugging.assert_equal(outputs['branch5x3x2_2'], desired_output_br2)

    # Test behavior when `output_names` is provided.
    branch = BranchGate(
        subnets=[Select(1), Select(0), Select(2)], gating_index=-1,
        name='branch5x3x2', output_names=['br_a', 'br_b', 'br_c']
    )
    outputs = branch(inputs)
    tf.debugging.assert_equal(outputs['br_a'], desired_output_br0)
    tf.debugging.assert_equal(outputs['br_b'], desired_output_br1)
    tf.debugging.assert_equal(outputs['br_c'], desired_output_br2)


@pytest.mark.parametrize(
    "is_eager", [True, False]
)
def test_fit_5x3_functional(
        inputs_5x3_v0, inputs_5x3_v1, groups_v0_2,
        is_eager):
    """Test call.

    Expect outputs to have zeros as placeholders in batches that were routed
    to a different branch.

    """
    tf.config.run_functions_eagerly(is_eager)
    inputs_v0 = inputs_5x3_v0
    inputs_v1 = inputs_5x3_v1

    # Define model.
    input_0 = tf.keras.Input(shape=(3,), name="data_0")
    input_1 = tf.keras.Input(shape=(3,), name="data_1")
    input_2 = tf.keras.Input(
        type_spec=tf.TensorSpec((None, 1), dtype=tf.dtypes.int32),
        name="groups"
    )

    # NOTE: You have to know TF's default naming scheme to suppy these
    # names ahead of time, i.e, `['<layer.name>', '<layer_name>_0',
    # '<layer_name>_1', ...]`. The best we can do here is anticipate and
    # set the layer name.

    name_branch_0 = "branch"
    name_branch_1 = "branch_1"
    outputs = BranchGate(
        subnets=[AddPairs(-0.1), AddPairs(0.1)], gating_index=-1,
        output_names=[name_branch_0, name_branch_1], name="branch"
    )([input_0, input_1, input_2])
    model = tf.keras.Model(
        inputs=[input_0, input_1, input_2],
        outputs=[outputs[name_branch_0], outputs[name_branch_1]]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=[
            tf.keras.losses.MeanAbsoluteError(name='mae_0'),
            tf.keras.losses.MeanAbsoluteError(name='mae_1'),
        ],
        loss_weights=[1.0, 1.0],
    )

    # Test dataset.
    n_data = 5
    x = {
        'data_0': inputs_v0,
        'data_1': inputs_v1,
        'groups': groups_v0_2,
    }

    # NOTE: When sample weights are appropriately set for each branch, the
    # loss (branch and total loss) should be unaffected by the `holder` value
    # in the target tensor.
    holder = tf.constant([0., 0., 0.])

    # holder = tf.constant([0., 0., 0.]) - tf.constant(0.1)
    added = inputs_v0 + inputs_v1 - tf.constant(0.1)
    targets_0 = tf.stack(
        [added[0], added[1], added[2], holder, holder],
        axis=0
    )
    # holder = tf.constant([0., 0., 0.]) + tf.constant(0.1)
    added = inputs_v0 + inputs_v1 + tf.constant(0.1)
    targets_1 = tf.stack(
        [holder, holder, holder, added[3], added[4]],
        axis=0
    )

    y = {name_branch_0: targets_0, name_branch_1: targets_1}

    w_0 = tf.constant([1., 1., 1., 0., 0.])
    w_1 = tf.constant([0., 0., 0., 1., 1.])
    w = {name_branch_0: w_0, name_branch_1: w_1}

    # Wrap it all together as a Dataset and fit for two epochs.
    tfds = tf.data.Dataset.from_tensor_slices((x, y, w)).batch(
        n_data, drop_remainder=False
    )
    history = model.fit(tfds, epochs=2, verbose=1)

    # NOTE: There are no trainable parameters, so the loss is predictable and
    # does not change across epochs.
    zeros_2epochs = [0.0, 0.0]
    assert history.history['loss'] == zeros_2epochs
    assert history.history[name_branch_0 + '_loss'] == zeros_2epochs
    assert history.history[name_branch_1 + '_loss'] == zeros_2epochs


@pytest.mark.parametrize(
    "is_eager", [True, False]
)
def test_fit_5x3_subclass(inputs_5x3_v0, inputs_5x3_v1, groups_v0_2, is_eager):
    """Test call.

    Expect outputs to have zeros as placeholders in batches that were
    routed to a different branch.

    """
    tf.config.run_functions_eagerly(is_eager)

    inputs_v0 = inputs_5x3_v0
    inputs_v1 = inputs_5x3_v1

    name_branch_0 = "branch_0"
    name_branch_1 = "branch_1"

    # Define model.
    class MyModel(tf.keras.Model):
        """Custom Model"""
        def __init__(self):
            super(MyModel, self).__init__()
            self.branch = BranchGate(
                subnets=[AddPairs(-0.1), AddPairs(0.1)],
                gating_index=-1,
                output_names=[name_branch_0, name_branch_1],
                name="branch"
            )

        def call(self, inputs):
            outputs = self.branch(
                [inputs['data_0'], inputs['data_1'], inputs['groups']]
            )
            return outputs

    model = MyModel()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss={
            name_branch_0: tf.keras.losses.MeanAbsoluteError(name='mae_0'),
            name_branch_1: tf.keras.losses.MeanAbsoluteError(name='mae_1'),
        },
        loss_weights={name_branch_0: 1.0, name_branch_1: 1.0},
    )

    # Create test dataset.
    n_data = 5
    x = {
        'data_0': inputs_v0,
        'data_1': inputs_v1,
        'groups': groups_v0_2
    }

    # NOTE: When sample weights are appropriately set for each branch, the
    # loss (branch and total loss) should be unaffected by the `holder` value
    # in the target tensor.
    holder = tf.constant([0., 0., 0.])

    # holder = tf.constant([0., 0., 0.]) - tf.constant(0.1)
    added = inputs_v0 + inputs_v1 - tf.constant(0.1)
    targets_0 = tf.stack(
        [added[0], added[1], added[2], holder, holder],
        axis=0
    )
    # holder = tf.constant([0., 0., 0.]) + tf.constant(0.1)
    added = inputs_v0 + inputs_v1 + tf.constant(0.1)
    targets_1 = tf.stack(
        [holder, holder, holder, added[3], added[4]],
        axis=0
    )

    y = {name_branch_0: targets_0, name_branch_1: targets_1}

    w_0 = tf.constant([1., 1., 1., 0., 0.])
    w_1 = tf.constant([0., 0., 0., 1., 1.])
    w = {name_branch_0: w_0, name_branch_1: w_1}

    # Wrap it all together as a Dataset and fit for two epochs.
    tfds = tf.data.Dataset.from_tensor_slices((x, y, w)).batch(
        n_data, drop_remainder=False
    )
    history = model.fit(tfds, epochs=2, verbose=1)

    # NOTE: There are no trainable parameters, so the loss is predictable and
    # does not change across epochs.
    zeros_2epochs = [0.0, 0.0]
    assert history.history['loss'] == zeros_2epochs
    assert history.history[name_branch_0 + '_loss'] == zeros_2epochs
    assert history.history[name_branch_1 + '_loss'] == zeros_2epochs


def test_call_2g_5x3x2_timestep(inputs_5x3x2_v0, groups_5x3x3_index_v0_2):
    """Test call using inputs with timestep axis.

    Use `gate_weights` that yield:
    [
        [0, 0, 0],
        [1, 1, 1],
        [0, 0, 0],
        [1, 1, 0],
        [1, 0, 1],
    ]

    """
    inputs_0 = inputs_5x3x2_v0
    groups = groups_5x3x3_index_v0_2

    # Branch 0.
    holder = tf.constant([-0.1, -0.1])
    added = inputs_0 - tf.constant(0.1)
    desired_output_br0 = tf.stack(
        [
            tf.stack([added[0, 0], added[0, 1], added[0, 2]], axis=0),
            tf.stack([holder, holder, holder], axis=0),
            tf.stack([added[2, 0], added[2, 1], added[2, 2]], axis=0),
            tf.stack([holder, holder, added[3, 2]], axis=0),
            tf.stack([holder, added[4, 1], holder], axis=0),
        ], axis=0
    )

    # Branch 1.
    holder = tf.constant([0.1, 0.1])
    added = inputs_0 + tf.constant(0.1)
    desired_output_br1 = tf.stack(
        [
            tf.stack([holder, holder, holder], axis=0),
            tf.stack([added[1, 0], added[1, 1], added[1, 2]], axis=0),
            tf.stack([holder, holder, holder], axis=0),
            tf.stack([added[3, 0], added[3, 1], holder], axis=0),
            tf.stack([added[4, 0], holder, added[4, 2]], axis=0),
        ], axis=0
    )

    inputs = [inputs_0, groups]

    # Test default behavior when `output_names` is not provided.
    branch = BranchGate(
        subnets=[Increment(-0.1), Increment(0.1)], gating_index=-1,
        name='branch5x3'
    )
    outputs = branch(inputs)
    tf.debugging.assert_equal(outputs['branch5x3_0'], desired_output_br0)
    tf.debugging.assert_equal(outputs['branch5x3_1'], desired_output_br1)

    # Test behavior when `output_names` is provided.
    branch = BranchGate(
        subnets=[Increment(-0.1), Increment(0.1)], gating_index=-1,
        name='branch5x3', output_names=['br_a', 'br_b']
    )
    outputs = branch(inputs)
    tf.debugging.assert_equal(outputs['br_a'], desired_output_br0)
    tf.debugging.assert_equal(outputs['br_b'], desired_output_br1)


def test_bad_instantiation_dict(inputs_5x1_v0, groups_v0_2):
    """Test bad instantiation."""
    inputs_v0 = inputs_5x1_v0
    inputs = {'inputs_0': inputs_v0, 'groups': groups_v0_2}

    # Check bad instantiation that is missing `gating_key` argument.
    branch = BranchGate(
        subnets=[IncrementDict(-0.1), IncrementDict(0.1)],
        name='branch5x1'
    )
    with pytest.raises(Exception) as e_info:
        _ = branch(inputs)
    assert e_info.type == ValueError


def test_call_dictinputs_2g_5x1_disjoint_viaindex(inputs_5x1_v0, groups_v0_2):
    """Test call.

    Use `gate_weights` [0, 0, 0, 1, 1].

    """
    inputs_v0 = inputs_5x1_v0

    holder = tf.constant([0.]) - tf.constant(0.1)
    incremented = inputs_v0 - tf.constant(0.1)
    desired_output_br0 = tf.stack(
        [incremented[0], incremented[1], incremented[2], holder, holder],
        axis=0
    )
    holder = tf.constant([0.]) + tf.constant(0.1)
    incremented = inputs_v0 + tf.constant(0.1)
    desired_output_br1 = tf.stack(
        [holder, holder, holder, incremented[3], incremented[4]],
        axis=0
    )

    inputs = {'inputs_0': inputs_v0, 'groups': groups_v0_2}

    # Test default behavior when `output_names` is not provided.
    branch = BranchGate(
        subnets=[IncrementDict(-0.1), IncrementDict(0.1)],
        gating_key='groups', name='branch5x1'
    )
    outputs = branch(inputs)
    tf.debugging.assert_equal(outputs['branch5x1_0'], desired_output_br0)
    tf.debugging.assert_equal(outputs['branch5x1_1'], desired_output_br1)

    # Test behavior when `output_names` is provided.
    branch = BranchGate(
        subnets=[IncrementDict(-0.1), IncrementDict(0.1)],
        gating_key='groups', name='branch5x1',
        output_names=['br_a', 'br_b']
    )
    outputs = branch(inputs)
    tf.debugging.assert_equal(outputs['br_a'], desired_output_br0)
    tf.debugging.assert_equal(outputs['br_b'], desired_output_br1)


def test_call_dictinputs_2g_5x3x2_timestep(
        inputs_5x3x2_v0, groups_5x3x3_index_v0_2):
    """Test call using inputs with timestep axis.

    Use `gate_weights` which yields:
    [
        [0, 0, 0],
        [1, 1, 1],
        [0, 0, 0],
        [1, 1, 0],
        [1, 0, 1],
    ]

    """
    inputs_0 = inputs_5x3x2_v0
    groups = groups_5x3x3_index_v0_2

    # Branch 0.
    holder = tf.constant([-0.1, -0.1])
    added = inputs_0 - tf.constant(0.1)
    desired_output_br0 = tf.stack(
        [
            tf.stack([added[0, 0], added[0, 1], added[0, 2]], axis=0),
            tf.stack([holder, holder, holder], axis=0),
            tf.stack([added[2, 0], added[2, 1], added[2, 2]], axis=0),
            tf.stack([holder, holder, added[3, 2]], axis=0),
            tf.stack([holder, added[4, 1], holder], axis=0),
        ], axis=0
    )

    # Branch 1.
    holder = tf.constant([0.1, 0.1])
    added = inputs_0 + tf.constant(0.1)
    desired_output_br1 = tf.stack(
        [
            tf.stack([holder, holder, holder], axis=0),
            tf.stack([added[1, 0], added[1, 1], added[1, 2]], axis=0),
            tf.stack([holder, holder, holder], axis=0),
            tf.stack([added[3, 0], added[3, 1], holder], axis=0),
            tf.stack([added[4, 0], holder, added[4, 2]], axis=0),
        ], axis=0
    )

    inputs = {'inputs_0': inputs_0, 'groups': groups}

    # Test default behavior when `output_names` is not provided.
    branch = BranchGate(
        subnets=[IncrementDict(-0.1), IncrementDict(0.1)],
        gating_key='groups', name='branch5x3'
    )
    outputs = branch(inputs)
    tf.debugging.assert_equal(outputs['branch5x3_0'], desired_output_br0)
    tf.debugging.assert_equal(outputs['branch5x3_1'], desired_output_br1)

    # Test behavior when `output_names` is provided.
    branch = BranchGate(
        subnets=[IncrementDict(-0.1), IncrementDict(0.1)],
        gating_key='groups',
        name='branch5x3',
        output_names=['br_a', 'br_b']
    )
    outputs = branch(inputs)
    tf.debugging.assert_equal(outputs['br_a'], desired_output_br0)
    tf.debugging.assert_equal(outputs['br_b'], desired_output_br1)
