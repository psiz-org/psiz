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
"""Module for a TensorFlow layers.

Classes:
    BraidGate: A layer that routes inputs to group-specific
        subnetworks.

"""

import tensorflow as tf

from psiz.keras.sparse_dispatcher import SparseDispatcher
from psiz.keras.layers.gates.split_gate import SplitGate


@tf.keras.utils.register_keras_serializable(package="psiz.keras", name="BraidGate")
class BraidGate(SplitGate):
    """A layer that routes inputs to subnetworks.

    In a `BraidGate` the subnetworks are re-combined at the end (in
    contrast to a `BranchGate`).

    The subnetworks can take a list of inputs, but each subnetwork must
    output a single tensor. The final output shape must be the same for
    all subnetworks.

    For more information see `psiz.keras.layers.Gate`

    """

    def __init__(
        self, subnets=None, pass_gate_weights=None, strip_inputs=None, **kwargs
    ):
        """Initialize.

        Args:
            subnets: see `Gate`.
            pass_gate_weights (optional): see `Gate`.
            strip_inputs (optional): see `Gate`.

        Raises:
            ValueError if subnetwork's non-batch output shape is not
            fully defined.

        """
        super(BraidGate, self).__init__(
            subnets=subnets,
            pass_gate_weights=pass_gate_weights,
            strip_inputs=strip_inputs,
            **kwargs
        )

    def call(self, inputs, mask=None):
        """Call.

        Args:
            inputs: Data Tensors. Can be an n-tuple or single-level
                dictionary containing Tensors. If n-tuple, the trailing
                Tensor is assumed to be a "gate weights" Tensor. If a
                dictionary, one of the fields is assumed to be
                `gate_weights`. The tuple format follows
                [data Tensor, [data Tensor, ...], gate_weights Tensor].
                The data Tensor(s) follows shape=(batch, m, [n, ...]).
                The gate_weights Tensor follows shape=(batch, g)
            mask (optional): A Tensor indicating which timesteps should
                be masked.

        """
        gate_weights = self._process_gate_weights(inputs)

        # Run inputs through dispatcher that routes inputs to correct subnet.
        dispatcher = SparseDispatcher(
            self.n_subnet, gate_weights, has_timestep_axis=self._has_timestep_axis
        )
        subnet_inputs = dispatcher.dispatch_multi_pad(inputs)
        subnet_outputs = []
        for i in range(self.n_subnet):
            if mask is None:
                out = self._processed_subnets[i](subnet_inputs[i])
            else:
                out = self._processed_subnets[i](subnet_inputs[i], mask=mask)
            out, lost_shape = self._pre_combine(out)
            # Weight outputs appropriately.
            if self._has_timestep_axis:
                out = out * tf.expand_dims(gate_weights[:, :, i], axis=2)
            else:
                out = out * tf.expand_dims(gate_weights[:, i], axis=1)
            subnet_outputs.append(out)

        outputs = tf.math.add_n(subnet_outputs)

        # Handle reshaping of output.
        outputs = tf.reshape(outputs, lost_shape)

        return outputs

    def compute_output_shape(self, input_shape):
        """Computes the output shape of the layer.

        Args:
            input_shape: Shape tuple (tuple of integers) or list of
                shape tuples (one per output tensor of the layer).
                Shape tuples can include None for free dimensions,
                instead of an integer.

        Returns:
            A tf.TensorShape representing the output shape.

        NOTE: This method overrides the TF default, since the default
        cannot infer the correct output shape.

        """
        # Compute output shape for a subnetwork.
        return self._processed_subnets[0].compute_output_shape(input_shape)

    def _pre_combine(self, x):
        """Prepare Tensor for combine operation.

        All non-batch dimensions must be flattened together to create a
        final 2D Tensor.

        Args:
            x: Data Tensor.

        Returns:
            x: Transformed data Tensor.
            lost_shape: A tensor recording the shape lost during
                reshape.

        """
        # Flatten non-batch dimensions.
        x_shape = tf.shape(x)
        if self._has_timestep_axis:
            x = tf.reshape(x, [x_shape[0], x_shape[1], -1])
        else:
            x = tf.reshape(x, [x_shape[0], -1])
        return x, x_shape
