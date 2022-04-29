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
from psiz.keras.layers.experimental.gate import Gate


@tf.keras.utils.register_keras_serializable(
    package='psiz.keras', name='BraidGate'
)
class BraidGate(Gate):
    """A layer that routes inputs to group-specific subnetworks.

    In a `BraidGate` the subnetworks are re-combined at the end (in
    contrast to a `BranchGate`).

    The subnetworks can take a list of inputs, but each subnetwork must
    output a single tensor. The final output shape must be the same for
    all subnetworks.

    For more information see: `psiz.keras.layers.Gate`

    """
    def __init__(
            self, subnets=None, group_col=0, pass_groups=None,
            strip_inputs=None, **kwargs):
        """Initialize.

        Args:
            subnets: see `Gate`.
            group_col (optional): see `Gate`.
            pass_groups (optional): see `Gate`.
            strip_inputs (optional): see `Gate`.

        Raises:
            ValueError if subnetwork's non-batch output shape is not
            fully defined.

        """
        super(BraidGate, self).__init__(
            subnets=subnets, group_col=group_col, pass_groups=pass_groups,
            strip_inputs=strip_inputs, **kwargs)

    def call(self, inputs):
        """Call.

        Args:
            inputs: a n-tuple containing a data Tensor and a trailing
                group Tensor.
                list format: [data Tensor, [data Tensor, ...], groups Tensor]
                data Tensor(s): shape=(batch, m, [n, ...])
                groups Tensor: shape=(batch, g)

        """
        idx_group = inputs[-1][:, self.group_col]
        idx_group = tf.one_hot(
            idx_group, self.n_subnet, on_value=1.0, off_value=0.0
        )

        # Run inputs through dispatcher that routes inputs to correct subnet.
        dispatcher = SparseDispatcher(self.n_subnet, idx_group)
        subnet_inputs = dispatcher.dispatch_multi(inputs)
        subnet_outputs = []
        for i in range(self.n_subnet):
            out = self._processed_subnets[i](subnet_inputs[i])
            out, lost_shape = self._pre_combine(out)
            subnet_outputs.append(out)

        outputs = dispatcher.combine(subnet_outputs, multiply_by_gates=True)

        # Handle reshaping of output.
        outputs = self._post_combine(outputs, lost_shape)

        return outputs

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
        lost_shape = x_shape[1:]
        x = tf.reshape(
            x, [x_shape[0], tf.reduce_prod(lost_shape)]
        )
        return x, lost_shape

    def _post_combine(self, x, lost_shape):
        """Handle post-combine operations."""
        batch_size = tf.expand_dims(tf.shape(x)[0], axis=0)
        # Unflatten non-batch dimensions.
        desired_shape = tf.concat(
            (batch_size, lost_shape), 0
        )
        return tf.reshape(x, desired_shape)
