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
"""Module for a TensorFlow GroupEmbedding.

Classes:
    EmbeddingGroup: A layer that manages group-specific subnetworks.

"""

import numpy as np
import tensorflow as tf
from psiz.keras.sparse_dispatcher import SparseDispatcher


@tf.keras.utils.register_keras_serializable(
    package='psiz.keras', name='GroupSpecific'
)
class GroupSpecific(tf.keras.layers.Layer):
    """A layer that manages group-specific subnetworks.

    The output shape must be the same for all subnetworks, this
    includes the sample shape (if any).

    """
    def __init__(self, subnets=None, group_col=0, **kwargs):
        """Initialize."""
        super(GroupSpecific, self).__init__(**kwargs)
        self.subnets = subnets
        self.n_subnet = len(subnets)
        self.group_col = group_col

    def build(self, inputs_shape):
        """Build."""
        # Probe first subnet for `sample_shape` information.
        if hasattr(self.subnets[0], 'n_sample'):
            self.sample_shape = tf.TensorShape(self.subnets[0].n_sample)
        else:
            self.sample_shape = tf.TensorShape(())

        input_shape_less_group = inputs_shape[0]
        output_shape = self.subnets[0].compute_output_shape(
            input_shape_less_group
        )

        # Pre-determine permutation order.
        # Example with sample_shape.rank=2. This example illustrates how
        # the permute order and unpermute order are not the same in general.
        #       [s s b d d ]
        # start [0 1 2 3 4]
        # perm  [2 0 1 3 4]
        # unper [1 2 0 3 4]
        perm_order = [i for i in range(output_shape.rank)]
        unperm_order = [i for i in range(output_shape.rank)]
        current_batch_position = self.sample_shape.rank
        if current_batch_position > 0:
            # Perm order.
            idx = perm_order.pop(current_batch_position)
            perm_order.insert(0, idx)
            # Unperm order.
            idx = unperm_order.pop(0)
            unperm_order.insert(current_batch_position, idx)
        self._perm_order = perm_order
        self._unperm_order = unperm_order

    def call(self, inputs):
        """Call."""
        # Split inputs.
        inputs_less_group = inputs[0]
        idx_group = inputs[-1][:, self.group_col]
        idx_group = tf.one_hot(idx_group, self.n_subnet)

        # Run inputs through group-specific dispatcher.
        dispatcher = SparseDispatcher(self.n_subnet, idx_group)
        subnet_inputs = dispatcher.dispatch(inputs_less_group)
        subnet_outputs = []
        for i in range(self.n_subnet):
            out, lost_shape = self._pre_combine(
                self.subnets[i](subnet_inputs[i])
            )
            subnet_outputs.append(out)
        outputs = dispatcher.combine(subnet_outputs, multiply_by_gates=True)

        # Handle reshaping of output.
        outputs = self._post_combine(outputs, lost_shape)

        return outputs

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        subnets_serial = []
        for i in range(self.n_subnet):
            subnets_serial.append(
                tf.keras.utils.serialize_keras_object(self.subnets[i])
            )
        config.update({
            'subnets': subnets_serial,
            'group_col': int(self.group_col)
        })
        return config

    @classmethod
    def from_config(cls, config):
        subnets_serial = config['subnets']
        subnets = []
        for subnet_serial in subnets_serial:
            subnets.append(
                tf.keras.layers.deserialize(subnet_serial)
            )
        config['subnets'] = subnets
        return super().from_config(config)

    def subnet(self, subnet_idx):
        return self.subnets[subnet_idx]

    def _pre_combine(self, x):
        """Prepare Tensor for combine operation.

        Must first move any `sample_shape` to the right of batch
        dimension. All non-batch dimensions must be flattened together
        to create a final 2D Tensor.

        Arguments:
            x: Data Tensor.

        Returns:
            x_2d: Transformed data Tensor.
            lost_shape: The original (i.e., lost) shape of the non-batch
                dimensions.

        """
        # Start by moving sample dimensions.
        x = tf.transpose(x, self._perm_order)

        # Now flatten non-batch dimensions.
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
            (batch_size, lost_shape), axis=0
        )
        x = tf.reshape(x, desired_shape)

        # Move sample dimensions back to left-hand side.
        x = tf.transpose(x, self._unperm_order)
        return x
