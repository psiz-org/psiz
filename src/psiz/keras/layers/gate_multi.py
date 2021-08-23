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
    GateMulti: A layer that manages group-specific subnetworks
        that consume a list of inputs.

"""

import tensorflow as tf
from psiz.keras.sparse_dispatcher import SparseDispatcher


@tf.keras.utils.register_keras_serializable(
    package='psiz.keras', name='GateMulti'
)
class GateMulti(tf.keras.layers.Layer):
    """A layer that manages group-specific subnetworks.

    The subnetworks can take a list of inputs, but each subnetwork must
    output a single tensor.The output shape must be the same for
    all subnetworks.

    Note: All subnetworks must be able to handle batch_size=0 calls. If
    a subnetwork cannot handle this, wierd error messages may be
    genreated. For example, if your layer makes a
    tensorflow-probability distribution sample call with a zero-sized
    shape, the program will crash. A simple work-around is to wrap
    these calls in a protective conditional. For example:
    ```
    x = inputs[0]
    x_shape = tf.shape(x)
    batch_size = tf.shape(x)[0]
    dist_samples = tf.cond(
        batch_size == 0,
        lambda: tf.zeros(x_shape),
        lambda: self.my_distribution.sample(x_shape)
    )
    ```

    """
    def __init__(self, subnets=None, group_col=0, **kwargs):
        """Initialize.

        Arguments:
            subnets: A non-empty list of sub-networks. It is assumed
                that all subnetworks have the same `input_shape` and
                `output_shape`.
            group_col: The group column on which to gate inputs to the
                subnetworks.

        Raises:
            ValueError if subnetwork's non-batch output shape is not
            fully defined.

        """
        super(GateMulti, self).__init__(**kwargs)
        self.subnets = subnets
        self.n_subnet = len(subnets)
        self.group_col = group_col

    def build(self, inputs_shape):
        """Build."""
        # Pop group tensor.
        input_shape_less_group = inputs_shape[0:-1]

        # Build subnets.
        for subnet in self.subnets:
            subnet.build(input_shape_less_group)

        super().build(inputs_shape)

    def call(self, inputs):
        """Call.

        Arguments:
            inputs: a n-tuple containing a data Tensor and a trialing
                group Tensor.
                data Tensors: shape=(batch, m, [n, ...])
                group Tensor: shape=(batch, g)

        """
        # Split inputs.
        inputs_less_group = inputs[0:-1]
        idx_group = inputs[-1][:, self.group_col]
        idx_group = tf.one_hot(
            idx_group, self.n_subnet, on_value=1.0, off_value=0.0
        )

        # Run inputs through group-specific dispatcher.
        dispatcher = SparseDispatcher(self.n_subnet, idx_group)
        subnet_inputs = dispatcher.dispatch_multi(inputs_less_group)
        subnet_outputs = []
        for i in range(self.n_subnet):
            out = self.subnets[i](subnet_inputs[i])
            out, lost_shape = self._pre_combine(out)
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

    def compute_output_shape(self, input_shape):
        """Computes the output shape of the layer.

        Arguments:
            input_shape: Shape tuple (tuple of integers) or list of
                shape tuples (one per output tensor of the layer).
                Shape tuples can include None for free dimensions,
                instead of an integer.

        Returns:
            A tf.TensorShape representing the output shape.

        NOTE: This method overrides the TF default, since the default
        cannot infer the correct output shape.

        """
        # Compute output shape for a subnetwork without passing in group
        # information.
        return self.subnets[0].compute_output_shape(input_shape[0:-1])

    def _pre_combine(self, x):
        """Prepare Tensor for combine operation.

        All non-batch dimensions must be flattened together to create a
        final 2D Tensor.

        Arguments:
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
