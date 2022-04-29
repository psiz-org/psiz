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
"""Module for a TensorFlow layers.

Classes:
    Gate: Abstract layer that routes inputs to group-specific
        subnetworks. Routing behavior depends on concrete class.

"""

import tensorflow as tf

from psiz.keras.layers.experimental.drop import Drop
from psiz.keras.layers.experimental.groups import Groups


@tf.keras.utils.register_keras_serializable(
    package='psiz.keras', name='Gate'
)
class Gate(Groups, tf.keras.layers.Layer):
    """Abstract layer that routes inputs to group-specific subnetworks.

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
    def __init__(
            self, subnets=None, group_col=0, pass_groups=None,
            strip_inputs=None, **kwargs):
        """Initialize.

        Args:
            subnets: A non-empty list of sub-networks. It is assumed
                that all subnetworks have the same `input_shape` and
                `output_shape`.
            group_col (optional): Integer indicating the group column
                on which to gate inputs to the subnetworks.
            pass_groups (optional): Boolean 1D array-like indicating if
                `groups` (last Tensor in `inputs`) should be passed
                to the subnets. By default, this information is not
                passed on to the subnetworks. If provided, the length
                must agree with the number of subnets.
                shape=(n_subnet)
            strip_inputs (optional): Boolean 1D array-like indicating
                if `inputs` to the subnetworks should be stripped to a
                single tensor if the list has only one element. This is
                useful if the subnet network is a native TF layer that
                expects a single Tensor for the  `inputs` arguments
                (e.g., Embedding). By default, this is True. If
                provided, the length must agree with the number of
                subnets.

        Raises:
            ValueError if subnetwork's non-batch output shape is not
            fully defined.

        """
        super(Gate, self).__init__(**kwargs)
        self.n_subnet = len(subnets)

        if pass_groups is None:
            pass_groups = [False] * self.n_subnet
        self.pass_groups = pass_groups

        if strip_inputs is None:
            strip_inputs = [True] * self.n_subnet
        self.strip_inputs = strip_inputs

        processed_subnets = []
        for idx, subnet in enumerate(subnets):
            processed_subnets.append(
                self._process_subnet(
                    subnet, pass_groups[idx], strip_inputs[idx]
                )
            )
        self._processed_subnets = processed_subnets
        self.group_col = group_col
        self.strip_inputs = strip_inputs

    def _process_subnet(self, subnet, pass_groups, strip_inputs):
        """Process subnet.

        Wraps subnet in `Drop` if `pass_group=False`.

        Args:
            subnet: A subnetwork.
            pass_groups: Boolean indicating if `groups` should be
                passed into the subnet.
            strip_inputs: Boolean indicating if `inputs` should be
                stripped to a single Tensor if `inputs` is a list
                containing only one item.

        Returns:
            processed_subnet

        """
        if pass_groups:
            return subnet
        else:
            return Drop(
                subnet=subnet, drop_index=-1, strip_inputs=strip_inputs
            )

    def _unprocess_subnet(self, subnet):
        """Unprocess subnet."""
        if isinstance(subnet, Drop):
            return subnet.subnet
        else:
            return subnet

    @property
    def subnets(self):
        subnets = []
        for subnet in self._processed_subnets:
            subnets.append(self._unprocess_subnet(subnet))
        return subnets

    def build(self, input_shape):
        """Build."""
        # Build subnets.
        for subnet in self._processed_subnets:
            subnet.build(input_shape)
        super().build(input_shape)

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        subnets_serial = []
        for i in range(self.n_subnet):
            subnets_serial.append(
                tf.keras.utils.serialize_keras_object(
                    self._unprocess_subnet(self._processed_subnets[i])
                )
            )
        config.update({
            'subnets': subnets_serial,
            'group_col': int(self.group_col),
            'pass_groups': list(self.pass_groups),
            'strip_inputs': list(self.strip_inputs),
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
