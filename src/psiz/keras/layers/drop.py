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
    Drop: A layer that drops a Tensor in the `inputs`.

"""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="psiz.keras", name="Drop")
class Drop(tf.keras.layers.Layer):
    """A wrapper layer that drops part of the inputs.

    Assumues `inputs` provided to call is a list of Tensors.

    """

    def __init__(self, subnet=None, drop_index=-1, strip_inputs=True, **kwargs):
        """Initialize.

        Args:
            subnet: A subnetwork.
            drop_index (optional): Integer indicating the part of the
                inputs to drop. Be default, the last Tensor is dropped.
            strip_inputs (optional): Boolean indicating if `inputs`
                should be stripped to a single Tensor if `inputs` is a
                list containing only one item.

        """
        super(Drop, self).__init__(**kwargs)
        self.subnet = subnet
        self.drop_index = drop_index
        self._drop_index = drop_index
        self.strip_inputs = strip_inputs  # Indicates user's preference.
        self._strip_inputs = None  # Indicates if actually necessary.
        self.supports_masking = True

    def build(self, input_shape):
        """Build."""
        # To make things easier when we use addition on the index, we convert
        # '-1' to the actual index.
        if self._drop_index == -1:
            self._drop_index = len(input_shape) - 1

        # Drop requested tensor from `inputs`.
        input_shape_w_drop = (
            input_shape[0 : self._drop_index] + input_shape[(self._drop_index + 1) :]
        )

        # Determine how inputs should be pre-processed.
        self._check_strip_inputs(input_shape_w_drop)

        if self._strip_inputs:
            input_shape_w_drop = input_shape_w_drop[0]

        # Build subnet.
        self.subnet.build(input_shape_w_drop)

        super().build(input_shape)

    def call(self, inputs, mask=None):
        """Call.

        Args:
            inputs: a n-tuple or list containing Tensors.
            mask (optional): A Tensor indicating which timesteps should
                be masked.

        """
        # Drop requested tensor from `inputs`.
        inputs_less_drop = (
            inputs[0 : self._drop_index] + inputs[(self._drop_index + 1) :]
        )

        if self._strip_inputs:
            inputs_less_drop = inputs_less_drop[0]

        if mask is not None and self.subnet.supports_masking:
            return self.subnet(inputs_less_drop, mask=mask)
        else:
            return self.subnet(inputs_less_drop)

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        config.update(
            {
                "subnet": tf.keras.utils.serialize_keras_object(self.subnet),
                "drop_index": int(self.drop_index),
                "strip_inputs": bool(self.strip_inputs),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        subnet_serial = config["subnet"]
        config["subnet"] = tf.keras.layers.deserialize(subnet_serial)
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
        # To make things easier when we use addition on the index, we convert
        # '-1' to the actual index.
        if self._drop_index == -1:
            self._drop_index = len(input_shape) - 1

        # Drop requested tensor from `input_shape`.
        input_shape_w_drop = (
            input_shape[0 : self._drop_index] + input_shape[(self._drop_index + 1) :]
        )

        # Determine how inputs should be pre-processed.
        self._check_strip_inputs(input_shape_w_drop)

        if self._strip_inputs:
            input_shape_w_drop = input_shape_w_drop[0]
        return self.subnet.compute_output_shape(input_shape_w_drop)

    def _check_strip_inputs(self, input_shape_w_drop):
        """Check if `_strip_inputs` has been set.

        Args:
            input_shape_w_drop: Shape tuple (tuple of integers) or list
                of shape tuples, but not including dropped tensor.

        """
        if self._strip_inputs is None:
            if self.strip_inputs and len(input_shape_w_drop) == 1:
                self._strip_inputs = True
            else:
                self._strip_inputs = False
