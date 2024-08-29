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
"""Module for a Keras layers.

Classes:
    Drop: A layer that drops a Tensor in the `inputs`.

"""


import keras


@keras.saving.register_keras_serializable(package="psiz.keras", name="Drop")
class Drop(keras.layers.Layer):
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
        if not self.subnet.built:
            self.subnet.build(input_shape_w_drop)

    def call(self, inputs):
        """Call.

        Args:
            inputs: a n-tuple or list containing Tensors.

        """
        # Drop requested tensor from `inputs`.
        inputs_less_drop = (
            inputs[0 : self._drop_index] + inputs[(self._drop_index + 1) :]
        )

        if self._strip_inputs:
            inputs_less_drop = inputs_less_drop[0]

        return self.subnet(inputs_less_drop)

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        config.update(
            {
                "subnet": keras.saving.serialize_keras_object(self.subnet),
                "drop_index": int(self.drop_index),
                "strip_inputs": bool(self.strip_inputs),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        subnet_serial = config["subnet"]
        config["subnet"] = keras.saving.deserialize_keras_object(subnet_serial)
        return super().from_config(config)

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
