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
    GateAdapter: A context-aware layer that regulates the pass-through
        of inputs based on the "plugged" layer.

"""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(
    package='psiz.keras', name='GateAdapter'
)
class GateAdapter(tf.keras.layers.Layer):
    """An input adapter."""
    def __init__(
            self, subnet=None, input_keys=None, gating_keys=None,
            format_inputs_as_tuple=None, **kwargs):
        """Initialize.

        Args:
            subnet: A Keras Layer.
            input_keys: List of strings indicating required dictionary
                keys.
            gating_keys (optional): List of strings indicating
                dictionary keys for gate weights.
            format_inputs_as_tuple (optional): Boolean indicating if
                inputs should be passed as tuple instead of dictionary.

        """
        super(GateAdapter, self).__init__(**kwargs)
        self.subnet = subnet
        if input_keys is None:
            raise ValueError(
                'The argument `input_keys` is not optional.'
            )
        elif not isinstance(input_keys, list):
            input_keys = [input_keys]
        self.input_keys = input_keys
        if gating_keys is None:
            gating_keys = []
        elif not isinstance(gating_keys, list):
            gating_keys = [gating_keys]
        self.gating_keys = gating_keys
        self.format_inputs_as_tuple = tf.constant(format_inputs_as_tuple)

        self._strip_inputs = None

    def build(self, input_shape):
        """Build."""
        super(GateAdapter, self).build(input_shape)

        # Make sure inputs are a dictionary.
        if not isinstance(input_shape, dict):
            raise ValueError(
                'GateAdapter layer only accepts dictionary-formatted '
                '`inputs`.'
            )

        # Add keys of "gate weights" to list of input keys.
        for key in self.gating_keys:
            self.input_keys.append(key)

        if len(self.input_keys) == 1:
            self._strip_inputs = tf.constant(True)
        else:
            self._strip_inputs = tf.constant(False)

    def call(self, inputs, training=None, mask=None):
        """Call.

        Args:
            inputs: A dictionary of Tensors.
            training (optional): see tf.keras.layers.Layer
            mask (optional): see tf.keras.layers.Layer

        """
        if self.format_inputs_as_tuple:
            formatted_inputs = []
            for key in self.input_keys:
                formatted_inputs.append(inputs[key])
            if self._strip_inputs:
                formatted_inputs = formatted_inputs[0]
            else:
                formatted_inputs = tuple(formatted_inputs)
        else:
            formatted_inputs = inputs
        return self.subnet(formatted_inputs)
