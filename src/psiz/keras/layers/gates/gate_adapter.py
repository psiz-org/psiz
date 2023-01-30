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
    GateAdapter: A layer that adapts inputs for networks with `Gates`.

"""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="psiz.keras", name="GateAdapter")
class GateAdapter(tf.keras.layers.Layer):
    """A layer that adapts inputs for networks with `Gates`.

    Attributes:
        input_keys: List of strings indicating required dictionary
            keys.
        gating_keys: See `__init__` method.
        format_inputs_as_tuple: See `__init__` method.

    """

    def __init__(self, gating_keys=None, format_inputs_as_tuple=None, **kwargs):
        """Initialize.

        Args:
            gating_keys (optional): List of strings indicating
                dictionary keys for gate weights.
            format_inputs_as_tuple (optional): Boolean indicating if
                inputs should be passed as tuple instead of dictionary.
                Default is `True`.

        """
        super(GateAdapter, self).__init__(**kwargs)
        self._all_keys = []
        self._input_keys = []
        if gating_keys is None:
            gating_keys = []
        elif not isinstance(gating_keys, list):
            gating_keys = [gating_keys]
        self.gating_keys = gating_keys
        if format_inputs_as_tuple is None:
            format_inputs_as_tuple = True
        self.format_inputs_as_tuple = tf.constant(format_inputs_as_tuple)

        self._strip_inputs = None

    @property
    def input_keys(self):
        return self._input_keys

    @input_keys.setter
    def input_keys(self, input_keys):
        if input_keys is None:
            raise ValueError("The argument `input_keys` cannot be `None`.")
        elif not isinstance(input_keys, list):
            input_keys = [input_keys]
        self._input_keys = input_keys

    def build(self, input_shape):
        """Build."""
        super(GateAdapter, self).build(input_shape)

        if self._input_keys is None:
            raise ValueError(
                "The attribute `input_keys` cannot be `None`. It should "
                "be set by an external caller."
            )
        # Make sure inputs are a dictionary.
        if not isinstance(input_shape, dict):
            raise ValueError(
                "GateAdapter layer only accepts dictionary-formatted " "`inputs`."
            )

        # Add input keys to list of all keys.
        for key in self._input_keys:
            self._all_keys.append(key)
        # Add keys of "gate weights" to list of all keys.
        for key in self.gating_keys:
            self._all_keys.append(key)

        if len(self._all_keys) == 1:
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
            for key in self._all_keys:
                formatted_inputs.append(inputs[key])
            if self._strip_inputs:
                formatted_inputs = formatted_inputs[0]
            else:
                formatted_inputs = tuple(formatted_inputs)
        else:
            formatted_inputs = inputs
        return formatted_inputs

    def get_config(self):
        """Get configuration."""
        config = super(GateAdapter, self).get_config()
        config.update(
            {
                "gating_keys": self.gating_keys,
                "format_inputs_as_tuple": bool(self.format_inputs_as_tuple),
            }
        )
        return config
