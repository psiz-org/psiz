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
    Gate: Abstract layer that routes inputs. Routing behavior depends
        on concrete class.

"""

import tensorflow as tf
from tensorflow.keras import backend as K

from psiz.keras.layers.drop import Drop


@tf.keras.utils.register_keras_serializable(package="psiz.keras", name="Gate")
class Gate(tf.keras.layers.Layer):
    """Abstract layer that routes inputs to subnetworks."""

    def __init__(self, gating_index=None, gating_key=None, **kwargs):
        """Initialize.

        Initialization must provide either `gating_index` (if `inputs`
        to call is a tuple) or `gating_key` (if `inputs` to call is a
        dictionary). If using a `GateAdapter` upstream that converts
        dictionary-formatted inputs to tuple-formated inputs, you must
        provide `gating_index`.

        Args:
            gating_index (optional): If inputs are tuple-formatte,
                the index position of the gate weights.
            gating_key (optional): If inputs are dictionary-
                formatted, the dictionary key name of the gate weights.

        Notes:
        It is assumed that gate weights are either a) integers
        respresenting indices or b) floats representing mixing
        coefficients. If gate weights axis=1 is a singleton dimension,
        then values are assumed to be indices. Otherwise assumed to be
        the raw mixing coefficients.

        """
        super(Gate, self).__init__(**kwargs)
        self.supports_gating = True

        # Handle "gate weights" attributes.
        self.gating_index = gating_index
        self.gating_key = gating_key
        self._gate_weights_are_indices = None

        # Handle masking support.
        self.supports_masking = True
        # Handle inputs information attributes.
        self.are_inputs_dict = None
        # Handle timestep axis attributes.
        self._has_timestep_axis = None

    def build(self, input_shape):
        """Build."""
        # Determine if inputs are a dictionary.
        # NOTE: Use TF Boolean because `are_inputs_dict` is used inside
        # `_process_gate_weights` which is part of `call`.
        are_inputs_dict = False
        if isinstance(input_shape, dict):
            are_inputs_dict = True
        self.are_inputs_dict = tf.constant(are_inputs_dict)

        # Check if appropriate argument has been provided.
        if are_inputs_dict:
            if self.gating_key is None:
                raise ValueError(
                    "When `inputs` to layer is a dictionary, user must "
                    "instantiate `Gate` layer with `gating_key` "
                    "argument."
                )
        else:
            if self.gating_index is None:
                raise ValueError(
                    "When `inputs` to layer is a tuple, user must "
                    "instantiate `Gate` layer with `gating_index` "
                    "argument."
                )

        # Get `gate_weights` shape.
        if self.are_inputs_dict:
            gate_weights_shape = input_shape[self.gating_key]
        else:
            gate_weights_shape = input_shape[self.gating_index]

        # Determine if `gate_weights` are indices based on the shape of the
        # last axis. If singleton, assume indices.
        gate_weights_are_indices = False
        if gate_weights_shape[-1] == 1:
            gate_weights_are_indices = True
        self._gate_weights_are_indices = tf.constant(gate_weights_are_indices)

        # Determine if input has timestep axis.
        # NOTE: rank==3 logic in below if statement requires using singleton
        # dimension when supplying indices instead of weights.
        has_timestep_axis = False
        if gate_weights_shape.rank == 3:
            has_timestep_axis = True
        self._has_timestep_axis = tf.constant(has_timestep_axis)

    def _process_subnet(self, subnet, pass_gate_weights, strip_inputs):
        """Process subnet.

        Wraps subnet in `Drop` if `pass_gate_weights=False`. No inputs
        are dropped when `inputs` are formatted as a dictionary.

        Args:
            subnet: A subnetwork.
            pass_gate_weights: Boolean indicating if gate weights
                should be passed into the subnet.
            strip_inputs: Boolean indicating if `inputs` should be
                stripped to a single Tensor if `inputs` is a list
                containing only one item.

        Returns:
            processed_subnet

        """
        if pass_gate_weights or self.are_inputs_dict:
            return subnet
        else:
            return Drop(
                subnet=subnet, drop_index=self.gating_index, strip_inputs=strip_inputs
            )

    def _unprocess_subnet(self, subnet):
        """Unprocess subnet."""
        if isinstance(subnet, Drop):
            return subnet.subnet
        else:
            return subnet

    def _process_gate_weights(self, inputs):
        """Process "gate weights" part of inputs.

        Args:
            inputs: A data structure of layer inputs, may be a tuple of
                Tensors or a single-level dictionary of Tensors.

        Returns:
            gate_weights: A float Tensor.

        """
        if self.are_inputs_dict:
            gate_weights = inputs[self.gating_key]
        else:
            gate_weights = inputs[self.gating_index]

        # Convert indices to one-hot encoding if necessary.
        if self._gate_weights_are_indices:
            # NOTE: Drop singleton gate axis, but do not call tf.squeeze in
            # case timestep axis is also singleton.
            if self._has_timestep_axis:
                gate_weights = gate_weights[:, :, 0]
            else:
                gate_weights = gate_weights[:, 0]
            # Make sure `gate_weights` are integer type before using `one_hot`.
            dtype = K.dtype(gate_weights)
            if dtype != "int32" and dtype != "int64":
                gate_weights = tf.cast(gate_weights, "int32")
            gate_weights = tf.one_hot(
                gate_weights, self.n_subnet, on_value=1.0, off_value=0.0
            )
        return gate_weights

    def get_config(self):
        """Return layer configuration."""
        config = super(Gate, self).get_config()
        if self.gating_index is not None:
            config.update({"gating_index": int(self.gating_index)})
        if self.gating_key is not None:
            config.update({"gating_key": self.gating_key})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
