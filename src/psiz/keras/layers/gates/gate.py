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

from psiz.keras.layers.drop import Drop


@tf.keras.utils.register_keras_serializable(
    package='psiz.keras', name='Gate'
)
class Gate(tf.keras.layers.Layer):
    """Abstract layer that routes inputs to subnetworks."""
    def __init__(self, gate_weights_idx=None, gate_weights_key=None, **kwargs):
        """Initialize.

        Initialization must provide either `gate_weights_idx` (if
        `inputs` to call is a tuple) or `gate_weights_key` (if `inputs`
        to call is a dictionary). If using a `GateAdapter` upstream
        that converts dictionary-formatted inputs to tuple-formated
        inputs, you must provide both arguments.

        Args:
            gate_weights_idx (optional): If inputs
            gate_weights_key (optional): If inputs are formatted as a
                dictionary, the dictionary key name containing the gate
                weights.

        Notes:
        Gate weights may be provided as integers respresenting indices
        or as floats representing mixing coefficients. If `gate_weights`
        axis=1 is a singleton dimension, then values are assumed to be
        indices. Otherwise assumed to be raw mixing coefficients.

        """
        super(Gate, self).__init__(**kwargs)
        self.supports_gating = True

        # Handle "gate weights" attributes.
        self.gate_weights_idx = gate_weights_idx
        self.gate_weights_key = gate_weights_key
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
            if self.gate_weights_key is None:
                raise ValueError(
                    'When `inputs` to layer is a dictionary, user must '
                    'instantiate `Gate` layer with `gate_weights_key` '
                    'argument.'
                )
        else:
            if self.gate_weights_idx is None:
                raise ValueError(
                    'When `inputs` to layer is a tuple, user must '
                    'instantiate `Gate` layer with `gate_weights_idx` '
                    'argument.'
                )

        # Get `gate_weights` shape.
        if self.are_inputs_dict:
            gate_weights_shape = input_shape[self.gate_weights_key]
        else:
            gate_weights_shape = input_shape[self.gate_weights_idx]

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
                subnet=subnet, drop_index=self.gate_weights_idx,
                strip_inputs=strip_inputs
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
            gate_weights = inputs[self.gate_weights_key]
        else:
            gate_weights = inputs[self.gate_weights_idx]

        # Convert indices to one-hot encoding if necessary.
        if self._gate_weights_are_indices:
            # NOTE: Drop singleton gate axis, but do not call tf.squeeze in
            # case timestep axis is also singleton.
            if self._has_timestep_axis:
                gate_weights = gate_weights[:, :, 0]
            else:
                gate_weights = gate_weights[:, 0]
            gate_weights = tf.one_hot(
                gate_weights, self.n_subnet, on_value=1.0, off_value=0.0
            )
        return gate_weights

    def get_config(self):
        """Return layer configuration."""
        config = super(Gate, self).get_config()
        if self.gate_weights_idx is not None:
            config.update({
                'gate_weights_idx': int(self.gate_weights_idx)
            })
        if self.gate_weights_key is not None:
            config.update({
                'gate_weights_key': self.gate_weights_key,
            })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def does_subnet_contain_gate(self, subnet):
        """Check if subnet contains a gate.

        Args:
            subnet: A Keras module (e.g., a Keras Layer).

        Returns:
            Boolean indicating if module supports gating.

        """
        contains_gate = False
        for submodule in subnet.submodules:
            contains_gate = contains_gate or isinstance(submodule, Gate)
        return contains_gate
