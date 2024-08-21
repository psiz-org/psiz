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
"""Module for a TensorFlow layers.

Classes:
    Gate: Abstract layer that routes inputs. Routing behavior depends
        on concrete class.

"""


import keras

from psiz.keras.layers.drop import Drop


@keras.saving.register_keras_serializable(package="psiz.keras", name="Gate")
class Gate(keras.layers.Layer):
    """Abstract layer that routes inputs to subnetworks."""

    def __init__(
        self,
        subnets=None,
        gating_index=None,
        gating_key=None,
        pass_gate_weights=None,
        strip_inputs=None,
        **kwargs
    ):
        """Initialize.

        Initialization must provide either `gating_index` (if `inputs`
        to call is a tuple) or `gating_key` (if `inputs` to call is a
        dictionary). If using a `GateAdapter` upstream that converts
        dictionary-formatted inputs to tuple-formated inputs, you must
        provide `gating_index`.

        Args:
            subnets: A non-empty list of sub-networks.
            gating_index (optional): If inputs are tuple-formatte,
                the index position of the gate weights.
            gating_key (optional): If inputs are dictionary-
                formatted, the dictionary key name of the gate weights.
            pass_gate_weights (optional): Applies to tuple-formatted
                `input` only. Boolean 1D array-like indicating if
                "gate weights"  should be passed to the subnets. By
                default, gate weights will not be passed. This argument
                can be used to override the default behavior. If
                provided by the user, the length must agree with the
                number of subnets.
                shape=(n_subnet)
            strip_inputs (optional): Applies to tuple-formatted
                `input` only. Boolean 1D array-like indicating if
                `inputs` to the subnetworks should be stripped to a
                single tensor if the tuple has only one element. This is
                useful if the subnet network is a TensorFlow layer that
                expects a single Tensor for the `inputs` argument
                (e.g., Embedding). By default, this is True. If
                provided, the length must agree with the number of
                subnets.

        Raises:
            ValueError if subnetwork's non-batch output shape is not
            fully defined.

        Notes:
        It is assumed that gate weights are floats representing mixing
        coefficients.

        """
        super(Gate, self).__init__(**kwargs)
        self.supports_gating = True

        # Handle "gate weights" attributes.
        self.gating_index = gating_index
        self.gating_key = gating_key

        # Handle masking support.
        self.supports_masking = True
        # Handle inputs information attributes.
        self.are_inputs_dict = None
        # Handle timestep axis attributes.
        self._has_timestep_axis = None

        self.n_subnet = len(subnets)
        self._subnets = subnets

        if pass_gate_weights is None:
            pass_gate_weights = [False] * self.n_subnet
        self.pass_gate_weights = pass_gate_weights

        if strip_inputs is None:
            strip_inputs = [True] * self.n_subnet
        self.strip_inputs = strip_inputs

    @property
    def subnets(self):
        return self._subnets

    def build(self, input_shape):
        """Build."""
        # Determine if inputs are a dictionary.
        # NOTE: Use TF Boolean because `are_inputs_dict` is used inside
        # `_process_gate_weights` which is part of `call`.
        are_inputs_dict = False
        if isinstance(input_shape, dict):
            are_inputs_dict = True
        self.are_inputs_dict = are_inputs_dict

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

        # Determine if input has timestep axis.
        has_timestep_axis = False
        if len(gate_weights_shape) == 3:
            has_timestep_axis = True
        self._has_timestep_axis = has_timestep_axis

        # Process subnets.
        processed_subnets = []
        for idx, subnet in enumerate(self._subnets):
            processed_subnets.append(
                self._process_subnet(
                    subnet, self.pass_gate_weights[idx], self.strip_inputs[idx]
                )
            )
        self._processed_subnets = processed_subnets

        # Build subnets.
        for subnet in self._processed_subnets:
            subnet.build(input_shape)

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

        return gate_weights

    def get_config(self):
        """Return layer configuration."""
        config = super(Gate, self).get_config()
        if self.gating_index is not None:
            config.update({"gating_index": int(self.gating_index)})
        if self.gating_key is not None:
            config.update({"gating_key": self.gating_key})

        subnets_serial = []
        for i in range(self.n_subnet):
            subnets_serial.append(
                keras.saving.serialize_keras_object(
                    self._unprocess_subnet(self._processed_subnets[i])
                )
            )
        config.update(
            {
                "subnets": subnets_serial,
                "pass_gate_weights": list(self.pass_gate_weights),
                "strip_inputs": list(self.strip_inputs),
            }
        )

        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
