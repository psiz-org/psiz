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
    SplitGate: Abstract layer that routes inputs to subnetworks.
        Routing behavior depends on concrete class.

"""

import tensorflow as tf

from psiz.keras.layers.gates.gate import Gate


@tf.keras.utils.register_keras_serializable(package="psiz.keras", name="SplitGate")
class SplitGate(Gate):
    """Abstract layer that routes inputs to subnetworks."""

    def __init__(
        self, subnets=None, pass_gate_weights=None, strip_inputs=None, **kwargs
    ):
        """Initialize.

        Args:
            subnets: A non-empty list of sub-networks.
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

        """
        super(SplitGate, self).__init__(**kwargs)

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
        super(SplitGate, self).build(input_shape)

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
        subnets_serial = config["subnets"]
        subnets = []
        for subnet_serial in subnets_serial:
            subnets.append(tf.keras.layers.deserialize(subnet_serial))
        config["subnets"] = subnets
        return cls(**config)
