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
    BraidGate: A layer that routes inputs to group-specific
        subnetworks.

"""


import keras

from psiz.keras.layers.splitter import Splitter
from psiz.keras.layers.gates.gate import Gate


@keras.saving.register_keras_serializable(package="psiz.keras", name="BraidGate")
class BraidGate(Gate):
    """A layer that routes inputs to subnetworks.

    In a `BraidGate` the subnetwork outputs are re-combined at the end.

    The subnetworks can take a list of inputs, but each subnetwork must
    output a single tensor. The final output shape must be the same for
    all subnetworks.

    For more information see `psiz.keras.layers.Gate`

    """

    def __init__(
        self, subnets=None, pass_gate_weights=None, strip_inputs=None, **kwargs
    ):
        """Initialize.

        Args:
            subnets: see `Gate`.
            pass_gate_weights (optional): see `Gate`.
            strip_inputs (optional): see `Gate`.

        Raises:
            ValueError if subnetwork's non-batch output shape is not
            fully defined.

        """
        super(BraidGate, self).__init__(
            subnets=subnets,
            pass_gate_weights=pass_gate_weights,
            strip_inputs=strip_inputs,
            **kwargs
        )

    def call(self, inputs):
        """Call.

        Args:
            inputs: Data Tensors. Can be an n-tuple or single-level
                dictionary containing Tensors. If n-tuple, the trailing
                Tensor is assumed to be a "gate weights" Tensor. If a
                dictionary, one of the fields is assumed to be
                `gate_weights`. The tuple format follows
                [data Tensor, [data Tensor, ...], gate_weights Tensor].
                The data Tensor(s) follows shape=(batch, m, [n, ...]).
                The gate_weights Tensor follows shape=(batch, g)

        """
        gate_weights = self._process_gate_weights(inputs)

        # Run inputs through splitter that routes inputs to correct subnet.
        splitter = Splitter(self.n_subnet, has_timestep_axis=self._has_timestep_axis)
        subnet_inputs = splitter(inputs)
        subnet_outputs = []
        for i in range(self.n_subnet):
            out = self._processed_subnets[i](subnet_inputs[i])
            out, lost_shape = self._pre_combine(out)
            # Weight outputs appropriately.
            if self._has_timestep_axis:
                out = out * keras.ops.expand_dims(gate_weights[:, :, i], axis=2)
            else:
                out = out * keras.ops.expand_dims(gate_weights[:, i], axis=1)
            subnet_outputs.append(out)

        outputs = subnet_outputs[0]
        for i in range(1, self.n_subnet):
            outputs = outputs + subnet_outputs[i]

        # Post combine: handle reshaping of output.
        outputs = keras.ops.reshape(outputs, lost_shape)

        return outputs

    def _pre_combine(self, x):
        """Prepare Tensor for combine operation.

        All non-batch dimensions must be flattened together to create a
        final 2D Tensor.

        Args:
            x: Data Tensor.

        Returns:
            x: Transformed data Tensor.
            lost_shape: A tensor recording the shape lost during
                reshape.

        """
        # Flatten non-batch dimensions.
        x_shape = keras.ops.shape(x)
        if self._has_timestep_axis:
            x = keras.ops.reshape(x, [x_shape[0], x_shape[1], -1])
        else:
            x = keras.ops.reshape(x, [x_shape[0], -1])
        return x, x_shape

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        subnets_serial = config["subnets"]
        subnets = []
        for subnet_serial in subnets_serial:
            subnets.append(keras.saving.deserialize_keras_object(subnet_serial))
        config["subnets"] = subnets
        return cls(**config)
