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
    BranchGate: A layer that routes inputs to group-specific
        subnetworks that do not rejoin.

"""

import tensorflow as tf

from psiz.keras.sparse_dispatcher import SparseDispatcher
from psiz.keras.layers.experimental.gate import Gate


@tf.keras.utils.register_keras_serializable(
    package='psiz.keras', name='BranchGate'
)
class BranchGate(Gate):
    """A layer that routes inputs to group-specific subnetworks.

    In a `BranchGate` the subnetworks are not combined at the end (in
    contrast to a `BraidGate`).

    For more information see: `psiz.keras.layers.Gate`

    """
    def __init__(
            self, subnets=None, groups_subset=0, pass_groups=None,
            strip_inputs=None, output_names=None, **kwargs):
        """Initialize.

        Args:
            subnets: see `Gate`.
            groups_subset (optional): see `Gate`.
            pass_groups (optional): see `Gate`.
            strip_inputs (optional): see `Gate`.
            output_names (optional): A list of names to apply to the
                outputs of each subnet. Useful if using different
                losses and sample weights for each branch. If provided,
                the list length must agree with the number of subnets.

        Raises:
            ValueError if subnetwork's non-batch output shape is not
            fully defined.

        """
        super(BranchGate, self).__init__(
            subnets=subnets, groups_subset=groups_subset,
            pass_groups=pass_groups, strip_inputs=strip_inputs, **kwargs)
        if output_names is None:
            output_names = []
            for i_subnet in range(self.n_subnet):
                output_names.append(self.name + '_{0}'.format(i_subnet))
        self.output_names = output_names

    def call(self, inputs):
        """Call.

        Args:
            inputs: a n-tuple containing a data Tensor and a trailing
                group Tensor.
                list format: [data Tensor, [data Tensor, ...], groups Tensor]
                data Tensor(s): shape=(batch, m, [n, ...])
                groups Tensor: shape=(batch, g)

        Returns:
            outputs: A dictionary of Tensors.
        """
        gates = self._process_groups(inputs[self.inputs_group_idx])

        # Run inputs through dispatcher that routes inputs to correct subnet.
        dispatcher = SparseDispatcher(
            self.n_subnet, gates, has_timestep_axis=self._has_timestep_axis
        )
        subnet_inputs = dispatcher.dispatch_multi_pad(inputs)
        subnet_outputs = {}

        for i in range(self.n_subnet):
            out = self._processed_subnets[i](tuple(subnet_inputs[i]))  # TODO
            subnet_outputs[self.output_names[i]] = out

        return subnet_outputs
