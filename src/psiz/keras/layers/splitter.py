# coding=utf-8
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
"""Module of Keras layers.

Classes:
    Splitter: A layer that splits an input into duplicate channels.

"""


import keras


@keras.saving.register_keras_serializable(package="psiz.keras", name="Splitter")
class Splitter(keras.layers.Layer):
    """A layer that splits an input in N copies.

    A `Splitter` layer duplicates inputs for each downstream
    consumer and can be useful for a mixture of experts.

    """

    def __init__(self, n_channel, has_timestep_axis=False, **kwargs):
        """Create a Splitter.

        Args:
            n_channel: The number of duplicate channels to create.
            has_timestep_axis (optional): Beolean indicating if second
                axis should be interpretted as a timestep axis (default is `False`).

        """
        super(Splitter, self).__init__(**kwargs)
        self._n_channel = n_channel
        self._has_timestep_axis = has_timestep_axis

    def build(self, input_shape):
        """Build."""
        # Determine if inputs are a dictionary.
        are_inputs_dict = False
        if isinstance(input_shape, dict):
            are_inputs_dict = True
        self.are_inputs_dict = are_inputs_dict

    def call(self, inputs):
        """Call.

        Args:
            inputs: a single tensor, list of tensors, or dictionary of
                tensors.
                shape=(batch_size, [n, m, ...])

        Returns:
            A list of `n_channel` inputs.
                shape=(batch_size, [n, m, ...])

        """
        expert_list = []
        for _ in range(self._n_channel):
            if not self.are_inputs_dict:
                # expert_list.append(tuple(inputs))  # TODO(roads) is wrapping with tuple necessary?
                expert_list.append(inputs)
            else:
                expert_list.append(inputs)

        return expert_list

    @property
    def n_channel(self):
        return self._n_channel

    def get_config(self):
        """Return layer configuration."""
        config = super(Splitter, self).get_config()
        config.update(
            {
                "n_channel": self._n_channel,
                "has_timestep_axis": self._has_timestep_axis,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
