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
"""Module for Keras layers.

Classes:
    Combiner: A channel combiner.

"""


import keras


@keras.saving.register_keras_serializable(package="psiz.keras", name="Combiner")
class Combiner(keras.layers.Layer):
    """A layer that combines multiple input channels based on provided mixing weights.

    A `Combiner` layer is useful for a mixture of experts.


    """

    def __init__(self, has_timestep_axis=False, **kwargs):
        """Initialize.

        Args:
            has_timestep_axis (optional): A boolean indicating whether the input
                has a timestep axis. Default is False.

        Returns:
            a Combiner layer.

        """
        super(Combiner, self).__init__(**kwargs)
        self._has_timestep_axis = has_timestep_axis

    def build(self, input_shape):
        """Build."""
        # NOTE: First input is assumed to be the mixing weights.
        self._n_channel = len(input_shape) - 1

    def call(self, inputs):
        """Call.

        Args:
            inputs: A list of inputs. First input is assumed to be the mixing weights.
                mixing_weights: A `Tensor` of:
                    shape=(batch_size, n_split) or
                    shape=(batch_size, sequence_length, n_split)

        Returns:
            A combined signal.

        """
        mixing_weights = inputs[0]
        channel_inputs = inputs[1:]

        # NOTE: Expand weight shape to account for input axis.
        mixing_weights = keras.ops.expand_dims(mixing_weights, axis=-1)

        if self._has_timestep_axis:
            outputs = mixing_weights[:, :, 0] * channel_inputs[0]
            for i in range(1, self._n_channel):
                outputs = outputs + (mixing_weights[:, :, i] * channel_inputs[i])
        else:
            outputs = mixing_weights[:, 0] * channel_inputs[0]
            for i in range(1, self._n_channel):
                outputs = outputs + (mixing_weights[:, i] * channel_inputs[i])
        return outputs

    @property
    def n_channel(self):
        return self._n_channel

    def get_config(self):
        """Return layer configuration."""
        config = super(Combiner, self).get_config()
        config.update(
            {
                "has_timestep_axis": self._has_timestep_axis,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
