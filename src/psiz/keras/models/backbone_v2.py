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
"""Module of PsiZ models.

Classes:
    BackboneV2:  A backbone-based psychological embedding model.

"""

import copy
from importlib.metadata import version

import tensorflow as tf

from psiz.keras.models.stochastic import Stochastic
from psiz.keras.mixins.groups_mixin import GroupsMixin


@tf.keras.utils.register_keras_serializable(
    package='psiz.keras.models', name='BackboneV2'
)
class BackboneV2(GroupsMixin, Stochastic):
    """A general-purpose model.

    This model is intended to be a convenience `Model` that covers a
    large number of pscyhological modeling use cases by bringing
    together a number of capabilities:
        1. Supports sequences. Assumes a "timestep axis" at axis=1. If
            your input data is not composed of sequences use a
            singleton dimension for this axis.
        2. Supports stochastic sampling. Assumes a "sample axis" at
            axis=2 and `StochasticMixin` is used appropriately.
        3. Supports multiple measures synthesis and hierarchical
            modeling via data routing. Data routing assumes `groups` is
            specified in the input dictionary and the `GroupsMixin` is
            used appropriately.

    If your use case is not covered, you can use this model as a guide
    to create a bespoke model.

    Attributes:
        net: The network.

    """

    def __init__(self, net=None, n_sample=1, **kwargs):
        """Initialize.

        Args:
            net: A Keras layer.
            n_sample (optional): See `psiz.keras.models.Stochastic`.
            kwargs:  Additional key-word arguments.

        Raises:
            ValueError: If arguments are invalid.

        """
        sample_axis = kwargs.pop('sample_axis', None)
        if sample_axis is None:
            sample_axis = 2
        else:
            if sample_axis != 2:
                raise ValueError('BackboneV2 requires sample_axis=2.')

        inputs_to_ignore = kwargs.pop('inputs_to_ignore', None)
        if inputs_to_ignore is None:
            inputs_to_ignore = ['groups']

        super().__init__(
            sample_axis=sample_axis, n_sample=n_sample,
            inputs_to_ignore=inputs_to_ignore, **kwargs
        )

        # Assign layers.
        self.net = net

        # Satisfy GroupsMixin contract.
        self._pass_groups = {
            'net': self.check_supports_groups(net)
        }

    def call(self, inputs, training=None):
        """Call.

        Args:
            inputs: A dictionary of inputs.
            training (optional): Boolean indicating if training mode.

        """
        return self.net(inputs, training=training)

    def get_config(self):
        """Return model configuration."""
        config = super().get_config()

        ver = version("psiz")
        ver = '.'.join(ver.split('.')[:3])

        layer_configs = {
            'net': tf.keras.utils.serialize_keras_object(self.net)
        }

        config.update({
            'psiz_version': ver,
            'name': self.name,
            'class_name': self.__class__.__name__,
            'layers': copy.deepcopy(layer_configs)
        })

        return config

    @classmethod
    def from_config(cls, config):
        """Create model from configuration.

        Args:
            config: A hierarchical configuration dictionary.

        Returns:
            An instantiated and configured TensorFlow model.

        """
        model_config = copy.deepcopy(config)
        model_config.pop('class_name', None)
        model_config.pop('psiz_version', None)

        # Deserialize layers.
        layer_configs = model_config.pop('layers', None)
        built_layers = {}
        for layer_name, layer_config in layer_configs.items():
            layer = tf.keras.layers.deserialize(layer_config)
            built_layers[layer_name] = layer

        model_config.update(built_layers)
        return cls(**model_config)
