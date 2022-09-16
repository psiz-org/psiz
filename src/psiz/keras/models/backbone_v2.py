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
from psiz.keras.layers.groups_mixin import GroupsMixin
from psiz.utils import expand_dim_repeat


class BackboneV2(GroupsMixin, Stochastic):
    """A backbone-based psychological model.

    This model is intended to cover a large number of pscyhological
    modeling use cases. If your use case is not covered, you can use
    this model as a guide to create a bespoke model.

    Attributes:
        net: The network.
        n_sample: Integer indicating the number of samples to draw for
            stochastic layers. Only useful if using stochastic layers
            (e.g., variational models).

    """

    def __init__(self, net=None, n_sample=1, **kwargs):
        """Initialize.

        Args:
            net: A TODO layer.
            n_sample (optional): See psiz.keras.models.Stochastic.
            kwargs:  Additional key-word arguments.

        Raises:
            ValueError: If arguments are invalid.

        """
        super().__init__(n_sample=n_sample, **kwargs)

        # Assign layers.
        self.net = net

        # TODO HACK Traverse network and set `n_sample` appropriately.
        # make this a more explicit contract somehow?
        try:
            self.net.cell.n_sample = self.n_sample
        except:
            self.net.subnets[0].cell.n_sample = self.n_sample
            self.net.subnets[1].cell.n_sample = self.n_sample

        # Satisfy GroupsMixin contract.
        self._pass_groups = {
            'net': self.check_supports_groups(net)
        }

    def call(self, inputs):
        """Call.

        Args:
            inputs: A dictionary of inputs.

        """
        # Repeat `stimulus_set` `n_sample` times in a newly inserted
        # "sample" axis (axis=2).
        inputs['stimulus_set'] = expand_dim_repeat(
            inputs['stimulus_set'], self.n_sample, axis=2
        )
        # TensorShape=(batch_size, n_timestep, n_sample, n, [m, ...])

        return self.net(inputs)

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
