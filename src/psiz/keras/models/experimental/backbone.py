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
    Backbone:  A backbone-based psychological embedding model.

"""

import copy
from importlib.metadata import version

import tensorflow as tf

from psiz.keras.models.experimental.stochastic import Stochastic
from psiz.keras.layers.experimental.groups import Groups
from psiz.utils import expand_dim_repeat


class Backbone(Groups, Stochastic):
    """A backbone-based psychological model.

    This model is intended to cover a large number of pscyhological
    modeling use cases. If your use case is not covered, you can use
    this model as a guide to create a bespoke model.

    Attributes:
        percept: A percept layer.
        behavior: A behavior layer.
        n_sample: Integer indicating the number of samples to draw for
            stochastic layers. Only useful if using stochastic layers
            (e.g., variational models).

    """

    def __init__(self, percept=None, behavior=None, n_sample=1, **kwargs):
        """Initialize.

        Args:
            percept: A percept layer.
            behavior: A behavior layer.
            n_sample (optional): See psiz.keras.models.Stochastic.
            kwargs:  Additional key-word arguments.

        Raises:
            ValueError: If arguments are invalid.

        """
        super().__init__(n_sample=n_sample, **kwargs)
        self.supports_groups = True

        # Assign layers.
        self.percept = percept
        self.behavior = behavior

        # Handle module switches.
        self._pass_groups = {
            'percept': self.check_supports_groups(percept),
            'behavior': self.check_supports_groups(behavior)
        }

    def call(self, inputs):
        """Call.

        Args:
            inputs: A dictionary of inputs. At a minimum, must contain:
                stimulus_set: dtype=tf.int32, consisting of the
                    integers on the interval [0, n_stimuli[
                    shape=(batch_size, n_max_reference + 1, n_outcome)
                groups: dtype=tf.int32, Integers indicating the
                    group membership of a trial.
                    shape=(batch_size, k)

        """
        # Pop universal inputs.
        stimulus_set = inputs.pop('stimulus_set')
        groups = inputs.pop('groups', None)

        # Repeat `stimulus_set` `n_sample` times in a newly inserted
        # "sample" axis (axis=1).
        stimulus_set = expand_dim_repeat(
            stimulus_set, self.n_sample, axis=1
        )
        # TensorShape=(batch_size, n_sample, [n, m, ...])

        # Embed stimuli indices in n-dimensional space.
        if self._pass_groups['percept']:
            z = self.percept([stimulus_set, groups])
        else:
            z = self.percept(stimulus_set)
        # TensorShape=(batch_size, n_sample, [n, m, ...] n_dim])

        # Convert remaining `inputs` dictionary to list, preserving order of
        # dictionary.
        inputs_list = self._unpack_inputs(inputs)

        if self._pass_groups['behavior']:
            y_pred = self.behavior([stimulus_set, z, *inputs_list, groups])
        else:
            y_pred = self.behavior([stimulus_set, z, *inputs_list])
        return y_pred

    def get_config(self):
        """Return model configuration."""
        config = super().get_config()

        ver = version("psiz")
        ver = '.'.join(ver.split('.')[:3])

        layer_configs = {
            'percept': tf.keras.utils.serialize_keras_object(self.percept),
            'behavior': tf.keras.utils.serialize_keras_object(self.behavior)
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
            # Convert old saved models.
            if layer_name == 'embedding':
                layer_name = 'percept'  # pragma: no cover
            built_layers[layer_name] = layer

        model_config.update(built_layers)
        return cls(**model_config)

    def _unpack_inputs(self, inputs):
        """Unpack inputs dictionary to list."""
        inputs_list = []
        for key, value in inputs.items():
            inputs_list.append(value)
        return inputs_list
