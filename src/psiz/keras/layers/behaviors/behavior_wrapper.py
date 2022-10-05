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
    BehaviorWrapper: A layer for wrapping behaviors.

"""

import copy

import tensorflow as tf

from psiz.keras.mixins.groups_mixin import GroupsMixin


@tf.keras.utils.register_keras_serializable(
    package='psiz.keras.layers', name='BehaviorWrapper'
)
class BehaviorWrapper(GroupsMixin, tf.keras.layers.Layer):
    """A rank similarity behavior layer."""
    def __init__(self, net, **kwargs):
        """Initialize.

        Args:
            net: An RNN cell compatible with the assumed dictionary of
                `inputs` for this layer.

        """
        super(BehaviorWrapper, self).__init__(**kwargs)
        self.net = net
        # Determine if RNN or not.
        self._net_is_rnn = tf.constant(False)
        if isinstance(net, tf.keras.layers.RNN):
            self._net_is_rnn = tf.constant(True)

    def call(
            self, inputs, mask=None, training=None, initial_state=None,
            constants=None):
        if self._net_is_rnn:
            mask = self.net.cell.get_mask(inputs)
            outputs = self.net.call(
                inputs, mask=mask, training=training,
                initial_state=initial_state,
                constants=constants
            )
        else:
            inputs_mod = self._drop_timestep_axis(inputs)
            states = self._get_initial_state(inputs_mod)
            outputs, _ = self.net.call(inputs_mod, states, training=training)
            outputs = self._add_timestep_axis(outputs)
        return outputs

    def _drop_timestep_axis(self, x):
        """Drop timestep axis."""
        # TODO Handle different data structures.
        # NOTE: Must make a copy since we are not allowed to modify any
        # lists or dicts passed as arguents to call().
        x_copied = copy.copy(x)
        groups = x_copied.pop('groups', None)
        key_list = x_copied.keys()
        for key in key_list:
            x_copied[key] = x_copied[key][:, 0]
        if groups is not None:
            x_copied['groups'] = groups
        return x_copied

    def _add_timestep_axis(self, x):
        """Add timestep axis."""
        # TODO Handle different data structures.
        x = tf.expand_dims(x, axis=1)
        return x

    def _get_initial_state(self, inputs):
        """Get initial state.

        NOTE: This method mimics TF RNN layer `get_initial_state`.

        """
        if tf.nest.is_nested(inputs):
            # The input are nested sequences. Use the first element in the seq
            # to get batch size and dtype.
            inputs = tf.nest.flatten(inputs)[0]

        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]  # TODO maybe add option time major axis?
        dtype = inputs.dtype

        states = self.net.get_initial_state(
            inputs=None, batch_size=batch_size, dtype=dtype
        )
        return states

    def get_config(self):
        """Return layer configuration."""
        config = super(BehaviorWrapper, self).get_config()
        config.update({
            'net': tf.keras.utils.serialize_keras_object(self.net)
        })
        return config

    @classmethod
    def from_config(cls, config):
        net_serial = config['net']
        config['net'] = tf.keras.layers.deserialize(net_serial)
        return cls(**config)
