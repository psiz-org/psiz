# -*- coding: utf-8 -*-
# Copyright 2020 The PsiZ Authors. All Rights Reserved.
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
"""Module of TensorFlow layers.

Classes:
    Select: A layer that selects input from list of inputs.

"""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(
    package='psiz.keras.layers', name='Select'
)
class Select(tf.keras.layers.Layer):
    """A layer."""
    def __init__(self, subnet=None, idx=0, **kwargs):
        """Initialize."""
        super(Select, self).__init__(**kwargs)
        self.subnet = subnet
        self.idx = idx

    def call(self, inputs):
        """Call.

        Select from n-tuple inputs.

        """
        return self.subnet(inputs[self.idx])

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        config.update({
            'subnet': tf.keras.utils.serialize_keras_object(self.subnet),
            'idx': int(self.idx)
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Create from configuration."""
        config['subnet'] = tf.keras.layers.deserialize(config['subnet'])
        return cls(**config)
