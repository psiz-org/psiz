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
"""Module of TensorFlow behavior layers.

Classes:
    Behavior2: An abstract behavior layer.

"""

import tensorflow as tf

from psiz.keras.layers.experimental.groups import Groups


class Behavior2(Groups, tf.keras.layers.Layer):
    """An abstract behavior layer."""
    def __init__(self, kernel=None, **kwargs):
        """Initialize.

        Args:
            kernel: A kernel layer.
            kwargs: Key-word arguments.

        """
        super(Behavior2, self).__init__(**kwargs)
        self.kernel = kernel

        # Handle module switches.
        self._pass_groups = {}
        self._pass_groups['kernel'] = self.check_supports_groups(kernel)

    def call(self, inputs):
        raise NotImplementedError

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        config.update({
            'kernel': tf.keras.utils.serialize_keras_object(self.kernel)
        })
        return config

    @classmethod
    def from_config(cls, config):
        kernel_serial = config['kernel']
        config['kernel'] = tf.keras.layers.deserialize(kernel_serial)
        return super().from_config(config)
