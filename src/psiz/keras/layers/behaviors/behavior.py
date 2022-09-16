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
    Behavior: An abstract behavior layer.

"""

import tensorflow as tf

from psiz.keras.layers.groups import GroupsMixin


class Behavior(GroupsMixin, tf.keras.layers.Layer):
    """An abstract behavior layer.

    Sub-classes of this layer are responsible for three things:
    1) set `self.supports_groups = True` if the layer supports `groups`.
    2) For each layer, set `self._pass_groups[<layer variable>]`. For
    example, if a layer named `kernel` is provided:
    3) Implementing a `call` method that routes inputs by using
    appropriate `self._pass_groups`.

    `self._pass_groups['kernel'] = self.check_supports_groups(kernel)`

    """
    def __init__(self, **kwargs):
        """Initialize.

        Args:
            kwargs: Key-word arguments.

        """
        super(Behavior, self).__init__(**kwargs)

        # Create placeholder for layer switches.
        self._pass_groups = {}

    def call(self, inputs):
        raise NotImplementedError

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return super().from_config(config)
