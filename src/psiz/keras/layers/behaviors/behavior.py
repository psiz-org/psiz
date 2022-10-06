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

from psiz.keras.mixins.gate_mixin import GateMixin
from psiz.keras.mixins.stochastic_mixin import StochasticMixin


class Behavior(StochasticMixin, GateMixin, tf.keras.layers.Layer):
    """An abstract behavior layer.

    Sub-classes of this layer are responsible for three things:
    1) Set `self.supports_groups = False` if the layer does not
    support `groups` input.
    2) For each layer, set `self._pass_groups[<layer variable>]`. For
    example, if a layer named `kernel` is provided:
    3) Implement a `call` method that routes inputs by using
    appropriate `self._pass_groups`.

    `self._pass_groups['kernel'] = self.check_supports_groups(kernel)`

    """
    def __init__(self, **kwargs):
        """Initialize.

        Args:
            kwargs: Key-word arguments.

        """
        super(Behavior, self).__init__(**kwargs)

    def get_mask(self, inputs, states, training=None):
        raise NotImplementedError

    def call(self, inputs):
        raise NotImplementedError

    def get_config(self):
        """Return layer configuration."""
        return super(Behavior, self).get_config()

    @classmethod
    def from_config(cls, config):
        return cls(**config)
