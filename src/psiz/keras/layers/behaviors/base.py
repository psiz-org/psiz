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


class Behavior(tf.keras.layers.Layer):
    """An abstract behavior layer."""

    def __init__(self, n_group=None, group_level=None, **kwargs):
        """Initialize.

        Arguments:
            n_group: DEPRECATED not used
            group_level: DEPRECATED not used
            kwargs (optional): Additional keyword arguments.

        """
        # TODO remove unused arguments
        # pylint: disable=unused-argument
        super().__init__(**kwargs)

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        return config

    def call(self, inputs):
        raise NotImplementedError
