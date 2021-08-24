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
"""Module of psychological embedding models.

Classes:
    GroupLevel: DEPRECATED An abstract TF Keras layer that handles
        group membership boilerplate.

"""

import tensorflow as tf


class GroupLevel(tf.keras.layers.Layer):
    """An abstract layer for managing group-specific semantics."""

    def __init__(self, n_group=1, group_level=0, **kwargs):
        """Initialize.

        Arguments:
            n_group (optional): Integer indicating the number of groups
                in the layer.
            group_level (optional): Ingeter indicating the group level
                of the layer. This will determine which column of
                `group` is used to route the forward pass.
            kwargs (optional): Additional keyword arguments.

        """
        super(GroupLevel, self).__init__(**kwargs)
        self.n_group = n_group
        self.group_level = group_level

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        config.update({
            'n_group': int(self.n_group),
            'group_level': int(self.group_level),
        })
        return config

    def call(self, inputs):
        raise NotImplementedError
