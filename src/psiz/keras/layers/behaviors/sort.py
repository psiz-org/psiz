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
    SortBehavior: A sort behavior layer.

"""

import tensorflow as tf

from psiz.keras.layers.behaviors.base import Behavior


@tf.keras.utils.register_keras_serializable(
    package='psiz.keras.layers', name='SortBehavior'
)
class SortBehavior(Behavior):
    """A sort behavior layer.

    TODO

    """

    def __init__(
            self, lower_initializer=None, upper_initializer=None,
            midpoint_initializer=None, rate_initializer=None,
            lower_trainable=True, upper_trainable=True,
            midpoint_trainable=True, rate_trainable=True, **kwargs):
        """Initialize.

        Arguments:
            TODO
            kwargs (optional): Additional keyword arguments.

        """
        # pylint: disable=unused-argument
        super(SortBehavior, self).__init__(**kwargs)
        raise NotImplementedError

    def call(self, inputs):
        """Return probability of outcome.

        Arguments:
            inputs:
                sim_qr: A tensor containing the precomputed
                    similarities between the query stimuli and
                    corresponding reference stimuli (only 1 reference).
                    shape = (batch_size, 1, 1)

        Returns:
            probs: The probabilites as determined by a parameterized
                logistic function.

        """
        raise NotImplementedError

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        # config.update({})  TODO
        return config
