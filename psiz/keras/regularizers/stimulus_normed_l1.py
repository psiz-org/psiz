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
"""Module of custom TensorFlow regularizers.

Classes:
    StimulusNormedL1:

"""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package='psiz.keras.regularizers')
class StimulusNormedL1(tf.keras.regularizers.Regularizer):
    """Stimulus-normed L1 regularization."""

    def __init__(self, l1=0.):
        """Initialize.

        Arguments:
            l1: Rate of L1 regularization.

        """
        self.l1 = l1

    def __call__(self, z):
        """Call."""
        return self.l1 * (tf.reduce_sum(tf.abs(z)) / z.shape[0])

    def get_config(self):
        """Return config."""
        return {'l1': float(self.l1)}
