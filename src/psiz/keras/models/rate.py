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
"""Module for Rate psychological embedding model.

Classes:
    Rate: Class that uses ratio observations between unanchored sets
        of stimulus (typically two stimuli).

"""

import tensorflow as tf

from psiz.keras.models.psych_embedding import PsychologicalEmbedding
import psiz.keras.layers


@tf.keras.utils.register_keras_serializable(
    package='psiz.keras.models', name='Rate'
)
class Rate(PsychologicalEmbedding):
    """Psychological embedding inferred from ranked similarity judgments.

    Attributes:
        See PsychologicalEmbedding.

    """

    def __init__(self, behavior=None, **kwargs):
        """Initialize.

        Arguments:
            See PschologicalEmbedding.

        Raises:
            ValueError: If arguments are invalid.

        """
        # Initialize behavioral component.
        if behavior is None:
            behavior = psiz.keras.layers.RateBehavior()
        kwargs.update({'behavior': behavior})
        super().__init__(**kwargs)

    def call(self, inputs):
        """Call.

        Arguments:
            inputs: A dictionary of inputs:
                stimulus_set: dtype=tf.int32, consisting of the
                    integers on the interval [0, n_stimuli[
                    shape=(batch_size, 2)
                groups: dtype=tf.int32, Integers indicating the
                    group membership of a trial.
                    shape=(batch_size, k)

        """
        # Grab inputs.
        stimulus_set = inputs['stimulus_set']
        groups = inputs['groups']

        # Repeat `stimulus_set` `n_sample` times in a newly inserted
        # axis (axis=1).
        # TensorShape([batch_size, n_sample, 2])
        stimulus_set = psiz.utils.expand_dim_repeat(
            stimulus_set, self.n_sample, axis=1
        )

        # Enbed stimuli indices in n-dimensional space.
        # TensorShape([batch_size, n_sample, 2, n_dim])
        if self._use_group['stimuli']:
            z = self.stimuli([stimulus_set, groups])
        else:
            z = self.stimuli(stimulus_set)

        # Divide up stimuli sets for kernel call.
        z_0 = z[:, :, 0]
        z_1 = z[:, :, 1]

        # Pass through similarity kernel.
        # TensorShape([batch_size, n_sample,])
        if self._use_group['kernel']:
            sim_01 = self.kernel([z_0, z_1, groups])
        else:
            sim_01 = self.kernel([z_0, z_1])

        # Predict rating of stimulus pair.
        if self._use_group['behavior']:
            rating = self.behavior([sim_01, groups])
        else:
            rating = self.behavior([sim_01])

        # Add singleton trailing dimension since MSE assumes rank-2 Tensors.
        rating = tf.expand_dims(rating, axis=-1)
        return rating
