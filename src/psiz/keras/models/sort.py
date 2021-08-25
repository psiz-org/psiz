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
"""Module for Rank psychological embedding model.

Classes:
    Sort: Class that uses ordinal observations that are unanchored by
        a designated query stimulus.

"""

import tensorflow as tf

from psiz.keras.models.psych_embedding import PsychologicalEmbedding
import psiz.keras.layers


@tf.keras.utils.register_keras_serializable(
    package='psiz.keras.models', name='Sort'
)
class Sort(PsychologicalEmbedding):
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
            behavior = psiz.keras.layers.SortBehavior()
        kwargs.update({'behavior': behavior})

        super().__init__(**kwargs)

    def call(self, inputs):
        """Call.

        Arguments:
            inputs: A dictionary of inputs:
                stimulus_set: dtype=tf.int32, consisting of the
                    integers on the interval [0, n_stimuli[
                    shape=(batch_size, TODO )
                is_select: dtype=tf.bool, the shape implies the
                    maximum number of selected stimuli in the data
                    shape=(batch_size, TODO)
                groups: dtype=tf.int32, Integers indicating the
                    group membership of a trial.
                    shape=(batch_size, k)

        """
        # pylint: disable=unused-variable
        # Grab inputs.
        stimulus_set = inputs['stimulus_set']
        is_select = inputs['is_select'][:, 1:, :]
        groups = inputs['groups']

        # Inflate coordinates.
        z = self.stimuli([stimulus_set, groups])
        # TensorShape([sample_size, batch_size, TODO, n_dim])
        raise NotImplementedError
