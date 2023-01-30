# -*- coding: utf-8 -*-
# Copyright 2022 The PsiZ Authors. All Rights Reserved.
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
    RankSimilarityCell: An RNN cell rank similarity layer.

"""

import copy

import tensorflow as tf

from psiz.keras.layers.behaviors.rank_similarity_base import RankSimilarityBase


@tf.keras.utils.register_keras_serializable(
    package="psiz.keras.layers", name="RankSimilarityCell"
)
class RankSimilarityCell(RankSimilarityBase):
    """A stateful rank similarity behavior layer."""

    def __init__(self, **kwargs):
        """Initialize.

        Args:
            kwargs: See `RankSimilarityBase`

        """
        super(RankSimilarityCell, self).__init__(**kwargs)

        # Satisfy RNNCell contract.
        # NOTE: A placeholder state.
        self.state_size = [tf.TensorShape([1])]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        """Get initial state."""
        initial_state = [tf.zeros([batch_size, 1], name="rank_cell_initial_state")]
        return initial_state

    def call(self, inputs, states, training=None):
        """Return probability of a ranked selection sequence.

        Args:
            inputs[".*_stimulus_set]: A tensor containing
                indices that define the stimuli used in each trial.
                shape=(batch_size, max_reference + 1)

        Returns:
            outcome_prob: Probability of different behavioral outcomes.

        NOTE: This computation takes advantage of log-probability
            space, exploiting the fact that log(prob=1)=1 to make
            vectorization cleaner.

        """
        # NOTE: The inputs are copied, because modifying the original `inputs`
        # is bad practice in TF. For example, it creates issues when saving
        # a model.
        inputs_copied = copy.copy(inputs)

        stimulus_set = inputs_copied[self.data_scope + "_stimulus_set"]
        is_reference_present = self._is_reference_present(stimulus_set)

        # Compute pairwise similarity between query and references.
        sim_qr = self._pairwise_similarity(inputs_copied)

        outcome_prob = self._compute_outcome_probability(is_reference_present, sim_qr)

        states_tplus1 = [states[0] + 1]
        return outcome_prob, states_tplus1
