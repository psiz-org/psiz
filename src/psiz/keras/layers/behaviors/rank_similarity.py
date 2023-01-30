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
    RankSimilarity: A (stateless) layer for rank-similarity judgments.

"""

import copy

import tensorflow as tf

from psiz.keras.layers.behaviors.rank_similarity_base import RankSimilarityBase


@tf.keras.utils.register_keras_serializable(
    package="psiz.keras.layers", name="RankSimilarity"
)
class RankSimilarity(RankSimilarityBase):
    """A rank similarity behavior layer."""

    def __init__(self, **kwargs):
        """Initialize.

        Args:
            kwargs: See `RankSimilarityBase`

        """
        super(RankSimilarity, self).__init__(**kwargs)

    def call(self, inputs, training=None):
        """Return probability of a ranked selection sequence.

        Args:
            inputs: A dictionary containing the following information
                (the actual keys will depend on how the user
                initialized the layer.

                stimulus_set: A tensor containing indices that define
                    the stimuli used in each trial.
                    shape=(batch_size, max_reference + 1)
                gate_weights (optional): Tensor(s) containing gate
                    weights.

        Returns:
            outcome_prob: Probability of different behavioral outcomes.

        """
        # NOTE: The inputs are copied, because modifying the original `inputs`
        # is bad practice in TF. For example, it creates issues when saving
        # a model.
        # TODO move into _pairwise_similarity
        inputs_copied = copy.copy(inputs)

        stimulus_set = inputs_copied[self.data_scope + "_stimulus_set"]
        is_reference_present = self._is_reference_present(stimulus_set)

        # Compute pairwise similarity between query and references.
        sim_qr = self._pairwise_similarity(inputs_copied)

        outcome_prob = self._compute_outcome_probability(is_reference_present, sim_qr)

        return outcome_prob
