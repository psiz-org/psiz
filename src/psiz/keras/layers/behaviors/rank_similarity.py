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
    package='psiz.keras.layers', name='RankSimilarity'
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
            inputs: A dictionary containing the following information:
                rank_similarity_stimulus_set: A tensor containing
                    indices that define the stimuli used in each trial.
                    shape=(batch_size, [1,] max_reference + 1, n_outcome)
                rank_similarity_is_select: A float tensor indicating if
                    a reference was selected, which corresponds to a
                    "true" probabilistic event.
                    shape = (batch_size, [1,] n_max_reference + 1, 1)
                gate_weights (optional): Tensor(s) containing gate
                    weights. The actual key value(s) will depend on how
                    the user initialized the layer.

        Returns:
            outcome_prob: Probability of different behavioral outcomes.

        """
        # NOTE: The inputs are copied, because modifying the original `inputs`
        # is bad practice in TF. For example, it creates issues when saving
        # a model.
        inputs_copied = copy.copy(inputs)

        stimulus_set = inputs_copied['rank_similarity_stimulus_set']
        # NOTE: We drop the "query" position in `is_select`.
        # NOTE: When a sample axis is present, equivalent to:
        #     is_select = inputs['rank_similarity_is_select'][:, :, 1:]
        is_select = tf.gather(
            inputs_copied['rank_similarity_is_select'],
            indices=self._reference_indices,
            axis=self._stimuli_axis
        )

        # TODO
        # Fill sample axis if necessary.
        # if self._has_sample_axis:
        #     stimulus_set = tf.repeat(
        #         stimulus_set, self.n_sample, axis=self.sample_axis
        #     )

        # Embed stimuli indices in n-dimensional space.
        inputs_copied.update({
            'rank_similarity_stimset_samples': stimulus_set
        })
        z = self._percept_adapter(inputs_copied)
        # TensorShape=(batch_size, [n_sample,] n, [m, ...] n_dim])

        # Prepare retrieved embeddings points for kernel and then compute
        # similarity.
        z_q, z_r = self._split_stimulus_set(z)
        inputs_copied.update({
            'rank_similarity_z_q': z_q,
            'rank_similarity_z_r': z_r
        })
        sim_qr = self._kernel_adapter(inputs_copied)

        outcome_prob = self._compute_outcome_probability(
            stimulus_set, is_select, sim_qr
        )

        return outcome_prob
