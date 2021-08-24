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
    RankBehavior: A rank behavior layer.

"""

import tensorflow as tf

from psiz.keras.layers.behaviors.base import Behavior


@tf.keras.utils.register_keras_serializable(
    package='psiz.keras.layers', name='RankBehavior'
)
class RankBehavior(Behavior):
    """A rank behavior layer.

    Embodies a `_tf_ranked_sequence_probability` call.

    """

    def __init__(self, **kwargs):
        """Initialize.

        Arguments:
            kwargs (optional): Additional keyword arguments.

        """
        super().__init__(**kwargs)

    def call(self, inputs):
        """Return probability of a ranked selection sequence.

        See: _ranked_sequence_probability for NumPy implementation.

        Arguments:
            inputs[0]: i.e., sim_qr: A tensor containing the
                precomputed similarities between the query stimuli and
                corresponding reference stimuli.
                shape=(batch_size, n_sample, n_max_reference, n_outcome)
            inputs[1]: i.e., is_select: A float tensor indicating if a
                reference was selected, which indicates a true event.
                shape = (batch_size, 1, n_max_reference, n_outcome)
            inputs[2]: i.e., is_outcome: A float tensor indicating if
                an outcome is real or a padded placeholder.
                shape = (batch_size, 1, n_outcome)

        NOTE: This computation takes advantage of log-probability
            space, exploiting the fact that log(prob=1)=1 to make
            vectorization cleaner.

        """
        sim_qr = inputs[0]
        is_select = inputs[1]
        is_outcome = inputs[2]

        # Compute denominator based on formulation of Luce's choice rule by
        # summing over the different references present in a trial. Note that
        # the similarity for placeholder references will be zero since they
        # were zeroed out by the caller.
        denom = tf.cumsum(sim_qr, axis=2, reverse=True)

        # Compute log-probability of each selection, assuming all selections
        # occurred. Add fuzz factor to avoid log(0)
        sim_qr = tf.maximum(sim_qr, tf.keras.backend.epsilon())
        denom = tf.maximum(denom, tf.keras.backend.epsilon())
        event_logprob = tf.math.log(sim_qr) - tf.math.log(denom)

        # Mask non-existent events (i.e, reference selections).
        event_logprob = is_select * event_logprob

        # Compute log-probability of outcome (i.e., a sequence of events).
        outcome_logprob = tf.reduce_sum(event_logprob, axis=2)
        outcome_prob = tf.math.exp(outcome_logprob)
        outcome_prob = is_outcome * outcome_prob

        # Clean up numerical errors in probabilities.
        total_outcome_prob = tf.reduce_sum(outcome_prob, axis=2, keepdims=True)
        outcome_prob = outcome_prob / total_outcome_prob
        return outcome_prob

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        return config
