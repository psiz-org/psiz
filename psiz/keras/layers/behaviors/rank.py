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
from tensorflow.python.keras import backend as K

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
        super(RankBehavior, self).__init__(**kwargs)

    def call(self, inputs):
        """Return probability of a ranked selection sequence.

        See: _ranked_sequence_probability for NumPy implementation.

        Arguments:
            inputs:
                sim_qr: A tensor containing the precomputed
                    similarities between the query stimuli and
                    corresponding reference stimuli.
                    shape=(sample_size, batch_size, n_max_reference, n_outcome)
                is_select: A Boolean tensor indicating if a reference
                    was selected.
                    shape = (batch_size, n_max_reference, n_outcome)

        """
        sim_qr = inputs[0]
        is_select = inputs[1]
        is_outcome = inputs[2]

        # Initialize sequence log-probability. Note that log(prob=1)=1.
        # sample_size = tf.shape(sim_qr)[0]
        # batch_size = tf.shape(sim_qr)[1]
        # n_outcome = tf.shape(sim_qr)[3]
        # seq_log_prob = tf.zeros(
        #     [sample_size, batch_size, n_outcome], dtype=K.floatx()
        # )

        # Compute denominator based on formulation of Luce's choice rule.
        denom = tf.cumsum(sim_qr, axis=2, reverse=True)

        # Compute log-probability of each selection, assuming all selections
        # occurred. Add fuzz factor to avoid log(0)
        sim_qr = tf.maximum(sim_qr, tf.keras.backend.epsilon())
        denom = tf.maximum(denom, tf.keras.backend.epsilon())
        log_prob = tf.math.log(sim_qr) - tf.math.log(denom)

        # Mask non-existent selections.
        log_prob = is_select * log_prob

        # Compute sequence log-probability
        seq_log_prob = tf.reduce_sum(log_prob, axis=2)
        seq_prob = tf.math.exp(seq_log_prob)
        seq_prob = is_outcome * seq_prob

        # Clean up probabilities
        total = tf.reduce_sum(seq_prob, axis=2, keepdims=True)
        seq_prob = seq_prob / total
        return seq_prob

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        return config
