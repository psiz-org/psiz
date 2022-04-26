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
    RankSimilarity: A rank behavior layer.

"""

import tensorflow as tf
from tensorflow.keras import backend as K

from psiz.keras.layers.behaviors.behavior2 import Behavior2


@tf.keras.utils.register_keras_serializable(
    package='psiz.keras.layers', name='RankSimilarity'
)
class RankSimilarity(Behavior2):
    """A rank similarity behavior layer.

    Embodies a `_tf_ranked_sequence_probability` call.

    """

    def on_kernel_begin(self, z):
        """Call at the start of kernel operation.

        Args:
            z: TODO
                shape=TensorShape(
                    [batch_size, n_sample, n_ref + 1, n_outcome, n_dim]
                )

        Returns:
            z_q: TODO
                shape=TensorShape(
                    [batch_size, n_sample, 1, n_outcome, n_dim]
                )
            z_r: TODO
                shape=TensorShape(
                    [batch_size, n_sample, n_ref, n_outcome, n_dim]
                )

        """
        # Define some useful variables before manipulating inputs.
        max_n_reference = tf.shape(z)[-3] - 1

        # Split query and reference embeddings:
        # z_q: TensorShape([batch_size, sample_size, 1, n_outcome, n_dim]
        # z_r: TensorShape([batch_size, sample_size, n_ref, n_outcome, n_dim]
        z_q, z_r = tf.split(z, [1, max_n_reference], -3)
        # The tf.split op does not infer split dimension shape. We know that
        # z_q will always have shape=1, but we don't know `max_n_reference`
        # ahead of time.
        z_q.set_shape([None, None, 1, None, None])

        return z_q, z_r

    def format_inputs(self, inputs):
        """TODO.

        Args:
            inputs: A dictionary of inputs. TODO   

        Returns:
            inputs_list: A list of inputs.

        """
        inputs_list = [inputs['stimulus_set'], inputs['is_select']]
        return inputs_list

    def call(self, inputs):
        """Return probability of a ranked selection sequence.

        Args:
            TODO
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

        Returns:
            outcome_prob: Probability of different behavioral outcomes.

        NOTE: This computation takes advantage of log-probability
            space, exploiting the fact that log(prob=1)=1 to make
            vectorization cleaner.

        """
        sim_qr = inputs[0]
        stimulus_set = inputs[1]
        is_select = inputs[2][:, 1:, :]
        # is_outcome = inputs[2]

        # Zero out similarities involving placeholder IDs by creating
        # a mask based on reference indices. We drop the query indices
        # because they have effectively been "consumed" by the similarity
        # operation.
        is_present = tf.cast(
            tf.math.not_equal(stimulus_set[:, :, 1:], 0), K.floatx()
        )
        sim_qr = sim_qr * is_present

        # Prepare for efficient probability computation by adding
        # singleton dimension for `n_sample`.
        is_select = tf.expand_dims(
            tf.cast(is_select, K.floatx()), axis=1
        )
        # Determine if outcome is legitamate by checking if at least one
        # reference is present. This is important because not all trials have
        # the same number of possible outcomes and we need to infer the
        # "zero-padding" of the outcome axis.
        is_outcome = is_present[:, :, 0, :]

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
