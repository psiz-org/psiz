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
    Rank: Class that uses ordinal observations that are anchored by a
        designated query stimulus.

"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

from psiz.keras.models.psych_embedding import PsychologicalEmbedding
import psiz.keras.layers


@tf.keras.utils.register_keras_serializable(
    package='psiz.keras.models', name='Rank'
)
class Rank(PsychologicalEmbedding):
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
            behavior = psiz.keras.layers.RankBehavior()
        kwargs.update({'behavior': behavior})
        super().__init__(**kwargs)

    def call(self, inputs):
        """Call.

        Arguments:
            inputs: A dictionary of inputs:
                stimulus_set: dtype=tf.int32, consisting of the
                    integers on the interval [0, n_stimuli[
                    shape=(batch_size, n_max_reference + 1, n_outcome)
                is_select: dtype=tf.bool, the shape implies the
                    maximum number of selected stimuli in the data
                    shape=(batch_size, n_max_select, n_outcome)
                groups: dtype=tf.int32, Integers indicating the
                    group membership of a trial.
                    shape=(batch_size, k)

        """
        # Grab inputs.
        stimulus_set = inputs['stimulus_set']
        is_select = inputs['is_select'][:, 1:, :]
        groups = inputs['groups']

        # Define some useful variables before manipulating inputs.
        max_n_reference = tf.shape(stimulus_set)[-2] - 1

        # Repeat `stimulus_set` `n_sample` times in a newly inserted
        # axis (axis=1).
        # TensorShape([batch_size, n_sample, n_ref + 1, n_outcome])
        stimulus_set = psiz.utils.expand_dim_repeat(
            stimulus_set, self.n_sample, axis=1
        )

        # Enbed stimuli indices in n-dimensional space:
        # TensorShape([batch_size, n_sample, n_ref + 1, n_outcome, n_dim])
        if self._use_group['stimuli']:
            z = self.stimuli([stimulus_set, groups])
        else:
            z = self.stimuli(stimulus_set)

        # Split query and reference embeddings:
        # z_q: TensorShape([batch_size, sample_size, 1, n_outcome, n_dim]
        # z_r: TensorShape([batch_size, sample_size, n_ref, n_outcome, n_dim]
        z_q, z_r = tf.split(z, [1, max_n_reference], -3)
        # The tf.split op does not infer split dimension shape. We know that
        # z_q will always have shape=1, but we don't know `max_n_reference`
        # ahead of time.
        z_q.set_shape([None, None, 1, None, None])

        # Pass through similarity kernel.
        # TensorShape([batch_size, sample_size, n_ref, n_outcome])
        if self._use_group['kernel']:
            sim_qr = self.kernel([z_q, z_r, groups])
        else:
            sim_qr = self.kernel([z_q, z_r])

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

        # Compute probability of different behavioral outcomes.
        if self._use_group['behavior']:
            probs = self.behavior([sim_qr, is_select, is_outcome, groups])
        else:
            probs = self.behavior([sim_qr, is_select, is_outcome])

        return probs


def _ranked_sequence_probability(sim_qr, n_select):
    """Return probability of a ranked selection sequence.

    Arguments:
        sim_qr: A 3D tensor containing pairwise similarity values.
            Each row (dimension 0) contains the similarity between
            a trial's query stimulus and reference stimuli. The
            tensor is arranged such that the first column
            corresponds to the first selection in a sequence, and
            the last column corresponds to the last selection
            (dimension 1). The third dimension indicates
            different samples.
            shape = (n_trial, n_reference, n_sample)
        n_select: Scalar indicating the number of selections made
            by an agent.

    Returns:
        A 2D tensor of probabilities.
        shape = (n_trial, n_sample)

    Notes:
        For example, given query Q and references A, B, and C, the
        probability of selecting reference A then B (in that order)
        would be:

        P(A)P(B|A) = s_QA/(s_QA + s_QB + s_QC) * s_QB/(s_QB + s_QC)

        where s_QA denotes the similarity between the query and
        reference A.

        The probability is computed by starting with the last
        selection for efficiency and numerical stability. In the
        provided example, this corresponds to first computing the
        probability of selecting B second, given that A was
        selected first.

    """
    n_trial = sim_qr.shape[0]
    n_sample = sim_qr.shape[2]

    # Initialize.
    seq_prob = np.ones((n_trial, n_sample), dtype=np.float64)
    selected_idx = n_select - 1
    denom = np.sum(sim_qr[:, selected_idx:, :], axis=1)

    for i_selected in range(selected_idx, -1, -1):
        # Compute selection probability.
        prob = np.divide(sim_qr[:, i_selected], denom)
        # Update sequence probability.
        # seq_prob = np.multiply(seq_prob, prob)
        seq_prob *= prob
        # Update denominator in preparation for computing the probability
        # of the previous selection in the sequence.
        if i_selected > 0:
            # denom = denom + sim_qr[:, i_selected-1, :]
            denom += sim_qr[:, i_selected - 1, :]
    return seq_prob
