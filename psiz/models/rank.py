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

import copy
import warnings

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

from psiz.models.base import PsychologicalEmbedding
import psiz.keras.layers


@tf.keras.utils.register_keras_serializable(
    package='psiz.models', name='Rank'
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
                group: dtype=tf.int32, Integers indicating the
                    group membership of a trial.
                    shape=(batch_size, k)

        """
        # Grab inputs.
        stimulus_set = inputs['stimulus_set']
        is_select = inputs['is_select'][:, 1:, :]
        group = inputs['group']

        # Inflate coordinates.
        z = self.stimuli([stimulus_set, group])

        # Check `z` shape is:
        # TensorShape([sample_size, batch_size, n_ref + 1, n_outcome, n_dim])
        if tf.math.equal(tf.rank(z), 4):
            z = tf.expand_dims(z, axis=0)
        max_n_reference = tf.shape(z)[-3] - 1
        z_q, z_r = tf.split(z, [1, max_n_reference], -3)

        # Pass through similarity kernel.
        sim_qr = self.kernel([z_q, z_r, group])
        # TensorShape([sample_size, batch_size, n_ref, n_outcome])

        # Zero out similarities involving placeholder IDs.
        is_present = tf.math.not_equal(stimulus_set, 0)
        is_present = tf.expand_dims(
            tf.cast(is_present[:, 1:, :], dtype=K.floatx()), axis=0
        )
        sim_qr = sim_qr * is_present

        # Compute probability of different behavioral outcomes.
        is_select = tf.expand_dims(
            tf.cast(is_select, dtype=K.floatx()), axis=0
        )
        is_outcome = tf.cast(is_present[:, :, 0, :], dtype=K.floatx())
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
            denom += sim_qr[:, i_selected-1, :]
    return seq_prob
