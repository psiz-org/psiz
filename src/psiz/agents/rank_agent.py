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
"""Module for simulating agent behavior.

Classes:
    RankAgent: An object that can be initialized using a psychological
        embedding and used to simulate Rank similarity judgments.

"""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from psiz.agents.base import Agent
from psiz.trials import RankObservations


class RankAgent(Agent):  # pylint: disable=too-few-public-methods
    """Agent that simulates Rank similarity judgments.

    Attributes:
        model: A PsychologicalEmbedding object that supplies a
            similarity function and embedding points.
        groups: Array of integers indicating group membership
            information.

    Methods:
        simulate: Stochastically simulate similarity judgments.

    """

    def __init__(self, model, groups=None):
        """Initialize.

        Arguments:
            model: A concrete instance of a PsychologicalEmedding
                object.
            groups (optional): Array-like integers indicating group
                membership information. For example, `[4, 3]` indicates
                that the first optional column has the value 4 and the
                second optional column has the value 3.

        """
        Agent.__init__(self)
        self.model = model
        if groups is None:
            groups = []
        self.groups = groups

    def simulate(self, docket, batch_size=None):
        """Stochastically simulate similarity judgments.

        Arguments:
            docket: A RankDocket object representing the
                to-be-judged trials. The order of the stimuli in the
                stimulus set is ignored for the simulations.
            batch_size (optional): If None, `batch_size` is equal to
                the total number of trials.

        Returns:
            RankObservations object representing the judged trials. The
                order of the stimuli is now informative.

        """
        if batch_size is None:
            batch_size = docket.n_trial

        if len(self.groups) == 0:
            group_matrix = None
        else:
            group_matrix = np.expand_dims(self.groups, axis=0)
            group_matrix = np.repeat(group_matrix, docket.n_trial, axis=0)

        # Create TF dataset.
        ds_docket = docket.as_dataset(group_matrix).batch(
            batch_size, drop_remainder=False
        )

        # Call model with TensorFlow formatted docket and
        # stochastically sample an outcome.
        stimulus_set = None
        for data in ds_docket:
            dict_x, _, _ = tf.keras.utils.unpack_x_y_sample_weight(data)

            batch_stimulus_set = _rank_sample(
                dict_x['stimulus_set'],
                tf.reduce_mean(self.model(dict_x, training=False), axis=1)
            )
            if stimulus_set is None:
                stimulus_set = [batch_stimulus_set]
            else:
                stimulus_set.append(batch_stimulus_set)
        stimulus_set = tf.concat(stimulus_set, 0).numpy() - 1

        obs = RankObservations(
            stimulus_set,
            n_select=docket.n_select,
            is_ranked=docket.is_ranked,
            groups=group_matrix
        )
        return obs


def _rank_sample(stimulus_set, probs):
    """Stochasatically select outcome.

    Arguments:
        stimulus_set:
            shape=(batch_size, n_reference + 1, n_outcome)
        probs:
            shape=(batch_size, n_outcome)

    Returns:
        stimulus_set_selected:
            shape=(batch_size, n_reference + 1)

    """
    outcome_distribution = tfp.distributions.Categorical(
        probs=probs
    )
    idx_sample = outcome_distribution.sample()
    idx_batch = tf.range(tf.shape(idx_sample)[0])
    idx_batch_sample = tf.stack([idx_batch, idx_sample], axis=1)

    # Retrieve stimulus set associated with particular outcome.
    stimulus_set = tf.transpose(stimulus_set, perm=[0, 2, 1])
    stimulus_set_selected = tf.gather_nd(stimulus_set, idx_batch_sample)
    return stimulus_set_selected
