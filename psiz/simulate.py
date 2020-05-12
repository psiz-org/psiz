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
# ==============================================================================

"""Module for simulating agent behavior.

Classes:
    Agent: An object that can be initialized using a psychological
        embedding and used to simulate similarity judgments.

"""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from psiz.trials import RankObservations


class Agent(object):
    """Agent that simulates similarity judgments.

    Attributes:
        embedding: A PsychologicalEmbedding object that supplies a
            similarity function and embedding points.
        group_id: An integer indicating which set of attention weights
            to use when simulating judgments.

    Methods:
        simulate: Stochastically simulate similarity judgments.

    """

    def __init__(self, embedding, group_id=0, agent_id=0):
        """Initialize.

        Arguments:
            embedding: A concrete instance of a PsychologicalEmedding
                object.
            group_id (optional): If the provided embedding was inferred
                for more than one group, an index can be provided to
                indicate which set of attention weights should be used.
            agent_id: An integer array indicating the agent ID of a
                trial. It is assumed that all IDs are non-negative and
                that observations with the same agent ID were judged by
                a single agent.
                shape = (n_trial,)

        """
        self.embedding = embedding
        self.group_id = group_id
        self.agent_id = agent_id

    def simulate(self, docket, session_id=None):
        """Stochastically simulate similarity judgments.

        Arguments:
            docket: A RankDocket object representing the
                to-be-judged trials. The order of the stimuli in the
                stimulus set is ignored for the simulations.
            session_id: An integer array indicating the session ID of a
                trial. It is assumed that all IDs are non-negative.
                Trials with different session IDs were obtained during
                different sessions.

        Returns:
            RankObservations object representing the judged trials. The
                order of the stimuli is now informative.

        """
        agent_id = self.agent_id * np.ones((docket.n_trial), dtype=np.int32)
        group_id = self.group_id * np.ones((docket.n_trial), dtype=np.int32)

        # Call model with TensorFlow formatted docket.
        membership = np.stack((group_id, agent_id), axis=-1)
        inputs = docket.as_dataset(membership, all_outcomes=True)
        prob_all = self.embedding.model(inputs)

        obs = self._select(
            docket, prob_all, inputs['stimulus_set'] - 1,
            session_id=session_id
        )

        return obs

    def _select(self, docket, prob_all, stimulus_set_expand, session_id=None):
        """Stochastically select from possible outcomes.

        Arguments:
            docket: An RankDocket object.
            prob_all: A 2D Tensor indicating the probabilites of all outcomes.
                shape=[n_trial, n_max_outcome]
            stimulus_set_expand: An expanded stimulus set 3D array.
                shape=[n_trial, n_max_reference+1, n_max_outcome]
            session_id (optional): The session ID.

        Returns:
            A RankObservations object.

        """
        stimulus_set_expand = stimulus_set_expand.numpy()

        # Clean up rounding errors.
        prob_all /= tf.reduce_sum(prob_all, axis=1, keepdims=True)
        # Sample from outcomes.
        dist = tfp.distributions.Multinomial(
            1, probs=prob_all, name='Multinomial'
        )
        outcome_mask = tf.cast(dist.sample(), dtype=tf.bool)
        outcome_mask = outcome_mask.numpy()

        stimulus_set = np.empty(stimulus_set_expand.shape[0:2], dtype=np.int32)
        for i_trial in range(docket.n_trial):
            stimulus_set[i_trial, :] = stimulus_set_expand[
                i_trial, :, outcome_mask[i_trial]
            ]

        group_id = np.full((docket.n_trial), self.group_id, dtype=np.int32)
        agent_id = np.full((docket.n_trial), self.agent_id, dtype=np.int32)
        obs = RankObservations(
            stimulus_set,
            n_select=docket.n_select,
            is_ranked=docket.is_ranked,
            group_id=group_id, agent_id=agent_id,
            session_id=session_id
        )
        return obs
