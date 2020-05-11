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
        group_id = self.group_id * np.ones((docket.n_trial), dtype=np.int32)
        prob_all = self.embedding.outcome_probability(
            docket, group_id=group_id
        )
        # TODO Expand docket for possible outcomes.
        
        # docket_expand = docket.add_outcomes()
        # prob_all = self.embedding.outcome_probability_new(
        #     docket_new, group_id=group_id
        # )
        (obs, _) = self._select(docket, prob_all, session_id=session_id)
        return obs

    def _select(self, docket, prob_all, session_id=None):
        """Stochastically select from possible outcomes.

        Arguments:
            docket: An RankDocket object.
            prob_all: A MaskedArray object.


        Returns:
            A RankObservations object.

        """
        outcome_idx_list = docket.outcome_idx_list

        n_trial_all = docket.n_trial
        trial_idx_all = np.arange(n_trial_all)
        max_n_ref = docket.stimulus_set.shape[1] - 1
        n_config = docket.config_list.shape[0]

        # Pre-allocate.
        chosen_outcome_idx = np.empty((n_trial_all), dtype=np.int32)
        stimulus_set = -1 * np.ones(
            (n_trial_all, 1 + max_n_ref), dtype=np.int32
        )
        stimulus_set[:, 0] = docket.stimulus_set[:, 0]
        for i_config in range(n_config):
            n_reference = docket.config_list.iloc[i_config]['n_reference']
            outcome_idx = outcome_idx_list[i_config]
            n_outcome = outcome_idx.shape[0]
            dummy_idx = np.arange(0, n_outcome)
            trial_locs = docket.config_idx == i_config
            n_trial = np.sum(trial_locs)
            trial_idx = trial_idx_all[trial_locs]
            prob = prob_all.data[trial_locs, 0:n_outcome]
            stimuli_set_ref = docket.stimulus_set[trial_locs, 1:]

            for i_trial in range(n_trial):
                outcome_loc = np.random.multinomial(
                    1, prob[i_trial, :]).astype(bool)
                chosen_outcome_idx[trial_idx[i_trial]] = dummy_idx[outcome_loc]
                stimulus_set[trial_idx[i_trial], 1:n_reference+1] = \
                    stimuli_set_ref[i_trial, outcome_idx[outcome_loc, :]]

        group_id = np.full((docket.n_trial), self.group_id, dtype=np.int32)
        agent_id = np.full((docket.n_trial), self.agent_id, dtype=np.int32)
        return (
            RankObservations(
                stimulus_set,
                n_select=docket.n_select,
                is_ranked=docket.is_ranked,
                group_id=group_id, agent_id=agent_id, session_id=session_id
            ), chosen_outcome_idx)
