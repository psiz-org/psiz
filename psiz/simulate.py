# -*- coding: utf-8 -*-
# Copyright 2018 The PsiZ Authors. All Rights Reserved.
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
from numpy.random import multinomial
import tensorflow as tf

from psiz.trials import JudgedTrials


class Agent(object):
    """Agent that simulates similarity judgments.

    Attributes:
        embedding: A PsychologicalEmbedding object that supplies a
            similarity function and embedding points.
        group_id: An integer idicating which set of attention weights
            to use when simulating judgments.
    Methods:
        simulate: Stochastically simulate similarity judgments.

    """

    def __init__(self, embedding, group_id=0):
        """Initialize.

        Arguments:
            embedding: A concrete instance of a PsychologicalEmedding
                object.
            group_id (optional): If the provided embedding was inferred
                for more than one group, an index can be provided to
                indicate which set of attention weights should be used.
        """
        self.embedding = embedding
        self.group_id = group_id

    def simulate(self, trials, override_group_id=None):
        """Stochastically simulate similarity judgments.

        Arguments:
            trials: UnjudgedTrials object representing the
                to-be-judged trials. The order of the stimuli in the
                stimulus set is ignored for the simulations.
            override_group_id (optional): A scalar indicating the group_id that
                should be used to mark the observations. This is a
                useful option when stacking observations generated
                from different Agents.

        Returns:
            JudgedTrials object representing the judged trials. The
                order of the stimuli is now informative.

        """
        group_id = self.group_id * np.ones((trials.n_trial), dtype=np.int32)    
        prob_all = self.embedding.outcome_probability(
            trials, group_id=group_id)
        judged_trials = self._select(trials, prob_all)

        if override_group_id is not None:
            override_group_id = (
                override_group_id * np.ones((trials.n_trial), dtype=np.int32)
            )
            judged_trials = JudgedTrials(
                judged_trials.stimulus_set, judged_trials.n_selected,
                judged_trials.is_ranked, override_group_id)
        return judged_trials

    def _select(self, trials, prob_all):
        """Stochastically select from possible outcomes.

        Arguments:
            trials:
            prob_all:


        Returns:
            A JudgedTrials object.

        """
        outcome_idx_list = trials.outcome_idx_list

        n_trial_all = trials.n_trial
        trial_idx_all = np.arange(n_trial_all)
        max_n_ref = trials.stimulus_set.shape[1] - 1
        n_config = trials.config_list.shape[0]

        # Pre-allocate.
        chosen_outcome_idx = np.empty((n_trial_all), dtype=np.int32)
        stimulus_set = -1 * np.ones(
            (n_trial_all, 1 + max_n_ref), dtype=np.int32
        )
        stimulus_set[:, 0] = trials.stimulus_set[:, 0]
        for i_config in range(n_config):
            n_reference = trials.config_list.iloc[i_config]['n_reference']
            outcome_idx = outcome_idx_list[i_config]
            n_outcome = outcome_idx.shape[0]
            dummy_idx = np.arange(0, n_outcome)
            trial_locs = trials.config_idx == i_config
            n_trial = np.sum(trial_locs)
            trial_idx = trial_idx_all[trial_locs]
            prob = prob_all[trial_locs, 0:n_outcome]
            stimuli_set_ref = trials.stimulus_set[trial_locs, 1:]

            for i_trial in range(n_trial):
                outcome_loc = multinomial(1, prob[i_trial, :]).astype(bool)
                chosen_outcome_idx[trial_idx[i_trial]] = dummy_idx[outcome_loc]
                stimulus_set[trial_idx[i_trial], 1:n_reference+1] = \
                    stimuli_set_ref[i_trial, outcome_idx[outcome_loc, :]]

        group_id = np.full((trials.n_trial), self.group_id, dtype=np.int32)
        return JudgedTrials(
                stimulus_set,
                n_selected=trials.n_selected,
                is_ranked=trials.is_ranked, group_id=group_id
            )
