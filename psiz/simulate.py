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

Todo:
    - wrapper similarity method

"""
import numpy as np
from numpy.random import multinomial

from psiz.trials import JudgedTrials
from psiz.utils import possible_outcomes


class Agent(object):
    """Agent that simulates similarity judgments.

    Attributes:
        embedding:
        group_idx:
    Methods:
        simulate:

    """

    def __init__(self, embedding, group_idx=0):
        """Initialize.

        Args:
            embedding: A concrete instance of a PsychologicalEmedding
                object.
            group_id (optional): If the provided embedding was inferred
                for more than one group, an index can be provided to
                indicate which set of attention weights should be used.
        """
        self.embedding = embedding
        self.group_idx = group_idx

    def simulate(self, unjudged_trials):
        """Stochastically simulate similarity judgments.

        Args:
            unjudged_trials: UnjudgedTrials object representing the
                to-be-judged trials. The order of the stimuli in the
                stimulus set is ignored for the simulations.

        Returns:
            JudgedTrials object representing the judged trials. The
                order of the stimuli is now informative.

        """
        (outcome_idx_list, prob) = self._probability(unjudged_trials)
        judged_trials = self._select(outcome_idx_list, prob, unjudged_trials)
        return judged_trials

    def _probability(self, trials):
        """Return probability of outcomes for each trial.

        Args:
            trials: A set of unjudged similarity trials.

        Returns:
            The probabilities associated with the different outcomes
                for each unjudged trial. In general, different trial
                configurations will have a different number of possible
                outcomes. Trials will a smaller number of possible
                outcomes are element padded with zeros to match the
                trial with the maximum number of outcomes.

        """
        n_trial = trials.n_trial
        max_n_ref = trials.stimulus_set.shape[1] - 1
        n_config = trials.config_list.shape[0]
        n_dim = self.embedding.z['value'].shape[1]

        outcome_idx_list = []
        n_outcome_list = []
        max_n_outcome = 0
        for i_config in range(n_config):
            outcome_idx_list.append(
                possible_outcomes(
                    trials.config_list.iloc[i_config]
                )
            )
            n_outcome = outcome_idx_list[i_config].shape[0]
            n_outcome_list.append(n_outcome)
            if n_outcome > max_n_outcome:
                max_n_outcome = n_outcome

        prob = np.zeros((n_trial, max_n_outcome))

        # Loop over different display configurations. TODO
        i_config = 0
        config = trials.config_list.iloc[i_config]
        outcome_idx = outcome_idx_list[i_config]
        n_outcome = max_n_outcome
        n_selected = int(config['n_selected'])  # TODO force int
        n_selected_idx = n_selected - 1

        z_q = self.embedding.z['value'][trials.stimulus_set[:, 0], :]
        z_q = np.expand_dims(z_q, axis=2)
        z_ref = np.empty((n_trial, n_dim, max_n_ref))
        for i_ref in range(max_n_ref):
            z_ref[:, :, i_ref] = \
                self.embedding.z['value'][trials.stimulus_set[:, 1+i_ref], :]

        # Precompute similarity between query and references.
        s_qref = self.embedding.similarity(z_q, z_ref)

        # Compute probability of each possible outcome.
        prob = np.ones((n_trial, n_outcome))
        for i_outcome in range(n_outcome):
            s_qref_perm = s_qref[:, outcome_idx[i_outcome, :]]
            # Start with last choice
            total = np.sum(s_qref_perm[:, n_selected_idx:], axis=1)
            # Compute sampling without replacement probability in reverse
            # order for numerical stabiltiy
            for i_selected in range(n_selected_idx, -1, -1):  # TODO verify
                # Grab similarity of selected reference and divide by total
                # similarity of all available references.
                prob[:, i_outcome] = np.multiply(
                    prob[:, i_outcome],
                    np.divide(s_qref_perm[:, n_selected_idx], total)
                )
                # Add similarity for "previous" selection
                if i_selected-1 > -1:
                    total = total + s_qref_perm[:, i_selected-1]  # TODO verify

        return (outcome_idx_list, prob)

    def _select(self, outcome_idx_list, prob, unjudged_trials):
        """Stochastically select from possible outcomes.

        Args:
            probs:

        Returns
            A JudgedTrials object.

        """
        n_trial = unjudged_trials.n_trial
        max_n_ref = unjudged_trials.stimulus_set.shape[1] - 1
        n_config = unjudged_trials.config_list.shape[0]
        n_dim = self.embedding.z['value'].shape[1]

        #  TODO loop over configurations
        i_config = 0
        config = unjudged_trials.config_list.iloc[i_config]
        outcome_idx = outcome_idx_list[i_config]
        stimuli_set_ref = unjudged_trials.stimulus_set[:, 1:]

        stimuli_set = -1 * np.ones((n_trial, 1 + max_n_ref), dtype=np.int64)
        stimuli_set[:, 0] = unjudged_trials.stimulus_set[:, 0]
        for i_trial in range(n_trial):
            outcome = multinomial(1, prob[i_trial, :]).astype(bool)
            stimuli_set[i_trial, 1:] = stimuli_set_ref[i_trial, outcome_idx[outcome,:]]  # TODO assign to 1:n_outcome NOT 1:

        # TODO redistribute back to global set

        return JudgedTrials(
            stimuli_set,
            n_selected=unjudged_trials.n_selected,
            is_ranked=unjudged_trials.is_ranked
        )