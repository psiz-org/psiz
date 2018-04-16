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

from psiz.trials import JudgedTrials


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
            embedding:
            group_id (optional): If the provided embedding was inferred
                for more than one group, an index can be provided to
                indicate which set of attention weights should be used.
        """
        self.embedding = embedding
        self.group_idx = group_idx

    def simulate(self, trials):
        """Simulate similarity judgments for provided trials.

        Args:
            displays: UnjudgedTrials object representing the
                to-be-judged trials. The order of the stimuli in the
                stimulus set is ignored for the simulations.

        Returns:
            JudgedTrials object representing the judged trials. The
                order of the stimuli is now informative.

        """
        probs = self._probability(trials)
        return JudgedTrials(trials.stimulus_set)  # HACK

    def _probability(self, trials):
        """Return probability of outcomes for each trial.
        
        Args:
            trials: A set of unjudged similarity trials.

        Returns:
            The probabilities associated with the different outcomes
                for each unjudged trial.

        """
        trial_configurations = trials.configurations

        Z_q = None
        Z_ref = None

        
        outcome_idx = possible_outcomes(display_configuration)

        # Precompute similarity between query and references.
        self.embedding.similarity(Z_q, Z_ref)

        # Compute probability of each possible outcome.