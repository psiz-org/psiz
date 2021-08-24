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
"""Generate rank-type unjudged similarity judgment trials.

Classes:
    RandomRank: Concrete class for generating random Rank similarity
        trials.

"""

import numpy as np

from psiz.trials.similarity.docket_generator import DocketGenerator
from psiz.trials.similarity.rank.rank_docket import RankDocket
from psiz.utils.choice_wo_replace import choice_wo_replace


class RandomRank(DocketGenerator):
    """A trial generator that blindly samples trials."""

    def __init__(self, n_stimuli, n_reference=2, n_select=1):
        """Initialize.

        Arguments:
            n_stimuli: A scalar indicating the total number of unique
                stimuli.
            n_reference (optional): A scalar indicating the number of
                references for each trial.
            n_select (optional): A scalar indicating the number of
                selections an agent must make.

        """
        DocketGenerator.__init__(self)

        self.n_stimuli = n_stimuli

        # Sanitize inputs.
        # TODO re-use sanitize methods from elsewhere
        self.n_reference = np.int32(n_reference)
        self.n_select = np.int32(n_select)
        self.is_ranked = True

    def generate(self, n_trial):
        """Return generated trials based on provided arguments.

        Arguments:
            n_trial: A scalar indicating the number of trials to
                generate.

        Returns:
            A RankDocket object.

        """
        n_reference = self.n_reference
        n_select = np.repeat(self.n_select, n_trial)
        is_ranked = np.repeat(self.is_ranked, n_trial)
        idx_eligable = np.arange(self.n_stimuli, dtype=np.int32)
        prob = np.ones([self.n_stimuli]) / self.n_stimuli
        stimulus_set = choice_wo_replace(
            idx_eligable, (n_trial, n_reference + 1), prob
        )

        return RankDocket(
            stimulus_set, n_select=n_select, is_ranked=is_ranked
        )
