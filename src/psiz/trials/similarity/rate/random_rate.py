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
"""Generating rate-type unjudged similarity judgment trials.

Classes:
    RandomRate: Concrete class for generating random Rate similarity
        trials.

"""

import numpy as np

from psiz.trials.similarity.docket_generator import DocketGenerator
from psiz.trials.similarity.rate.rate_docket import RateDocket
from psiz.utils import choice_wo_replace


class RandomRate(DocketGenerator):
    """A trial generator that blindly samples trials."""

    def __init__(self, n_stimuli, n_present=2):
        """Initialize.

        Arguments:
            n_stimuli: A scalar indicating the total number of unique
                stimuli.
            n_present: A scalar indicating the number of unique stimuli
                per trial.

        """
        DocketGenerator.__init__(self)

        self.n_stimuli = n_stimuli
        self.n_present = n_present

    def generate(self, n_trial):
        """Return generated trials based on provided arguments.

        Arguments:
            n_trial: A scalar indicating the number of trials to
                generate.

        Returns:
            A RateDocket object.

        """
        idx_eligable = np.arange(self.n_stimuli, dtype=np.int32)
        prob = np.ones([self.n_stimuli]) / self.n_stimuli
        stimulus_set = choice_wo_replace(
            idx_eligable, (n_trial, self.n_present), prob
        )
        return RateDocket(stimulus_set)
