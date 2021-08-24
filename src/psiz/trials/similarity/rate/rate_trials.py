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
"""Rate trials module.

On each similarity judgment trial, an agent rates the similarity
between a two stimuli.

Classes:
    RateTrials: Abstract base class for 'Rate' trials.

"""

from abc import ABCMeta

import numpy as np

from psiz.trials.similarity.similarity_trials import SimilarityTrials


class RateTrials(SimilarityTrials, metaclass=ABCMeta):
    """Abstract base class for rank-type trials."""

    def __init__(self, stimulus_set):
        """Initialize.

        Arguments:
            stimulus_set: An integer matrix containing indices that
                indicate the set of stimuli used in each trial. Each
                row indicates the stimuli used in one trial. It is
                assumed that stimuli indices are composed of integers
                from [0, N-1], where N is the number of unique stimuli.
                The value -1 can be used as a placeholder for
                non-existent stimuli.
                shape = (n_trial, max(n_present))

        """
        SimilarityTrials.__init__(self, stimulus_set)

        n_present = self._infer_n_present(stimulus_set)
        self.n_present = self._check_n_present(n_present)

        # Format stimulus set.
        self.max_n_present = np.amax(self.n_present)
        self.stimulus_set = self.stimulus_set[:, 0:self.max_n_present]

    def _check_n_present(self, n_present):
        """Check the argument `n_present`.

        Raises:
            ValueError

        """
        if np.sum(np.less(n_present, 2)) > 0:
            raise ValueError((
                "The argument `stimulus_set` must contain at least two "
                "non-negative integers per a row, i.e., at least "
                "two stimuli per trial."))
        return n_present
