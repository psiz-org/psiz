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
from psiz.utils import random_combinations


class RandomRate(DocketGenerator):
    """A trial generator that blindly samples trials."""

    def __init__(self, eligible_indices, n_present=2, mask_zero=False):
        """Initialize.

        Args:
            eligible_indices: A 1D array-like of integers indicating
                the eligible indices.
            n_present: A scalar indicating the number of unique stimuli
                per trial.
            mask_zero (optional): A Boolean indicating if zero should
                be interpretted as a mask value in `stimulus_set`. By
                default, `mask_zero=False`.

        """
        DocketGenerator.__init__(self)

        eligible_indices = np.array(eligible_indices, copy=False)
        if eligible_indices.ndim == 0:
            raise ValueError("Argument `eligible_indices` must be 1D.")
        elif eligible_indices.ndim != 1:
            raise ValueError("Argument `eligible_indices` must be 1D.")
        self.eligible_indices = eligible_indices
        self.n_stimuli = len(eligible_indices)

        if n_present > self.n_stimuli:
            raise ValueError("`n_present` must be less than `n_stimuli`")

        self.n_present = n_present
        self.mask_zero = mask_zero

    def generate(self, n_trial):
        """Return generated trials based on provided arguments.

        Args:
            n_trial: A scalar indicating the number of trials to
                generate.

        Returns:
            A RateDocket object.

        """
        idx_eligable = self.eligible_indices
        prob = np.ones([self.n_stimuli]) / self.n_stimuli
        stimulus_set = random_combinations(
            idx_eligable, self.n_present, n_trial, p=prob
        )
        return RateDocket(stimulus_set, mask_zero=self.mask_zero)
