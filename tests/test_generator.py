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

"""Module for testing generator.py.

Todo:
    - test ActiveGenerator
"""

import numpy as np
import pytest

from psiz.generator import RandomGenerator, ActiveGenerator


def test_random_generator():
    """Test random generator."""
    n_stimuli_desired = 10
    n_trial_desired = 50
    n_reference_desired = 4
    n_selected_desired = 2
    is_ranked_desired = True
    generator = RandomGenerator(n_stimuli_desired)
    trials = generator.generate(
        n_trial=n_trial_desired, n_reference=n_reference_desired,
        n_selected=n_selected_desired)

    assert trials.n_trial == n_trial_desired
    assert sum(trials.n_reference == n_reference_desired) == n_trial_desired
    assert trials.stimulus_set.shape[0] == n_trial_desired
    assert trials.stimulus_set.shape[1] == 9  # TODO n_reference_desired + 1
    min_actual = np.min(trials.stimulus_set)
    max_actual = np.max(trials.stimulus_set)
    assert min_actual >= 0
    assert max_actual < n_stimuli_desired
    n_unique_desired = n_reference_desired + 1
    for i_trial in range(n_trial_desired):
        assert (
            len(np.unique(
                trials.stimulus_set[i_trial, 0:n_reference_desired+1])  # TODO the padding zeros are counted as unique, thus the indexing
            ) == n_unique_desired
        )
    assert sum(trials.n_selected == n_selected_desired) == n_trial_desired
    assert sum(trials.is_ranked == is_ranked_desired) == n_trial_desired
