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

"""Module for testing simulate.py."""


import numpy as np
import pytest

from psiz.trials import UnjudgedTrials
from psiz.models import Exponential
from psiz.simulate import Agent


@pytest.fixture(scope="module")
def ground_truth():
    """Return a ground truth embedding."""
    n_stimuli = 10
    dimensionality = 2

    model = Exponential(n_stimuli)
    mean = np.ones((dimensionality))
    cov = np.identity(dimensionality)
    Z = np.random.multivariate_normal(mean, cov, (n_stimuli))
    freeze_options = {
        'rho': 2,
        'tau': 1,
        'beta': 1,
        'gamma': 0,
        'Z': Z
    }
    model.freeze(freeze_options)
    return model


@pytest.fixture(scope="module")
def unjudged_trials():
    """Return a set of unjudged trials."""
    stimulus_set = np.array(((0, 1, 2),
                            (3, 4, 5)))
    unjudged_trials = UnjudgedTrials(stimulus_set)
    return unjudged_trials


# def test_simulate(ground_truth, unjudged_trials):
#     """Test simulation of agent."""
#     agent = Agent(ground_truth)
#     obs = agent.simulate(unjudged_trials)
#     assert obs.n_trial is 2
