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

"""Module for testing simulate.py.

Todo:
    - test outcome_idx_list
    - test resultant group_id of judged trials
"""

import numpy as np
import tensorflow as tf
import pytest

from psiz.trials import UnjudgedTrials
from psiz.models import Exponential
from psiz.simulate import Agent, possible_outcomes


@pytest.fixture(scope="module")
def ground_truth():
    """Return a ground truth embedding."""
    n_stimuli = 10
    n_dim = 2

    model = Exponential(n_stimuli)
    mean = np.ones((n_dim))
    cov = np.identity(n_dim)
    z = np.random.multivariate_normal(mean, cov, (n_stimuli))
    freeze_options = {
        'rho': 2,
        'tau': 1,
        'beta': 1,
        'gamma': 0,
        'z': z
    }
    model.freeze(freeze_options)
    return model


@pytest.fixture(scope="module")
def unjudged_trials():
    """Return a set of unjudged trials."""
    stimulus_set = np.array((
        (0, 1, 2, 7, 3),
        (3, 4, 5, 9, 1),
        (1, 8, 9, 2, -1),
        (7, 3, 2, 7, -1),
        (6, 7, 5, 0, -1),
        (2, 1, 0, 6, -1),
        (3, 0, 2, 6, -1),
    ))
    n_selected = np.array((
        2, 2, 2, 2, 1, 1, 1
        ), dtype=np.int64)
    unjudged_trials = UnjudgedTrials(stimulus_set, n_selected=n_selected)
    return unjudged_trials


def test_simulate(ground_truth, unjudged_trials):
    """Test simulation of agent."""
    agent = Agent(ground_truth)
    obs = agent.simulate(unjudged_trials)
    np.testing.assert_array_equal(
        obs.stimulus_set[:, 0], unjudged_trials.stimulus_set[:, 0]
    )


def test_possible_outcomes_2c1():
    """Test outcomes 2 choose 1 ranked trial."""
    stimulus_set = np.array(((0, 1, 2), (9, 12, 7)))
    n_selected = 1 * np.ones((2))
    tasks = UnjudgedTrials(stimulus_set, n_selected=n_selected)

    po = possible_outcomes(tasks.config_list.iloc[0])

    correct = np.array(((0, 1), (1, 0)))
    np.testing.assert_array_equal(po, correct)


def test_possible_outcomes_3c2():
    """Test outcomes 3 choose 2 ranked trial."""
    stimulus_set = np.array(((0, 1, 2, 3), (33, 9, 12, 7)))
    n_selected = 2 * np.ones((2))
    tasks = UnjudgedTrials(stimulus_set, n_selected=n_selected)

    po = possible_outcomes(tasks.config_list.iloc[0])

    correct = np.array((
        (0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0),
        (2, 0, 1), (2, 1, 0)))
    np.testing.assert_array_equal(po, correct)


def test_possible_outcomes_4c2():
    """Test outcomes 4 choose 2 ranked trial."""
    stimulus_set = np.array(((0, 1, 2, 3, 4), (45, 33, 9, 12, 7)))
    n_selected = 2 * np.ones((2))
    tasks = UnjudgedTrials(stimulus_set, n_selected=n_selected)

    po = possible_outcomes(tasks.config_list.iloc[0])

    correct = np.array((
        (0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2),
        (1, 0, 2, 3), (1, 2, 0, 3), (1, 3, 0, 2),
        (2, 0, 1, 3), (2, 1, 0, 3), (2, 3, 0, 1),
        (3, 0, 1, 2), (3, 1, 0, 2), (3, 2, 0, 1)))
    np.testing.assert_array_equal(po, correct)


def test_possible_outcomes_8c1():
    """Test outcomes 8 choose 1 ranked trial."""
    stimulus_set = np.array((
        (0, 1, 2, 3, 4, 5, 6, 7, 8),
        (45, 33, 9, 12, 7, 2, 5, 4, 3)))
    n_selected = 1 * np.ones((2))
    tasks = UnjudgedTrials(stimulus_set, n_selected=n_selected)

    po = possible_outcomes(tasks.config_list.iloc[0])

    correct = np.array((
        (0, 1, 2, 3, 4, 5, 6, 7),
        (1, 0, 2, 3, 4, 5, 6, 7),
        (2, 0, 1, 3, 4, 5, 6, 7),
        (3, 0, 1, 2, 4, 5, 6, 7),
        (4, 0, 1, 2, 3, 5, 6, 7),
        (5, 0, 1, 2, 3, 4, 6, 7),
        (6, 0, 1, 2, 3, 4, 5, 7),
        (7, 0, 1, 2, 3, 4, 5, 6)))
    np.testing.assert_array_equal(po, correct)

