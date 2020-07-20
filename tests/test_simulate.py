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
"""Module for testing simulate.py.

Todo:
    - test outcome_idx_list
    - test resultant group_id of judged trials
    - test simulate and _select for a Trial object with different
        configurations.
"""

import numpy as np
import numpy.ma as ma
import pytest

from psiz import simulate
from psiz.trials import RankDocket
from psiz.models import Exponential


@pytest.fixture(scope="module")
def ground_truth():
    """Return a ground truth embedding."""
    n_stimuli = 10
    n_dim = 2

    model = Exponential(n_stimuli)
    mean = np.ones((n_dim))
    cov = np.identity(n_dim)
    z = np.random.multivariate_normal(mean, cov, (n_stimuli))
    model.z = z
    model.rho = 2
    model.tau = 1
    model.beta = 1
    model.gamma = 0
    return model


@pytest.fixture(scope="module")
def docket_0():
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
    n_select = np.array((
        2, 2, 2, 2, 1, 1, 1
        ), dtype=np.int32)
    docket = RankDocket(stimulus_set, n_select=n_select)
    return docket


def test_simulate(ground_truth, docket_0):
    """Test simulation of agent."""
    agent = simulate.Agent(ground_truth.model)
    obs = agent.simulate(docket_0)
    np.testing.assert_array_equal(
        obs.stimulus_set[:, 0], docket_0.stimulus_set[:, 0]
    )


def test_select(ground_truth):
    """Test _select method."""
    # TODO more cases, make assert safer
    # stimulus_set = np.array((
    #     (0, 1, 2, 7, 3),
    #     (3, 4, 5, 9, 1),
    #     (1, 8, 9, 2, -1),
    #     (7, 3, 2, 7, -1),
    #     (6, 7, 5, 0, -1),
    #     (2, 1, 0, 6, -1),
    # ))
    # n_select = np.array((
    #     2, 2, 2, 2, 1, 1
    #     ), dtype=np.int32)

    n_trial = 100000
    stimulus_set = np.array((
        (0, 1, 2, 7, 3),
        (3, 4, 5, 9, 1),
    ))
    stimulus_set = np.tile(stimulus_set, (int(n_trial/2), 1))
    n_select = 2 * np.ones(n_trial, dtype=np.int32)
    docket = RankDocket(stimulus_set, n_select=n_select)

    agent = simulate.Agent(ground_truth.model)
    prob_all = np.array((
        (.01, .01, .01, .01, .01, .8, .1, .01, .01, .01, .01, .01),
        (.01, .01, .01, .01, .01, .8, .1, .01, .01, .01, .01, .01),
    ))
    prob_all = np.tile(prob_all, (int(n_trial/2), 1))
    # prob_all = np.array((
    #     (.01, .01, .01, .01, .01, .8, .1, .01, .01, .01, .01, .01),
    #     (.01, .01, .01, .01, .01, .8, .1, .01, .01, .01, .01, .01),
    #     (.1, .8, .07, .01, .01, .01, -1, -1, -1, -1, -1, -1),
    #     (.1, .8, .07, .01, .01, .01, -1, -1, -1, -1, -1, -1),
    #     (.1, .8, .1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    #     (.1, .8, .1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    #     (.1, .8, .1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    # ))
    prob_all = ma.masked_values(prob_all, -1)
    (_, chosen_outcome_idx) = agent._select(docket, prob_all)
    _, counts = np.unique(chosen_outcome_idx, return_counts=True)
    prop = counts / np.sum(counts)

    x = np.array([.01, .01, .01, .01, .01, .8, .1, .01, .01, .01, .01, .01])
    np.testing.assert_allclose(x, prop, rtol=1e-6, atol=.005)

        # Old approach. TODO
        # prob_all_2 = self.stimuli.outcome_probability(
        #     docket, group_id=group_id
        # )
        # prob_all_old = prob_all_2.data
        # prob_all_old[prob_all_2.mask] = 0
        # np.testing.assert_array_almost_equal(prob_all.numpy(), prob_all_old, decimal=6)  # TODO
        # (obs, _) = self._select_old(docket, prob_all_2, session_id=session_id)  TODO
