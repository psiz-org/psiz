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

"""Module for testing trials.py.

Notes:
    It is critical that the function possible_outcomes returns the
        unaltered index first (as the test cases are written). Many
        downstream applications make this assumption.

Todo:
    - write test for subset and test config_idx, it is important that
    the config_idx and configuration is recomputed on initialization
    because down-stream code assumes that config_idx is from [0,N[ and
    corresponds to indices of config_list
    - test stack different config for JudgedTrials
"""

import pytest
import numpy as np
import pandas as pd

from psiz.trials import UnjudgedTrials, JudgedTrials, possible_outcomes
from psiz.generator import RandomGenerator
from psiz.simulate import Agent
from psiz.models import Exponential


@pytest.fixture(scope="module")
def setup_tasks_0():
    """
    """
    stimulus_set = np.array(((0, 1, 2, -1, -1, -1, -1, -1, -1),
                            (9, 12, 7, -1, -1, -1, -1, -1, -1),
                            (3, 4, 5, 6, 7, -1, -1, -1, -1),
                            (3, 4, 5, 6, 13, 14, 15, 16, 17)))
    n_trial = 4
    n_selected = np.array((1, 1, 1, 1))
    n_reference = np.array((2, 2, 4, 8))
    is_ranked = np.array((True, True, True, True))

    configurations = pd.DataFrame(
        {
            'n_reference': [2, 4, 8],
            'n_selected': [1, 1, 1],
            'is_ranked': [True, True, True]
        },
        index=[0, 2, 3])
    configuration_id = np.array((0, 0, 1, 2))

    tasks = UnjudgedTrials(stimulus_set)
    return {
        'n_trial': n_trial, 'stimulus_set': stimulus_set,
        'n_reference': n_reference, 'n_selected': n_selected,
        'is_ranked': is_ranked, 'tasks': tasks,
        'configurations': configurations,
        'configuration_id': configuration_id
        }


@pytest.fixture(scope="module")
def setup_tasks_1():
    """
    """
    stimulus_set = np.array(((0, 1, 2, -1, -1, -1, -1, -1, -1),
                            (9, 12, 7, -1, -1, -1, -1, -1, -1),
                            (3, 4, 5, 6, 7, -1, -1, -1, -1),
                            (3, 4, 5, 6, 13, 14, 15, 16, 17)))
    n_trial = 4
    n_selected = np.array((1, 1, 1, 2))
    n_reference = np.array((2, 2, 4, 8))
    is_ranked = np.array((True, True, True, True))

    configurations = pd.DataFrame(
        {
            'n_reference': [2, 4, 8],
            'n_selected': [1, 1, 2],
            'is_ranked': [True, True, True]
        },
        index=[0, 2, 3])
    configuration_id = np.array((0, 0, 1, 2))

    tasks = UnjudgedTrials(stimulus_set, n_selected=n_selected)
    return {
        'n_trial': n_trial, 'stimulus_set': stimulus_set,
        'n_reference': n_reference, 'n_selected': n_selected,
        'is_ranked': is_ranked, 'tasks': tasks,
        'configurations': configurations,
        'configuration_id': configuration_id
        }


@pytest.fixture(scope="module")
def setup_obs_0():
    """
    """
    stimulus_set = np.array(((0, 1, 2, -1, -1, -1, -1, -1, -1),
                            (9, 12, 7, -1, -1, -1, -1, -1, -1),
                            (3, 4, 5, 6, 7, -1, -1, -1, -1),
                            (3, 4, 5, 6, 13, 14, 15, 16, 17)))
    n_trial = 4
    n_selected = np.array((1, 1, 1, 2))
    n_reference = np.array((2, 2, 4, 8))
    is_ranked = np.array((True, True, True, True))

    configurations = pd.DataFrame(
        {
            'n_reference': [2, 4, 8],
            'n_selected': [1, 1, 2],
            'is_ranked': [True, True, True],
            'group_id': [0, 0, 0],
            'session_id': [0, 0, 0]
        },
        index=[0, 2, 3])
    configuration_id = np.array((0, 0, 1, 2))

    tasks = JudgedTrials(stimulus_set, n_selected=n_selected)
    return {
        'n_trial': n_trial, 'stimulus_set': stimulus_set,
        'n_reference': n_reference, 'n_selected': n_selected,
        'is_ranked': is_ranked, 'tasks': tasks,
        'configurations': configurations,
        'configuration_id': configuration_id
        }


@pytest.fixture(scope="module")
def setup_obs_1():
    """
    """
    stimulus_set = np.array(((0, 1, 2, -1, -1, -1, -1, -1, -1),
                            (9, 12, 7, -1, -1, -1, -1, -1, -1),
                            (3, 4, 5, 6, 7, -1, -1, -1, -1),
                            (3, 4, 5, 6, 13, 14, 15, 16, 17)))
    n_trial = 4
    n_selected = np.array((1, 1, 1, 2))
    n_reference = np.array((2, 2, 4, 8))
    is_ranked = np.array((True, True, True, True))
    group_id = np.array((0, 0, 1, 1))

    configurations = pd.DataFrame(
        {
            'n_reference': [2, 4, 8],
            'n_selected': [1, 1, 2],
            'is_ranked': [True, True, True],
            'group_id': [0, 1, 1],
            'session_id': [0, 0, 0]
        },
        index=[0, 2, 3])
    configuration_id = np.array((0, 0, 1, 2))

    tasks = JudgedTrials(stimulus_set, n_selected=n_selected,
                         group_id=group_id)
    return {
        'n_trial': n_trial, 'stimulus_set': stimulus_set,
        'n_reference': n_reference, 'n_selected': n_selected,
        'is_ranked': is_ranked, 'group_id': group_id, 'tasks': tasks,
        'configurations': configurations,
        'configuration_id': configuration_id
        }


# @pytest.fixture(scope="module")
def ground_truth(n_stimuli):
    """Return a ground truth model."""
    n_dim = 3
    n_group = 2

    model = Exponential(n_stimuli, n_dim, n_group)
    mean = np.ones((n_dim))
    cov = np.identity(n_dim)
    z = np.random.multivariate_normal(mean, cov, (n_stimuli))
    attention = np.array((
        (1.9, 1., .1),
        (.1, 1., 1.9)
    ))
    freeze_options = {
        'rho': 2,
        'tau': 1,
        'beta': 1,
        'gamma': 0,
        'z': z,
        'attention': attention
    }
    model.freeze(freeze_options)
    return model


class TestSimilarityTrials:
    """Test functionality of base class SimilarityTrials."""

    def test_invalid_n_selected(self):
        """Test handling of invalid 'n_selected' argument."""
        stimulus_set = np.array((
            (0, 1, 2, -1, -1, -1, -1, -1, -1),
            (9, 12, 7, -1, -1, -1, -1, -1, -1),
            (3, 4, 5, 6, 7, -1, -1, -1, -1),
            (3, 4, 5, 6, 13, 14, 15, 16, 17)))

        # Mismatch in number of trials
        n_selected = np.array((1, 1, 2))
        with pytest.raises(Exception) as e_info:
            trials = UnjudgedTrials(stimulus_set, n_selected=n_selected)

        # Below support.
        n_selected = np.array((1, 0, 1, 0))
        with pytest.raises(Exception) as e_info:
            trials = UnjudgedTrials(stimulus_set, n_selected=n_selected)
    
        # Above support.
        n_selected = np.array((2, 1, 1, 2))
        with pytest.raises(Exception) as e_info:
            trials = UnjudgedTrials(stimulus_set, n_selected=n_selected)

    def test_invalid_is_ranked(self):
        """Test handling of invalid 'is_ranked' argument."""
        stimulus_set = np.array((
            (0, 1, 2, -1, -1, -1, -1, -1, -1),
            (9, 12, 7, -1, -1, -1, -1, -1, -1),
            (3, 4, 5, 6, 7, -1, -1, -1, -1),
            (3, 4, 5, 6, 13, 14, 15, 16, 17)))

        # Mismatch in number of trials
        is_ranked = np.array((True, True, True))
        with pytest.raises(Exception) as e_info:
            trials = UnjudgedTrials(stimulus_set, is_ranked=is_ranked)

        is_ranked = np.array((True, False, True, False))
        with pytest.raises(Exception) as e_info:
            trials = UnjudgedTrials(stimulus_set, is_ranked=is_ranked)


class TestUnjudgedTrials:
    """Test class UnjudgedTrials."""

    def test_subset_config_idx(self):
        """Test if config_idx is updated correctly after subset."""
        stimulus_set = np.array((
            (0, 1, 2, -1, -1, -1, -1, -1, -1),
            (9, 12, 7, -1, -1, -1, -1, -1, -1),
            (3, 4, 5, 6, 7, -1, -1, -1, -1),
            (3, 4, 2, 6, 7, -1, -1, -1, -1),
            (3, 4, 5, 6, 13, 14, 15, 16, 17)))

        # Create original trials.
        n_selected = np.array((1, 1, 1, 1, 2))
        trials = UnjudgedTrials(stimulus_set, n_selected=n_selected)
        desired_config_idx = np.array((0, 0, 1, 1, 2))
        np.testing.assert_array_equal(trials.config_idx, desired_config_idx)
        # Grab subset and check that config_idx is updated to start at 0.
        trials_subset = trials.subset(np.array((2, 3, 4)))
        desired_config_idx = np.array((0, 0, 1))
        np.testing.assert_array_equal(
            trials_subset.config_idx, desired_config_idx)

    def test_stack_config_idx(self):
        """Test if config_idx is updated correctly after stack."""
        stimulus_set = np.array((
            (0, 1, 2, 3, -1, -1, -1, -1, -1),
            (9, 12, 7, 1, -1, -1, -1, -1, -1),
            (3, 4, 5, 6, 7, -1, -1, -1, -1),
            (3, 4, 2, 6, 7, -1, -1, -1, -1),
            (3, 4, 5, 6, 13, 14, 15, 16, 17)))

        # Create first set of original trials.
        n_selected = np.array((1, 1, 1, 1, 1))
        trials_0 = UnjudgedTrials(stimulus_set, n_selected=n_selected)
        desired_config_idx = np.array((0, 0, 1, 1, 2))
        np.testing.assert_array_equal(trials_0.config_idx, desired_config_idx)

        # Create second set of original trials, with non-overlapping
        # configuration.
        n_selected = np.array((2, 2, 2, 2, 2))
        trials_1 = UnjudgedTrials(stimulus_set, n_selected=n_selected)
        desired_config_idx = np.array((0, 0, 1, 1, 2))
        np.testing.assert_array_equal(trials_1.config_idx, desired_config_idx)

        # Stack trials
        trials_stack = UnjudgedTrials.stack((trials_0, trials_1))
        desired_config_idx = np.array((0, 0, 1, 1, 2, 3, 3, 4, 4, 5))
        np.testing.assert_array_equal(
            trials_stack.config_idx, desired_config_idx)

    def test_n_trial_0(self, setup_tasks_0):
        assert setup_tasks_0['n_trial'] == setup_tasks_0['tasks'].n_trial

    def test_stimulus_set_0(self, setup_tasks_0):
        np.testing.assert_array_equal(
            setup_tasks_0['stimulus_set'],
            setup_tasks_0['tasks'].stimulus_set)

    def test_n_reference_0(self, setup_tasks_0):
        np.testing.assert_array_equal(
            setup_tasks_0['n_reference'], setup_tasks_0['tasks'].n_reference)

    def test_n_selected_0(self, setup_tasks_0):
        np.testing.assert_array_equal(
            setup_tasks_0['n_selected'], setup_tasks_0['tasks'].n_selected)

    def test_is_ranked_0(self, setup_tasks_0):
        np.testing.assert_array_equal(
            setup_tasks_0['is_ranked'], setup_tasks_0['tasks'].is_ranked)

    def test_configurations_0(self, setup_tasks_0):
        pd.testing.assert_frame_equal(
            setup_tasks_0['configurations'],
            setup_tasks_0['tasks'].config_list)

    def test_configuration_id_0(self, setup_tasks_0):
        np.testing.assert_array_equal(
            setup_tasks_0['configuration_id'],
            setup_tasks_0['tasks'].config_idx)

    def test_n_trial_1(self, setup_tasks_1):
        assert setup_tasks_1['n_trial'] == setup_tasks_1['tasks'].n_trial

    def test_stimulus_set_1(self, setup_tasks_1):
        np.testing.assert_array_equal(
            setup_tasks_1['stimulus_set'], setup_tasks_1['tasks'].stimulus_set)

    def test_n_reference_1(self, setup_tasks_1):
        np.testing.assert_array_equal(
            setup_tasks_1['n_reference'], setup_tasks_1['tasks'].n_reference)

    def test_n_selected_1(self, setup_tasks_1):
        np.testing.assert_array_equal(
            setup_tasks_1['n_selected'], setup_tasks_1['tasks'].n_selected)

    def test_is_ranked_1(self, setup_tasks_1):
        np.testing.assert_array_equal(
            setup_tasks_1['is_ranked'], setup_tasks_1['tasks'].is_ranked)

    def test_configurations_1(self, setup_tasks_1):
        pd.testing.assert_frame_equal(
            setup_tasks_1['configurations'],
            setup_tasks_1['tasks'].config_list)

    def test_configuration_id_1(self, setup_tasks_1):
        np.testing.assert_array_equal(
            setup_tasks_1['configuration_id'],
            setup_tasks_1['tasks'].config_idx)


class TestJudgedTrials:
    """Test class JudgedTrials."""

    def test_invalid_group_id(self):
        """Test handling of invalid 'group_id' argument."""
        stimulus_set = np.array((
            (0, 1, 2, -1, -1, -1, -1, -1, -1),
            (9, 12, 7, -1, -1, -1, -1, -1, -1),
            (3, 4, 5, 6, 7, -1, -1, -1, -1),
            (3, 4, 5, 6, 13, 14, 15, 16, 17)))

        # Mismatch in number of trials
        group_id = np.array((0, 0, 1))
        with pytest.raises(Exception) as e_info:
            trials = JudgedTrials(stimulus_set, group_id=group_id)

        # Below support.
        group_id = np.array((0, -1, 1, 0))
        with pytest.raises(Exception) as e_info:
            trials = JudgedTrials(stimulus_set, group_id=group_id)

    def test_subset_config_idx(self):
        """Test if config_idx is updated correctly after subset."""
        stimulus_set = np.array((
            (0, 1, 2, -1, -1, -1, -1, -1, -1),
            (9, 12, 7, -1, -1, -1, -1, -1, -1),
            (3, 4, 5, 6, 7, -1, -1, -1, -1),
            (3, 4, 2, 6, 7, -1, -1, -1, -1),
            (3, 4, 5, 6, 13, 14, 15, 16, 17)))

        # Create original trials.
        n_selected = np.array((1, 1, 1, 1, 2))
        trials = JudgedTrials(stimulus_set, n_selected=n_selected)
        desired_config_idx = np.array((0, 0, 1, 1, 2))
        np.testing.assert_array_equal(trials.config_idx, desired_config_idx)
        # Grab subset and check that config_idx is updated to start at 0.
        trials_subset = trials.subset(np.array((2, 3, 4)))
        desired_config_idx = np.array((0, 0, 1))
        np.testing.assert_array_equal(
            trials_subset.config_idx, desired_config_idx)

    def test_stack_config_idx(self):
        """Test if config_idx is updated correctly after stack."""
        stimulus_set = np.array((
            (0, 1, 2, 3, -1, -1, -1, -1, -1),
            (9, 12, 7, 1, -1, -1, -1, -1, -1),
            (3, 4, 5, 6, 7, -1, -1, -1, -1),
            (3, 4, 2, 6, 7, -1, -1, -1, -1),
            (3, 4, 5, 6, 13, 14, 15, 16, 17)))

        # Create first set of original trials.
        n_selected = np.array((1, 1, 1, 1, 1))
        trials_0 = JudgedTrials(stimulus_set, n_selected=n_selected)
        desired_config_idx = np.array((0, 0, 1, 1, 2))
        np.testing.assert_array_equal(trials_0.config_idx, desired_config_idx)

        # Create second set of original trials, with non-overlapping
        # configuration.
        n_selected = np.array((2, 2, 2, 2, 2))
        trials_1 = JudgedTrials(stimulus_set, n_selected=n_selected)
        desired_config_idx = np.array((0, 0, 1, 1, 2))
        np.testing.assert_array_equal(trials_1.config_idx, desired_config_idx)

        # Stack trials
        trials_stack = JudgedTrials.stack((trials_0, trials_1))
        desired_config_idx = np.array((0, 0, 1, 1, 2, 3, 3, 4, 4, 5))
        np.testing.assert_array_equal(
            trials_stack.config_idx, desired_config_idx)

    def test_n_trial_0(self, setup_obs_0):
        assert setup_obs_0['n_trial'] == setup_obs_0['tasks'].n_trial

    def test_stimulus_set_0(self, setup_obs_0):
        np.testing.assert_array_equal(
            setup_obs_0['stimulus_set'], setup_obs_0['tasks'].stimulus_set)

    def test_n_reference_0(self, setup_obs_0):
        np.testing.assert_array_equal(
            setup_obs_0['n_reference'], setup_obs_0['tasks'].n_reference)

    def test_n_selected_0(self, setup_obs_0):
        np.testing.assert_array_equal(
            setup_obs_0['n_selected'], setup_obs_0['tasks'].n_selected)

    def test_is_ranked_0(self, setup_obs_0):
        np.testing.assert_array_equal(
            setup_obs_0['is_ranked'], setup_obs_0['tasks'].is_ranked)

    def test_configurations_0(self, setup_obs_0):
        pd.testing.assert_frame_equal(
            setup_obs_0['configurations'],
            setup_obs_0['tasks'].config_list)

    def test_configuration_id_0(self, setup_obs_0):
        np.testing.assert_array_equal(
            setup_obs_0['configuration_id'],
            setup_obs_0['tasks'].config_idx)

    def test_n_trial_1(self, setup_obs_1):
        assert setup_obs_1['n_trial'] == setup_obs_1['tasks'].n_trial

    def test_stimulus_set_1(self, setup_obs_1):
        np.testing.assert_array_equal(
            setup_obs_1['stimulus_set'], setup_obs_1['tasks'].stimulus_set)

    def test_n_reference_1(self, setup_obs_1):
        np.testing.assert_array_equal(
            setup_obs_1['n_reference'], setup_obs_1['tasks'].n_reference)

    def test_n_selected_1(self, setup_obs_1):
        np.testing.assert_array_equal(
            setup_obs_1['n_selected'], setup_obs_1['tasks'].n_selected)

    def test_is_ranked_1(self, setup_obs_1):
        np.testing.assert_array_equal(
            setup_obs_1['is_ranked'], setup_obs_1['tasks'].is_ranked)

    def test_configurations_1(self, setup_obs_1):
        pd.testing.assert_frame_equal(
            setup_obs_1['configurations'],
            setup_obs_1['tasks'].config_list)

    def test_configuration_id_1(self, setup_obs_1):
        np.testing.assert_array_equal(
            setup_obs_1['configuration_id'],
            setup_obs_1['tasks'].config_idx)


class TestStack:
    """Test stack static method."""

    def test_stack_same_config(self):
        n_stimuli = 10
        model_truth = ground_truth(n_stimuli)

        n_trial = 50
        n_reference = 8
        n_selected = 2
        generator = RandomGenerator(n_stimuli)
        trials = generator.generate(n_trial, n_reference, n_selected)

        double_trials = UnjudgedTrials.stack((trials, trials))

        assert double_trials.n_trial == 2 * n_trial
        np.testing.assert_array_equal(
            double_trials.n_reference[0:n_trial], trials.n_reference)
        np.testing.assert_array_equal(
            double_trials.n_reference[n_trial:], trials.n_reference)

        np.testing.assert_array_equal(
            double_trials.n_selected[0:n_trial], trials.n_selected)
        np.testing.assert_array_equal(
            double_trials.n_selected[n_trial:], trials.n_selected)

        np.testing.assert_array_equal(
            double_trials.is_ranked[0:n_trial], trials.is_ranked)
        np.testing.assert_array_equal(
            double_trials.is_ranked[n_trial:], trials.is_ranked)

        agent_novice = Agent(model_truth, group_id=0)
        agent_expert = Agent(model_truth, group_id=1)
        obs_novice = agent_novice.simulate(trials)
        obs_expert = agent_expert.simulate(trials)
        obs_all = JudgedTrials.stack((obs_novice, obs_expert))

        assert obs_all.n_trial == 2 * n_trial
        np.testing.assert_array_equal(
            obs_all.n_reference[0:n_trial], obs_novice.n_reference)
        np.testing.assert_array_equal(
            obs_all.n_reference[n_trial:], obs_expert.n_reference)

        np.testing.assert_array_equal(
            obs_all.n_selected[0:n_trial], obs_novice.n_selected)
        np.testing.assert_array_equal(
            obs_all.n_selected[n_trial:], obs_expert.n_selected)

        np.testing.assert_array_equal(
            obs_all.is_ranked[0:n_trial], obs_novice.is_ranked)
        np.testing.assert_array_equal(
            obs_all.is_ranked[n_trial:], obs_expert.is_ranked)

        np.testing.assert_array_equal(
            obs_all.group_id[0:n_trial], obs_novice.group_id)
        np.testing.assert_array_equal(
            obs_all.group_id[n_trial:], obs_expert.group_id)

    def test_stack_different_config(self):
        """Test stack static method with different configurations."""
        n_stimuli = 20
        generator = RandomGenerator(n_stimuli)

        n_reference1 = 2
        n_selected1 = 1
        trials1 = generator.generate(5, n_reference1, n_selected1)

        n_reference2 = 4
        n_selected2 = 2
        trials2 = generator.generate(5, n_reference2, n_selected2)

        n_reference3 = 6
        n_selected3 = 2
        trials3 = generator.generate(5, n_reference3, n_selected3)
        
        trials_all = UnjudgedTrials.stack((trials1, trials2, trials3))

        desired_n_reference = np.hstack((
            n_reference1 * np.ones((5), dtype=np.int),
            n_reference2 * np.ones((5), dtype=np.int),
            n_reference3 * np.ones((5), dtype=np.int),
        ))

        np.testing.assert_array_equal(
            trials_all.n_reference, desired_n_reference
        )

    def test_padding(self):
        """Test padding values when using stack method."""
        n_stimuli = 20
        generator = RandomGenerator(n_stimuli)

        n_reference1 = 2
        n_selected1 = 1
        trials1 = generator.generate(5, n_reference1, n_selected1)

        n_reference2 = 4
        n_selected2 = 2
        trials2 = generator.generate(5, n_reference2, n_selected2)

        n_reference3 = 8
        n_selected3 = 2
        trials3 = generator.generate(5, n_reference3, n_selected3)

        trials_all = UnjudgedTrials.stack((trials1, trials2, trials3))

        # Check padding values of first set (non-padded and then padded values).
        assert np.sum(np.equal(trials_all.stimulus_set[1:5, 0:3], -1)) == 0
        np.testing.assert_array_equal(
            trials_all.stimulus_set[0:5, 3:], -1 * np.ones((5, 6), dtype=np.int)
        )
        # Check padding values of second set (non-padded and then padded values).
        assert np.sum(np.equal(trials_all.stimulus_set[5:10, 0:5], -1)) == 0
        np.testing.assert_array_equal(
            trials_all.stimulus_set[5:10, 5:], -1 * np.ones((5, 4), dtype=np.int)
        )
        # Check padding values of third set (non-padded and then padded values).
        assert np.sum(np.equal(trials_all.stimulus_set[10:15, :], -1)) == 0


class TestPossibleOutcomes:

    def test_possible_outcomes_2c1(self):
        """Test outcomes 2 choose 1 ranked trial."""
        stimulus_set = np.array(((0, 1, 2), (9, 12, 7)))
        n_selected = 1 * np.ones((2))
        tasks = UnjudgedTrials(stimulus_set, n_selected=n_selected)

        po = possible_outcomes(tasks.config_list.iloc[0])

        correct = np.array(((0, 1), (1, 0)))
        np.testing.assert_array_equal(po, correct)

    def test_possible_outcomes_3c2(self):
        """Test outcomes 3 choose 2 ranked trial."""
        stimulus_set = np.array(((0, 1, 2, 3), (33, 9, 12, 7)))
        n_selected = 2 * np.ones((2))
        tasks = UnjudgedTrials(stimulus_set, n_selected=n_selected)

        po = possible_outcomes(tasks.config_list.iloc[0])

        correct = np.array((
            (0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0),
            (2, 0, 1), (2, 1, 0)))
        np.testing.assert_array_equal(po, correct)

    def test_possible_outcomes_4c2(self):
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

    def test_possible_outcomes_8c1(self):
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
