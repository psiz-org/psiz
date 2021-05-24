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
"""Test `trials` module."""

import pytest
import numpy as np
import pandas as pd
import tensorflow as tf

from psiz import trials
import psiz.keras.models
import psiz.keras.layers

from psiz.trials import RandomRate
from psiz.trials.similarity.rate.rate_docket import RateDocket
from psiz.trials.similarity.rate.rate_observations import RateObservations
from psiz.trials.similarity.rate.rate_trials import RateTrials


@pytest.fixture(scope="module")
def setup_docket_0():
    """All trials have two stimuli.
    """
    stimulus_set = np.array(
        (
            (0, 1),
            (9, 12),
            (3, 4),
            (3, 4)
        ), dtype=np.int32
    )
    n_trial = 4
    n_present = np.array((2, 2, 2, 2), dtype=np.int32)

    configurations = pd.DataFrame(
        {
            'n_present': np.array([2], dtype=np.int32),
        },
        index=[0])
    configuration_id = np.array((0, 0, 0, 0))

    docket = RateDocket(stimulus_set)
    return {
        'n_trial': n_trial, 'stimulus_set': stimulus_set,
        'n_present': n_present, 'docket': docket,
        'configurations': configurations,
        'configuration_id': configuration_id
    }


@pytest.fixture(scope="module")
def setup_docket_1():
    """Trial with three stimuli.
    """
    stimulus_set = np.array(
        (
            (0, 1, -1),
            (9, 12, -1),
            (3, 4, -1),
            (3, 4, 5)
        ), dtype=np.int32
    )
    n_trial = 4
    n_present = np.array((2, 2, 2, 3), dtype=np.int32)

    configurations = pd.DataFrame(
        {
            'n_present': np.array([2, 3], dtype=np.int32),
        },
        index=[0, 3])
    configuration_id = np.array((0, 0, 0, 1))

    docket = RateDocket(stimulus_set)
    return {
        'n_trial': n_trial, 'stimulus_set': stimulus_set,
        'n_present': n_present, 'docket': docket,
        'configurations': configurations,
        'configuration_id': configuration_id
    }


@pytest.fixture(scope="module")
def setup_obs_0():
    """Default group information.
    """
    stimulus_set = np.array(
        (
            (0, 1, -1),
            (9, 12, -1),
            (3, 4, -1),
            (3, 4, 5)
        ), dtype=np.int32
    )
    rating = np.array(
        (.1, .2, .3, .14)
    )
    n_trial = 4
    n_present = np.array((2, 2, 2, 3), dtype=np.int32)
    groups = np.zeros([n_trial, 1], dtype=np.int32)
    configurations = pd.DataFrame(
        {
            'n_present': np.array([2, 3], dtype=np.int32),
            'session_id': np.array([0, 0], dtype=np.int32),
            'groups_0': np.array([0, 0], dtype=np.int32),
        },
        index=[0, 3])
    configuration_id = np.array((0, 0, 0, 1))

    obs = RateObservations(stimulus_set, rating=rating)
    return {
        'n_trial': n_trial, 'stimulus_set': stimulus_set, 'rating': rating,
        'n_present': n_present, 'groups': groups, 'obs': obs,
        'configurations': configurations,
        'configuration_id': configuration_id
        }


@pytest.fixture(scope="module")
def setup_obs_1():
    """Varying group information.
    """
    stimulus_set = np.array(
        (
            (0, 1, -1),
            (9, 12, -1),
            (3, 4, -1),
            (3, 4, 5)
        ), dtype=np.int32
    )
    rating = np.array(
        (.1, .2, .3, .14)
    )
    n_trial = 4
    n_present = np.array((2, 2, 2, 3), dtype=np.int32)
    groups = np.array(([0], [0], [1], [1]), dtype=np.int32)

    configurations = pd.DataFrame(
        {
            'n_present': np.array([2, 2, 3], dtype=np.int32),
            'session_id': np.array([0, 0, 0], dtype=np.int32),
            'groups_0': np.array([0, 1, 1], dtype=np.int32),
        },
        index=[0, 2, 3])
    configuration_id = np.array((0, 0, 1, 2), dtype=np.int32)

    obs = RateObservations(
        stimulus_set, rating=rating, groups=groups)
    return {
        'n_trial': n_trial, 'stimulus_set': stimulus_set, 'rating': rating,
        'n_present': n_present, 'groups': groups, 'obs': obs,
        'configurations': configurations,
        'configuration_id': configuration_id
        }


def ground_truth(n_stimuli):
    """Return a ground truth model."""
    n_dim = 3
    n_group = 2

    stimuli = tf.keras.layers.Embedding(
        n_stimuli+1, n_dim, mask_zero=True
    )

    shared_similarity = psiz.keras.layers.ExponentialSimilarity(
        trainable=False,
        beta_initializer=tf.keras.initializers.Constant(1.),
        tau_initializer=tf.keras.initializers.Constant(1.),
        gamma_initializer=tf.keras.initializers.Constant(0.)
    )

    # Define group-specific kernels.
    kernel_0 = psiz.keras.layers.DistanceBased(
        distance=psiz.keras.layers.Minkowski(
            rho_trainable=False,
            rho_initializer=tf.keras.initializers.Constant(2.),
            w_initializer=tf.keras.initializers.Constant(
                [1.9, 1., .1]
            ),
        ),
        similarity=shared_similarity
    )

    kernel_1 = psiz.keras.layers.DistanceBased(
        distance=psiz.keras.layers.Minkowski(
            rho_trainable=False,
            rho_initializer=tf.keras.initializers.Constant(2.),
            w_initializer=tf.keras.initializers.Constant(
                [.1, 1., 1.9]
            ),
        ),
        similarity=shared_similarity
    )

    kernel_group = psiz.keras.layers.GateMulti(
        subnets=[kernel_0, kernel_1], group_col=0
    )

    model = psiz.keras.models.Rate(
        stimuli=stimuli, kernel=kernel_group, use_group_kernel=True
    )
    return model


class TestRateDocket:
    """Test class RateDocket."""

    def test_invalid_stimulus_set(self):
        """Test handling of invalid `stimulus_set` argument."""
        # Non-integer input.
        stimulus_set = np.array((
            (0., 1, 2, -1, -1, -1, -1, -1, -1),
            (9, 12, 7, -1, -1, -1, -1, -1, -1),
            (3, 4, 5, 6, 7, -1, -1, -1, -1),
            (3, 4, 5, 6, 13, 14, 15, 16, 17)))
        with pytest.raises(Exception) as e_info:
            docket = RateDocket(stimulus_set)

        # Contains integers below -1.
        stimulus_set = np.array((
            (0, 1, -2, -1, -1, -1, -1, -1, -1),
            (9, 12, 7, -1, -1, -1, -1, -1, -1),
            (3, 4, 5, 6, 7, -1, -1, -1, -1),
            (3, 4, 5, 6, 13, 14, 15, 16, 17)))
        with pytest.raises(Exception) as e_info:
            docket = RateDocket(stimulus_set)

        # Does not contain enough stimuli for each trial.
        stimulus_set = np.array((
            (0, 1, 2, -1, -1, -1, -1, -1, -1),
            (9, 12, 7, -1, -1, -1, -1, -1, -1),
            (3, -1, -1, -1, -1, -1, -1, -1, -1),
            (3, 4, 5, 6, 13, 14, 15, 16, 17)))
        with pytest.raises(Exception) as e_info:
            docket = RateDocket(stimulus_set)

    def test_subset_config_idx(self):
        """Test if config_idx is updated correctly after subset."""
        stimulus_set = np.array((
            (0, 1, 2, -1, -1, -1, -1, -1, -1),
            (9, 12, 7, -1, -1, -1, -1, -1, -1),
            (3, 4, 5, 6, 7, -1, -1, -1, -1),
            (3, 4, 2, 6, 7, -1, -1, -1, -1),
            (3, 4, 5, 6, 13, 14, 15, 16, 17)))

        # Create original trials.
        docket = RateDocket(stimulus_set)
        desired_config_idx = np.array((0, 0, 1, 1, 2))
        np.testing.assert_array_equal(docket.config_idx, desired_config_idx)
        # Grab subset and check that config_idx is updated to start at 0.
        trials_subset = docket.subset(np.array((2, 3, 4)))
        desired_config_idx = np.array((0, 0, 1))
        np.testing.assert_array_equal(
            trials_subset.config_idx, desired_config_idx
        )

    def test_stack_config_idx(self):
        """Test if config_idx is updated correctly after stack."""
        # Create first set of original trials.
        stimulus_set = np.array((
            (0, 1, 2, 3, -1, -1, -1, -1, -1),
            (9, 12, 7, 1, -1, -1, -1, -1, -1),
            (3, 4, 5, 6, 7, -1, -1, -1, -1),
            (3, 4, 2, 6, 7, -1, -1, -1, -1),
            (3, 4, 5, 6, 13, 14, 15, 16, 17)
        ))
        trials_0 = RateDocket(stimulus_set)
        desired_config_idx = np.array((0, 0, 1, 1, 2))
        np.testing.assert_array_equal(trials_0.config_idx, desired_config_idx)

        # Create second set of original trials, with non-overlapping
        # configuration.
        stimulus_set = np.array((
            (0, 1, -1, -1, -1, -1, -1, -1, -1),
            (9, 12, -1, -1, -1, -1, -1, -1, -1),
            (3, 4, -1, -1, -1, -1, -1, -1, -1),
            (5, 6, -1, -1, -1, -1, -1, -1, -1),
            (7, 8, -1, -1, -1, -1, -1, -1, -1),
        ))
        trials_1 = RateDocket(stimulus_set)
        desired_config_idx = np.array((0, 0, 0, 0, 0))
        np.testing.assert_array_equal(trials_1.config_idx, desired_config_idx)

        # Stack trials
        trials_stack = trials.stack((trials_0, trials_1))
        desired_config_idx = np.array((0, 0, 1, 1, 2, 3, 3, 3, 3, 3))
        np.testing.assert_array_equal(
            trials_stack.config_idx, desired_config_idx
        )

    def test_n_trial_0(self, setup_docket_0):
        assert setup_docket_0['n_trial'] == setup_docket_0['docket'].n_trial

    def test_stimulus_set_0(self, setup_docket_0):
        np.testing.assert_array_equal(
            setup_docket_0['stimulus_set'],
            setup_docket_0['docket'].stimulus_set
        )

    def test_n_present_0(self, setup_docket_0):
        np.testing.assert_array_equal(
            setup_docket_0['n_present'],
            setup_docket_0['docket'].n_present
        )

    def test_configurations_0(self, setup_docket_0):
        pd.testing.assert_frame_equal(
            setup_docket_0['configurations'],
            setup_docket_0['docket'].config_list
        )

    def test_configuration_id_0(self, setup_docket_0):
        np.testing.assert_array_equal(
            setup_docket_0['configuration_id'],
            setup_docket_0['docket'].config_idx
        )

    def test_n_trial_1(self, setup_docket_1):
        assert setup_docket_1['n_trial'] == setup_docket_1['docket'].n_trial

    def test_stimulus_set_1(self, setup_docket_1):
        np.testing.assert_array_equal(
            setup_docket_1['stimulus_set'],
            setup_docket_1['docket'].stimulus_set
        )

    def test_n_present_1(self, setup_docket_1):
        np.testing.assert_array_equal(
            setup_docket_1['n_present'],
            setup_docket_1['docket'].n_present
        )

    def test_configurations_1(self, setup_docket_1):
        pd.testing.assert_frame_equal(
            setup_docket_1['configurations'],
            setup_docket_1['docket'].config_list
        )

    def test_configuration_id_1(self, setup_docket_1):
        np.testing.assert_array_equal(
            setup_docket_1['configuration_id'],
            setup_docket_1['docket'].config_idx
        )

    def test_persistence(self, setup_docket_0, tmpdir):
        """Test persistence of RankDocket."""
        # Save docket.
        fn = tmpdir.join('docket_test.hdf5')
        setup_docket_0['docket'].save(fn)
        # Load the saved docket.
        loaded_docket = trials.load_trials(fn)
        # Check that the loaded RankDocket object is correct.
        assert setup_docket_0['n_trial'] == loaded_docket.n_trial
        np.testing.assert_array_equal(
            setup_docket_0['stimulus_set'], loaded_docket.stimulus_set
        )
        np.testing.assert_array_equal(
            setup_docket_0['n_present'], loaded_docket.n_present
        )
        pd.testing.assert_frame_equal(
            setup_docket_0['configurations'], loaded_docket.config_list
        )
        np.testing.assert_array_equal(
            setup_docket_0['configuration_id'], loaded_docket.config_idx
        )


class TestRateObservations:
    """Test class RankObservations."""

    def test_invalid_stimulus_set(self):
        """Test handling of invalid `stimulus_set` argument."""
        # Non-integer input.
        stimulus_set = np.array((
            (0., 1, 2, -1, -1, -1, -1, -1, -1),
            (9, 12, 7, -1, -1, -1, -1, -1, -1),
            (3, 4, 5, 6, 7, -1, -1, -1, -1),
            (3, 4, 5, 6, 13, 14, 15, 16, 17)
        ))
        rating = np.array([.1, .2, .3, .4])
        with pytest.raises(Exception) as e_info:
            obs = RateObservations(stimulus_set, rating=rating)

        # Contains integers below -1.
        stimulus_set = np.array((
            (0, 1, -2, -1, -1, -1, -1, -1, -1),
            (9, 12, 7, -1, -1, -1, -1, -1, -1),
            (3, 4, 5, 6, 7, -1, -1, -1, -1),
            (3, 4, 5, 6, 13, 14, 15, 16, 17)))
        with pytest.raises(Exception) as e_info:
            obs = RateObservations(stimulus_set, rating=rating)

        # Does not contain enough references for each trial.
        stimulus_set = np.array((
            (0, 1, 2, -1, -1, -1, -1, -1, -1),
            (9, 12, 7, -1, -1, -1, -1, -1, -1),
            (3, -1, -1, -1, -1, -1, -1, -1, -1),
            (3, 4, 5, 6, 13, 14, 15, 16, 17)))
        with pytest.raises(Exception) as e_info:
            obs = RateObservations(stimulus_set)

    def test_invalid_groups(self):
        """Test handling of invalid `groups` argument."""
        stimulus_set = np.array((
            (0, 1, 2, -1, -1, -1, -1, -1, -1),
            (9, 12, 7, -1, -1, -1, -1, -1, -1),
            (3, 4, 5, 6, 7, -1, -1, -1, -1),
            (3, 4, 5, 6, 13, 14, 15, 16, 17)
        ))
        rating = np.array([.1, .2, .3, .4])

        # Mismatch in number of trials
        groups = np.array(([0], [0], [1]))
        with pytest.raises(Exception) as e_info:
            obs = RateObservations(stimulus_set, rating=rating, groups=groups)

        # Below support.
        groups = np.array(([0], [-1], [1], [0]))
        with pytest.raises(Exception) as e_info:
            obs = RateObservations(stimulus_set, rating=rating, groups=groups)

    def test_subset_config_idx(self):
        """Test if config_idx is updated correctly after subset."""
        stimulus_set = np.array((
            (0, 1, 2, -1, -1, -1, -1, -1, -1),
            (9, 12, 7, -1, -1, -1, -1, -1, -1),
            (3, 4, 5, 6, 7, -1, -1, -1, -1),
            (3, 4, 2, 6, 7, -1, -1, -1, -1),
            (3, 4, 5, 6, 13, 14, 15, 16, 17)
        ))
        rating = np.array([.1, .2, .3, .4, .5])

        # Create original trials.
        n_select = np.array((1, 1, 1, 1, 2))
        obs = RateObservations(stimulus_set, rating=rating)
        desired_config_idx = np.array((0, 0, 1, 1, 2))
        np.testing.assert_array_equal(obs.config_idx, desired_config_idx)
        # Grab subset and check that config_idx is updated to start at 0.
        trials_subset = obs.subset(np.array((2, 3, 4)))
        desired_config_idx = np.array((0, 0, 1))
        np.testing.assert_array_equal(
            trials_subset.config_idx, desired_config_idx
        )

    def test_stack_config_idx(self):
        """Test if config_idx is updated correctly after stack."""
        stimulus_set = np.array((
            (0, 1, 2, 3, -1, -1, -1, -1, -1),
            (9, 12, 7, 1, -1, -1, -1, -1, -1),
            (3, 4, 5, 6, 7, -1, -1, -1, -1),
            (3, 4, 2, 6, 7, -1, -1, -1, -1),
            (3, 4, 5, 6, 13, 14, 15, 16, 17)))

        # Create first set of original trials.
        rating = np.array([.1, .2, .3, .4, .5])
        trials_0 = RateObservations(stimulus_set, rating=rating)
        desired_config_idx = np.array((0, 0, 1, 1, 2))
        np.testing.assert_array_equal(trials_0.config_idx, desired_config_idx)

        # Create second set of original trials, with non-overlapping
        # configuration.
        stimulus_set = np.array((
            (0, 1, -1, -1, -1, -1, -1, -1, -1),
            (9, 12, -1, -1, -1, -1, -1, -1, -1),
            (3, 4, -1, -1, -1, -1, -1, -1, -1),
            (5, 6, -1, -1, -1, -1, -1, -1, -1),
            (7, 8, -1, -1, -1, -1, -1, -1, -1),
        ))
        rating = np.array([.6, .7, .8, .9, 1.])
        trials_1 = RateObservations(stimulus_set, rating=rating)
        desired_config_idx = np.array((0, 0, 0, 0, 0))
        np.testing.assert_array_equal(trials_1.config_idx, desired_config_idx)

        # Stack trials
        trials_stack = trials.stack((trials_0, trials_1))
        desired_config_idx = np.array((0, 0, 1, 1, 2, 3, 3, 3, 3, 3))
        np.testing.assert_array_equal(
            trials_stack.config_idx, desired_config_idx
        )

    def test_n_trial_0(self, setup_obs_0):
        assert setup_obs_0['n_trial'] == setup_obs_0['obs'].n_trial

    def test_stimulus_set_0(self, setup_obs_0):
        np.testing.assert_array_equal(
            setup_obs_0['stimulus_set'], setup_obs_0['obs'].stimulus_set
        )

    def test_n_present_0(self, setup_obs_0):
        np.testing.assert_array_equal(
            setup_obs_0['n_present'], setup_obs_0['obs'].n_present
        )

    def test_rating_0(self, setup_obs_0):
        np.testing.assert_array_almost_equal(
            setup_obs_0['rating'], setup_obs_0['obs'].rating
        )

    def test_configurations_0(self, setup_obs_0):
        pd.testing.assert_frame_equal(
            setup_obs_0['configurations'],
            setup_obs_0['obs'].config_list
        )

    def test_configuration_id_0(self, setup_obs_0):
        np.testing.assert_array_equal(
            setup_obs_0['configuration_id'],
            setup_obs_0['obs'].config_idx)

    def test_n_trial_1(self, setup_obs_1):
        assert setup_obs_1['n_trial'] == setup_obs_1['obs'].n_trial

    def test_stimulus_set_1(self, setup_obs_1):
        np.testing.assert_array_equal(
            setup_obs_1['stimulus_set'], setup_obs_1['obs'].stimulus_set)

    def test_n_present_1(self, setup_obs_1):
        np.testing.assert_array_equal(
            setup_obs_1['n_present'], setup_obs_1['obs'].n_present
        )

    def test_rating_1(self, setup_obs_1):
        np.testing.assert_array_almost_equal(
            setup_obs_1['rating'], setup_obs_1['obs'].rating
        )

    def test_configurations_1(self, setup_obs_1):
        pd.testing.assert_frame_equal(
            setup_obs_1['configurations'],
            setup_obs_1['obs'].config_list)

    def test_configuration_id_1(self, setup_obs_1):
        np.testing.assert_array_equal(
            setup_obs_1['configuration_id'],
            setup_obs_1['obs'].config_idx
        )

    def test_set_groups(self, setup_obs_1):
        obs = setup_obs_1['obs']
        # Test initial configuration.
        np.testing.assert_array_equal(
            setup_obs_1['groups'], obs.groups)
        # Test setting groups using scalar.
        new_groups_0 = np.array(([3], [3], [3], [3]), dtype=np.int32)
        obs.set_groups(new_groups_0)
        expected_groups_0 = np.array(([3], [3], [3], [3]), dtype=np.int32)
        np.testing.assert_array_equal(expected_groups_0, obs.groups)
        # Test setting groups using correct-sized array.
        new_groups_1 = np.array(([1], [1], [2], [2]), dtype=np.int32)
        obs.set_groups(new_groups_1)
        expected_groups_1 = np.array(([1], [1], [2], [2]), dtype=np.int32)
        np.testing.assert_array_equal(expected_groups_1, obs.groups)
        # Test setting groups using incorrect-sized array.
        new_groups_2 = np.array(([1], [1], [2]), dtype=np.int32)
        with pytest.raises(Exception) as e_info:
            obs.set_groups(new_groups_2)

    def test_save_load_file(self, setup_obs_0, tmpdir):
        """Test saving and loading of RankObservations."""
        # Save observations.
        fn = tmpdir.join('obs_test.hdf5')
        setup_obs_0['obs'].save(fn)
        # Load the saved observations.
        loaded_obs = trials.load_trials(fn)
        # Check that the loaded RankObservations object is correct.
        assert setup_obs_0['n_trial'] == loaded_obs.n_trial
        np.testing.assert_array_equal(
            setup_obs_0['stimulus_set'], loaded_obs.stimulus_set
        )
        np.testing.assert_array_equal(
            setup_obs_0['n_present'], loaded_obs.n_present
        )
        np.testing.assert_array_almost_equal(
            setup_obs_0['rating'], loaded_obs.rating
        )
        np.testing.assert_array_equal(
            setup_obs_0['groups'], loaded_obs.groups
        )
        pd.testing.assert_frame_equal(
            setup_obs_0['configurations'], loaded_obs.config_list
        )
        np.testing.assert_array_equal(
            setup_obs_0['configuration_id'], loaded_obs.config_idx
        )


class TestStack:
    """Test stack static method."""

    def test_stack_same_config(self):
        """Test stack method with same configuration."""
        n_stimuli = 10
        model_truth = ground_truth(n_stimuli)

        n_trial = 50
        n_present = 2
        generator = RandomRate(n_stimuli, n_present=n_present)
        docket = generator.generate(n_trial)

        double_trials = trials.stack((docket, docket))

        assert double_trials.n_trial == 2 * n_trial
        np.testing.assert_array_equal(
            double_trials.n_present[0:n_trial], docket.n_present)
        np.testing.assert_array_equal(
            double_trials.n_present[n_trial:], docket.n_present)

        rating = np.random.uniform(size=n_trial)
        obs_novice = RateObservations(
            docket.stimulus_set, rating=rating, groups=np.zeros([n_trial, 1])
        )
        rating = np.random.uniform(size=n_trial)
        obs_expert = RateObservations(
            docket.stimulus_set, rating=rating, groups=np.ones([n_trial, 1])
        )
        obs_all = trials.stack((obs_novice, obs_expert))

        assert obs_all.n_trial == 2 * n_trial
        np.testing.assert_array_equal(
            obs_all.n_present[0:n_trial], obs_novice.n_present
        )
        np.testing.assert_array_equal(
            obs_all.n_present[n_trial:], obs_expert.n_present
        )

        np.testing.assert_array_equal(
            obs_all.groups[0:n_trial], obs_novice.groups
        )
        np.testing.assert_array_equal(
            obs_all.groups[n_trial:], obs_expert.groups
        )

    def test_stack_different_config(self):
        """Test stack static method with different configurations."""
        n_stimuli = 20
        n_trial = 5

        n_present1 = 2
        generator = RandomRate(n_stimuli, n_present=n_present1)
        trials1 = generator.generate(n_trial)

        n_present2 = 3
        generator = RandomRate(n_stimuli, n_present=n_present2)
        trials2 = generator.generate(n_trial)

        n_present3 = 4
        generator = RandomRate(n_stimuli, n_present=n_present3)
        trials3 = generator.generate(n_trial)

        trials_all = trials.stack((trials1, trials2, trials3))

        desired_n_present = np.hstack((
            n_present1 * np.ones((n_trial), dtype=np.int32),
            n_present2 * np.ones((n_trial), dtype=np.int32),
            n_present3 * np.ones((n_trial), dtype=np.int32),
        ))

        np.testing.assert_array_equal(
            trials_all.n_present, desired_n_present
        )

        # Check padding values of first set (non-padded and then padded
        # values).
        assert np.sum(np.equal(trials_all.stimulus_set[0:5, 0:2], -1)) == 0
        np.testing.assert_array_equal(
            trials_all.stimulus_set[0:5, 2:],
            -1 * np.ones((5, 2), dtype=np.int32)
        )
        # Check padding values of second set (non-padded and then padded
        # values).
        assert np.sum(np.equal(trials_all.stimulus_set[5:10, 0:3], -1)) == 0
        np.testing.assert_array_equal(
            trials_all.stimulus_set[5:10, 3:],
            -1 * np.ones((5, 1), dtype=np.int32)
        )
        # Check padding values of third set (non-padded and then padded
        # values).
        assert np.sum(np.equal(trials_all.stimulus_set[10:15, :], -1)) == 0

        # Check padding when taking subset.
        trials_subset = trials_all.subset(np.arange(10))
        assert trials_subset.stimulus_set.shape[1] == 3
        # Check padding values of first set (non-padded and then padded
        # values).
        assert np.sum(np.equal(trials_subset.stimulus_set[0:5, 0:2], -1)) == 0
        np.testing.assert_array_equal(
            trials_subset.stimulus_set[0:5, 2:],
            -1 * np.ones((5, 1), dtype=np.int32)
        )
        # Check padding values of second set (non-padded and then padded
        # values).
        assert np.sum(
            np.equal(trials_subset.stimulus_set[5:10, 0:3], -1)
        ) == 0
