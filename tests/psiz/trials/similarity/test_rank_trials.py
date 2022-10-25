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
"""Test `trials` module.

Notes:
    It is critical that the function `_possible_rank_outcomes` returns the
        unaltered index first (as the test cases are written). Many
        downstream applications make this assumption.

"""

import pytest
import numpy as np
import pandas as pd
import tensorflow as tf

from psiz import trials
from psiz.trials import RandomRank
from psiz.agents import RankAgent
import psiz.keras.models
import psiz.keras.layers

from psiz.trials.similarity.rank.rank_trials import RankTrials


@pytest.fixture(scope="module")
def setup_docket_0():
    """Simplest n_select."""
    stimulus_set = np.array(
        (
            (1, 2, 3, 0, 0, 0, 0, 0, 0),
            (10, 13, 8, 0, 0, 0, 0, 0, 0),
            (4, 5, 6, 7, 8, 0, 0, 0, 0),
            (4, 5, 6, 7, 14, 15, 16, 17, 18)
        ), dtype=np.int32
    )
    n_trial = 4
    n_select = np.array((1, 1, 1, 1), dtype=np.int32)
    n_reference = np.array((2, 2, 4, 8), dtype=np.int32)
    is_ranked = np.array((True, True, True, True))

    configurations = pd.DataFrame(
        {
            'n_reference': np.array([2, 4, 8], dtype=np.int32),
            'n_select': np.array([1, 1, 1], dtype=np.int32),
            'is_ranked': [True, True, True],
            'n_outcome': np.array([2, 4, 8], dtype=np.int32)
        },
        index=[0, 2, 3])
    configuration_id = np.array((0, 0, 1, 2))

    docket = trials.RankDocket(stimulus_set, mask_zero=True)
    return {
        'n_trial': n_trial, 'stimulus_set': stimulus_set,
        'n_reference': n_reference, 'n_select': n_select,
        'is_ranked': is_ranked, 'docket': docket,
        'configurations': configurations,
        'configuration_id': configuration_id
    }


@pytest.fixture(scope="module")
def setup_docket_1():
    """Varying n_select."""
    stimulus_set = np.array(
        (
            (1, 2, 3, 0, 0, 0, 0, 0, 0),
            (10, 13, 8, 0, 0, 0, 0, 0, 0),
            (4, 5, 6, 7, 8, 0, 0, 0, 0),
            (4, 5, 6, 7, 14, 15, 16, 17, 18)
        ), dtype=np.int32
    )
    n_trial = 4
    n_select = np.array((1, 1, 1, 2), dtype=np.int32)
    n_reference = np.array((2, 2, 4, 8), dtype=np.int32)
    is_ranked = np.array((True, True, True, True))

    configurations = pd.DataFrame(
        {
            'n_reference': np.array([2, 4, 8], dtype=np.int32),
            'n_select': np.array([1, 1, 2], dtype=np.int32),
            'is_ranked': [True, True, True],
            'n_outcome': np.array([2, 4, 56], dtype=np.int32)
        },
        index=[0, 2, 3])
    configuration_id = np.array((0, 0, 1, 2))

    docket = trials.RankDocket(
        stimulus_set, n_select=n_select, mask_zero=True
    )
    return {
        'n_trial': n_trial, 'stimulus_set': stimulus_set,
        'n_reference': n_reference, 'n_select': n_select,
        'is_ranked': is_ranked, 'docket': docket,
        'configurations': configurations,
        'configuration_id': configuration_id
    }


@pytest.fixture(scope="module")
def setup_docket_3():
    """Simplest n_select."""
    stimulus_set = np.array(
        (
            (1, 2, 3, 0),
            (10, 13, 8, 0),
            (4, 5, 6, 7),
            (4, 5, 6, 17)
        ), dtype=np.int32
    )
    n_trial = 4
    n_select = np.array((1, 1, 1, 1), dtype=np.int32)
    n_reference = np.array((2, 2, 3, 3), dtype=np.int32)
    docket = trials.RankDocket(stimulus_set, mask_zero=True)
    return {
        'n_trial': n_trial, 'stimulus_set': stimulus_set,
        'n_reference': n_reference, 'n_select': n_select,
        'docket': docket
    }


@pytest.fixture(scope="module")
def setup_obs_0():
    """Default group information.
    """
    stimulus_set = np.array(
        (
            (1, 2, 3, 0, 0, 0, 0, 0, 0),
            (10, 13, 8, 0, 0, 0, 0, 0, 0),
            (4, 5, 6, 7, 8, 0, 0, 0, 0),
            (4, 5, 6, 7, 14, 15, 16, 17, 18)
        ), dtype=np.int32
    )
    n_trial = 4
    n_select = np.array((1, 1, 1, 2), dtype=np.int32)
    n_reference = np.array((2, 2, 4, 8), dtype=np.int32)
    is_ranked = np.array((True, True, True, True))
    groups = np.zeros([n_trial, 1], dtype=np.int32)
    configurations = pd.DataFrame(
        {
            'n_reference': np.array([2, 4, 8], dtype=np.int32),
            'n_select': np.array([1, 1, 2], dtype=np.int32),
            'is_ranked': [True, True, True],
            'groups_0': np.array([0, 0, 0], dtype=np.int32),
            'n_outcome': np.array([2, 4, 56], dtype=np.int32)
        },
        index=[0, 2, 3])
    configuration_id = np.array((0, 0, 1, 2))

    obs = trials.RankObservations(
        stimulus_set, n_select=n_select, mask_zero=True
    )
    return {
        'n_trial': n_trial, 'stimulus_set': stimulus_set,
        'n_reference': n_reference, 'n_select': n_select,
        'is_ranked': is_ranked, 'groups': groups, 'obs': obs,
        'configurations': configurations,
        'configuration_id': configuration_id
    }


@pytest.fixture(scope="module")
def setup_obs_1():
    """Varying group information.
    """
    stimulus_set = np.array(
        (
            (1, 2, 3, 0, 0, 0, 0, 0, 0),
            (10, 13, 8, 0, 0, 0, 0, 0, 0),
            (4, 5, 6, 7, 8, 0, 0, 0, 0),
            (4, 5, 6, 7, 14, 15, 16, 17, 18)
        ), dtype=np.int32
    )
    n_trial = 4
    n_select = np.array((1, 1, 1, 2), dtype=np.int32)
    n_reference = np.array((2, 2, 4, 8), dtype=np.int32)
    is_ranked = np.array((True, True, True, True))
    groups = np.array(([0], [0], [1], [1]), dtype=np.int32)

    configurations = pd.DataFrame(
        {
            'n_reference': np.array([2, 4, 8], dtype=np.int32),
            'n_select': np.array([1, 1, 2], dtype=np.int32),
            'is_ranked': [True, True, True],
            'groups_0': np.array([0, 1, 1], dtype=np.int32),
            'n_outcome': np.array([2, 4, 56], dtype=np.int32)
        },
        index=[0, 2, 3])
    configuration_id = np.array((0, 0, 1, 2), dtype=np.int32)

    obs = trials.RankObservations(
        stimulus_set, n_select=n_select, mask_zero=True, groups=groups
    )
    return {
        'n_trial': n_trial, 'stimulus_set': stimulus_set,
        'n_reference': n_reference, 'n_select': n_select,
        'is_ranked': is_ranked, 'groups': groups, 'obs': obs,
        'configurations': configurations,
        'configuration_id': configuration_id
    }


@pytest.fixture(scope="module")
def setup_obs_2():
    """Varying group information.
    """
    stimulus_set = np.array(
        (
            (1, 2, 3, 0, 0, 0, 0, 0, 0),
            (10, 13, 8, 0, 0, 0, 0, 0, 0),
            (4, 5, 6, 7, 8, 0, 0, 0, 0),
            (4, 5, 6, 7, 14, 15, 16, 17, 18)
        ), dtype=np.int32
    )
    n_trial = 4
    n_select = np.array((1, 1, 1, 2), dtype=np.int32)
    n_reference = np.array((2, 2, 4, 8), dtype=np.int32)
    groups = np.array(([0], [0], [1], [1]), dtype=np.int32)
    session_id = np.array([0, 1, 0, 0], dtype=np.int32)

    obs = trials.RankObservations(
        stimulus_set, n_select=n_select, mask_zero=True, groups=groups,
        session_id=session_id
    )
    return {
        'n_trial': n_trial, 'stimulus_set': stimulus_set,
        'n_reference': n_reference, 'n_select': n_select,
        'groups': groups, 'obs': obs,
    }


def ground_truth(n_stimuli, mask_zero):
    """Return a ground truth model."""
    n_dim = 3

    if mask_zero:
        stimuli = tf.keras.layers.Embedding(
            n_stimuli + 1, n_dim, mask_zero=True
        )
    else:
        stimuli = tf.keras.layers.Embedding(
            n_stimuli, n_dim, mask_zero=False
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

    kernel_group = psiz.keras.layers.BraidGate(
        subnets=[kernel_0, kernel_1], gating_index=-1  # TODO verify idx
    )

    model = psiz.keras.models.Rank(
        stimuli=stimuli, kernel=kernel_group, use_group_kernel=True
    )
    return model


class TestRankSimilarityTrials:
    """Test functionality of base class SimilarityTrials."""

    def test_invalid_n_select_nomaskzero(self):
        """Test handling of invalid 'n_select' argument."""
        stimulus_set = np.array(
            (
                (1, 2, 3),
                (10, 13, 8),
                (4, 5, 6),
                (16, 17, 18)
            )
        )

        # Mismatch in number of trials
        n_select = np.array((1, 1, 2))
        with pytest.raises(Exception) as e_info:
            trials.RankDocket(stimulus_set, n_select=n_select)
        assert e_info.type == ValueError

        # Below support.
        n_select = np.array((1, 0, 1))
        with pytest.raises(Exception) as e_info:
            trials.RankDocket(stimulus_set, n_select=n_select)
        assert e_info.type == ValueError

        # Above support.
        n_select = np.array((2, 1, 1))
        with pytest.raises(Exception) as e_info:
            trials.RankDocket(stimulus_set, n_select=n_select)
        assert e_info.type == ValueError

    def test_invalid_n_select_maskzero(self):
        """Test handling of invalid 'n_select' argument."""
        stimulus_set = np.array(
            (
                (1, 2, 3, 0, 0, 0, 0, 0, 0),
                (10, 13, 8, 0, 0, 0, 0, 0, 0),
                (4, 5, 6, 7, 8, 0, 0, 0, 0),
                (4, 5, 6, 7, 14, 15, 16, 17, 18)
            )
        )

        # Mismatch in number of trials
        n_select = np.array((1, 1, 2))
        with pytest.raises(Exception) as e_info:
            trials.RankDocket(stimulus_set, n_select=n_select, mask_zero=True)
        assert e_info.type == ValueError

        # Below support.
        n_select = np.array((1, 0, 1, 0))
        with pytest.raises(Exception) as e_info:
            trials.RankDocket(stimulus_set, n_select=n_select, mask_zero=True)
        assert e_info.type == ValueError

        # Above support.
        n_select = np.array((2, 1, 1, 2))
        with pytest.raises(Exception) as e_info:
            trials.RankDocket(stimulus_set, n_select=n_select, mask_zero=True)
        assert e_info.type == ValueError

    def test_invalid_is_ranked(self):
        """Test handling of invalid 'is_ranked' argument."""
        stimulus_set = np.array(
            (
                (1, 2, 3, 0, 0, 0, 0, 0, 0),
                (10, 13, 8, 0, 0, 0, 0, 0, 0),
                (4, 5, 6, 7, 8, 0, 0, 0, 0),
                (4, 5, 6, 7, 14, 15, 16, 17, 18)
            )
        )

        # Mismatch in number of trials
        is_ranked = np.array((True, True, True))
        with pytest.raises(Exception) as e_info:
            trials.RankDocket(
                stimulus_set, is_ranked=is_ranked, mask_zero=True
            )
        assert e_info.type == ValueError

        is_ranked = np.array((True, False, True, False))
        with pytest.raises(Exception) as e_info:
            trials.RankDocket(
                stimulus_set, is_ranked=is_ranked, mask_zero=True
            )
        assert e_info.type == ValueError


class TestRankDocket:
    """Test class RankDocket."""

    def test_invalid_stimulus_set_nomaskzero(self):
        """Test handling of invalid `stimulus_set` argument."""
        # Non-integer input.
        stimulus_set = np.array(
            (
                (1., 2, 3),
                (10, 13, 8),
                (4, 5, 6),
                (16, 17, 18)
            )
        )
        with pytest.raises(Exception) as e_info:
            trials.RankDocket(stimulus_set)
        assert e_info.type == ValueError

        # Contains integers below 0.
        stimulus_set = np.array(
            (
                (1, 2, -1),
                (10, 13, 8),
                (4, 5, 6),
                (16, 17, 18)
            )
        )
        with pytest.warns(Warning):
            trials.RankDocket(stimulus_set)

        # Does not contain enough references for each trial.
        stimulus_set = np.array(
            (
                (1, 2),
                (10, 13),
                (4, 5),
                (17, 18)
            )
        )
        with pytest.raises(Exception) as e_info:
            trials.RankDocket(stimulus_set)
        assert e_info.type == ValueError

    def test_invalid_stimulus_set_maskzero(self):
        """Test handling of invalid `stimulus_set` argument."""
        # Non-integer input.
        stimulus_set = np.array(
            (
                (1., 2, 3, 0, 0, 0, 0, 0, 0),
                (10, 13, 8, 0, 0, 0, 0, 0, 0),
                (4, 5, 6, 7, 8, 0, 0, 0, 0),
                (4, 5, 6, 7, 14, 15, 16, 17, 18)
            )
        )
        with pytest.raises(Exception) as e_info:
            trials.RankDocket(stimulus_set, mask_zero=True)
        assert e_info.type == ValueError

        # Contains integers below 0.
        stimulus_set = np.array(
            (
                (1, 2, -1, 0, 0, 0, 0, 0, 0),
                (10, 13, 8, 0, 0, 0, 0, 0, 0),
                (4, 5, 6, 7, 8, 0, 0, 0, 0),
                (4, 5, 6, 7, 14, 15, 16, 17, 18)
            )
        )
        with pytest.warns(Warning):
            trials.RankDocket(stimulus_set, mask_zero=True)

        # Does not contain enough references for each trial.
        stimulus_set = np.array(
            (
                (1, 2, 3, 0, 0, 0, 0, 0, 0),
                (10, 13, 8, 0, 0, 0, 0, 0, 0),
                (4, 5, 0, 0, 0, 0, 0, 0, 0),
                (4, 5, 6, 7, 14, 15, 16, 17, 18)
            )
        )
        with pytest.raises(Exception) as e_info:
            trials.RankDocket(stimulus_set, mask_zero=True)
        assert e_info.type == ValueError

    def test_subset_config_idx(self):
        """Test if config_idx is updated correctly after subset."""
        stimulus_set = np.array((
            (1, 2, 3, 0, 0, 0, 0, 0, 0),
            (10, 13, 8, 0, 0, 0, 0, 0, 0),
            (4, 5, 6, 7, 8, 0, 0, 0, 0),
            (4, 5, 6, 7, 8, 0, 0, 0, 0),
            (4, 5, 6, 7, 14, 15, 16, 17, 18)
        ))

        # Create original trials.
        n_select = np.array((1, 1, 1, 1, 2))
        docket = trials.RankDocket(
            stimulus_set, n_select=n_select, mask_zero=True
        )
        desired_config_idx = np.array((0, 0, 1, 1, 2))
        np.testing.assert_array_equal(docket.config_idx, desired_config_idx)
        # Grab subset and check that config_idx is updated to start at 0.
        trials_subset = docket.subset(np.array((2, 3, 4)))
        desired_config_idx = np.array((0, 0, 1))
        np.testing.assert_array_equal(
            trials_subset.config_idx, desired_config_idx)

    def test_stack_config_idx(self):
        """Test if config_idx is updated correctly after stack."""
        stimulus_set = np.array((
            (1, 2, 3, 4, 0, 0, 0, 0, 0),
            (10, 13, 8, 2, 0, 0, 0, 0, 0),
            (4, 5, 6, 7, 8, 0, 0, 0, 0),
            (4, 5, 6, 7, 8, 0, 0, 0, 0),
            (4, 5, 6, 7, 14, 15, 16, 17, 18),
        ))

        # Create first set of original trials.
        n_select = np.array((1, 1, 1, 1, 1))
        trials_0 = trials.RankDocket(
            stimulus_set, n_select=n_select, mask_zero=True
        )
        desired_config_idx = np.array((0, 0, 1, 1, 2))
        np.testing.assert_array_equal(trials_0.config_idx, desired_config_idx)

        # Create second set of original trials, with non-overlapping
        # configuration.
        n_select = np.array((2, 2, 2, 2, 2))
        trials_1 = trials.RankDocket(
            stimulus_set, n_select=n_select, mask_zero=True
        )
        desired_config_idx = np.array((0, 0, 1, 1, 2))
        np.testing.assert_array_equal(trials_1.config_idx, desired_config_idx)

        # Stack trials
        trials_stack = trials.stack((trials_0, trials_1))
        desired_config_idx = np.array((0, 0, 1, 1, 2, 3, 3, 4, 4, 5))
        np.testing.assert_array_equal(
            trials_stack.config_idx, desired_config_idx)

    def test_n_trial_0(self, setup_docket_0):
        assert setup_docket_0['n_trial'] == setup_docket_0['docket'].n_trial

    def test_stimulus_set_0(self, setup_docket_0):
        np.testing.assert_array_equal(
            setup_docket_0['stimulus_set'],
            setup_docket_0['docket'].stimulus_set
        )

    def test_n_reference_0(self, setup_docket_0):
        np.testing.assert_array_equal(
            setup_docket_0['n_reference'],
            setup_docket_0['docket'].n_reference
        )

    def test_n_select_0(self, setup_docket_0):
        np.testing.assert_array_equal(
            setup_docket_0['n_select'],
            setup_docket_0['docket'].n_select
        )

    def test_is_ranked_0(self, setup_docket_0):
        np.testing.assert_array_equal(
            setup_docket_0['is_ranked'],
            setup_docket_0['docket'].is_ranked
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

    def test_n_reference_1(self, setup_docket_1):
        np.testing.assert_array_equal(
            setup_docket_1['n_reference'],
            setup_docket_1['docket'].n_reference
        )

    def test_n_select_1(self, setup_docket_1):
        np.testing.assert_array_equal(
            setup_docket_1['n_select'],
            setup_docket_1['docket'].n_select
        )

    def test_is_ranked_1(self, setup_docket_1):
        np.testing.assert_array_equal(
            setup_docket_1['is_ranked'],
            setup_docket_1['docket'].is_ranked
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
            setup_docket_0['stimulus_set'], loaded_docket.stimulus_set)
        np.testing.assert_array_equal(
            setup_docket_0['n_reference'], loaded_docket.n_reference)
        np.testing.assert_array_equal(
            setup_docket_0['n_select'], loaded_docket.n_select)
        np.testing.assert_array_equal(
            setup_docket_0['is_ranked'], loaded_docket.is_ranked)
        pd.testing.assert_frame_equal(
            setup_docket_0['configurations'],
            loaded_docket.config_list)
        np.testing.assert_array_equal(
            setup_docket_0['configuration_id'],
            loaded_docket.config_idx)
        assert loaded_docket.mask_zero

    def test_as_dataset(self, setup_docket_3):
        docket = setup_docket_3['docket']
        groups = np.array([[0], [0], [0], [0]])
        # Zero is a placeholder, each outcome is a column (as displayed
        # below). Since the first trial only has two references, there are
        # only two different outcomes. The first outcome corresponds to the
        # original ordering. Values are absolute stimulus indices.
        stimulus_set_0_desired = tf.constant(
            [
                [1, 1, 0],
                [2, 3, 0],
                [3, 2, 0],
                [0, 0, 0]
            ], dtype=tf.int32
        )
        stimulus_set_3_desired = tf.constant(
            [
                [4, 4, 4],
                [5, 6, 17],
                [6, 5, 5],
                [17, 17, 6]
            ], dtype=tf.int32
        )

        ds_docket = docket.as_dataset(groups)
        docket_list = list(ds_docket)
        # Grab the first trial.
        stimulus_set_0 = docket_list[0]['stimulus_set']
        tf.debugging.assert_equal(
            stimulus_set_0, stimulus_set_0_desired
        )
        # Grab the fourth trial.
        stimulus_set_3 = docket_list[3]['stimulus_set']
        tf.debugging.assert_equal(
            stimulus_set_3, stimulus_set_3_desired
        )

        # Check that behavior is still the same if groups information is not
        # provided.
        ds_docket = docket.as_dataset()
        docket_list = list(ds_docket)
        stimulus_set_0 = docket_list[0]['stimulus_set']
        tf.debugging.assert_equal(
            stimulus_set_0, stimulus_set_0_desired
        )
        stimulus_set_3 = docket_list[3]['stimulus_set']
        tf.debugging.assert_equal(
            stimulus_set_3, stimulus_set_3_desired
        )


class TestRankObservations:
    """Test class RankObservations."""

    def test_invalid_stimulus_set_nomaskzero(self):
        """Test handling of invalid `stimulus_set` argument."""
        # Non-integer input.
        stimulus_set = np.array((
            (1., 2, 3),
            (10, 13, 8),
            (4, 5, 6),
            (16, 17, 18)
        ))
        with pytest.raises(Exception) as e_info:
            trials.RankObservations(stimulus_set)
        assert e_info.type == ValueError

        # Contains integers below 0.
        stimulus_set = np.array((
            (1, 2, -1),
            (10, 13, 8),
            (4, 5, 6),
            (16, 17, 18)
        ))
        with pytest.warns(Warning):
            trials.RankObservations(stimulus_set)

        # Does not contain enough references for each trial.
        stimulus_set = np.array((
            (1, 2),
            (10, 13),
            (4, 5),
            (17, 18)
        ))
        with pytest.raises(Exception) as e_info:
            trials.RankObservations(stimulus_set)
        assert e_info.type == ValueError

    def test_invalid_stimulus_set_maskzero(self):
        """Test handling of invalid `stimulus_set` argument."""
        # Non-integer input.
        stimulus_set = np.array((
            (1., 2, 3, 0, 0, 0, 0, 0, 0),
            (10, 13, 8, 0, 0, 0, 0, 0, 0),
            (4, 5, 6, 7, 8, 0, 0, 0, 0),
            (4, 5, 6, 7, 14, 15, 16, 17, 18)
        ))
        with pytest.raises(Exception) as e_info:
            trials.RankObservations(stimulus_set, mask_zero=True)
        assert e_info.type == ValueError

        # Contains integers below 0.
        stimulus_set = np.array((
            (1, 2, -1, 0, 0, 0, 0, 0, 0),
            (10, 13, 8, 0, 0, 0, 0, 0, 0),
            (4, 5, 6, 7, 8, 0, 0, 0, 0),
            (4, 5, 6, 7, 14, 15, 16, 17, 18)
        ))
        # with pytest.raises(Exception) as e_info:
        #     trials.RankObservations(stimulus_set, mask_zero=True)
        # assert e_info.type == ValueError
        with pytest.warns(Warning):
            trials.RankObservations(stimulus_set, mask_zero=True)

        # Does not contain enough references for each trial.
        stimulus_set = np.array((
            (1, 2, 3, 0, 0, 0, 0, 0, 0),
            (10, 13, 8, 0, 0, 0, 0, 0, 0),
            (4, 5, 0, 0, 0, 0, 0, 0, 0),
            (4, 5, 6, 7, 14, 15, 16, 17, 18)
        ))
        with pytest.raises(Exception) as e_info:
            trials.RankObservations(stimulus_set, mask_zero=True)
        assert e_info.type == ValueError

    def test_invalid_groups(self):
        """Test handling of invalid `groups` argument."""
        stimulus_set = np.array((
            (1, 2, 3, 0, 0, 0, 0, 0, 0),
            (10, 13, 8, 0, 0, 0, 0, 0, 0),
            (4, 5, 6, 7, 8, 0, 0, 0, 0),
            (4, 5, 6, 7, 14, 15, 16, 17, 18)
        ))

        # Mismatch in number of trials
        groups = np.array(([0], [0], [1]))
        with pytest.raises(Exception) as e_info:
            trials.RankObservations(
                stimulus_set, mask_zero=True, groups=groups
            )
        assert e_info.type == ValueError

        # Below support.
        groups = np.array(([0], [-1], [1], [0]))
        with pytest.raises(Exception) as e_info:
            trials.RankObservations(
                stimulus_set, mask_zero=True, groups=groups
            )
        assert e_info.type == ValueError

    def test_invalid_agent_id(self):
        """Test invalid agent_id."""
        stimulus_set = np.array((
            (1, 2, 3, 0, 0, 0, 0, 0, 0),
            (10, 13, 8, 0, 0, 0, 0, 0, 0),
            (4, 5, 6, 7, 8, 0, 0, 0, 0),
            (4, 5, 6, 7, 14, 15, 16, 17, 18)
        ))

        # Mismatch in number of trials.
        agent_id = np.array([0, 0, 1], dtype=np.int32)
        with pytest.raises(Exception) as e_info:
            trials.RankObservations(
                stimulus_set, mask_zero=True, agent_id=agent_id
            )
        assert e_info.type == ValueError
        assert str(e_info.value) == (
            "The argument 'agent_id' must have the same length as the "
            "number of rows in the argument 'stimulus_set'."
        )

        # Value is below support.
        agent_id = np.array([0, 0, -1, 0], dtype=np.int32)
        with pytest.raises(Exception) as e_info:
            trials.RankObservations(
                stimulus_set, mask_zero=True, agent_id=agent_id
            )
        assert e_info.type == ValueError
        assert str(e_info.value) == (
            "The parameter 'agent_id' contains integers less than 0. "
            "Found 1 bad trial(s)."
        )

    def test_invalid_session_id(self):
        """Test invalid session_id."""
        stimulus_set = np.array((
            (1, 2, 3, 0, 0, 0, 0, 0, 0),
            (10, 13, 8, 0, 0, 0, 0, 0, 0),
            (4, 5, 6, 7, 8, 0, 0, 0, 0),
            (4, 5, 6, 7, 14, 15, 16, 17, 18)
        ))

        # Mismatch in number of trials.
        session_id = np.array([0, 0, 1], dtype=np.int32)
        with pytest.raises(Exception) as e_info:
            trials.RankObservations(
                stimulus_set, mask_zero=True, session_id=session_id
            )
        assert e_info.type == ValueError
        assert str(e_info.value) == (
            "The argument 'session_id' must have the same length as the "
            "number of rows in the argument 'stimulus_set'."
        )

        # Value is below support.
        session_id = np.array([0, 0, -1, 0], dtype=np.int32)
        with pytest.raises(Exception) as e_info:
            trials.RankObservations(
                stimulus_set, mask_zero=True, session_id=session_id
            )
        assert e_info.type == ValueError
        assert str(e_info.value) == (
            "The parameter 'session_id' contains integers less than 0. "
            "Found 1 bad trial(s)."
        )

    def test_invalid_weight(self):
        """Test invalid weight."""
        stimulus_set = np.array((
            (1, 2, 3, 0, 0, 0, 0, 0, 0),
            (10, 13, 8, 0, 0, 0, 0, 0, 0),
            (4, 5, 6, 7, 8, 0, 0, 0, 0),
            (4, 5, 6, 7, 14, 15, 16, 17, 18)
        ))

        # Mismatch in number of trials.
        weight = np.array([0.5, 0.5, 1], dtype=np.int32)
        with pytest.raises(Exception) as e_info:
            trials.RankObservations(
                stimulus_set, mask_zero=True, weight=weight
            )
        assert e_info.type == ValueError
        assert str(e_info.value) == (
            "The argument 'weight' must have the same length as the "
            "number of rows in the argument 'stimulus_set'."
        )

    def test_invalid_rt(self):
        """Test invalid rt."""
        stimulus_set = np.array((
            (1, 2, 3, 0, 0, 0, 0, 0, 0),
            (10, 13, 8, 0, 0, 0, 0, 0, 0),
            (4, 5, 6, 7, 8, 0, 0, 0, 0),
            (4, 5, 6, 7, 14, 15, 16, 17, 18)
        ))

        # Mismatch in number of trials.
        rt_ms = np.array([123, 99, 144], dtype=np.int32)
        with pytest.raises(Exception) as e_info:
            trials.RankObservations(stimulus_set, mask_zero=True, rt_ms=rt_ms)
        assert e_info.type == ValueError
        assert str(e_info.value) == (
            "The argument 'rt_ms' must have the same length as the "
            "number of rows in the argument 'stimulus_set'."
        )

    def test_subset_config_idx(self):
        """Test if config_idx is updated correctly after subset."""
        stimulus_set = np.array((
            (1, 2, 3, 0, 0, 0, 0, 0, 0),
            (10, 13, 8, 0, 0, 0, 0, 0, 0),
            (4, 5, 6, 7, 8, 0, 0, 0, 0),
            (4, 5, 6, 7, 8, 0, 0, 0, 0),
            (4, 5, 6, 7, 14, 15, 16, 17, 18)
        ))

        # Create original trials.
        n_select = np.array((1, 1, 1, 1, 2))
        obs = trials.RankObservations(
            stimulus_set, n_select=n_select, mask_zero=True
        )
        desired_config_idx = np.array((0, 0, 1, 1, 2))
        np.testing.assert_array_equal(obs.config_idx, desired_config_idx)
        # Grab subset and check that config_idx is updated to start at 0.
        trials_subset = obs.subset(np.array((2, 3, 4)))
        desired_config_idx = np.array((0, 0, 1))
        np.testing.assert_array_equal(
            trials_subset.config_idx, desired_config_idx
        )
        assert trials_subset.mask_zero

    def test_stack_config_idx(self):
        """Test if config_idx is updated correctly after stack."""
        stimulus_set = np.array((
            (1, 2, 3, 4, 0, 0, 0, 0, 0),
            (10, 13, 8, 2, 0, 0, 0, 0, 0),
            (4, 5, 6, 7, 8, 0, 0, 0, 0),
            (4, 5, 6, 7, 8, 0, 0, 0, 0),
            (4, 5, 6, 7, 14, 15, 16, 17, 18)
        ))

        # Create first set of original trials.
        n_select = np.array((1, 1, 1, 1, 1))
        trials_0 = trials.RankObservations(
            stimulus_set, n_select=n_select, mask_zero=True
        )
        desired_config_idx = np.array((0, 0, 1, 1, 2))
        np.testing.assert_array_equal(trials_0.config_idx, desired_config_idx)

        # Create second set of original trials, with non-overlapping
        # configuration.
        n_select = np.array((2, 2, 2, 2, 2))
        trials_1 = trials.RankObservations(
            stimulus_set, n_select=n_select, mask_zero=True
        )
        desired_config_idx = np.array((0, 0, 1, 1, 2))
        np.testing.assert_array_equal(trials_1.config_idx, desired_config_idx)

        # Stack trials
        trials_stack = trials.stack((trials_0, trials_1))
        desired_config_idx = np.array((0, 0, 1, 1, 2, 3, 3, 4, 4, 5))
        np.testing.assert_array_equal(
            trials_stack.config_idx, desired_config_idx
        )
        assert trials_stack.mask_zero

    def test_n_trial_0(self, setup_obs_0):
        assert setup_obs_0['n_trial'] == setup_obs_0['obs'].n_trial

    def test_stimulus_set_0(self, setup_obs_0):
        np.testing.assert_array_equal(
            setup_obs_0['stimulus_set'], setup_obs_0['obs'].stimulus_set
        )

    def test_n_reference_0(self, setup_obs_0):
        np.testing.assert_array_equal(
            setup_obs_0['n_reference'], setup_obs_0['obs'].n_reference
        )

    def test_n_select_0(self, setup_obs_0):
        np.testing.assert_array_equal(
            setup_obs_0['n_select'], setup_obs_0['obs'].n_select
        )

    def test_is_ranked_0(self, setup_obs_0):
        np.testing.assert_array_equal(
            setup_obs_0['is_ranked'], setup_obs_0['obs'].is_ranked
        )

    def test_configurations_0(self, setup_obs_0):
        pd.testing.assert_frame_equal(
            setup_obs_0['configurations'],
            setup_obs_0['obs'].config_list
        )

    def test_configuration_id_0(self, setup_obs_0):
        np.testing.assert_array_equal(
            setup_obs_0['configuration_id'],
            setup_obs_0['obs'].config_idx
        )

    def test_n_trial_1(self, setup_obs_1):
        assert setup_obs_1['n_trial'] == setup_obs_1['obs'].n_trial

    def test_stimulus_set_1(self, setup_obs_1):
        np.testing.assert_array_equal(
            setup_obs_1['stimulus_set'], setup_obs_1['obs'].stimulus_set
        )

    def test_n_reference_1(self, setup_obs_1):
        np.testing.assert_array_equal(
            setup_obs_1['n_reference'], setup_obs_1['obs'].n_reference
        )

    def test_n_select_1(self, setup_obs_1):
        np.testing.assert_array_equal(
            setup_obs_1['n_select'], setup_obs_1['obs'].n_select
        )

    def test_is_ranked_1(self, setup_obs_1):
        np.testing.assert_array_equal(
            setup_obs_1['is_ranked'], setup_obs_1['obs'].is_ranked
        )

    def test_configurations_1(self, setup_obs_1):
        pd.testing.assert_frame_equal(
            setup_obs_1['configurations'],
            setup_obs_1['obs'].config_list
        )

    def test_configuration_id_1(self, setup_obs_1):
        np.testing.assert_array_equal(
            setup_obs_1['configuration_id'],
            setup_obs_1['obs'].config_idx
        )

    def test_set_groups(self, setup_obs_1):
        obs = setup_obs_1['obs']
        # Test initial configuration.
        np.testing.assert_array_equal(
            setup_obs_1['groups'], obs.groups
        )
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
        assert e_info.type == ValueError

    def test_init_2(self, setup_obs_2):
        """Test initialization."""
        obs = setup_obs_2['obs']
        session_id_desired = np.array([0, 1, 0, 0], dtype=np.int32)
        np.testing.assert_equal(obs.session_id, session_id_desired)

    def test_set_weight(self, setup_obs_2):
        """Test set_weight."""
        obs = setup_obs_2['obs']

        obs.set_weight(.9)
        weight_desired = np.array([.9, .9, .9, .9])
        np.testing.assert_array_equal(
            obs.weight, weight_desired
        )

        weight_desired = np.array([.7, .8, .9, 1.])
        obs.set_weight(weight_desired)
        np.testing.assert_array_equal(
            obs.weight, weight_desired
        )

    def test_as_dataset(self, setup_obs_2):
        obs = setup_obs_2['obs']

        ds_obs = obs.as_dataset()
        obs_list = list(ds_obs)
        groups_0 = obs_list[0][0]['groups']

        groups_0_desired = tf.constant(
            [0], dtype=tf.int32
        )
        tf.debugging.assert_equal(
            groups_0, groups_0_desired
        )

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
            setup_obs_0['n_reference'], loaded_obs.n_reference
        )
        np.testing.assert_array_equal(
            setup_obs_0['n_select'], loaded_obs.n_select
        )
        np.testing.assert_array_equal(
            setup_obs_0['is_ranked'], loaded_obs.is_ranked
        )
        np.testing.assert_array_equal(
            setup_obs_0['groups'], loaded_obs.groups
        )
        pd.testing.assert_frame_equal(
            setup_obs_0['configurations'],
            loaded_obs.config_list
        )
        np.testing.assert_array_equal(
            setup_obs_0['configuration_id'],
            loaded_obs.config_idx
        )
        assert loaded_obs.mask_zero


class TestStack:
    """Test stack static method."""

    def test_stack_same_config_nozeromask(self):
        """Test stack method with same configuration.

        Using `zero_mask = False`.

        """
        n_stimuli = 10
        mask_zero = False
        model_truth = ground_truth(n_stimuli, mask_zero)

        n_trial = 50
        n_reference = 8
        n_select = 2
        gen = RandomRank(
            n_stimuli, n_reference=n_reference, n_select=n_select
        )
        docket = gen.generate(n_trial)

        double_trials = trials.stack((docket, docket))

        assert double_trials.n_trial == 2 * n_trial
        np.testing.assert_array_equal(
            double_trials.n_reference[0:n_trial], docket.n_reference)
        np.testing.assert_array_equal(
            double_trials.n_reference[n_trial:], docket.n_reference)

        np.testing.assert_array_equal(
            double_trials.n_select[0:n_trial], docket.n_select
        )
        np.testing.assert_array_equal(
            double_trials.n_select[n_trial:], docket.n_select
        )

        np.testing.assert_array_equal(
            double_trials.is_ranked[0:n_trial], docket.is_ranked
        )
        np.testing.assert_array_equal(
            double_trials.is_ranked[n_trial:], docket.is_ranked
        )
        assert not double_trials.mask_zero

        agent_novice = RankAgent(model_truth, groups=[0])
        agent_expert = RankAgent(model_truth, groups=[1])
        obs_novice = agent_novice.simulate(docket)
        obs_expert = agent_expert.simulate(docket)
        obs_all = trials.stack((obs_novice, obs_expert))

        assert obs_all.n_trial == 2 * n_trial
        np.testing.assert_array_equal(
            obs_all.n_reference[0:n_trial], obs_novice.n_reference
        )
        np.testing.assert_array_equal(
            obs_all.n_reference[n_trial:], obs_expert.n_reference
        )

        np.testing.assert_array_equal(
            obs_all.n_select[0:n_trial], obs_novice.n_select
        )
        np.testing.assert_array_equal(
            obs_all.n_select[n_trial:], obs_expert.n_select
        )

        np.testing.assert_array_equal(
            obs_all.is_ranked[0:n_trial], obs_novice.is_ranked
        )
        np.testing.assert_array_equal(
            obs_all.is_ranked[n_trial:], obs_expert.is_ranked
        )

        np.testing.assert_array_equal(
            obs_all.groups[0:n_trial], obs_novice.groups
        )
        np.testing.assert_array_equal(
            obs_all.groups[n_trial:], obs_expert.groups
        )

    def test_stack_same_config_zeromask(self):
        """Test stack method with same configuration.

        Using `zero_mask = True`.

        """
        n_stimuli = 10
        mask_zero = True
        model_truth = ground_truth(n_stimuli, mask_zero)

        n_trial = 50
        n_reference = 8
        n_select = 2
        gen = RandomRank(
            np.arange(1, n_stimuli + 1), n_reference=n_reference,
            n_select=n_select, mask_zero=True
        )
        docket = gen.generate(n_trial)

        double_trials = trials.stack((docket, docket))

        assert double_trials.n_trial == 2 * n_trial
        np.testing.assert_array_equal(
            double_trials.n_reference[0:n_trial], docket.n_reference)
        np.testing.assert_array_equal(
            double_trials.n_reference[n_trial:], docket.n_reference)

        np.testing.assert_array_equal(
            double_trials.n_select[0:n_trial], docket.n_select
        )
        np.testing.assert_array_equal(
            double_trials.n_select[n_trial:], docket.n_select
        )

        np.testing.assert_array_equal(
            double_trials.is_ranked[0:n_trial], docket.is_ranked
        )
        np.testing.assert_array_equal(
            double_trials.is_ranked[n_trial:], docket.is_ranked
        )
        assert double_trials.mask_zero

        agent_novice = RankAgent(model_truth, groups=[0])
        agent_expert = RankAgent(model_truth, groups=[1])
        obs_novice = agent_novice.simulate(docket)
        obs_expert = agent_expert.simulate(docket)
        obs_all = trials.stack((obs_novice, obs_expert))

        assert obs_all.n_trial == 2 * n_trial
        np.testing.assert_array_equal(
            obs_all.n_reference[0:n_trial], obs_novice.n_reference
        )
        np.testing.assert_array_equal(
            obs_all.n_reference[n_trial:], obs_expert.n_reference
        )

        np.testing.assert_array_equal(
            obs_all.n_select[0:n_trial], obs_novice.n_select
        )
        np.testing.assert_array_equal(
            obs_all.n_select[n_trial:], obs_expert.n_select
        )

        np.testing.assert_array_equal(
            obs_all.is_ranked[0:n_trial], obs_novice.is_ranked
        )
        np.testing.assert_array_equal(
            obs_all.is_ranked[n_trial:], obs_expert.is_ranked
        )

        np.testing.assert_array_equal(
            obs_all.groups[0:n_trial], obs_novice.groups
        )
        np.testing.assert_array_equal(
            obs_all.groups[n_trial:], obs_expert.groups
        )
        assert obs_all.mask_zero

    def test_stack_different_config(self):
        """Test stack static method with different configurations."""
        n_stimuli = 20
        n_trial = 5

        n_reference1 = 2
        n_select1 = 1
        gen = RandomRank(
            np.arange(1, n_stimuli + 1), n_reference=n_reference1,
            n_select=n_select1, mask_zero=True
        )
        trials1 = gen.generate(n_trial)

        n_reference2 = 4
        n_select2 = 2
        gen = RandomRank(
            np.arange(1, n_stimuli + 1), n_reference=n_reference2,
            n_select=n_select2, mask_zero=True
        )
        trials2 = gen.generate(n_trial)

        n_reference3 = 6
        n_select3 = 2
        gen = RandomRank(
            np.arange(1, n_stimuli + 1), n_reference=n_reference3,
            n_select=n_select3, mask_zero=True
        )
        trials3 = gen.generate(n_trial)

        trials_all = trials.stack((trials1, trials2, trials3))

        desired_n_reference = np.hstack((
            n_reference1 * np.ones((n_trial), dtype=np.int32),
            n_reference2 * np.ones((n_trial), dtype=np.int32),
            n_reference3 * np.ones((n_trial), dtype=np.int32),
        ))

        np.testing.assert_array_equal(
            trials_all.n_reference, desired_n_reference
        )

    def test_stack_different_config_error(self):
        """Test stack static method with different configurations."""
        n_stimuli = 20
        n_trial = 5

        n_reference1 = 2
        n_select1 = 1
        gen = RandomRank(
            n_stimuli, n_reference=n_reference1, n_select=n_select1
        )
        trials1 = gen.generate(n_trial)

        n_reference2 = 4
        n_select2 = 2
        gen = RandomRank(
            n_stimuli, n_reference=n_reference2, n_select=n_select2
        )
        trials2 = gen.generate(n_trial)

        n_reference3 = 6
        n_select3 = 2
        gen = RandomRank(
            n_stimuli, n_reference=n_reference3, n_select=n_select3
        )
        trials3 = gen.generate(n_trial)

        with pytest.raises(Exception) as e_info:
            trials.stack((trials1, trials2, trials3))
        assert e_info.type == ValueError

    def test_padding(self):
        """Test padding values when using stack and subset method."""
        n_stimuli = 20
        n_trial = 5
        mask_value = 0

        n_reference1 = 2
        n_select1 = 1
        gen = RandomRank(
            np.arange(1, n_stimuli + 1), n_reference=n_reference1,
            n_select=n_select1, mask_zero=True
        )
        trials1 = gen.generate(n_trial)

        n_reference2 = 4
        n_select2 = 2
        gen = RandomRank(
            np.arange(1, n_stimuli + 1), n_reference=n_reference2,
            n_select=n_select2, mask_zero=True
        )
        trials2 = gen.generate(n_trial)

        n_reference3 = 8
        n_select3 = 2
        gen = RandomRank(
            np.arange(1, n_stimuli + 1), n_reference=n_reference3,
            n_select=n_select3, mask_zero=True
        )
        trials3 = gen.generate(n_trial)

        trials_all = trials.stack((trials1, trials2, trials3))

        # Check padding values of first set (non-padded and then padded
        # values).
        assert np.sum(
            np.equal(trials_all.stimulus_set[0:5, 0:3], mask_value)
        ) == 0
        np.testing.assert_array_equal(
            trials_all.stimulus_set[0:5, 3:],
            mask_value * np.ones((5, 6), dtype=np.int32)
        )
        # Check padding values of second set (non-padded and then padded
        # values).
        assert np.sum(
            np.equal(trials_all.stimulus_set[5:10, 0:5], mask_value)
        ) == 0
        np.testing.assert_array_equal(
            trials_all.stimulus_set[5:10, 5:],
            mask_value * np.ones((5, 4), dtype=np.int32)
        )
        # Check padding values of third set (non-padded and then padded
        # values).
        assert np.sum(
            np.equal(trials_all.stimulus_set[10:15, :], mask_value)
        ) == 0
        assert trials_all.mask_zero

        # Check padding when taking subset.
        trials_subset = trials_all.subset(np.arange(10))
        assert trials_subset.stimulus_set.shape[1] == 5
        # Check padding values of first set (non-padded and then padded
        # values).
        assert np.sum(
            np.equal(trials_subset.stimulus_set[1:5, 0:3], mask_value)
        ) == 0
        np.testing.assert_array_equal(
            trials_subset.stimulus_set[0:5, 3:],
            mask_value * np.ones((5, 2), dtype=np.int32)
        )
        # Check padding values of second set (non-padded and then padded
        # values).
        assert np.sum(
            np.equal(trials_subset.stimulus_set[5:10, 0:5], mask_value)
        ) == 0
        assert trials_subset.mask_zero


class TestPossibleOutcomes:
    """Test possible outcomes."""

    def test_possible_outcomes_2c1(self):
        """Test outcomes 2 choose 1 ranked trial."""
        stimulus_set = np.array(((1, 2, 3), (10, 13, 8)))
        n_select = 1 * np.ones((2))
        tasks = trials.RankDocket(stimulus_set, n_select=n_select)

        po = RankTrials._possible_rank_outcomes(tasks.config_list.iloc[0])

        correct = np.array(((0, 1), (1, 0)))
        np.testing.assert_array_equal(po, correct)

    def test_possible_outcomes_3c2(self):
        """Test outcomes 3 choose 2 ranked trial."""
        stimulus_set = np.array(((1, 2, 3, 4), (34, 10, 13, 8)))
        n_select = 2 * np.ones((2))
        tasks = trials.RankDocket(stimulus_set, n_select=n_select)

        po = RankTrials._possible_rank_outcomes(tasks.config_list.iloc[0])

        correct = np.array((
            (0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0),
            (2, 0, 1), (2, 1, 0)))
        np.testing.assert_array_equal(po, correct)

    def test_possible_outcomes_4c2(self):
        """Test outcomes 4 choose 2 ranked trial."""
        stimulus_set = np.array(((1, 2, 3, 4, 5), (46, 34, 10, 13, 8)))
        n_select = 2 * np.ones((2))
        tasks = trials.RankDocket(stimulus_set, n_select=n_select)

        po = RankTrials._possible_rank_outcomes(tasks.config_list.iloc[0])

        correct = np.array((
            (0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2),
            (1, 0, 2, 3), (1, 2, 0, 3), (1, 3, 0, 2),
            (2, 0, 1, 3), (2, 1, 0, 3), (2, 3, 0, 1),
            (3, 0, 1, 2), (3, 1, 0, 2), (3, 2, 0, 1)))
        np.testing.assert_array_equal(po, correct)

    def test_possible_outcomes_8c1(self):
        """Test outcomes 8 choose 1 ranked trial."""
        stimulus_set = np.array((
            (1, 2, 3, 4, 5, 6, 7, 8, 9),
            (46, 34, 10, 13, 8, 3, 6, 5, 4)
        ))
        n_select = 1 * np.ones((2))
        tasks = trials.RankDocket(stimulus_set, n_select=n_select)

        po = RankTrials._possible_rank_outcomes(tasks.config_list.iloc[0])

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
