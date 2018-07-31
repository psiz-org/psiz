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
    - more information gain tests
"""

import numpy as np
import pytest

from psiz import generator
from psiz.models import Exponential
from psiz.trials import stack, UnjudgedTrials
from psiz.simulate import Agent


@pytest.fixture(scope="module")
def ground_truth():
    """Return a ground truth embedding."""
    n_stimuli = 10
    n_dim = 2

    model = Exponential(n_stimuli)
    mean = np.zeros((n_dim))
    cov = .1 * np.identity(n_dim)
    z = np.random.multivariate_normal(mean, cov, (n_stimuli))
    freeze_options = {
        'z': z,
        'theta': {
            'rho': 2,
            'tau': 1,
            'beta': 10,
            'gamma': 0
        }
    }
    model.freeze(freeze_options)
    return model


def simulated_samples(z):
    """Simulate posterior samples for a set of embedding points."""
    n_stimuli = z.shape[0]
    n_dim = z.shape[1]
    n_sample = 1000

    stim_cov = np.array((.08, .01, .01, .01, .01, .01, .01, .01, .01, .01))
    # Draw samples
    z_samples = np.empty((n_sample, n_stimuli, n_dim))
    for i_stimulus in range(n_stimuli):
        z_samples[:, i_stimulus, :] = np.random.multivariate_normal(
            z[i_stimulus], stim_cov[i_stimulus] * np.identity(n_dim),
            (n_sample)
        )
    z_samples = np.transpose(z_samples, axes=[1, 2, 0])
    return {'z': z_samples}


def test_random_generator():
    """Test random generator."""
    n_stimuli_desired = 10
    n_trial_desired = 50
    n_reference_desired = 4
    n_selected_desired = 2
    is_ranked_desired = True
    gen = generator.RandomGenerator(n_stimuli_desired)
    trials = gen.generate(
        n_trial=n_trial_desired, n_reference=n_reference_desired,
        n_selected=n_selected_desired)

    assert trials.n_trial == n_trial_desired
    assert sum(trials.n_reference == n_reference_desired) == n_trial_desired
    assert trials.stimulus_set.shape[0] == n_trial_desired
    assert trials.stimulus_set.shape[1] == n_reference_desired + 1
    min_actual = np.min(trials.stimulus_set)
    max_actual = np.max(trials.stimulus_set)
    assert min_actual >= -1  # Need -1 for padding.
    assert max_actual < n_stimuli_desired
    n_unique_desired = n_reference_desired + 1
    for i_trial in range(n_trial_desired):
        # NOTE: The padding padding values (-1)'s are counted as unique, thus
        # the indexing into stimulus set.
        assert (
            len(np.unique(
                trials.stimulus_set[i_trial, 0:n_reference_desired+1])
                ) == n_unique_desired
        )
    assert sum(trials.n_selected == n_selected_desired) == n_trial_desired
    assert sum(trials.is_ranked == is_ranked_desired) == n_trial_desired


def test_information_gain(ground_truth):
    """Test expected information gain computation."""
    z = ground_truth.z['value']
    samples = simulated_samples(z)

    n_stimuli = 10
    gen = generator.ActiveGenerator(n_stimuli)
    candidate_trial_0 = UnjudgedTrials(
        np.array([[0, 1, 2]], dtype=np.int32),
        np.array([1], dtype=np.int32)
        )
    candidate_trial_1 = UnjudgedTrials(
        np.array([[3, 1, 2]], dtype=np.int32),
        np.array([1], dtype=np.int32)
        )
    # Compute expected informatin gain.
    ig_0 = gen._information_gain(ground_truth, samples, candidate_trial_0)
    ig_1 = gen._information_gain(ground_truth, samples, candidate_trial_1)

    assert ig_0 > ig_1

    # Check case for multiple candidate trials.
    candidate_trial_01 = UnjudgedTrials(
        np.array([[0, 1, 2], [3, 1, 2]], dtype=np.int32),
        np.array([1, 1], dtype=np.int32)
        )
    ig_01 = gen._information_gain(ground_truth, samples, candidate_trial_01)
    assert ig_01[0] == ig_0
    assert ig_01[1] == ig_1

    # Check case for multiple candidate trials with different configurations.
    candidate_trial_23 = UnjudgedTrials(
        np.array([[0, 1, 2, 4, 9], [3, 1, 5, 6, 8]], dtype=np.int32),
        np.array([2, 2], dtype=np.int32)
        )
    ig_23 = gen._information_gain(ground_truth, samples, candidate_trial_23)

    candidate_trial_0123 = UnjudgedTrials(
        np.array([
            [0, 1, 2, -1, -1],
            [3, 1, 2, -1, -1],
            [0, 1, 2, 4, 9],
            [3, 1, 5, 6, 8]
        ], dtype=np.int32),
        np.array([1, 1, 2, 2], dtype=np.int32)
        )
    ig_0123 = gen._information_gain(ground_truth, samples, candidate_trial_0123)
    assert ig_0123[0] == ig_0
    assert ig_0123[1] == ig_1
    assert ig_0123[2] == ig_23[0]
    assert ig_0123[3] == ig_23[1]

# @pytest.fixture(scope="module")
# def unbalanced_trials():
#     """Return a set of unbalanced trials.

#     In this context, unbalanced means trials in which a stimulus
#     occurs rarely.
#     """

#     n_stimuli_desired = 9
#     n_trial_desired = 200
#     n_reference_desired = 2
#     n_selected_desired = 1
#     gen = generator.RandomGenerator(n_stimuli_desired)
#     unjudged_trials_0 = gen.generate(
#         n_trial=n_trial_desired, n_reference=n_reference_desired,
#         n_selected=n_selected_desired)
#     n_stimuli_desired = 10
#     n_trial_desired = 50
#     gen = generator.RandomGenerator(n_stimuli_desired)
#     unjudged_trials_1 = gen.generate(
#         n_trial=n_trial_desired, n_reference=n_reference_desired,
#         n_selected=n_selected_desired)
#     unjudged_trials = stack((unjudged_trials_0, unjudged_trials_1))    
#     return unjudged_trials

