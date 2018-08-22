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

"""Example that demonstrates that selection heursitics work well."""

import copy
import itertools

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from psiz.trials import UnjudgedTrials
from psiz.models import Exponential
from psiz.simulate import Agent
from psiz.generator import RandomGenerator, ActiveGenerator
from psiz.utils import similarity_matrix, matrix_correlation


def main():
    """Examine validity of heuristics used during active selection."""
    # Settings.
    np.random.seed(123)
    n_sample = 2000
    n_reference = 2
    n_selected = 1
    n_dim = 2
    n_stimuli = 30
    n_scenario = 100
    n_keep = 3

    eligable_list = np.arange(n_stimuli, dtype=np.int32)
    stimulus_set = candidate_list(eligable_list, n_reference)
    n_candidate = stimulus_set.shape[0]

    diff_ig = np.empty((n_scenario))
    for i_scenario in range(n_scenario):
        print("scenario {0}".format(i_scenario))
        model = ground_truth(n_dim, n_stimuli)
        samples = simulated_samples(model.z['value'], n_sample)

        config_list = pd.DataFrame({
            'n_reference': np.array([2], dtype=np.int32),
            'n_selected': np.array([1], dtype=np.int32),
            'is_ranked': [True],
            'n_outcome': np.array([2], dtype=np.int32)
        })
        gen = ActiveGenerator(n_stimuli, config_list=config_list)

        # Exhausive search.
        # Exhaustive set of trials.
        candidate_trial = UnjudgedTrials(
            stimulus_set, n_selected * np.ones(n_candidate, dtype=np.int32)
        )
        ig = gen._information_gain(model, samples, candidate_trial)
        min_ig = np.min(ig)
        rel_ig = ig - min_ig
        max_rel_ig = np.max(rel_ig)
        rel_ig = rel_ig / max_rel_ig
        # Find best trial for each stimulus when serving as query.
        exha_rel_ig_stim = np.empty((n_stimuli))
        for i_stim in range(n_stimuli):
            locs = np.equal(candidate_trial.stimulus_set[:, 0], i_stim)
            curr_rel_ig = rel_ig[locs]
            sorted_idx = np.argsort(-curr_rel_ig)
            sorted_rel_ig = curr_rel_ig[sorted_idx]
            exha_rel_ig_stim[i_stim] = sorted_rel_ig[0]
        exha_rel_ig_stim = -np.sort(-exha_rel_ig_stim)
        exha_rel_ig_stim = exha_rel_ig_stim[0:n_keep]

        # Hueristic search.
        (_, heur_ig_stim) = gen.generate(
            n_keep, model, samples, n_query=n_stimuli)
        heur_rel_ig_stim = (heur_ig_stim - min_ig) / max_rel_ig

        # Compute difference between heuristic and exhaustive search.
        diff_ig[i_scenario] = (
            np.sum(exha_rel_ig_stim) - np.sum(heur_rel_ig_stim)) / n_stimuli

    print('Average difference in total expected information gain: {0:.4f}'.format(np.mean(diff_ig)))

    fig = plt.figure(figsize=(6.5, 2), dpi=200)
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(1. - diff_ig, bins=np.linspace(.9, 1., 11))
    ax.set_xlabel('Proportion of Global Optimum')
    ax.set_ylabel('Frequency')
    plt.tight_layout()
    plt.show()


def ground_truth(n_dim, n_stimuli):
    """Return a ground truth embedding."""
    # Sample embeddingp points from Gaussian.
    mean = np.zeros((n_dim))
    cov = .3 * np.identity(n_dim)
    z = np.random.multivariate_normal(mean, cov, (n_stimuli))

    # Create embedding model.
    n_group = 1
    (n_stimuli, n_dim) = z.shape
    model = Exponential(n_stimuli, n_dim=n_dim, n_group=n_group)
    freeze_options = {
        'z': z,
        'theta': {
            'rho': 2,
            'tau': 1,
            'beta': 7,
            'gamma': 0
        }
    }
    model.freeze(freeze_options)

    # sim_mat = similarity_matrix(model.similarity, z)
    # idx_upper = np.triu_indices(n_stimuli, 1)
    # plt.hist(sim_mat[idx_upper])
    # plt.show()
    return model


def simulated_samples(z, n_sample):
    """Simulate posterior samples for a set of embedding points."""
    n_stimuli = z.shape[0]
    n_dim = z.shape[1]

    stim_cov = np.random.uniform(low=.0001, high=.01, size=n_stimuli)
    # stim_cov = np.random.uniform(low=.0001, high=.0005, size=n_stimuli)
    # cov_high = np.random.uniform(low=.001, high=.01, size=4)
    # stim_cov[1] = cov_high[0]
    # stim_cov[5] = cov_high[1]
    # stim_cov[9] = cov_high[2]
    # stim_cov[15] = cov_high[3]
    stim_cov = np.expand_dims(stim_cov, axis=1)
    stim_cov = np.expand_dims(stim_cov, axis=1)
    stim_cov = stim_cov * np.expand_dims(np.identity(n_dim), axis=0)

    # Draw samples
    z_samples = np.empty((n_sample, n_stimuli, n_dim))
    for i_stimulus in range(n_stimuli):
        z_samples[:, i_stimulus, :] = np.random.multivariate_normal(
            z[i_stimulus], stim_cov[i_stimulus], (n_sample)
        )
    z_samples = np.transpose(z_samples, axes=[1, 2, 0])
    return {'z': z_samples}


def candidate_list(eligable_list, n_reference):
    """Determine all possible trials."""
    n_stimuli = len(eligable_list)
    stimulus_set = np.empty([0, n_reference + 1], dtype=np.int32)
    for i_stim in range(n_stimuli):
        locs = np.not_equal(eligable_list, i_stim)
        sub_list = itertools.combinations(eligable_list[locs], n_reference)
        for item in sub_list:
            item = np.hstack((i_stim * np.ones(1), item))
            stimulus_set = np.vstack((stimulus_set, item))
    stimulus_set = stimulus_set.astype(dtype=np.int32)
    return stimulus_set


def normal_entropy(sigma):
    """Return entropy of multivariate normal distribution."""
    n_dim = sigma.shape[0]
    h = (
        (n_dim / 2) +
        (n_dim / 2 * np.log(2 * np.pi)) +
        (1 / 2 * np.log(np.linalg.det(sigma)))
    )
    return h


if __name__ == "__main__":
    main()
