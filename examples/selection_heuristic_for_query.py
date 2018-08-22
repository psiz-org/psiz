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

"""Example shoing that query entropy is a good selection heuristic."""

import copy
import itertools

import numpy as np
from scipy.stats import sem
import matplotlib
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from psiz.trials import UnjudgedTrials
from psiz.models import Exponential
from psiz.generator import ActiveGenerator, stimulus_entropy
from psiz.utils import similarity_matrix


def main():
    """Inspect the validity of the query selection heuristic."""
    # Settings.
    np.random.seed(123)
    n_sample = 2000
    n_reference = 2
    n_selected = 1
    n_dim = 2
    n_stimuli = 20  # 25 TODO
    n_scenario = 100

    eligable_list = np.arange(n_stimuli, dtype=np.int32)
    stimulus_set = candidate_list(eligable_list, n_reference)
    n_candidate = stimulus_set.shape[0]
    candidate_trial = UnjudgedTrials(
        stimulus_set, n_selected * np.ones(n_candidate, dtype=np.int32)
    )

    rel_entropy_all = np.empty((0))
    best_ig_all = np.empty((0))
    for i_scenario in range(n_scenario):
        print("scenario {0}".format(i_scenario))
        model = ground_truth(n_dim, n_stimuli)

        # Simulate posterior samples.
        samples = simulated_samples(model.z['value'], n_sample)
        # Compute entropy associated with each stimulus.
        entropy = stimulus_entropy(samples)
        rel_entropy = entropy - np.min(entropy)
        rel_entropy = rel_entropy / np.max(rel_entropy)

        # Compute expected information gain.
        gen = ActiveGenerator(n_stimuli)
        ig = gen._information_gain(model, samples, candidate_trial)
        rel_ig = ig - np.min(ig)
        rel_ig = rel_ig / np.max(rel_ig)

        # Find best trial for each stimulus when serving as query.
        best_ig = np.empty((n_stimuli))
        for i_stim in range(n_stimuli):
            locs = np.equal(candidate_trial.stimulus_set[:, 0], i_stim)
            curr_ig = rel_ig[locs]
            sorted_idx = np.argsort(-curr_ig)
            sorted_ig = curr_ig[sorted_idx]
            best_ig[i_stim] = sorted_ig[0]

        rel_entropy_all = np.concatenate((rel_entropy_all, rel_entropy))
        best_ig_all = np.concatenate((best_ig_all, best_ig))

    n_bin = 10
    bin_edges = np.linspace(0., 1., (n_bin + 1))
    bin_xg = np.empty((n_bin))
    bin_mean = np.empty((n_bin))
    bin_sem = np.empty((n_bin))
    for i_bin in range(n_bin):
        if i_bin < (n_bin - 1):
            locs = np.logical_and(
                np.greater_equal(rel_entropy_all, bin_edges[i_bin]),
                np.less(rel_entropy_all, bin_edges[i_bin + 1])
            )
        else:
            locs = np.logical_and(
                np.greater_equal(rel_entropy_all, bin_edges[i_bin]),
                np.less_equal(rel_entropy_all, bin_edges[i_bin + 1])
            )
        bin_xg[i_bin] = (bin_edges[i_bin] + bin_edges[i_bin + 1]) / 2
        bin_mean[i_bin] = np.mean(best_ig_all[locs])
        bin_sem[i_bin] = sem(best_ig_all[locs])

    fig = plt.figure(figsize=(6.5, 2), dpi=200)
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(rel_entropy_all, best_ig_all, s=4, alpha=.25, edgecolors='none')
    ax.errorbar(bin_xg, bin_mean, yerr=bin_sem)
    ax.set_xlabel('Rel. Query Entropy')
    ax.set_ylabel('Rel. Information Gain')
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


if __name__ == "__main__":
    main()
