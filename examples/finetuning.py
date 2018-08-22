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

"""Example exploring MCMC finetuning of gradient-decent solution."""

import copy
import itertools

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from psiz.models import Exponential
from psiz.simulate import Agent
from psiz.generator import RandomGenerator
from psiz.utils import similarity_matrix, matrix_correlation


def main():
    """Compare hueristic and exhaustive search method for trial selection."""
    # Settings.
    np.random.seed(123)
    n_stimuli = 30
    n_dim = 2
    n_trial = 1000
    n_reference = 8
    n_selected = 2
    n_scenario = 10

    r2_gd = np.empty((n_scenario))
    r2_mcmc = np.empty((n_scenario))
    for i_scenario in range(n_scenario):
        print("scenario {0}".format(i_scenario))
        model_true = ground_truth(n_dim, n_stimuli)
        simmat_true = similarity_matrix(
            model_true.similarity, model_true.z['value'])

        # Create a random set of trials.
        generator = RandomGenerator(n_stimuli)
        trials = generator.generate(n_trial, n_reference, n_selected)

        # Simulate similarity judgments.
        agent = Agent(model_true)
        obs = agent.simulate(trials)

        # Gradient decent solution.
        model_gd = Exponential(n_stimuli, n_dim)
        model_gd.freeze({'theta': {'beta': 10, 'rho': 2, 'tau': 1}})
        model_gd.fit(obs, n_restart=10, verbose=1)
        simmat_gd = similarity_matrix(
            model_gd.similarity, model_gd.z['value'])
        r2_gd[i_scenario] = matrix_correlation(simmat_gd, simmat_true)

        # MCMC fine-tuned solution.
        samples = model_gd.posterior_samples(obs, n_burn=200, verbose=1)
        z_samp = samples['z']
        z_mcmc = np.median(z_samp, axis=2)
        simmat_mcmc = similarity_matrix(
            model_gd.similarity, z_mcmc)
        r2_mcmc[i_scenario] = matrix_correlation(simmat_mcmc, simmat_true)

    # Compute difference between GD and MCMC solution.
    r2_diff = r2_mcmc - r2_gd

    fig, ax = plt.subplots()
    ind = np.arange(n_scenario)
    width = 0.35         # the width of the bars
    p1 = ax.bar(ind, r2_gd, width, color='r', bottom=0)
    p2 = ax.bar(ind + width, r2_mcmc, width, color='b', bottom=0)

    # ax.set_title('R^2 Similarity Matrix Correlation')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(np.arange(1, n_scenario + 1))
    ax.set_ylim(bottom=.5, top=1.)
    ax.set_xlabel('Scenario')
    ax.set_ylabel('R^2')
    ax.legend((p1[0], p2[0]), ('GD', 'MCMC'))
    plt.show()

    print('R^2 Difference')
    print('  Minimum: {0:.1f}'.format(np.min(r2_diff)))
    print('  Mean:    {0:.1f}'.format(np.mean(r2_diff)))
    print('  Median:  {0:.1f}'.format(np.median(r2_diff)))
    print('  Maximum: {0:.1f}'.format(np.max(r2_diff)))
    print('')


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
