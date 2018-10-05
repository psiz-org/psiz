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

"""Example demonstrating the selection heuristic for references."""

import copy
import itertools

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from psiz.trials import Docket
from psiz.models import Exponential
from psiz.simulate import Agent
from psiz.generator import ActiveGenerator
from psiz.utils import similarity_matrix, matrix_correlation


def main():
    """Examine validity of heuristics used during active selection."""
    # Settings.
    np.random.seed(123)
    n_sample = 2000
    n_reference = 2
    n_select = 1
    n_dim = 2
    n_stimuli = 25
    n_scenario = 100
    n_plot = 10

    cmap = matplotlib.cm.get_cmap('jet')
    norm = matplotlib.colors.Normalize(vmin=0., vmax=n_stimuli)
    color_array = cmap(norm(range(n_stimuli)))

    eligable_list = np.arange(n_stimuli, dtype=np.int32)
    stimulus_set = candidate_list(eligable_list, n_reference)
    n_candidate = stimulus_set.shape[0]

    rel_entropy_all = np.empty((0))
    best_ig_all = np.empty((0))
    nn_sum_all = np.empty((0))
    ig_all_2 = np.empty((0))

    fig = plt.figure(figsize=(10, 3), dpi=200)
    plot_idx = 1
    for i_scenario in range(n_scenario):
        print("scenario {0}".format(i_scenario))
        model = ground_truth(n_dim, n_stimuli)
        z_true = model.z['value']
        (n_stimuli, n_dim) = z_true.shape

        samples = simulated_samples(model.z['value'], n_sample)
        z_samp = samples['z']
        z_samp = np.transpose(z_samp, axes=[2, 0, 1])

        mu = np.empty((n_stimuli, n_dim))
        sigma = np.empty((n_stimuli, n_dim, n_dim))
        entropy = np.empty((n_stimuli))
        for i_stim in range(n_stimuli):
            gmm = GaussianMixture(
                n_components=1, covariance_type='full')
            gmm.fit(z_samp[:, i_stim, :])
            mu[i_stim, :] = gmm.means_[0]
            sigma[i_stim, :, :] = gmm.covariances_[0]
            entropy[i_stim] = normal_entropy(gmm.covariances_[0])
        rel_entropy = entropy - np.min(entropy)
        rel_entropy = rel_entropy / np.max(rel_entropy)

        rho = 2.
        nbrs = NearestNeighbors(
            n_neighbors=n_stimuli, algorithm='auto', p=rho
        ).fit(mu)
        (_, nn_idx) = nbrs.kneighbors(mu)

        candidate_docket = Docket(
            stimulus_set, n_select * np.ones(n_candidate, dtype=np.int32)
        )

        # Compute expected information gain.
        gen = ActiveGenerator()
        ig = gen._information_gain(model, samples, candidate_docket)

        ig = ig - np.min(ig)
        ig = ig / np.max(ig)

        # Find best trial for each stimulus when serving as query.
        best_ig = np.empty((n_stimuli))
        # nn_sum = np.empty((n_stimuli))
        for i_stim in range(n_stimuli):
            locs = np.equal(candidate_docket.stimulus_set[:, 0], i_stim)
            curr_ig = ig[locs]
            # curr_stim_set = candidate_docket.stimulus_set[locs]

            sorted_idx = np.argsort(-curr_ig)

            sorted_ig = curr_ig[sorted_idx]
            best_ig[i_stim] = sorted_ig[0]
            # sorted_stimulus_set = curr_stim_set[sorted_idx]
            # best_stimulus_set = sorted_stimulus_set[0]
            # nn_sum[i_stim] = check_neighbor_distance(best_stimulus_set, nn_idx)

        sorted_idx = np.argsort(-ig)
        sorted_stim_set = candidate_docket.stimulus_set[sorted_idx]
        curr_ig_2 = ig[sorted_idx]
        curr_ig_2 = curr_ig_2[0]
        nn_sum = [
            check_neighbor_distance(sorted_stim_set[0], nn_idx),
            check_neighbor_distance(sorted_stim_set[1], nn_idx),
            check_neighbor_distance(sorted_stim_set[3], nn_idx),
            check_neighbor_distance(sorted_stim_set[4], nn_idx),
            check_neighbor_distance(sorted_stim_set[5], nn_idx)
        ]
        # locs = np.equal(candidate_docket.stimulus_set[:, 0], best_stim_set[0])
        # curr_ig_2 = ig[locs]
        # curr_stim_set = candidate_docket.stimulus_set[locs]
        # n_candidate_trial = curr_stim_set.shape[0]
        # nn_sum = np.empty((n_candidate_trial))
        # for i_trial in range(n_candidate_trial):
        #     nn_sum[i_trial] = check_neighbor_distance(
        #         curr_stim_set[i_trial], nn_idx)

        rel_entropy_all = np.concatenate((rel_entropy_all, rel_entropy))
        best_ig_all = np.concatenate((best_ig_all, best_ig))
        nn_sum_all = np.append(nn_sum_all, nn_sum)
        ig_all_2 = np.append(ig_all_2, curr_ig_2)

        if i_scenario < n_plot:
            ax = fig.add_subplot(3, n_plot, plot_idx)
            # z_samp = np.transpose(z_samp, axes=[2, 0, 1])
            z_samp = np.reshape(z_samp, (n_sample * n_stimuli, n_dim))

            limits = {
                'x': [np.min(z_samp[:, 0]), np.max(z_samp[:, 0])],
                'y': [np.min(z_samp[:, 1]), np.max(z_samp[:, 1])]
            }
            color_array_samp = np.matlib.repmat(color_array, n_sample, 1)
            ax.scatter(
                z_samp[:, 0], z_samp[:, 1],
                s=5, c=color_array_samp, alpha=.01, edgecolors='none')
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(limits['x'][0], limits['x'][1])
            ax.set_ylim(limits['y'][0], limits['y'][1])

            ax = fig.add_subplot(3, n_plot, plot_idx + n_plot)
            candidate_subplot(
                ax, mu, sorted_stim_set[0], n_select, color_array, limits)

            ax = fig.add_subplot(3, n_plot, plot_idx + 2 * n_plot)
            candidate_subplot(
                ax, mu, sorted_stim_set[1], n_select, color_array, limits)

            plot_idx = plot_idx + 1

    plt.show()

    fig = plt.figure(figsize=(6.5, 2), dpi=200)
    ax = fig.add_subplot(1, 2, 1)
    ax.scatter(rel_entropy_all, best_ig_all, s=4, alpha=.25, edgecolors='none')
    ax.set_xlabel('Rel. Query Entropy')
    ax.set_ylabel('Rel. Information Gain')

    ax = fig.add_subplot(1, 2, 2)
    ax.hist(nn_sum_all, bins=n_stimuli, density=True)
    ax.set_xlabel('Nearest Neighbor Rank of References')
    ax.set_ylabel('Proportion')

    plt.tight_layout()
    plt.show()


def check_neighbor_distance(stimulus_set, nn_idx):
    """"""
    curr_nn_idx = nn_idx[stimulus_set[0], 1:]
    n_ref = np.sum(np.greater(stimulus_set[1:], 0))
    dmy_idx = np.arange(len(curr_nn_idx))
    nn_sum = 0
    for i_ref in range(n_ref):
        loc = np.equal(curr_nn_idx, stimulus_set[1 + i_ref])
        nn_sum = nn_sum + dmy_idx[loc]
    return nn_sum / n_ref


def candidate_subplot(
        ax, z, stimulus_set, n_select, color_array, limits):
    """Plot subplots for candidate trials."""
    locs = np.not_equal(stimulus_set, -1)
    stimulus_set = stimulus_set[locs]

    # fontdict = {
    #     'fontsize': 4,
    #     'verticalalignment': 'center',
    #     'horizontalalignment': 'center'
    # }

    ax.scatter(
        z[stimulus_set[0], 0],
        z[stimulus_set[0], 1],
        s=15, c=color_array[stimulus_set[0]], marker=r'$q$')
    ax.scatter(
        z[stimulus_set[1:], 0],
        z[stimulus_set[1:], 1],
        s=15, c=color_array[stimulus_set[1:]], marker=r'$r$')
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(limits['x'][0], limits['x'][1])
    ax.set_ylim(limits['y'][0], limits['y'][1])

    # rect_back = matplotlib.patches.Rectangle(
    #     (.55, -.55), .06, 1.1, clip_on=False, color=[.9, .9, .9])
    # ax.add_patch(rect_back)
    # rect_val = matplotlib.patches.Rectangle(
    #     (.55, -.55), .06, rel_ig * 1.1, clip_on=False, color=[.3, .3, .3])
    # ax.add_patch(rect_val)
    # plt.text(-.45, .45, "{0}".format(n_select), fontdict=fontdict)


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
