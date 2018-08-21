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

"""Example that demonstrates that query entropy is a good predictor."""

import copy
import itertools

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

from psiz.trials import UnjudgedTrials
from psiz.models import Exponential
from psiz.simulate import Agent
from psiz.generator import RandomGenerator, ActiveGenerator
from psiz.utils import similarity_matrix, matrix_correlation


def main():
    """Sample from posterior of pre-defined embedding model."""
    # Settings.
    np.random.seed(123)
    n_sample = 2000
    n_reference = 3
    n_selected = 1
    n_stimuli = 20
    n_scenario = 10

    eligable_list = np.arange(n_stimuli, dtype=np.int32)
    stimulus_set = candidate_list(eligable_list, n_reference)
    n_candidate = stimulus_set.shape[0]

    rel_entropy_all = np.empty((0))
    best_ig_all = np.empty((0))

    for _ in range(n_scenario):
        model = ground_truth()
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

        z_samp = np.reshape(z_samp, (n_sample * n_stimuli, n_dim))

        gen = ActiveGenerator(n_stimuli)
        candidate_trial = UnjudgedTrials(
            stimulus_set, n_selected * np.ones(n_candidate, dtype=np.int32)
        )

        # Compute expected information gain.
        ig = gen._information_gain(model, samples, candidate_trial)
        # score = gen._query_priority(samples)

        ig = ig - np.min(ig)
        ig = ig / np.max(ig)

        # Find best trial for each stimulus when serving as query.
        best_ig = np.empty((n_stimuli))
        for i_stim in range(n_stimuli):
            locs = np.equal(candidate_trial.stimulus_set[:, 0], i_stim)
            sorted_ig = np.sort(-ig[locs])
            best_ig[i_stim] = -1 * sorted_ig[0]

        rel_entropy_all = np.concatenate((rel_entropy_all, rel_entropy))
        best_ig_all = np.concatenate((best_ig_all, best_ig))

    plt.scatter(rel_entropy_all, best_ig_all)
    plt.xlabel('Relative Query Entropy')
    plt.ylabel('Relative IG')
    plt.show()

    # # Sort
    # sorted_indices = np.argsort(-ig)
    # ig = ig[sorted_indices]
    # v = copy.copy(ig)
    # v = v - np.min(v)
    # v = v / np.max(v)
    # stimulus_set = stimulus_set[sorted_indices]

    # # Select the 10 best trials and a 30 trials evenly spaced from best to
    # # worst.
    # # intermediate_trials = np.linspace(10, n_candidate-1, 30, dtype=np.int32)
    # # intermediate_trials = np.linspace(20, n_candidate-1, 20, dtype=np.int32)
    # display_idx = np.concatenate(
    #     (
    #         np.arange(40),
    #         # intermediate_trials
    #     ), axis=0
    # )

    # # Visualize.
    # cmap = matplotlib.cm.get_cmap('jet')
    # norm = matplotlib.colors.Normalize(vmin=0., vmax=model.n_stimuli)
    # color_array = cmap(norm(range(model.n_stimuli)))
    # color_array_samp = np.matlib.repmat(color_array, n_sample, 1)

    # fig = plt.figure(figsize=(6.5, 2), dpi=200)

    # fontdict = {
    #     'fontsize': 4,
    #     # 'fontweight': rcParams[‘axes.titleweight’],
    #     'verticalalignment': 'baseline',
    #     'horizontalalignment': 'center'
    # }

    # ax1 = fig.add_subplot(1, 6, 1)
    # ax1.scatter(
    #     z_samp[:, 0], z_samp[:, 1],
    #     s=5, c=color_array_samp, alpha=.01, edgecolors='none')
    # ax1.set_title('Posterior')
    # ax1.set_aspect('equal')
    # ax1.set_xlim(-.55, .55)
    # ax1.set_xticks([])
    # ax1.set_ylim(-.55, .55)
    # ax1.set_yticks([])

    # idx_list = np.concatenate(
    #     (
    #         np.arange(3, 13), np.arange(15, 25), np.arange(27, 37),
    #         np.arange(39, 49)
    #     ), axis=0
    # )

    # n_subplot = 40
    # for i_subplot in range(n_subplot):
    #     candidate_subplot(
    #         fig, idx_list[i_subplot], z_true,
    #         stimulus_set[display_idx[i_subplot]],
    #         ig[display_idx[i_subplot]], v[display_idx[i_subplot]],
    #         color_array, fontdict)
    # plt.suptitle('Candidate Trials', x=.58)
    # plt.show()


def candidate_subplot(fig, idx, z, stimulus_set, ig, v, color_array, fontdict):
    """Plot subplot of candidate trial."""
    ax = fig.add_subplot(4, 12, idx)
    ax.scatter(
        z[stimulus_set[0], 0],
        z[stimulus_set[0], 1],
        s=15, c=color_array[stimulus_set[0]], marker=r'$q$')
    ax.scatter(
        z[stimulus_set[1:], 0],
        z[stimulus_set[1:], 1],
        s=15, c=color_array[stimulus_set[1:]], marker=r'$r$')
    ax.set_aspect('equal')
    ax.set_xlim(-.55, .55)
    ax.set_xticks([])
    ax.set_ylim(-.55, .55)
    ax.set_yticks([])
    # plt.text(
    #     .25, .47, "{0:.4f}".format(ig),
    #     fontdict=fontdict)
    rect_back = matplotlib.patches.Rectangle(
        (.55, -.55), .06, 1.1, clip_on=False, color=[.9, .9, .9])
    ax.add_patch(rect_back)
    rect_val = matplotlib.patches.Rectangle(
        (.55, -.55), .06, v * 1.1, clip_on=False, color=[.3, .3, .3])
    ax.add_patch(rect_val)


def ground_truth():
    """Return a ground truth embedding."""
    # Create embeddingp points arranged on a grid.
    x, y = np.meshgrid([-.45, -.15, 0, .15, .45], [-.45, -.15, .15, .45])
    x = np.expand_dims(x.flatten(), axis=1)
    y = np.expand_dims(y.flatten(), axis=1)
    z = np.hstack((x, y))

    (n_stimuli, n_dim) = z.shape
    # Add some Gaussian noise to the embedding points.
    mean = np.zeros((n_dim))
    cov = .1 * np.identity(n_dim)
    z_noise = np.random.multivariate_normal(mean, cov, (n_stimuli))
    # z = (.8 * z) + (.2 * z_noise)
    z = (.7 * z) + (.3 * z_noise)

    # Create embedding model.
    n_group = 1
    (n_stimuli, n_dim) = z.shape
    model = Exponential(n_stimuli, n_dim=n_dim, n_group=n_group)
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
