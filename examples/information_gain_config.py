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

"""Example that demonstrates information gain computation.

A comparison of the expected information gain for different display
configurations.
"""

import copy
import itertools

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from psiz.trials import Docket, stack
from psiz.models import Exponential
from psiz.generator import ActiveGenerator


def main():
    """Sample from posterior of pre-defined embedding model."""
    # Settings.
    np.random.seed(126)
    n_sample = 2000

    model = ground_truth()
    z_true = model.z['value']
    n_stimuli = z_true.shape[0]

    eligable_list = np.arange(n_stimuli, dtype=np.int32)
    stimulus_set_2 = candidate_stimulus_sets(eligable_list, 2)
    stimulus_set_4 = candidate_stimulus_sets(eligable_list, 4)

    n_select = 1
    n_candidate = stimulus_set_2.shape[0]
    candidate_docket_2c1 = Docket(
        stimulus_set_2, n_select * np.ones(n_candidate, dtype=np.int32)
    )

    n_select = 1
    n_candidate = stimulus_set_4.shape[0]
    candidate_docket_4c1 = Docket(
        stimulus_set_4, n_select * np.ones(n_candidate, dtype=np.int32)
    )

    n_select = 2
    n_candidate = stimulus_set_4.shape[0]
    candidate_docket_4c2 = Docket(
        stimulus_set_4, n_select * np.ones(n_candidate, dtype=np.int32)
    )

    candidate_docket = stack((
        candidate_docket_2c1,
        candidate_docket_4c1, candidate_docket_4c2
    ))

    origin_cov = np.array([
        [[.03, 0], [0, .03]],
        [[.003, 0], [0, .003]],
        [[.0003, 0], [0, .0003]],
        [[.0003, 0], [0, .003]],
        [[.003, 0], [0, .0003]],
        [[.003, -.0025], [-.0025, .003]]
    ])
    n_scenario = origin_cov.shape[0]

    scenario_trials = []
    scenario_stim_set = []
    scenario_ig = []
    scenario_rel_ig = []
    for i_scenario in range(n_scenario):
        (curr_z_samp, curr_stim_set, curr_ig, curr_v) = process_scenario(
            model, origin_cov[i_scenario], n_sample, candidate_docket)
        scenario_trials.append(curr_z_samp)
        scenario_stim_set.append(curr_stim_set)
        scenario_ig.append(curr_ig)
        scenario_rel_ig.append(curr_v)

    # Visualize.
    cmap = matplotlib.cm.get_cmap('jet')
    norm = matplotlib.colors.Normalize(vmin=0., vmax=model.n_stimuli)
    color_array = cmap(norm(range(model.n_stimuli)))

    idx_list = np.array([1, 6, 10, 15, 19, 24])

    fig = plt.figure(figsize=(6.5, 2), dpi=200)
    for i_scenario in range(n_scenario):
        scenario_subplot(
            fig, idx_list[i_scenario], z_true,
            scenario_trials[i_scenario],
            scenario_stim_set[i_scenario],
            scenario_ig[i_scenario],
            scenario_rel_ig[i_scenario],
            color_array)

    plt.show()


def process_scenario(model, origin_cov, n_sample, candidate_docket):
    """Evaluate scenario."""
    z_true = model.z['value']
    (n_stimuli, n_dim) = z_true.shape

    samples = simulated_samples(z_true, n_sample, origin_cov)
    z_samp = samples['z']

    gen = ActiveGenerator()
    # Compute expected information gain.
    ig = gen._information_gain(model, samples, candidate_docket)
    # Relative information gain.
    rel_ig = copy.copy(ig)
    rel_ig = rel_ig - np.min(rel_ig)
    rel_ig = rel_ig / np.max(rel_ig)

    # Sort
    sorted_indices = np.argsort(-ig)
    ig = ig[sorted_indices]
    rel_ig = rel_ig[sorted_indices]
    candidate_docket = candidate_docket.subset(sorted_indices)

    # Select best of each trial configuration.
    n_config = 3
    for i_config in range(n_config):
        keep_locs = np.not_equal(
            candidate_docket.config_idx, candidate_docket.config_idx[i_config])
        keep_locs[i_config] = True
        ig = ig[keep_locs]
        rel_ig = rel_ig[keep_locs]
        candidate_docket = candidate_docket.subset(keep_locs)
    # Select best three.
    # sorted_indices = np.argsort(-ig)
    # ig = ig[sorted_indices]
    # trials_sub = candidate_docket.subset(sorted_indices)
    # trials_sub = trials_sub.subset(np.arange(0, 4, dtype=np.int32))
    # ig = ig[0:4]
    # rel_ig = rel_ig[0:4]
    return (z_samp, candidate_docket, ig, rel_ig)


def scenario_subplot(
        fig, idx, z_true, z_samp, docket, ig, rel_ig, color_array):
    """Plot scenario (posterior samples and trial candidates)."""
    # Plot posterior samples of scenario.
    posterior_subplot(fig, idx, z_samp, color_array)

    # Plot visualization of candidate trials and information gain.
    n_subplot = 3
    for i_subplot in range(n_subplot):
        candidate_subplot(
            fig, idx + i_subplot + 1, z_true,
            docket.stimulus_set[i_subplot],
            docket.n_select[i_subplot],
            ig[i_subplot], rel_ig[i_subplot],
            color_array)


def posterior_subplot(fig, idx, z_samp, color_array):
    """Plot posterior samples."""
    (n_stimuli, n_dim, n_sample) = z_samp.shape
    z_samp = np.transpose(z_samp, axes=[2, 0, 1])
    z_samp = np.reshape(z_samp, (n_sample * n_stimuli, n_dim))
    color_array_samp = np.matlib.repmat(color_array, n_sample, 1)

    ax = fig.add_subplot(3, 9, idx)
    ax.scatter(
        z_samp[:, 0], z_samp[:, 1],
        s=5, c=color_array_samp, alpha=.01, edgecolors='none')
    ax.set_aspect('equal')
    ax.set_xlim(-.55, .55)
    ax.set_xticks([])
    ax.set_ylim(-.55, .55)
    ax.set_yticks([])


def candidate_subplot(
        fig, idx, z, stimulus_set, n_select, ig, rel_ig, color_array):
    """Plot subplots for candidate trials."""
    locs = np.not_equal(stimulus_set, -1)
    stimulus_set = stimulus_set[locs]

    fontdict = {
        'fontsize': 4,
        'verticalalignment': 'center',
        'horizontalalignment': 'center'
    }

    ax = fig.add_subplot(3, 9, idx)
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
    rect_back = matplotlib.patches.Rectangle(
        (.55, -.55), .06, 1.1, clip_on=False, color=[.9, .9, .9])
    ax.add_patch(rect_back)
    rect_val = matplotlib.patches.Rectangle(
        (.55, -.55), .06, rel_ig * 1.1, clip_on=False, color=[.3, .3, .3])
    ax.add_patch(rect_val)
    plt.text(-.45, .45, "{0}".format(n_select), fontdict=fontdict)


def ground_truth():
    """Return a ground truth embedding."""
    # Create embeddingp points arranged in circle.
    origin = np.array([[0.0, 0.0]])

    theta = np.linspace(0, 2 * np.pi, 9)
    theta = theta[0:-1]

    # r = .2
    # x = r * np.cos(theta)
    # y = r * np.sin(theta)
    # x = np.expand_dims(x, axis=1)
    # y = np.expand_dims(y, axis=1)
    # z_inner = np.hstack((x, y))

    r = .4
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    x = np.expand_dims(x, axis=1)
    y = np.expand_dims(y, axis=1)
    z_outer = np.hstack((x, y))
    z = np.concatenate((origin, z_outer), axis=0)

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


def simulated_samples(z, n_sample, origin_cov):
    """Simulate posterior samples for a set of embedding points."""
    n_stimuli = z.shape[0]
    n_dim = z.shape[1]

    stim_cov = .0003 * np.ones((n_stimuli))
    stim_cov = np.expand_dims(stim_cov, axis=1)
    stim_cov = np.expand_dims(stim_cov, axis=1)
    stim_cov = stim_cov * np.expand_dims(np.identity(n_dim), axis=0)

    stim_cov[0, :, :] = origin_cov
    # Draw samples
    z_samples = np.empty((n_sample, n_stimuli, n_dim))
    for i_stimulus in range(n_stimuli):
        z_samples[:, i_stimulus, :] = np.random.multivariate_normal(
            z[i_stimulus], stim_cov[i_stimulus], (n_sample)
        )
    z_samples = np.transpose(z_samples, axes=[1, 2, 0])
    return {'z': z_samples}


def candidate_stimulus_sets(eligable_list, n_reference):
    """Determine all possible stimulus set combinations."""
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
