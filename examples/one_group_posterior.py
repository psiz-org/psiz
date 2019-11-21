# -*- coding: utf-8 -*-
# Copyright 2019 The PsiZ Authors. All Rights Reserved.
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

"""Example that samples from the posterior distribution.

Fake data is generated from a ground truth model.

"""

from pathlib import Path

import matplotlib
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import numpy as np
import pickle

from psiz.trials import stack
from psiz.models import Exponential
from psiz.simulate import Agent
from psiz.generator import RandomGenerator
from psiz.utils import similarity_matrix, matrix_comparison


def main():
    """Run the simulation."""
    # Settings.
    n_trial = 1000
    fp_samples = Path(
        'examples', 'one_group_posterior_samples_{0}.p'.format(n_trial)
    )
    n_stimuli = 25
    n_dim = 2
    n_group = 1
    n_restart = 20
    n_reference = 8
    n_select = 2

    emb_true = ground_truth(n_stimuli, n_dim, n_group)

    # Generate a random docket of trials.
    generator = RandomGenerator(n_reference, n_select)
    docket = generator.generate(n_trial, n_stimuli)

    # Simulate similarity judgments.
    agent = Agent(emb_true)
    obs = agent.simulate(docket)

    # Sample from embedding posterior.
    samples = emb_true.posterior_samples(obs, verbose=1)
    print('{0:.0f} s'.format(emb_true.posterior_duration))
    # pickle.dump(samples, open(fp_samples, 'wb'))
    # orig
    #     100: 0:00:57
    #    1000: 0:02:07 +/- 0:00:01
    #   10000: 0:11:53
    # new
    #     100: 0:00:56
    #    1000: 0:02:00
    #   10000: 0:11:37

    # samples = pickle.load(open(fp_samples, 'rb'))
    # Visualize posterior.
    # cmap = matplotlib.cm.get_cmap('jet')
    # norm = matplotlib.colors.Normalize(vmin=0., vmax=emb_true.n_stimuli)
    # color_array = cmap(norm(range(emb_true.n_stimuli)))

    # fig = plt.figure(figsize=(5.5, 2), dpi=300)
    # [lim_x, lim_y] = determine_limits(emb_true.z)

    # ax1 = fig.add_subplot(1, 2, 1)
    # ax1.scatter(
    #     emb_true.z[:, 0], emb_true.z[:, 1], s=5, c=color_array, marker='o'
    # )
    # ax1.set_title('Ground Truth')
    # ax1.set_aspect('equal')
    # ax1.set_xlim(lim_x)
    # ax1.set_ylim(lim_y)
    # ax1.set_xticks([], [])
    # ax1.set_yticks([], [])

    # ax2 = fig.add_subplot(1, 2, 2)
    # visualize_samples(ax2, samples, s=5, c=color_array)
    # ax2.set_title('Posterior')
    # ax2.set_aspect('equal')
    # ax2.set_xlim(lim_x)
    # ax2.set_ylim(lim_y)
    # ax2.set_xticks([], [])
    # ax2.set_yticks([], [])

    # plt.show()
    # fname = None
    # plt.savefig(os.fspath(fname), format='pdf', bbox_inches="tight", dpi=300)


def determine_limits(z, pad=.2):
    """Determine limits of plot."""
    x_min = np.min(z[:, 0])
    x_max = np.max(z[:, 0])
    y_min = np.min(z[:, 1])
    y_max = np.max(z[:, 1])

    x_pad = pad * (x_max - x_min)
    y_pad = pad * (y_max - y_min)

    x_min = x_min - x_pad
    x_max = x_max + x_pad
    y_min = y_min - y_pad
    y_max = y_max + y_pad

    return ([x_min, x_max], [y_min, y_max])


def visualize_samples(ax, samples, s=5, c=None):
    """Visualize distribution of posterior using confidence ellipses.

    Arguments:
        samples:
        fname (optional): The pdf filename to save the figure,
            otherwise the figure is displayed. Can be either a path
            string or a pathlib Path object.

    """
    # Settings
    nstd = 2

    z_samp = samples['z']
    (n_stim, n_dim, _) = z_samp.shape
    z_samp = np.transpose(z_samp, axes=[0, 2, 1])

    if c is None:
        cmap = matplotlib.cm.get_cmap('jet')
        norm = matplotlib.colors.Normalize(vmin=0., vmax=n_stim)
        c = cmap(norm(range(emb_true.n_stim)))

    # Plot images.
    for i_stim in range(n_stim):
        curr_samples = z_samp[i_stim]
        mu = np.mean(curr_samples, axis=0)
        ax.scatter(
            mu[0], mu[1], s=s, c=c[np.newaxis, i_stim], edgecolors='none'
        )
        ell = error_ellipse(curr_samples, nstd)
        ell.set_facecolor('none')
        ell.set_edgecolor(c[i_stim])
        ax.add_artist(ell)

    # plt.tick_params(
    #         axis='x',
    #         which='both',
    #         bottom='off',
    #         top='off',
    #         labelbottom='off')
    # plt.tick_params(
    #     axis='y',
    #     which='both',
    #     left='off',
    #     right='off',
    #     labelleft='off')
    # ax.xaxis.get_offset_text().set_visible(False)


def eigsorted(cov):
    """Sort eigenvalues."""
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:, order]


def error_ellipse(samples, nstd):
    """Return artist of error ellipse.

    SEE: https://stackoverflow.com/questions/20126061/
    creating-a-confidence-ellipses-in-a-sccatterplot-using-matplotlib

    Arguments:
        samples:
        nstd: The number of standard deviations.

    """
    x = samples[:, 0]
    y = samples[:, 1]
    cov = np.cov(x, y)
    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2 * nstd * np.sqrt(vals)
    ell = Ellipse(
        xy=(np.mean(x), np.mean(y)), width=w, height=h, angle=theta,
        color='black'
    )
    return ell


def ground_truth(n_stimuli, n_dim, n_group):
    """Return a ground truth embedding."""
    emb = Exponential(
        n_stimuli, n_dim=n_dim, n_group=n_group)
    mean = np.zeros((n_dim))
    cov = .03 * np.identity(n_dim)
    np.random.seed(123)
    z = np.random.multivariate_normal(mean, cov, (n_stimuli))
    emb.z = z
    emb.rho = 2
    emb.tau = 1
    emb.beta = 10
    emb.gamma = 0.001
    emb.trainable("freeze")
    return emb


if __name__ == "__main__":
    main()
