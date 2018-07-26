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

"""Example that samples from the posterior of an embedding model.

Synthetic data is generated from a ground truth embedding model. For
simplicity, the ground truth model is also used as the inferred
model in this example. In practice the judged trials would be used to
infer an embedding model since the ground truth is not known. In this
example, using the ground truth allows us to see how the posterior
sampling algorithm works under ideal conditions.

Notes:
    - Handling invariance to affine transformations (translation, scale,
      and rotation).

"""

import copy

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf

from psiz.trials import UnjudgedTrials, stack
from psiz.models import Exponential, HeavyTailed, StudentsT
from psiz.simulate import Agent
from psiz.generator import RandomGenerator, ActiveGenerator
from psiz.utils import similarity_matrix, matrix_correlation


def main():
    """Sample from posterior of pre-defined embedding model."""
    # Settings
    n_trial = 10000  # 5000 10000
    n_frame = 30
    n_sample = 1000
    n_burn = 100  # 1000
    thin_step = 1  # 3

    # Ground truth model.
    model_truth = ground_truth()
    n_stimuli = model_truth.z['value'].shape[0]
    n_dim = model_truth.z['value'].shape[1]
    z_true = model_truth.z['value'].astype(np.float64)
    simmat_truth = similarity_matrix(
        model_truth.similarity, model_truth.z['value'])

    # Create some random trials.
    generator = RandomGenerator(model_truth.n_stimuli)
    n_reference = 2
    n_selected = 1
    trials = generator.generate(n_trial, n_reference, n_selected)

    # Remove data for stimulus 8
    # locs = np.equal(trials.stimulus_set, 6)
    # locs = np.sum(locs, axis=1)
    # n_loc = np.sum(locs)
    # locs[0:int(np.floor(n_trial/20))] = False
    # print('dropped: {0}'.format(np.sum(locs) / n_loc))
    # locs = np.logical_not(locs)
    # trials = trials.subset(locs)

    # Simulate similarity judgements using ground truth model.
    agent = Agent(model_truth)
    obs = agent.simulate(trials)

    # Infer an embedding model.  # TODO
    # model_inferred = Exponential(
    #     model_truth.n_stimuli, model_truth.n_dim)
    # model_inferred.freeze({'theta': {'beta': 10, 'rho': 2, 'tau': 1})
    # model_inferred.fit(obs, 10, verbose=1)
    model_inferred = model_truth
    z_original = copy.copy(model_inferred.z['value'])

    z_inferred = copy.copy(model_inferred.z['value'].astype(np.float64))
    simmat_infer = similarity_matrix(
        model_inferred.similarity, model_inferred.z['value'])
    r_squared = matrix_correlation(simmat_infer, simmat_truth)
    print('R^2 | {0: >6.2f}'.format(r_squared))

    z_samp_list = n_frame * [None]
    z_central_list = n_frame * [None]
    r_squared_list = n_frame * [None]
    n_obs = np.floor(np.linspace(20, n_trial, n_frame)).astype(np.int64)
    for i_frame in range(n_frame):
        include_idx = np.arange(0, n_obs[i_frame])
        z_samp = model_inferred.posterior_samples(
            obs.subset(include_idx), n_sample, n_burn, thin_step)
        z_central = np.median(z_samp, axis=0)

        z_samp_list[i_frame] = np.reshape(
            z_samp, (n_sample * n_stimuli, n_dim))
        z_central_list[i_frame] = z_central

        model_inferred.z['value'] = z_central
        simmat_infer = similarity_matrix(
            model_inferred.similarity, model_inferred.z['value'])
        r_squared = matrix_correlation(simmat_infer, simmat_truth)
        r_squared_list[i_frame] = r_squared
        print('R^2 | {0: >6.2f}'.format(r_squared))
        model_inferred.z['value'] = z_original

    cmap = matplotlib.cm.get_cmap('jet')
    norm = matplotlib.colors.Normalize(vmin=0., vmax=model_truth.n_stimuli)
    color_array = cmap(norm(range(model_truth.n_stimuli)))
    color_array_samp = np.matlib.repmat(color_array, n_sample, 1)

    # Set up formatting for the movie files
    Writer = animation.writers['ffmpeg']
    metadata = dict(
            title='Embedding Inference Evolution', artist='Matplotlib')
    writer = Writer(fps=3, metadata=metadata)

    # Initialize first frame.
    i_frame = 0

    fig = plt.figure(figsize=(5.5, 2), dpi=200)

    ax1 = fig.add_subplot(1, 3, 1)
    scat1 = ax1.scatter(
        z_true[:, 0], z_true[:, 1], s=15, c=color_array, marker='o')
    ax1.set_title('Ground Truth')
    ax1.set_aspect('equal')
    ax1.set_xlim(-.05, .55)
    ax1.set_xticks([])
    ax1.set_ylim(-.05, .55)
    ax1.set_yticks([])

    ax2 = fig.add_subplot(1, 3, 2)
    scat2 = ax2.scatter(
        z_central_list[i_frame][:, 0], z_central_list[i_frame][:, 1],
        s=15, c=color_array, marker='X')
    ax2.set_title('Point Estimate')
    ax2.set_aspect('equal')
    ax2.set_xlim(-.05, .55)
    ax2.set_xticks([])
    ax2.set_ylim(-.05, .55)
    ax2.set_yticks([])

    ax3 = fig.add_subplot(1, 3, 3)
    scat3 = ax3.scatter(
        z_samp_list[i_frame][:, 0], z_samp_list[i_frame][:, 1],
        s=5, c=color_array_samp, alpha=.01, edgecolors='none')
    ax3.set_title('Posterior Estimate')
    ax3.set_aspect('equal')
    ax3.set_xlim(-.05, .55)
    ax3.set_xticks([])
    ax3.set_ylim(-.05, .55)
    ax3.set_yticks([])

    def update(frame_number):
        scat2.set_offsets(
            z_central_list[frame_number])
        scat3.set_offsets(
            z_samp_list[frame_number])
    ani = animation.FuncAnimation(fig, update, frames=n_frame)
    ani.save('posterior.mp4', writer=writer)


def ground_truth():
    """Return a ground truth embedding."""
    n_stimuli = 16
    n_dim = 2
    # Create embeddingp points arranged on a grid.
    x, y = np.meshgrid([.1, .2, .3, .4], [.1, .2, .3, .4])
    x = np.expand_dims(x.flatten(), axis=1)
    y = np.expand_dims(y.flatten(), axis=1)
    z = np.hstack((x, y))
    # Add some Gaussian noise to the embedding points.
    mean = np.zeros((n_dim))
    cov = .01 * np.identity(n_dim)
    z_noise = .1 * np.random.multivariate_normal(mean, cov, (n_stimuli))
    z = z + z_noise
    # Create embedding model.
    n_group = 1
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


if __name__ == "__main__":
    main()
