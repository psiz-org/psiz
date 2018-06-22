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
example, using the ground truth allows us to see how well the algorithm
works in the best case scenario.

Notes:
    - Handling invarianc to affine transformations (translation, scale,
      and rotation).

"""

import copy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf

from psiz.trials import UnjudgedTrials
from psiz.models import Exponential, HeavyTailed, StudentsT
from psiz.simulate import Agent
from psiz.generator import RandomGenerator, ActiveGenerator
from psiz.utils import matrix_correlation


def main():
    """Sample from posterior of pre-defined embedding model."""

    # Ground truth model.
    model_truth = ground_truth()
    z_true = model_truth.z['value'].astype(np.float64)
    simmat_truth = model_truth.similarity_matrix()

    # Create some random trials.
    generator = RandomGenerator(model_truth.n_stimuli)
    n_trial = 10000
    n_reference = 2
    n_selected = 1
    trials = generator.generate(n_trial, n_reference, n_selected)

    # Remove data for stimulus 8
    # locs = np.equal(trials.stimulus_set, 8)
    # locs = np.sum(locs, axis=1)
    # n_loc = np.sum(locs)
    # locs[0:int(np.floor(n_trial/20))] = False
    # print('dropped: {0}'.format(np.sum(locs) / n_loc))
    # locs = np.logical_not(locs)
    # trials = trials.subset(locs)

    # Simulate similarity judgements using ground truth model.
    agent = Agent(model_truth)
    obs = agent.simulate(trials)

    # Infer an embedding model.
    model_inferred = Exponential(
        model_truth.n_stimuli, model_truth.dimensionality)
    model_inferred.fit(obs, 10, verbose=1)  # TODO
    # print('rho_0:', model_inferred.theta['rho']['value'])  # TODO
    # print('tau_0:', model_inferred.theta['tau']['value'])  # TODO
    # print('gamma_0:', model_inferred.theta['gamma']['value'])  # TODO
    # print('beta_0:', model_inferred.theta['beta']['value'])  # TODO
    # tf.reset_default_graph()
    # print('rho_0:', model_inferred.theta['rho']['value'])  # TODO
    # print('tau_0:', model_inferred.theta['tau']['value'])  # TODO
    # print('gamma_0:', model_inferred.theta['gamma']['value'])  # TODO
    # print('beta_0:', model_inferred.theta['beta']['value'])  # TODO
    z_inferred = copy.copy(model_inferred.z['value'].astype(np.float64))
    simmat_infer = model_inferred.similarity_matrix()
    r_squared = matrix_correlation(simmat_infer, simmat_truth)
    print('R^2 | {0: >6.2f}'.format(r_squared))
    # ==== TODO
    # model_inferred.z['value'] = 2 * model_inferred.z['value']
    # simmat_infer = model_inferred.similarity_matrix()
    # r_squared = matrix_correlation(simmat_infer, simmat_truth)
    # print('R^2 | {0: >6.2f}'.format(r_squared))
    # ====
    # Is it necessary to freeze the parameters?
    # freeze_options = {
    #     'rho': model_inferred.theta['rho']['value'],
    #     'tau': model_inferred.theta['tau']['value'],
    #     'beta': model_inferred.theta['beta']['value'],
    #     'gamma': model_inferred.theta['gamma']['value'],
    #     # 'z': model_inferred.z['value']
    # }
    # model_inferred.freeze(freeze_options)

    z_samp = model_inferred.posterior_samples(obs)
    z_central = np.median(z_samp, axis=0)

    model_inferred.z['value'] = z_central
    simmat_infer = model_inferred.similarity_matrix()
    r_squared = matrix_correlation(simmat_infer, simmat_truth)
    print('R^2 | {0: >6.2f}'.format(r_squared))

    cmap = matplotlib.cm.get_cmap('jet')
    norm = matplotlib.colors.Normalize(vmin=0., vmax=model_truth.n_stimuli)
    color_array = cmap(norm(range(model_truth.n_stimuli)))

    fig, ax = plt.subplots()

    plt.subplot(2, 2, 1)
    for i_stimulus in range(model_truth.n_stimuli):
        plt.scatter(
            z_true[i_stimulus, 0], z_true[i_stimulus, 1],
            c=color_array[i_stimulus, :])
    plt.axis('equal')
    plt.title('Ground Truth Locations')

    plt.subplot(2, 2, 2)
    for i_stimulus in range(model_truth.n_stimuli):
        plt.scatter(
            z_inferred[i_stimulus, 0], z_inferred[i_stimulus, 1],
            c=color_array[i_stimulus, :])
    plt.axis('equal')
    plt.title('Inferred Locations')

    plt.subplot(2, 2, 3)
    for i_stimulus in range(model_truth.n_stimuli):
        plt.scatter(
            z_samp[:, i_stimulus, 0], z_samp[:, i_stimulus, 1],
            c=color_array[i_stimulus, :], alpha=.01, edgecolors='none')
    plt.axis('equal')
    plt.title('Posterior Samples')

    plt.subplot(2, 2, 4)
    for i_stimulus in range(model_truth.n_stimuli):
        plt.scatter(
            z_central[i_stimulus, 0], z_central[i_stimulus, 1],
            c=color_array[i_stimulus, :])
    plt.axis('equal')
    plt.title('Mean of the Posterior Samples')
    plt.show()


def ground_truth():
    """Return a ground truth embedding."""
    n_stimuli = 16
    n_dim = 2
    # Create embeddingp points arranged on a grid.
    x, y = np.meshgrid([1, 2, 3, 4], [-2, -1, 0, 1])
    x = np.expand_dims(x.flatten(), axis=1)
    y = np.expand_dims(y.flatten(), axis=1)
    z = np.hstack((x, y))
    # Add some Gaussian noise to the embedding points.
    mean = np.ones((n_dim))
    cov = .01 * np.identity(n_dim)
    z_noise = np.random.multivariate_normal(mean, cov, (n_stimuli))
    z = z + z_noise
    # Create embedding model.
    n_group = 1
    model = Exponential(n_stimuli, dimensionality=n_dim, n_group=n_group)
    freeze_options = {
        'rho': 2,
        'tau': 1,
        'beta': 1,
        'gamma': 0,
        'z': z
    }
    model.freeze(freeze_options)
    return model


if __name__ == "__main__":
    main()

