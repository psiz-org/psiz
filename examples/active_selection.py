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

"""Example comparing random and active selection.

This example using simulated behavior to illustrate the theoretical
advantage of using active selection over random trial selection. The
simulation is time intensive.

"""

import copy
import itertools

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from psiz.trials import stack
from psiz.models import Exponential
from psiz.simulate import Agent
from psiz.generator import RandomGenerator, ActiveGenerator
from psiz.utils import similarity_matrix, matrix_correlation


def main():
    """Sample from posterior of pre-defined embedding model."""
    # Settings.
    np.random.seed(123)
    n_stimuli = 30
    n_dim = 2
    n_trial = 1000
    n_reference = 8
    n_select = 2

    model_true = ground_truth(n_dim, n_stimuli)
    simmat_true = similarity_matrix(
        model_true.similarity, model_true.z['value'])

    # Generate a random docket of trials.
    generator = RandomGenerator(n_stimuli)
    docket = generator.generate(n_trial, n_reference, n_select)

    # Simulate similarity judgments.
    agent = Agent(model_true)
    obs = agent.simulate(docket)

    # Gradient decent solution.
    model_gd = Exponential(n_stimuli, n_dim)
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


if __name__ == "__main__":
    main()
