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

import numpy as np

from psiz.trials import stack
from psiz.models import Exponential
from psiz.simulate import Agent
from psiz.generator import RandomGenerator
from psiz.utils import similarity_matrix, matrix_comparison


def main():
    """Run the simulation."""
    # Settings.
    n_stimuli = 25
    n_dim = 2
    n_group = 1
    n_restart = 20

    emb_true = ground_truth(n_stimuli, n_dim, n_group)
    simmat_true = similarity_matrix(emb_true.similarity, emb_true.z)

    # Generate a random docket of trials.
    n_trial = 1000
    n_reference = 8
    n_select = 2
    generator = RandomGenerator(n_reference, n_select)
    docket = generator.generate(n_trial, n_stimuli)

    # Simulate similarity judgments.
    agent = Agent(emb_true)
    obs = agent.simulate(docket)

    # Sample from embedding posterior.
    samples = emb_true.posterior_samples_special(obs, verbose=1)
    print('{0:.0f} s'.format(emb_true.posterior_duration))

    # Visualize posterior.
    visualize_posterior(emb_true.z, samples)


def visualize_posterior():
    """Visualize posterior."""
    x = 0


def ground_truth(n_stimuli, n_dim, n_group):
    """Return a ground truth embedding."""
    emb = Exponential(
        n_stimuli, n_dim=n_dim, n_group=n_group)
    mean = np.ones((n_dim))
    cov = .03 * np.identity(n_dim)
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
