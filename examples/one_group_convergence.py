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

"""Example that infers an embedding with an increasing amount of data.

Fake data is generated from a ground truth model assuming one group.
An embedding is inferred with an increasing amount of data,
demonstrating how the inferred model improves and asymptotes as more
data is added.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from psiz.models import Exponential
from psiz.simulate import Agent
from psiz.generator import RandomGenerator
from psiz.utils import similarity_matrix, matrix_comparison


def main():
    """Run the simulation that infers an embedding for two groups."""
    n_stimuli = 25
    n_dim = 3
    n_group = 1
    emb_true = ground_truth(n_stimuli, n_dim, n_group)
    simmat_true = similarity_matrix(
        emb_true.similarity, emb_true.z['value'])

    # Generate a random docket of trials.
    n_trial = 1000
    n_reference = 8
    n_select = 2
    generator = RandomGenerator(n_reference, n_select)
    docket = generator.generate(n_trial, n_stimuli)

    # Simulate similarity judgments.
    agent = Agent(emb_true)
    obs = agent.simulate(docket)

    # Infer independent models with increasing amounts of data.
    n_step = 10
    n_obs = np.floor(np.linspace(20, n_trial, n_step)).astype(np.int64)
    r_pearson = np.empty((n_step))
    loss = np.empty((n_step))
    for i_round in range(n_step):
        emb_inferred = Exponential(n_stimuli, n_dim, n_group)
        include_idx = np.arange(0, n_obs[i_round])
        loss[i_round], _ = emb_inferred.fit(obs.subset(include_idx), 40)
        # Compare the inferred model with ground truth by comparing the
        # similarity matrices implied by each model.
        simmat_infer = similarity_matrix(
            emb_inferred.similarity, emb_inferred.z['value'])
        r_pearson[i_round] = matrix_comparison(
            simmat_infer, simmat_true, score='pearson'
        )
        print(
            'Round {0} ({1} trials) | Loss: {2:.2f} | '
            'Correlation (r): {3:.2f}'.format(
                i_round, n_obs[i_round], loss[i_round], r_pearson[i_round]
            )
        )

    # Plot comparison results.
    plt.plot(n_obs, r_pearson, 'ro-')
    plt.title('Model Convergence to Ground Truth')
    plt.xlabel('Number of Judged Trials')
    plt.ylabel(r'Pearson Correlation ($\rho$)')
    plt.ylim(-0.05, 1.05)
    plt.show()


def ground_truth(n_stimuli, n_dim, n_group):
    """Return a ground truth embedding."""
    model = Exponential(
        n_stimuli, n_dim=n_dim, n_group=n_group)
    mean = np.ones((n_dim))
    cov = .03 * np.identity(n_dim)
    z = np.random.multivariate_normal(mean, cov, (n_stimuli))
    freeze_options = {
        'z': z,
        'theta': {
            'rho': 2,
            'tau': 1,
            'beta': 10,
            'gamma': 0.001
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
