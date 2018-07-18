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

"""Example that infers a shared embedding for three groups.

Fake data is generated from a ground truth model for three different
groups. In this example, these groups represent groups of agents with
varying levels of skill: novices, intermediates, and experts. Each group
has a different set of attention weights. An embedding model is
inferred from the simulated data and compared to the ground truth
model.
"""

import numpy as np
import tensorflow as tf

from psiz.trials import JudgedTrials
from psiz.models import Exponential
from psiz.simulate import Agent
from psiz.generator import RandomGenerator
from psiz.utils import similarity_matrix, matrix_correlation


def main():
    """Run the simulation that infers an embedding for three groups."""
    n_stimuli = 10
    n_dim = 3
    n_group = 3
    model_truth = ground_truth(n_stimuli, n_dim, n_group)

    # Create a random set of trials to show to every group.
    n_trial = 1000
    n_reference = 8
    n_selected = 2
    generator = RandomGenerator(n_stimuli)
    trials = generator.generate(n_trial, n_reference, n_selected)

    # Simulate similarity judgments for the three groups.
    agent_novice = Agent(model_truth, group_id=0)
    agent_interm = Agent(model_truth, group_id=1)
    agent_expert = Agent(model_truth, group_id=2)
    obs_novice = agent_novice.simulate(trials)
    obs_interm = agent_interm.simulate(trials)
    obs_expert = agent_expert.simulate(trials)
    obs_all = JudgedTrials.stack((obs_novice, obs_interm, obs_expert))

    model_inferred = Exponential(
        model_truth.n_stimuli, n_dim, n_group)
    model_inferred.fit(obs_all, 20, verbose=1)

    # Compare the inferred model with ground truth by comparing the
    # similarity matrices implied by each model.
    def truth_sim_func0(z_q, z_ref):
        return model_truth.similarity(
            z_q, z_ref, attention=model_truth.attention['value'][0])

    def truth_sim_func1(z_q, z_ref):
        return model_truth.similarity(
            z_q, z_ref, attention=model_truth.attention['value'][1])

    def truth_sim_func2(z_q, z_ref):
        return model_truth.similarity(
            z_q, z_ref, attention=model_truth.attention['value'][2])

    simmat_truth = (
        similarity_matrix(truth_sim_func0, model_truth.z['value']),
        similarity_matrix(truth_sim_func1, model_truth.z['value']),
        similarity_matrix(truth_sim_func2, model_truth.z['value'])
    )

    def infer_sim_func0(z_q, z_ref):
        return model_inferred.similarity(
            z_q, z_ref, attention=model_inferred.attention['value'][0])

    def infer_sim_func1(z_q, z_ref):
        return model_inferred.similarity(
            z_q, z_ref, attention=model_inferred.attention['value'][1])

    def infer_sim_func2(z_q, z_ref):
        return model_inferred.similarity(
            z_q, z_ref, attention=model_inferred.attention['value'][2])

    simmat_infer = (
        similarity_matrix(infer_sim_func0, model_inferred.z['value']),
        similarity_matrix(infer_sim_func1, model_inferred.z['value']),
        similarity_matrix(infer_sim_func2, model_inferred.z['value'])
    )
    r_squared = np.empty((n_group, n_group))
    for i_truth in range(n_group):
        for j_infer in range(n_group):
            r_squared[i_truth, j_infer] = matrix_correlation(
                simmat_truth[i_truth], simmat_infer[j_infer]
            )

    # Display comparison results. A good infferred model will have a high
    # R^2 value on the diagonal elements (max is 1) and relatively low R^2
    # values on the off-diagonal elements.
    print('\nModel Comparison (R^2)')
    print('================================')
    print('  True  |        Inferred')
    print('        | Novice  Interm  Expert')
    print('--------+-----------------------')
    print(' Novice | {0: >6.2f}  {1: >6.2f}  {2: >6.2f}'.format(
        r_squared[0, 0], r_squared[0, 1], r_squared[0, 2]))
    print(' Interm | {0: >6.2f}  {1: >6.2f}  {2: >6.2f}'.format(
        r_squared[1, 0], r_squared[1, 1], r_squared[1, 2]))
    print(' Expert | {0: >6.2f}  {1: >6.2f}  {2: >6.2f}'.format(
        r_squared[2, 0], r_squared[2, 1], r_squared[2, 2]))
    print('\n')


def ground_truth(n_stimuli, n_dim, n_group):
    """Return a ground truth embedding."""
    model = Exponential(
        n_stimuli, n_dim=n_dim, n_group=n_group)
    mean = np.ones((n_dim))
    cov = np.identity(n_dim)
    z = np.random.multivariate_normal(mean, cov, (n_stimuli))
    attention = np.array((
        (1.9, 1., .1),
        (1., 1., 1.),
        (.1, 1., 1.9)
    ))
    freeze_options = {
        'rho': 2,
        'tau': 1,
        'beta': 1,
        'gamma': 0,
        'z': z,
        'attention': attention
    }
    model.freeze(freeze_options)
    return model

if __name__ == "__main__":
    main()
