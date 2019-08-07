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

"""Example that infers a shared embedding for three groups.

Fake data is generated from a ground truth model for three different
groups. In this example, these groups represent groups of agents with
varying levels of skill: novices, intermediates, and experts. Each group
has a different set of attention weights. An embedding model is
inferred from the simulated data and compared to the ground truth
model.

Example output:

    Attention weights:
          Novice | [0.81 0.20 0.26 2.72]
    Intermediate | [0.42 0.92 1.12 1.54]
          Expert | [0.08 1.80 1.80 0.32]

    Model Comparison (r^2)
    ================================
      True  |        Inferred
            | Novice  Interm  Expert
    --------+-----------------------
     Novice |   0.99    0.76    0.25
     Interm |   0.77    0.99    0.63
     Expert |   0.25    0.63    0.98

"""

import numpy as np
import matplotlib.pyplot as plt

from psiz.trials import stack
from psiz.models import Exponential
from psiz.simulate import Agent
from psiz.generator import RandomGenerator
from psiz.utils import similarity_matrix, matrix_comparison


def main():
    """Run the simulation that infers an embedding for three groups."""
    n_stimuli = 25
    n_dim = 4
    n_group = 3
    emb_true = ground_truth(n_stimuli, n_dim, n_group)

    # Generate a random docket of trials to show each group.
    n_trial = 3000
    n_reference = 8
    n_select = 2
    generator = RandomGenerator(n_reference, n_select)
    docket = generator.generate(n_trial, n_stimuli)

    # Simulate similarity judgments for the three groups.
    agent_novice = Agent(emb_true, group_id=0)
    agent_interm = Agent(emb_true, group_id=1)
    agent_expert = Agent(emb_true, group_id=2)
    obs_novice = agent_novice.simulate(docket)
    obs_interm = agent_interm.simulate(docket)
    obs_expert = agent_expert.simulate(docket)
    obs_all = stack((obs_novice, obs_interm, obs_expert))

    emb_inferred = Exponential(emb_true.n_stimuli, n_dim, n_group)
    emb_inferred.fit(obs_all, 20, verbose=1)

    # Compare the inferred model with ground truth by comparing the
    # similarity matrices implied by each model.
    def truth_sim_func0(z_q, z_ref):
        return emb_true.similarity(z_q, z_ref, group_id=0)

    def truth_sim_func1(z_q, z_ref):
        return emb_true.similarity(z_q, z_ref, group_id=1)

    def truth_sim_func2(z_q, z_ref):
        return emb_true.similarity(z_q, z_ref, group_id=2)

    simmat_truth = (
        similarity_matrix(truth_sim_func0, emb_true.z),
        similarity_matrix(truth_sim_func1, emb_true.z),
        similarity_matrix(truth_sim_func2, emb_true.z)
    )

    def infer_sim_func0(z_q, z_ref):
        return emb_inferred.similarity(z_q, z_ref, group_id=0)

    def infer_sim_func1(z_q, z_ref):
        return emb_inferred.similarity(z_q, z_ref, group_id=1)

    def infer_sim_func2(z_q, z_ref):
        return emb_inferred.similarity(z_q, z_ref, group_id=2)

    simmat_infer = (
        similarity_matrix(infer_sim_func0, emb_inferred.z),
        similarity_matrix(infer_sim_func1, emb_inferred.z),
        similarity_matrix(infer_sim_func2, emb_inferred.z)
    )
    r_squared = np.empty((n_group, n_group))
    for i_truth in range(n_group):
        for j_infer in range(n_group):
            r_squared[i_truth, j_infer] = matrix_comparison(
                simmat_truth[i_truth], simmat_infer[j_infer],
                score='r2'
            )

    # Display attention weights.
    attention_weight = emb_inferred.w
    group_labels = ["Novice", "Intermediate", "Expert"]
    print("\n    Attention weights:")
    for i_group in range(emb_inferred.n_group):
        print("    {0:>12} | {1}".format(
            group_labels[i_group],
            np.array2string(
                attention_weight[i_group, :],
                formatter={'float_kind': lambda x: "%.2f" % x})
            )
        )

    # Display comparison results. A good infferred model will have a high
    # R^2 value on the diagonal elements (max is 1) and relatively low R^2
    # values on the off-diagonal elements.
    print('\n    Model Comparison (r^2)')
    print('    ================================')
    print('      True  |        Inferred')
    print('            | Novice  Interm  Expert')
    print('    --------+-----------------------')
    print('     Novice | {0: >6.2f}  {1: >6.2f}  {2: >6.2f}'.format(
        r_squared[0, 0], r_squared[0, 1], r_squared[0, 2]))
    print('     Interm | {0: >6.2f}  {1: >6.2f}  {2: >6.2f}'.format(
        r_squared[1, 0], r_squared[1, 1], r_squared[1, 2]))
    print('     Expert | {0: >6.2f}  {1: >6.2f}  {2: >6.2f}'.format(
        r_squared[2, 0], r_squared[2, 1], r_squared[2, 2]))
    print('\n')


def ground_truth(n_stimuli, n_dim, n_group):
    """Return a ground truth embedding."""
    emb = Exponential(n_stimuli, n_dim=n_dim, n_group=n_group)
    mean = np.zeros((n_dim))
    cov = .03 * np.identity(n_dim)
    emb.z = np.random.multivariate_normal(mean, cov, (n_stimuli))
    emb.w = np.array((
        (1.8, 1.8, .2, .2),
        (1., 1., 1., 1.),
        (.2, .2, 1.8, 1.8)
    ))
    emb.rho = 2
    emb.tau = 1
    emb.beta = 10
    emb.gamma = 0.001
    emb.trainable("freeze")
    return emb


if __name__ == "__main__":
    main()
