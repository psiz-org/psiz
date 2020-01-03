# -*- coding: utf-8 -*-
# Copyright 2020 The PsiZ Authors. All Rights Reserved.
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

"""Benchmark embedding inference.

Synthetic observations are generated using a ground truth model.
"""

import json
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]= "2"

import numpy as np
import matplotlib.pyplot as plt

from psiz.models import Exponential
from psiz.simulate import Agent
from psiz.generator import RandomGenerator
from psiz.utils import similarity_matrix, matrix_comparison
from psiz.benchmark import system_info, benchmark_filename


def main():
    """Run the simulation that infers an embedding for two groups."""
    # Settings.
    n_stimuli = 1000
    n_dim = 3
    n_group = 1
    n_restart = 1
    n_trial = 100000

    sys_info = system_info()

    emb_true = ground_truth(n_stimuli, n_dim, n_group)
    simmat_true = similarity_matrix(emb_true.similarity, emb_true.z)

    # Generate a random docket of trials.
    n_reference = 8
    n_select = 2
    generator = RandomGenerator(n_reference, n_select)
    docket = generator.generate(n_trial, n_stimuli)

    # Simulate similarity judgments.
    agent = Agent(emb_true)
    obs = agent.simulate(docket)

    emb_inferred = Exponential(n_stimuli, n_dim, n_group)

    # Set parameters for consistent benchmarking.
    np.random.seed(354)
    emb_inferred.rho = np.random.uniform(1., 3.)
    emb_inferred.tau = np.random.uniform(1., 2.)
    emb_inferred.gamma = np.random.uniform(0., .001)
    emb_inferred.beta = np.random.uniform(1., 30.)

    mu = np.zeros((n_dim))
    std = np.ones((n_dim)) * 10**np.random.uniform(-3, 0)
    emb_inferred.z = np.random.normal(mu, std, (n_stimuli, n_dim))

    loss_train, loss_val = emb_inferred.fit(
        obs, n_restart=n_restart, init_mode='hot'
    )
    # Compare the inferred model with ground truth by comparing the
    # similarity matrices implied by each model.
    simmat_infer = similarity_matrix(
        emb_inferred.similarity, emb_inferred.z
    )
    r2 = matrix_comparison(simmat_infer, simmat_true, score='r2')
    print(
        '{0} trials | Loss train: {1:.2f} | '
        'Correlation (R^2): {2:.2f} | {3:.0f} s'.format(
            obs.n_trial, loss_train, r2, emb_inferred.fit_duration
        )
    )

    test_info = {}
    test_info['n_stimuli'] = n_stimuli
    test_info['n_dim'] = n_dim
    test_info['n_group'] = n_group
    test_info['n_restart'] = n_restart
    test_info['n_trial'] = obs.n_trial
    test_info['loss_train'] = str(loss_train)
    test_info['loss_val'] = str(loss_val)
    test_info['r2'] = str(r2)
    test_info['duration_fit'] = emb_inferred.fit_duration

    report = {
        'system': sys_info,
        'test': test_info
    }
    fp_report = benchmark_filename()
    with open(fp_report, 'w') as outfile:
        json.dump(report, outfile)


def ground_truth(n_stimuli, n_dim, n_group):
    """Return a ground truth embedding."""
    np.random.seed(57439)
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
