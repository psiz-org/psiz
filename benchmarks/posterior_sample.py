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

"""Benchmark sampling from posterior distribution.

Synthetic observations are generated using a ground truth model.
"""

import json
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]= "2"
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
from psiz.benchmark import system_info, benchmark_filename


def main():
    """Run the simulation."""
    # Settings.
    n_dim = 3
    n_stimuli = 1000
    n_trial = 10000
    n_group = 1
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
    # Prime.
    samples = emb_true.posterior_samples(obs, verbose=1, n_final_sample=10)
    # Do actual test.
    samples = emb_true.posterior_samples(obs, verbose=1)
    best_time = emb_true.posterior_duration
    print('{0:.0f} s'.format(best_time))

    test_info = {}
    test_info['n_dim'] = n_dim
    test_info['n_trial'] = n_trial
    test_info['n_reference'] = n_reference
    test_info['n_select'] = n_select
    test_info['duration'] = best_time

    sys_info = system_info()

    report = {
        'test': test_info,
        'system': sys_info
    }
    fp_report = benchmark_filename()
    with open(fp_report, 'w') as outfile:
        json.dump(report, outfile)


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
