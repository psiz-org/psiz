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
import time

import numpy as np
import matplotlib.pyplot as plt

from psiz.models import _mink_distance
from psiz.simulate import Agent
from psiz.generator import RandomGenerator
from psiz.utils import similarity_matrix, matrix_comparison
from psiz.benchmark import system_info, benchmark_filename


def main():
    """Run the simulation that infers an embedding for two groups."""
    # Settings.
    n_dim = 10
    # n_trial > 100000 sees advantage
    # n_trial = 100000
    n_trial = 1000000
    # n_trial = 10000000
    n_ref = 8
    rho = 2.0

    z_q = np.random.randn(n_trial, n_dim, n_ref)
    z_r = np.random.randn(n_trial, n_dim, n_ref)
    attention = np.ones([n_trial, n_dim, 1])

    n_restart = 20
    best_time = np.inf
    for i_restart in range(n_restart):
        start_time = time.time()
        d = _mink_distance(z_q, z_r, rho, attention)
        duration = time.time() - start_time
        if duration < best_time:
            best_time = duration

    sys_info = system_info()

    test_info = {}
    test_info['n_dim'] = n_dim
    test_info['n_trial'] = n_trial
    test_info['n_ref'] = n_ref
    test_info['n_restart'] = n_restart
    test_info['duration'] = best_time

    report = {
        'test': test_info,
        'system': sys_info
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
