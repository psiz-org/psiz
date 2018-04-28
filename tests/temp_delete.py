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

"""Module for testing models.py."""

import numpy as np
import tensorflow as tf

from psiz.trials import UnjudgedTrials
from psiz.models import Exponential, HeavyTailed, StudentsT
from psiz.simulate import Agent
from psiz.generator import RandomGenerator, ActiveGenerator
from psiz.utils import matrix_correlation


def main():
    """Main docstring."""
    test_simulate()


def ground_truth():
    """Return a ground truth embedding."""
    n_stimuli = 10
    dimensionality = 2

    model = Exponential(n_stimuli)
    mean = np.ones((dimensionality))
    cov = np.identity(dimensionality)
    z = np.random.multivariate_normal(mean, cov, (n_stimuli))
    freeze_options = {
        'rho': 2,
        'tau': 1,
        'beta': 1,
        'gamma': 0,
        'z': z
    }
    model.freeze(freeze_options)
    return model


def get_unjudged_trials():
    """Return a set of unjudged trials."""
    stimulus_set = np.array((
        (0, 1, 2, 7, 3),
        (3, 4, 5, 9, 1),
        (1, 8, 9, 2, -1),
        (7, 3, 2, 7, -1),
        (6, 7, 5, 0, -1),
        (2, 1, 0, 6, -1),
        (3, 0, 2, 6, -1),
    ))
    n_selected = np.array((
        2, 2, 2, 2, 1, 1, 1
        ), dtype=np.int64)
    unjudged_trials = UnjudgedTrials(stimulus_set, n_selected=n_selected)
    return unjudged_trials


def test_simulate():
    """Testing."""
    model_truth = ground_truth()
    s_truth = model_truth.similarity_matrix()

    agent = Agent(model_truth)
    generator = RandomGenerator(model_truth.n_stimuli)
    n_trial = 1000
    n_reference = 8
    n_selected = 2
    trials = generator.generate(n_trial, n_reference, n_selected)
    obs = agent.simulate(trials)

    model_inferred = Exponential(model_truth.n_stimuli)
    model_inferred.fit(obs, 5, verbose=1)
    s_inferred = model_inferred.similarity_matrix()

    # Compare inferred model with ground truth.
    r_squared = matrix_correlation(s_inferred, s_truth)
    print('R^2: {0:.2f}'.format(r_squared))
    # Where should similarity matrix creation go?


if __name__ == "__main__":
    main()
