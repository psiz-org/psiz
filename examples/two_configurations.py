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

"""Example that infers an embedding from two trial configurations.

Fake data is generated from a ground truth model for two different
trial configurations: 2-choose-1 and 8-choose-2. This example
demonstrates how one can use data collected in a variety of formats to
infer a single embedding.
"""

import numpy as np

from psiz.trials import stack
from psiz.models import Exponential
from psiz.simulate import Agent
from psiz.generator import RandomGenerator
from psiz.utils import similarity_matrix, matrix_comparison
import sklearn.model_selection
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping


def main():
    """Run the simulation that infers an embedding for three groups."""
    # Settings.
    n_stimuli = 25
    n_dim = 3
    n_restart = 3  # 20

    # Ground truth embedding.
    emb_true = ground_truth(n_stimuli, n_dim)

    # Generate a random docket of trials using two different trial
    # configurations.
    # Generate 1000 2-choose-1 trials.
    n_reference = 2
    n_select = 1
    gen_2c1 = RandomGenerator(
        n_stimuli, n_reference=n_reference, n_select=n_select
    )
    n_trial = 1000
    docket_2c1 = gen_2c1.generate(n_trial)
    # Generate 1000 8-choose-2 trials.
    n_reference = 8
    n_select = 2
    gen_8c2 = RandomGenerator(
        n_stimuli, n_reference=n_reference, n_select=n_select
    )
    n_trial = 1000
    docket_8c2 = gen_8c2.generate(n_trial)
    # Merge both sets of trials into a single docket.
    docket = stack([docket_2c1, docket_8c2])

    # Simulate similarity judgments for the three groups.
    agent = Agent(emb_true)
    obs = agent.simulate(docket)

    # Partition observations into train and validation set. TODO
    # sklearn.model_selection.train_test_split(

    # )
    skf = sklearn.model_selection.StratifiedKFold(n_splits=10)
    (train_idx, val_idx) = list(
        skf.split(obs.stimulus_set, obs.config_idx)
    )[0]
    obs_train = obs.subset(train_idx)
    obs_val = obs.subset(val_idx)

    # Use early stopping.
    early_stop = EarlyStopping(
        'val_loss', patience=10, mode='min', restore_best_weights=True
    )

    # Infer embedding.
    emb_inferred = Exponential(n_stimuli, n_dim)
    emb_inferred.log_freq = 10
    emb_inferred.compile()
    emb_inferred.fit(
        obs_train, obs_val=obs_val, epochs=1000, verbose=3,
        callbacks=[early_stop]
    )

    # Compare the inferred model with ground truth by comparing the
    # similarity matrices implied by each model.
    simmat_truth = similarity_matrix(emb_true.similarity, emb_true.z)
    simmat_infer = similarity_matrix(emb_inferred.similarity, emb_inferred.z)
    r_squared = matrix_comparison(simmat_truth, simmat_infer, score='r2')

    # Display comparison results. A good inferred model will have a high
    # R^2 value on the diagonal elements (max is 1) and relatively low R^2
    # values on the off-diagonal elements.
    print(
        '\n    R^2 Model Comparison: {0: >6.2f}\n'.format(r_squared)
    )


def ground_truth(n_stimuli, n_dim):
    """Return a ground truth embedding."""
    emb = Exponential(
        n_stimuli, n_dim=n_dim)
    mean = np.ones((n_dim))
    cov = .03 * np.identity(n_dim)
    z = np.random.multivariate_normal(mean, cov, (n_stimuli))
    emb.z = z
    emb.rho = 2
    emb.tau = 1
    emb.beta = 10
    emb.gamma = 0.001
    return emb


# TODO
def custom_regularizer(model):
    """Compute regularization penalty of model."""
    tf_z = model.get_layer(name='core_layer').z

    # L1 penalty on coordinates (adjusted for n_stimuli).
    l1_penalty = tf.reduce_sum(tf.abs(tf_z)) / tf_z.shape[0]
    return tf.constant(0.1, dtype=tf.keras.backend.floatx()) * l1_penalty


if __name__ == "__main__":
    main()
