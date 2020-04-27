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
from sklearn.model_selection import StratifiedKFold

import psiz.keras.callbacks
import psiz.keras.layers
from psiz.generator import RandomGenerator
import psiz.models
from psiz.simulate import Agent
from psiz.trials import stack
from psiz.utils import pairwise_matrix, matrix_comparison

# Uncomment the following line to force eager execution.
# tf.config.experimental_run_functions_eagerly(True)


def main():
    """Run script."""
    # Settings.
    n_stimuli = 25
    n_dim = 3
    n_restart = 20

    # Ground truth embedding.
    emb_true = ground_truth(n_stimuli, n_dim)

    # Generate a random docket of trials using two different trial
    # configurations.
    # Generate 1500 2-choose-1 trials.
    n_reference = 2
    n_select = 1
    gen_2c1 = RandomGenerator(
        n_stimuli, n_reference=n_reference, n_select=n_select
    )
    n_trial = 1500
    docket_2c1 = gen_2c1.generate(n_trial)
    # Generate 1500 8-choose-2 trials.
    n_reference = 8
    n_select = 2
    gen_8c2 = RandomGenerator(
        n_stimuli, n_reference=n_reference, n_select=n_select
    )
    n_trial = 1500
    docket_8c2 = gen_8c2.generate(n_trial)
    # Merge both sets of trials into a single docket.
    docket = stack([docket_2c1, docket_8c2])

    # Simulate similarity judgments for the three groups.
    agent = Agent(emb_true)
    obs = agent.simulate(docket)

    # Partition observations into train and validation set.
    skf = StratifiedKFold(n_splits=10)
    (train_idx, val_idx) = list(
        skf.split(obs.stimulus_set, obs.config_idx)
    )[0]
    obs_train = obs.subset(train_idx)
    obs_val = obs.subset(val_idx)

    # Use early stopping.
    cb_early = psiz.keras.callbacks.EarlyStoppingRe(
        'val_loss', patience=10, mode='min', restore_best_weights=True
    )
    # Visualize using TensorBoard.
    cb_board = psiz.keras.callbacks.TensorBoardRe(
        log_dir='/tmp/psiz/tensorboard_logs', histogram_freq=0,
        write_graph=False, write_images=False, update_freq='epoch',
        profile_batch=0, embeddings_freq=0, embeddings_metadata=None
    )

    # Infer embedding.
    embedding = psiz.keras.layers.EmbeddingRe(n_stimuli, n_dim=n_dim)
    kernel = psiz.keras.layers.ExponentialKernel()
    rankModel = psiz.models.Rank(
        embedding=embedding, kernel=kernel
    )
    emb_inferred = psiz.models.Proxy(model=rankModel)
    emb_inferred.compile()
    restart_record = emb_inferred.fit(
        obs_train, obs_val=obs_val, epochs=1000, verbose=2,
        callbacks=[cb_early, cb_board], n_restart=n_restart,
        monitor='val_loss'
    )

    # Compare the inferred model with ground truth by comparing the
    # similarity matrices implied by each model.
    simmat_truth = pairwise_matrix(emb_true.similarity, emb_true.z)
    simmat_infer = pairwise_matrix(emb_inferred.similarity, emb_inferred.z)
    r_squared = matrix_comparison(simmat_truth, simmat_infer, score='r2')

    # Display comparison results. A good inferred model will have a high
    # R^2 value on the diagonal elements (max is 1) and relatively low R^2
    # values on the off-diagonal elements.
    print(
        '\n    R^2 Model Comparison: {0: >6.2f}\n'.format(r_squared)
    )


def ground_truth(n_stimuli, n_dim):
    """Return a ground truth embedding."""
    kernel = psiz.keras.layers.ExponentialKernel()

    embedding = psiz.keras.layers.EmbeddingRe(n_stimuli, n_dim=n_dim)
    rankModel = psiz.models.Rank(embedding=embedding, kernel=kernel)

    emb = psiz.models.Proxy(rankModel)

    mean = np.zeros((n_dim))
    cov = .03 * np.identity(n_dim)
    z = np.random.multivariate_normal(mean, cov, (n_stimuli))
    emb.z = z

    emb.theta = {
        'rho': 2.,
        'tau': 1.,
        'beta': 10.,
        'gamma': 0.001
    }

    return emb


if __name__ == "__main__":
    main()
