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

"""Example that infers an embedding with an increasing amount of data.

Fake data is generated from a ground truth model assuming one group.
An embedding is inferred with an increasing amount of data,
demonstrating how the inferred model improves and asymptotes as more
data is added.

"""

import numpy as np
import matplotlib.pyplot as plt

from psiz.generator import RandomGenerator
import psiz.models
import psiz.restart
from psiz.simulate import Agent
from psiz.utils import pairwise_matrix, matrix_comparison
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import EarlyStopping


def main():
    """Run script."""
    # Settings.
    n_stimuli = 25
    n_dim = 3
    n_restart = 10

    emb_true = ground_truth(n_stimuli, n_dim)
    simmat_true = pairwise_matrix(emb_true.similarity, emb_true.z)

    # Generate a random docket of trials.
    n_trial = 1000
    n_reference = 8
    n_select = 2
    generator = RandomGenerator(
        n_stimuli, n_reference=n_reference, n_select=n_select
    )
    docket = generator.generate(n_trial)

    # Simulate similarity judgments.
    agent = Agent(emb_true)
    obs = agent.simulate(docket)

    # Partition observations into train and test set.
    skf = StratifiedKFold(n_splits=10)
    (train_idx, test_idx) = list(
        skf.split(obs.stimulus_set, obs.config_idx)
    )[0]
    obs_train = obs.subset(train_idx)
    obs_test = obs.subset(test_idx)

    # Use early stopping.
    early_stop = EarlyStopping(
        'val_loss', patience=10, mode='min', restore_best_weights=True
    )

    # Infer independent models with increasing amounts of data.
    n_step = 8
    n_obs = np.floor(np.linspace(15, obs_train.n_trial, n_step)).astype(np.int64)
    r2 = np.empty((n_step))
    train_loss = np.empty((n_step))
    val_loss = np.empty((n_step))
    test_loss = np.empty((n_step))
    for i_round in range(n_step):
        include_idx = np.arange(0, n_obs[i_round])
        obs_round = obs_train.subset(include_idx)

        # Partition training observations into train and validation set.
        skf = StratifiedKFold(n_splits=10)
        (train_idx, val_idx) = list(
            skf.split(obs_round.stimulus_set, obs_round.config_idx)
        )[0]
        obs_round_train = obs_round.subset(train_idx)
        obs_round_val = obs_round.subset(val_idx)

        # Infer embedding.
        kernel = psiz.models.ExponentialKernel()
        emb_inferred = psiz.models.AnchoredOrdinal(
            n_stimuli, n_dim=n_dim, kernel=kernel
        )
        emb_inferred.compile()
        restarter = psiz.restart.Restarter(
            emb_inferred, 'val_loss', n_restart=n_restart
        )
        restart_record = restarter.fit(
            obs_round_train, obs_val=obs_round_val, epochs=1000, verbose=1,
            callbacks=[early_stop]
        )

        train_loss[i_round] = restart_record.record['train_loss'][0]
        val_loss[i_round] = restart_record.record['val_loss'][0]
        test_loss[i_round] = emb_inferred.evaluate(obs_test)

        # Compare the inferred model with ground truth by comparing the
        # similarity matrices implied by each model.
        simmat_infer = pairwise_matrix(
            emb_inferred.similarity, emb_inferred.z
        )
        r2[i_round] = matrix_comparison(
            simmat_infer, simmat_true, score='r2'
        )
        print(
            '  Round {0} | n_obs: {1:4d} | train_loss: {2:.2f} | '
            'val_loss: {3:.2f} | test_loss: {4:.2f} | '
            'Correlation (R^2): {5:.2f}'.format(
                i_round, n_obs[i_round], train_loss[i_round],
                val_loss[i_round], test_loss[i_round], r2[i_round]
            )
        )

    # Plot comparison results.
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3))

    axes[0].plot(n_obs, train_loss, 'bo-', label='Train Loss')
    axes[0].plot(n_obs, val_loss, 'go-', label='Val. Loss')
    axes[0].plot(n_obs, test_loss, 'ro-', label='Test Loss')
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Number of Judged Trials')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    axes[1].plot(n_obs, r2, 'ro-')
    axes[1].set_title('Model Convergence to Ground Truth')
    axes[1].set_xlabel('Number of Judged Trials')
    axes[1].set_ylabel(r'Squared Pearson Correlation ($R^2$)')
    axes[1].set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.show()


def ground_truth(n_stimuli, n_dim):
    """Return a ground truth embedding."""
    kernel = psiz.models.ExponentialKernel()
    kernel.rho = 2.
    kernel.tau = 1.
    kernel.beta = 10.
    kernel.gamma = 0.001

    emb = psiz.models.AnchoredOrdinal(
        n_stimuli, n_dim=n_dim, kernel=kernel
    )

    mean = np.zeros((n_dim))
    cov = .03 * np.identity(n_dim)
    z = np.random.multivariate_normal(mean, cov, (n_stimuli))
    emb.z = z

    return emb


if __name__ == "__main__":
    main()
