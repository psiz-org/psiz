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

"""Example that infers an embedding using variation inference.

Fake data is generated from a ground truth model for rank 8-choose-2
trial configurations.

Example output:

    R^2 Model Comparison:   0.94

"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf

import psiz

# Uncomment the following line to force eager execution.
# tf.config.experimental_run_functions_eagerly(True)


def main():
    """Run script."""
    # Settings.
    n_stimuli = 30
    n_dim = 3
    n_trial = 2000
    batch_size = 100
    n_restart = 1  # TODO

    # Ground truth embedding.
    emb_true = ground_truth(n_stimuli, n_dim)

    # Generate a random docket of 8-choose-2 trials.
    gen_8c2 = psiz.generator.RandomGenerator(
        n_stimuli, n_reference=8, n_select=2
    )
    docket = gen_8c2.generate(n_trial)

    # Simulate similarity judgments.
    agent = psiz.simulate.Agent(emb_true)
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
        'val_cce', patience=15, mode='min', restore_best_weights=True
    )
    cb_board = psiz.keras.callbacks.TensorBoardRe(
        log_dir='/tmp/psiz/tensorboard_logs', histogram_freq=0,
        write_graph=False, write_images=False, update_freq='epoch',
        profile_batch=0, embeddings_freq=0, embeddings_metadata=None
    )
    callbacks = [cb_early]

    compile_kwargs = {
        'loss': tf.keras.losses.CategoricalCrossentropy(),
        'optimizer': tf.keras.optimizers.Adam(lr=.001),
        'weighted_metrics': [
            tf.keras.metrics.CategoricalCrossentropy(name='cce')
        ]
    }

    # Define model.
    # embedding = tf.keras.layers.Embedding(n_stimuli+1, n_dim, mask_zero=True)
    kl_weight = (1.0 / obs_train.n_trial)
    embedding = psiz.keras.layers.EmbeddingVariational(
        n_stimuli+1, n_dim, mask_zero=True, kl_weight=kl_weight
    )
    kernel = psiz.keras.layers.Kernel(
        similarity=psiz.keras.layers.ExponentialSimilarityVariational(),
        distance=psiz.keras.layers.WeightedMinkowskiVariational()
    )
    model = psiz.models.Rank(embedding=embedding, kernel=kernel)
    emb_inferred = psiz.models.Proxy(model=model)

    # Infer embedding.
    restart_record = emb_inferred.fit(
        obs_train, validation_data=obs_val, epochs=10000, batch_size=batch_size,
        callbacks=callbacks, n_restart=n_restart, monitor='val_cce', verbose=2,
        compile_kwargs=compile_kwargs
    )

    # Compare the inferred model with ground truth by comparing the
    # similarity matrices implied by each model.
    simmat_truth = psiz.utils.pairwise_matrix(emb_true.similarity, emb_true.z)
    simmat_infer = psiz.utils.pairwise_matrix(
        emb_inferred.similarity, emb_inferred.z
    )
    r_squared = psiz.utils.matrix_comparison(
        simmat_truth, simmat_infer, score='r2'
    )

    # Display comparison results. A good inferred model will have a high
    # R^2 value (max is 1).
    print(
        '\n    R^2 Model Comparison: {0: >6.2f}\n'.format(r_squared)
    )
    plot_posterior(emb_inferred)


def plot_posterior(emb_inferred):
    """Plot posteriors."""
    fig, ax = plt.subplots(figsize=(12, 5))

    # Rho.
    ax = plt.subplot(1, 3, 1)
    xg = np.linspace(0, 5, 1000)
    y = emb_inferred.model.kernel.distance.rho_posterior.distribution.prob(xg)
    x_map = emb_inferred.model.kernel.distance.rho_posterior.distribution.loc.numpy()
    ax.plot(xg, y)
    ax.text(x_map, np.max(y), '{0:.2f}'.format(x_map))
    ax.set_xlabel('x')
    ax.set_ylabel('p(x)')
    ax.set_title('rho')

    # Tau.
    ax = plt.subplot(1, 3, 2)
    xg = np.linspace(0, 5, 1000)
    y = emb_inferred.model.kernel.similarity.tau_posterior.distribution.prob(xg)
    x_map = emb_inferred.model.kernel.similarity.tau_posterior.distribution.loc.numpy()
    ax.plot(xg, y)
    ax.text(x_map, np.max(y), '{0:.2f}'.format(x_map))
    ax.set_xlabel('x')
    ax.set_ylabel('p(x)')
    ax.set_title('tau')

    # Gamma.
    x_map = emb_inferred.model.kernel.similarity.gamma_posterior.distribution.loc.numpy()
    ax = plt.subplot(1, 3, 3)
    xg = np.linspace(x_map - 1, x_map + 1, 1000)
    y = emb_inferred.model.kernel.similarity.gamma_posterior.distribution.prob(xg)
    ax.plot(xg, y)
    ax.text(x_map, np.max(y), '{0:.2f}'.format(x_map))
    ax.set_xlabel('x')
    ax.set_ylabel('p(x)')
    ax.set_title('gamma')

    # Beta.
    # ax = plt.subplot(1, 4, 4)
    # xg = np.linspace(0, 5, 1000)
    # y = emb_inferred.model.kernel.similarity.rho_posterior.distribution.prob(xg)
    # ax.text(x_map, np.max(y), '{0:.2f}'.format(x_map))
    # ax.set_xlabel('x')
    # ax.set_ylabel('p(x)')
    # ax.set_title('beta')

    plt.tight_layout()
    plt.show()


def ground_truth(n_stimuli, n_dim):
    """Return a ground truth embedding."""
    embedding = tf.keras.layers.Embedding(
        n_stimuli+1, n_dim, mask_zero=True,
        embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=.17)
    )
    kernel = psiz.keras.layers.Kernel(
        similarity=psiz.keras.layers.ExponentialSimilarity()
    )
    model = psiz.models.Rank(embedding=embedding, kernel=kernel)
    emb = psiz.models.Proxy(model)

    emb.theta = {
        'rho': 2.,
        'tau': 1.,
        'beta': 10.,
        'gamma': 0.001
    }

    return emb


if __name__ == "__main__":
    main()
