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

Example output:

    Restart Summary
    n_valid_restart 3 | total_duration: 33 s
    best | n_epoch: 65 | val_cce: 1.8201
    mean ±stddev | n_epoch: 60 ±10 | val_cce: 1.8359 ±0.0113 | 10 ±2 s | 180 ±3 ms/epoch

    R^2 Model Comparison:   0.95

"""

from pathlib import Path
import shutil

import numpy as np
import tensorflow as tf

import psiz

# Uncomment the following line to force eager execution.
# tf.config.experimental_run_functions_eagerly(True)


def main():
    """Run script."""
    # Settings.
    n_stimuli = 30
    n_dim = 3
    n_restart = 3
    batch_size = 100
    log_dir='/tmp/psiz/tensorboard_logs'

    # Ground truth embedding.
    emb_true = ground_truth(n_stimuli, n_dim)

    # Generate a random docket of trials using two different trial
    # configurations.
    # Generate 1500 2-choose-1 trials.
    gen_2c1 = psiz.generator.RandomGenerator(
        n_stimuli, n_reference=2, n_select=1
    )
    n_trial = 1500
    docket_2c1 = gen_2c1.generate(n_trial)
    # Generate 1500 8-choose-2 trials.
    gen_8c2 = psiz.generator.RandomGenerator(
        n_stimuli, n_reference=8, n_select=2
    )
    n_trial = 1500
    docket_8c2 = gen_8c2.generate(n_trial)
    # Merge both sets of trials into a single docket.
    docket = psiz.trials.stack([docket_2c1, docket_8c2])

    # Simulate similarity judgments.
    agent = psiz.simulate.Agent(emb_true)
    obs = agent.simulate(docket)

    # Partition observations into 80% train, 10% validation and 10% test set.
    obs_train, obs_val, obs_test = psiz.utils.standard_split(obs)

    # Use early stopping.
    cb_early = psiz.keras.callbacks.EarlyStoppingRe(
        'val_cce', patience=10, mode='min', restore_best_weights=True
    )
    # Visualize using TensorBoard.
    # Remove existing TensorBoard logs.
    if Path(log_dir).exists():
        shutil.rmtree(log_dir)
    cb_board = psiz.keras.callbacks.TensorBoardRe(
        log_dir=log_dir, histogram_freq=0,
        write_graph=False, write_images=False, update_freq='epoch',
        profile_batch=0, embeddings_freq=0, embeddings_metadata=None
    )
    callbacks = [cb_early, cb_board]

    compile_kwargs = {
        'loss': tf.keras.losses.CategoricalCrossentropy(),
        'optimizer': tf.keras.optimizers.Adam(lr=.001),
        'weighted_metrics': [
            tf.keras.metrics.CategoricalCrossentropy(name='cce')
        ]
    }

    # Define model.
    embedding = tf.keras.layers.Embedding(
        n_stimuli+1, n_dim, mask_zero=True
    )
    kernel = psiz.keras.layers.Kernel(
        similarity=psiz.keras.layers.ExponentialSimilarity()
    )
    model = psiz.models.Rank(embedding=embedding, kernel=kernel)
    emb_inferred = psiz.models.Proxy(model=model)

    # Infer embedding.
    restart_record = emb_inferred.fit(
        obs_train, validation_data=obs_val, epochs=1000, batch_size=batch_size,
        callbacks=callbacks, n_restart=n_restart, monitor='val_cce', verbose=1,
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
    # R^2 value on the diagonal elements (max is 1) and relatively low R^2
    # values on the off-diagonal elements.
    print(
        '\n    R^2 Model Comparison: {0: >6.2f}\n'.format(r_squared)
    )


def ground_truth(n_stimuli, n_dim):
    """Return a ground truth embedding."""
    embedding = tf.keras.layers.Embedding(
        n_stimuli+1, n_dim, mask_zero=True,
        embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=.17)
    )
    kernel = psiz.keras.layers.Kernel(
        distance=psiz.keras.layers.WeightedMinkowski(
            fit_rho=False,
            rho_initializer=tf.keras.initializers.Constant(2.),
        ),
        similarity=psiz.keras.layers.ExponentialSimilarity(
            fit_tau=False, fit_gamma=False,
            tau_initializer=tf.keras.initializers.Constant(1.),
            gamma_initializer=tf.keras.initializers.Constant(0.001),
        )
    )
    model = psiz.models.Rank(embedding=embedding, kernel=kernel)
    emb = psiz.models.Proxy(model)
    return emb


if __name__ == "__main__":
    main()
