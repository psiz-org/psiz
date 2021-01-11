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
# ============================================================================
"""Module for testing models.py."""

import os
from pathlib import Path
import pytest
import shutil
import time

import numpy as np
from scipy.stats import pearsonr
import tensorflow as tf

import psiz


# TODO a argument based fixture?
def ground_truth(n_stimuli, n_dim):
    """Return a ground truth embedding."""
    stimuli = psiz.keras.layers.Stimuli(
        embedding=tf.keras.layers.Embedding(
            n_stimuli+1, n_dim, mask_zero=True,
            embeddings_initializer=tf.keras.initializers.RandomNormal(
                stddev=.17, seed=4
            )
        )
    )
    kernel = psiz.keras.layers.Kernel(
        distance=psiz.keras.layers.WeightedMinkowski(
            rho_initializer=tf.keras.initializers.Constant(2.),
            trainable=False
        ),
        similarity=psiz.keras.layers.ExponentialSimilarity(
            fit_tau=False, fit_gamma=False,
            tau_initializer=tf.keras.initializers.Constant(1.),
            gamma_initializer=tf.keras.initializers.Constant(0.001),
        )
    )
    model = psiz.models.Rank(stimuli=stimuli, kernel=kernel)

    return model


def build_model(n_stimuli, n_dim):
    """Build model.

    Arguments:
        n_stimuli: Integer indicating the number of stimuli in the
            embedding.
        n_dim: Integer indicating the dimensionality of the embedding.

    Returns:
        model: A TensorFlow Keras model.

    """
    stimuli = psiz.keras.layers.Stimuli(
        embedding=tf.keras.layers.Embedding(
            n_stimuli+1, n_dim, mask_zero=True
        )
    )
    kernel = psiz.keras.layers.Kernel(
        similarity=psiz.keras.layers.ExponentialSimilarity()
    )
    model = psiz.models.Rank(stimuli=stimuli, kernel=kernel)
    return model


def evaluate_pairs(stimuli, kernel, ds_pairs):
    """Evaluate similarity pairs."""
    simmat_unr = []
    for x_batch in ds_pairs:
        z_0 = stimuli([x_batch[0], x_batch[2]])
        z_1 = stimuli([x_batch[1], x_batch[2]])
        simmat_unr.append(
            kernel([z_0, z_1, x_batch[2]])
        )
    simmat_unr = tf.concat(simmat_unr, axis=0)
    return simmat_unr


@pytest.mark.slow
def test_example_0_execution():
    # Settings.
    n_stimuli = 30
    n_dim = 3
    n_restart = 3
    epochs = 1000
    n_trial = 2000
    batch_size = 128
    n_frame = 2

    # Assemble dataset of stimuli pairs for comparing similarity matrices.
    ds_pairs, ds_info = psiz.utils.pairwise_index_dataset(
        n_stimuli, mask_zero=True
    )

    model_true = ground_truth(n_stimuli, n_dim)

    # Generate a random docket of trials.
    generator = psiz.generators.RandomRank(
        n_stimuli, n_reference=8, n_select=2
    )
    docket = generator.generate(n_trial)

    # Simulate similarity judgments.
    agent = psiz.agents.RankAgent(model_true)
    obs = agent.simulate(docket)

    simmat_true = evaluate_pairs(
        model_true.stimuli, model_true.kernel, ds_pairs
    ).numpy()

    # Partition observations into 80% train, 10% validation and 10% test set.
    obs_train, obs_val, obs_test = psiz.utils.standard_split(obs)
    # Convert validation and test to dataset. Convert train dataset
    # inside frame loop.
    ds_obs_val = obs_val.as_dataset().batch(
        batch_size, drop_remainder=False
    )
    ds_obs_test = obs_val.as_dataset().batch(
        batch_size, drop_remainder=False
    )

    # Use early stopping.
    early_stop = psiz.keras.callbacks.EarlyStoppingRe(
        'val_cce', patience=30, mode='min', restore_best_weights=True
    )

    compile_kwargs = {
        'loss': tf.keras.losses.CategoricalCrossentropy(),
        'optimizer': tf.keras.optimizers.Adam(lr=.001),
        'weighted_metrics': [
            tf.keras.metrics.CategoricalCrossentropy(name='cce')
        ]
    }

    # Infer independent models with increasing amounts of data.
    if n_frame == 1:
        n_obs = np.array([obs_train.n_trial], dtype=int)
    else:
        n_obs = np.round(
            np.linspace(15, obs_train.n_trial, n_frame)
        ).astype(np.int64)
    r2 = np.empty((n_frame))
    train_cce = np.empty((n_frame))
    val_cce = np.empty((n_frame))
    test_cce = np.empty((n_frame))
    for i_frame in range(n_frame):
        include_idx = np.arange(0, n_obs[i_frame])
        obs_round_train = obs_train.subset(include_idx)
        print(
            '\n  Frame {0} ({1} obs)'.format(i_frame, obs_round_train.n_trial)
        )

        # Convert obs to dataset.
        ds_obs_train = obs_round_train.as_dataset().shuffle(
            buffer_size=obs_round_train.n_trial, reshuffle_each_iteration=True
        ).batch(batch_size, drop_remainder=False)

        # Use Tensorboard callback.
        callbacks = [early_stop]

        model_inferred = build_model(n_stimuli, n_dim)
        model_inferred.compile(**compile_kwargs)

        model_inferred.fit(
            ds_obs_train, validation_data=ds_obs_val, epochs=epochs,
            callbacks=callbacks, verbose=0
        )

        # d_train = model_inferred.evaluate(ds_obs_train, return_dict=True)
        # d_val = model_inferred.evaluate(ds_obs_val, return_dict=True)
        # d_test = model_inferred.evaluate(ds_obs_test, return_dict=True)

        # Compare the inferred model with ground truth by comparing the
        # similarity matrices implied by each model.
        simmat_infer = evaluate_pairs(
            model_inferred.stimuli, model_inferred.kernel, ds_pairs
        ).numpy()
        rho, _ = pearsonr(simmat_true, simmat_infer)
        r2[i_frame] = rho**2

    # Assert that more data helps inference.
    assert r2[0] < r2[-1]

    # Assert that the last frame (the most data) has an R^2 value greater
    # than .9. This indicates that inference has found a model that closely
    # matches the ground truth (which is never directly observed).
    assert r2[-1] > .9
