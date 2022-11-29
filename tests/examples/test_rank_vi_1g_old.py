# -*- coding: utf-8 -*-
# Copyright 2022 The PsiZ Authors. All Rights Reserved.
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

import numpy as np
import pytest
from scipy.stats import pearsonr
import tensorflow as tf
import tensorflow_probability as tfp

import psiz


def ground_truth(n_stimuli, n_dim, mask_zero):
    """Return a ground truth embedding."""
    # Settings.
    scale_request = .17

    if mask_zero:
        stimuli = tf.keras.layers.Embedding(
            n_stimuli + 1, n_dim, mask_zero=True,
            embeddings_initializer=tf.keras.initializers.RandomNormal(
                stddev=scale_request, seed=58
            )
        )
    else:
        stimuli = tf.keras.layers.Embedding(
            n_stimuli, n_dim,
            embeddings_initializer=tf.keras.initializers.RandomNormal(
                stddev=scale_request, seed=58
            )
        )

    kernel = psiz.keras.layers.DistanceBased(
        distance=psiz.keras.layers.Minkowski(
            rho_initializer=tf.keras.initializers.Constant(2.),
            w_initializer=tf.keras.initializers.Constant(1.),
            trainable=False
        ),
        similarity=psiz.keras.layers.ExponentialSimilarity(
            trainable=False,
            beta_initializer=tf.keras.initializers.Constant(10.),
            tau_initializer=tf.keras.initializers.Constant(1.),
            gamma_initializer=tf.keras.initializers.Constant(0.),
        )
    )

    model = psiz.keras.models.Rank(stimuli=stimuli, kernel=kernel)

    return model


def build_model(n_stimuli, n_dim, n_group, n_obs_train, mask_zero):
    """Build model.

    Args:
        n_stimuli: Integer indicating the number of stimuli in the
            embedding.
        n_dim: Integer indicating the dimensionality of the embedding.
        n_group: Integer indicating the number of groups.
        n_obs_train: Integer indicating the number of training
            observations. Used to determine KL weight for variational
            inference.

    Returns:
        model: A TensorFlow Keras model.

    """
    kl_weight = 1. / n_obs_train

    # Note that scale of the prior can be misspecified. The true scale
    # is .17, but halving (.085) or doubling (.34) still works well. When
    # the prior scale is much smaller than appropriate and there is
    # little data, the posterior *will* be driven by the incorrect prior.
    prior_scale = .2  # Mispecified to demonstrate robustness.

    if mask_zero:
        n_stimuli_emb = n_stimuli + 1
    else:
        n_stimuli_emb = n_stimuli

    # Create variational stimuli layer.
    embedding_posterior = psiz.keras.layers.EmbeddingNormalDiag(
        n_stimuli_emb, n_dim, mask_zero=mask_zero,
        scale_initializer=tf.keras.initializers.Constant(
            tfp.math.softplus_inverse(prior_scale).numpy()
        )
    )
    embedding_prior = psiz.keras.layers.EmbeddingShared(
        n_stimuli_emb, n_dim, mask_zero=mask_zero,
        embedding=psiz.keras.layers.EmbeddingNormalDiag(
            1, 1,
            loc_initializer=tf.keras.initializers.Constant(0.),
            scale_initializer=tf.keras.initializers.Constant(
                tfp.math.softplus_inverse(prior_scale).numpy()
            ),
            loc_trainable=False,
        )
    )
    stimuli = psiz.keras.layers.EmbeddingVariational(
        posterior=embedding_posterior, prior=embedding_prior,
        kl_weight=kl_weight, kl_n_sample=30
    )

    kernel = psiz.keras.layers.DistanceBased(
        distance=psiz.keras.layers.Minkowski(
            rho_initializer=tf.keras.initializers.Constant(2.),
            w_initializer=tf.keras.initializers.Constant(1.),
            trainable=False
        ),
        similarity=psiz.keras.layers.ExponentialSimilarity(
            trainable=False,
            beta_initializer=tf.keras.initializers.Constant(10.),
            tau_initializer=tf.keras.initializers.Constant(1.),
            gamma_initializer=tf.keras.initializers.Constant(0.),
        )
    )

    model = psiz.keras.models.Rank(stimuli=stimuli, kernel=kernel, n_sample=1)
    return model


@pytest.mark.xfail(
    reason="Uses deprecated functionality."
)
@pytest.mark.slow
@pytest.mark.parametrize(
    "similarity_func", ["Exponential"]
)
@pytest.mark.parametrize(
    "mask_zero", [True]
)
def test_rank_1g_vi_execution(similarity_func, mask_zero, tmpdir):
    """A crude VI functional test that asserts more data helps."""
    # Settings.
    n_stimuli = 30
    n_dim = 2
    n_group = 1
    epochs = 1000
    n_trial = 2000
    batch_size = 128
    n_frame = 2

    # Assemble dataset of stimuli pairs for comparing similarity matrices.
    ds_pairs, _ = psiz.utils.pairwise_index_dataset(
        np.arange(n_stimuli) + 1, elements='upper'
    )

    model_true = ground_truth(n_stimuli, n_dim, mask_zero)

    # Generate a random docket of trials.
    if mask_zero:
        generator = psiz.trials.RandomRank(
            np.arange(n_stimuli) + 1, n_reference=8, n_select=2, mask_zero=True
        )
    else:
        generator = psiz.trials.RandomRank(
            n_stimuli, n_reference=8, n_select=2
        )
    docket = generator.generate(n_trial)

    # Simulate similarity judgments.
    agent = psiz.agents.RankAgent(model_true)
    obs = agent.simulate(docket)

    simmat_true = psiz.utils.pairwise_similarity(
        model_true.stimuli, model_true.kernel, ds_pairs
    ).numpy()

    # Partition observations into 80% train, 10% validation and 10% test set.
    obs_train, obs_val, obs_test = psiz.utils.standard_split(obs)
    # Convert validation and test to TF Dataset. Convert train dataset
    # inside frame loop.
    # Convert observations to TF dataset.
    ds_obs_val = obs_val.as_dataset().batch(
        batch_size, drop_remainder=False
    )
    # ds_obs_test = obs_test.as_dataset().batch(
    #     batch_size, drop_remainder=False
    # )

    # Use Tensorboard callback.
    cb_board = tf.keras.callbacks.TensorBoard(
        log_dir=tmpdir, histogram_freq=0,
        write_graph=False, write_images=False, update_freq='epoch',
        profile_batch=0, embeddings_freq=0, embeddings_metadata=None
    )
    cb_early = tf.keras.callbacks.EarlyStopping(
        'loss', patience=100, mode='min', restore_best_weights=False,
        verbose=1
    )

    compile_kwargs = {
        'loss': tf.keras.losses.CategoricalCrossentropy(),
        'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
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

    r2 = np.empty((n_frame)) * np.nan
    for i_frame in range(n_frame):
        include_idx = np.arange(0, n_obs[i_frame])
        obs_round_train = obs_train.subset(include_idx)

        # Convert obs to dataset.
        ds_obs_round_train = obs_round_train.as_dataset().shuffle(
            buffer_size=obs_round_train.n_trial, reshuffle_each_iteration=True
        ).batch(batch_size, drop_remainder=False)

        callbacks = [cb_early, cb_board]

        # Define model.
        model_inferred = build_model(
            n_stimuli, n_dim, n_group, obs_round_train.n_trial, mask_zero
        )
        model_inferred.compile(**compile_kwargs)

        # Infer embedding.
        # MAYBE keras-tuner 3 restarts, monitor='val_loss'
        model_inferred.fit(
            x=ds_obs_round_train, validation_data=ds_obs_val, epochs=epochs,
            callbacks=callbacks, verbose=0
        )

        # Compare the inferred model with ground truth by comparing the
        # similarity matrices implied by each model.
        simmat_infer = tf.reduce_mean(
            psiz.utils.pairwise_similarity(
                model_inferred.stimuli, model_inferred.kernel, ds_pairs,
                n_sample=100
            ), axis=1
        ).numpy()

        rho, _ = pearsonr(simmat_true, simmat_infer)
        if np.isnan(rho):
            rho = 0
        r2[i_frame] = rho**2

    # Assert that more data helps inference.
    assert r2[0] < r2[-1]
