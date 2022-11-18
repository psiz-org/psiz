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
from psiz.utils import choice_wo_replace


class BehaviorModel(tf.keras.Model):
    """A behavior model.

    No Gates.

    """

    def __init__(self, behavior=None, **kwargs):
        """Initialize."""
        super(BehaviorModel, self).__init__(**kwargs)
        self.behavior = behavior

    def call(self, inputs):
        """Call."""
        return self.behavior(inputs)


def build_ground_truth_model(n_stimuli, n_dim, similarity_func, mask_zero):
    """Return a ground truth embedding."""
    if mask_zero:
        n_stimuli_emb = n_stimuli + 1
    else:
        n_stimuli_emb = n_stimuli

    percept = tf.keras.layers.Embedding(
        n_stimuli_emb, n_dim, mask_zero=mask_zero,
        embeddings_initializer=tf.keras.initializers.RandomNormal(
            stddev=.17, seed=4
        )
    )

    # Set similarity function.
    if similarity_func == 'Exponential':
        similarity = psiz.keras.layers.ExponentialSimilarity(
            fit_tau=False, fit_gamma=False, fit_beta=False,
            tau_initializer=tf.keras.initializers.Constant(1.),
            gamma_initializer=tf.keras.initializers.Constant(0.001),
        )
    elif similarity_func == 'StudentsT':
        similarity = psiz.keras.layers.StudentsTSimilarity(
            fit_tau=False, fit_alpha=False,
            tau_initializer=tf.keras.initializers.Constant(2.),
            alpha_initializer=tf.keras.initializers.Constant(1.),
        )
    elif similarity_func == 'HeavyTailed':
        similarity = psiz.keras.layers.HeavyTailedSimilarity(
            fit_tau=False, fit_kappa=False, fit_alpha=False,
            tau_initializer=tf.keras.initializers.Constant(2.),
            kappa_initializer=tf.keras.initializers.Constant(2.),
            alpha_initializer=tf.keras.initializers.Constant(10.),
        )
    elif similarity_func == "Inverse":
        similarity = psiz.keras.layers.InverseSimilarity(
            fit_tau=False, fit_mu=False,
            tau_initializer=tf.keras.initializers.Constant(2.),
            mu_initializer=tf.keras.initializers.Constant(0.000001)
        )

    kernel = psiz.keras.layers.DistanceBased(
        distance=psiz.keras.layers.Minkowski(
            rho_initializer=tf.keras.initializers.Constant(2.),
            w_initializer=tf.keras.initializers.Constant(1.),
            trainable=False
        ),
        similarity=similarity
    )

    rank = psiz.keras.layers.RankSimilarity(
        n_reference=8, n_select=2, percept=percept, kernel=kernel
    )

    model = BehaviorModel(behavior=rank)

    compile_kwargs = {
        'loss': tf.keras.losses.CategoricalCrossentropy(),
        'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
        'weighted_metrics': [
            tf.keras.metrics.CategoricalCrossentropy(name='cce')
        ]
    }
    model.compile(**compile_kwargs)
    return model


def build_model(n_stimuli, n_dim, similarity_func, mask_zero):
    """Build model.

    Args:
        n_stimuli: Integer indicating the number of stimuli in the
            embedding.
        n_dim: Integer indicating the dimensionality of the embedding.

    Returns:
        model: A TensorFlow Keras model.

    """
    if mask_zero:
        n_stimuli_emb = n_stimuli + 1
    else:
        n_stimuli_emb = n_stimuli

    percept = tf.keras.layers.Embedding(
        n_stimuli_emb, n_dim, mask_zero=mask_zero
    )

    # Set similarity function.
    if similarity_func == 'Exponential':
        similarity = psiz.keras.layers.ExponentialSimilarity(
            beta_initializer=tf.keras.initializers.Constant(10.),
            tau_initializer=tf.keras.initializers.Constant(1.),
            gamma_initializer=tf.keras.initializers.Constant(0.001),
            fit_beta=False,
            fit_tau=False,
        )
    elif similarity_func == 'StudentsT':
        similarity = psiz.keras.layers.StudentsTSimilarity()
    elif similarity_func == 'HeavyTailed':
        similarity = psiz.keras.layers.HeavyTailedSimilarity()
    elif similarity_func == "Inverse":
        similarity = psiz.keras.layers.InverseSimilarity()

    kernel = psiz.keras.layers.DistanceBased(
        distance=psiz.keras.layers.Minkowski(),
        similarity=similarity
    )
    rank = psiz.keras.layers.RankSimilarity(
        n_reference=8, n_select=2, percept=percept, kernel=kernel
    )

    model = BehaviorModel(behavior=rank)
    compile_kwargs = {
        'loss': tf.keras.losses.CategoricalCrossentropy(),
        'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
        'weighted_metrics': [
            tf.keras.metrics.CategoricalCrossentropy(name='cce')
        ]
    }
    model.compile(**compile_kwargs)
    return model


# TODO The coordinate space may need to be scaled so that it is
# "learnable" by the other similarity functions.
# TODO ideally use `tf.keras.utils.split_dataset`, but it's brittle.
@pytest.mark.slow
@pytest.mark.parametrize(
    "similarity_func", ["Exponential"]
)
@pytest.mark.parametrize(
    "mask_zero", [True]
)
@pytest.mark.parametrize(
    "is_eager", [True]
)
def test_rank_1g_mle_execution(similarity_func, mask_zero, tmpdir, is_eager):
    """A crude MLE functional test that asserts more data helps."""
    tf.config.run_functions_eagerly(is_eager)

    # Settings.
    n_stimuli = 30
    n_dim = 3
    epochs = 1000
    batch_size = 128
    n_trial_train = batch_size * 10
    n_trial = batch_size * 12
    n_frame = 2

    # Assemble dataset of stimuli pairs for comparing similarity matrices.
    if mask_zero:
        ds_pairs, _ = psiz.data.pairwise_index_dataset(
            np.arange(n_stimuli) + 1, elements='upper'
        )
    else:
        ds_pairs, _ = psiz.data.pairwise_index_dataset(
            n_stimuli, elements='upper'
        )

    model_true = build_ground_truth_model(
        n_stimuli, n_dim, similarity_func, mask_zero
    )

    # Generate a random docket of trials.
    rng = np.random.default_rng()
    if mask_zero:
        eligibile_indices = np.arange(n_stimuli) + 1
    else:
        eligibile_indices = np.arange(n_stimuli)
    p = np.ones_like(eligibile_indices) / len(eligibile_indices)
    stimulus_set = choice_wo_replace(
        eligibile_indices, (n_trial, 9), p, rng=rng
    )
    content = psiz.data.Rank(stimulus_set, n_select=2)
    pds = psiz.data.Dataset([content])
    ds_content = pds.export(export_format='tfds')

    # Simulate similarity judgments.
    def simulate_agent(x):
        depth = content.n_outcome
        outcome_probs = model_true(x)
        outcome_distribution = tfp.distributions.Categorical(
            probs=outcome_probs
        )
        outcome_idx = outcome_distribution.sample()
        outcome_one_hot = tf.one_hot(outcome_idx, depth)
        return outcome_one_hot

    ds = ds_content.map(lambda x: (x, simulate_agent(x))).cache()

    simmat_true = np.squeeze(
        psiz.utils.pairwise_similarity(
            model_true.behavior.percept, model_true.behavior.kernel, ds_pairs
        ).numpy()
    )

    # Partition observations into train and validation/test set.
    # ds_content_train, ds_content_val = tf.keras.utils.split_dataset(
    #     ds_content, left_size=n_trial_train, shuffle=False
    # )
    # ds_outcome_train, ds_outcome_val = tf.keras.utils.split_dataset(
    #     ds_outcome, left_size=n_trial_train, shuffle=False
    # )

    # Combine and batch val, since there are no more transformations.
    # ds_val = tf.data.Dataset.zip((ds_content_val, ds_outcome_val))
    # ds_val = ds_val.batch(batch_size, drop_remainder=False)
    ds_train = ds.take(n_trial_train)
    ds_val = ds.skip(n_trial_train).cache().batch(
        batch_size, drop_remainder=False
    )

    # Use early stopping.
    early_stop = tf.keras.callbacks.EarlyStopping(
        'val_cce', patience=30, mode='min', restore_best_weights=True
    )
    cb_board = tf.keras.callbacks.TensorBoard(
        log_dir=tmpdir, histogram_freq=0,
        write_graph=False, write_images=False, update_freq='epoch',
        profile_batch=0, embeddings_freq=0, embeddings_metadata=None
    )

    # Infer independent models with increasing amounts of data.
    if n_frame == 1:
        n_trial_train_frame = np.array([n_trial_train], dtype=int)
    else:
        n_trial_train_frame = np.round(
            np.linspace(15, n_trial_train, n_frame)
        ).astype(np.int64)

    r2 = np.empty((n_frame)) * np.nan
    for i_frame in range(n_frame):
        # Prepare `ds_train_sub` for training.
        # ds_content_train_sub, _ = tf.keras.utils.split_dataset(
        #     ds_content_train, left_size=int(n_trial_train_frame[i_frame])
        # )
        # ds_outcome_train_sub, _ = tf.keras.utils.split_dataset(
        #     ds_outcome_train, left_size=int(n_trial_train_frame[i_frame])
        # )
        # ds_train_sub = tf.data.Dataset.zip(
        #     (ds_content_train_sub, ds_outcome_train_sub)
        # ).cache().shuffle(
        #     buffer_size=n_trial_train_frame[i_frame], reshuffle_each_iteration=True
        # ).batch(
        #     batch_size, drop_remainder=False
        # )
        ds_train_sub = ds_train.take(int(n_trial_train_frame[i_frame])).cache().shuffle(
            buffer_size=n_trial_train_frame[i_frame], reshuffle_each_iteration=True
        ).batch(
            batch_size, drop_remainder=False
        )

        # Use Tensorboard callback.
        callbacks = [early_stop, cb_board]

        model_inferred = build_model(
            n_stimuli, n_dim, similarity_func, mask_zero
        )

        # Infer embedding.
        # MAYBE keras-tuner 3 restarts, monitor='val_loss'
        model_inferred.fit(
            x=ds_train_sub, validation_data=ds_val, epochs=epochs,
            callbacks=callbacks, verbose=0
        )

        # Compare the inferred model with ground truth by comparing the
        # similarity matrices implied by each model.
        simmat_infer = np.squeeze(
            psiz.utils.pairwise_similarity(
                model_inferred.behavior.percept,
                model_inferred.behavior.kernel, ds_pairs
            ).numpy()
        )
        rho, _ = pearsonr(simmat_true, simmat_infer)
        if np.isnan(rho):
            rho = 0
        r2[i_frame] = rho**2

    # Assert that more data helps inference.
    assert r2[-1] > r2[0]

    # Strong test: Assert that the last frame (the most data) has an R^2 value
    # greater than 0.9. This indicates that inference has found a model that
    # closely matches the ground truth (which is never directly observed).
    # assert r2[-1] > .9
