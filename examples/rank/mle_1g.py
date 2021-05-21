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
"""Example that infers an embedding with an increasing amount of data.

Fake data is generated from a ground truth model assuming one group.
An embedding is inferred with an increasing amount of data,
demonstrating how the inferred model improves and asymptotes as more
data is added.

Results are saved in the directory specified by `fp_example`. By
default, a `psiz_examples` directory is created in your home directory.

"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # noqa
from pathlib import Path
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
import tensorflow as tf

import psiz

# Uncomment the following line to force eager execution.
# tf.config.run_functions_eagerly(True)

# Uncomment and edit the following to control GPU visibility.
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    """Run script."""
    # Settings.
    fp_example = Path.home() / Path('psiz_examples', 'rank', 'mle_1g')
    fp_board = fp_example / Path('logs', 'fit')
    n_stimuli = 30
    n_dim = 3
    n_restart = 3
    epochs = 1000
    n_trial = 2000
    batch_size = 100
    n_frame = 1  # Set to 8 to observe convergence behavior.

    # Directory preparation.
    fp_example.mkdir(parents=True, exist_ok=True)
    # Remove existing TensorBoard logs.
    if fp_board.exists():
        shutil.rmtree(fp_board)

    # Assemble dataset of stimuli pairs for comparing similarity matrices.
    ds_pairs, ds_info = psiz.utils.pairwise_index_dataset(
        n_stimuli, mask_zero=True
    )

    model_true = ground_truth(n_stimuli, n_dim)

    simmat_true = psiz.utils.pairwise_similarity(
        model_true.stimuli, model_true.kernel, ds_pairs
    ).numpy()

    # Generate a random docket of trials.
    generator = psiz.trials.RandomRank(
        n_stimuli, n_reference=8, n_select=2
    )
    docket = generator.generate(n_trial)

    # Simulate similarity judgments.
    agent = psiz.agents.RankAgent(model_true)
    obs = agent.simulate(docket)

    # Partition observations into 80% train, 10% validation and 10% test set.
    obs_train, obs_val, obs_test = psiz.utils.standard_split(obs)
    # Convert validation and test to dataset. Convert train dataset
    # inside frame loop.
    ds_obs_val = obs_val.as_dataset().batch(
        batch_size, drop_remainder=False
    )
    ds_obs_test = obs_test.as_dataset().batch(
        batch_size, drop_remainder=False
    )

    # Use early stopping.
    early_stop = psiz.keras.callbacks.EarlyStoppingRe(
        'val_cce', patience=30, mode='min', restore_best_weights=True
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
    r2 = np.empty((n_frame))
    train_cce = np.empty((n_frame))
    val_cce = np.empty((n_frame))
    test_cce = np.empty((n_frame))
    for i_frame in range(n_frame):
        include_idx = np.arange(0, n_obs[i_frame])
        obs_round_train = obs_train.subset(include_idx)
        ds_obs_round_train = obs_round_train.as_dataset().shuffle(
            buffer_size=obs_round_train.n_trial, reshuffle_each_iteration=True
        ).batch(batch_size, drop_remainder=False)

        print(
            '\n  Frame {0} ({1} obs)'.format(i_frame, obs_round_train.n_trial)
        )

        # Use Tensorboard callback.
        fp_board_frame = fp_board / Path('frame_{0}'.format(i_frame))
        cb_board = psiz.keras.callbacks.TensorBoardRe(
            log_dir=fp_board_frame, histogram_freq=0,
            write_graph=False, write_images=False, update_freq='epoch',
            profile_batch=0, embeddings_freq=0, embeddings_metadata=None
        )
        callbacks = [early_stop, cb_board]

        model = build_model(n_stimuli, n_dim)

        # Infer embedding with restarts.
        restarter = psiz.keras.Restarter(
            model, compile_kwargs=compile_kwargs, monitor='val_loss',
            n_restart=n_restart
        )
        restart_record = restarter.fit(
            x=ds_obs_round_train, validation_data=ds_obs_val, epochs=epochs,
            callbacks=callbacks, verbose=0
        )
        model = restarter.model

        train_cce[i_frame] = restart_record.record['cce'][0]
        val_cce[i_frame] = restart_record.record['val_cce'][0]
        test_metrics = model.evaluate(
            ds_obs_test, verbose=0, return_dict=True
        )
        test_cce[i_frame] = test_metrics['cce']

        # Compare the inferred model with ground truth by comparing the
        # similarity matrices implied by each model.
        simmat_infer = psiz.utils.pairwise_similarity(
            model.stimuli, model.kernel, ds_pairs
        ).numpy()
        rho, _ = pearsonr(simmat_true, simmat_infer)
        r2[i_frame] = rho**2

        print(
            '    n_obs: {0:4d} | train_cce: {1:.2f} | '
            'val_cce: {2:.2f} | test_cce: {3:.2f} | '
            'Correlation (R^2): {4:.2f}'.format(
                n_obs[i_frame], train_cce[i_frame],
                val_cce[i_frame], test_cce[i_frame], r2[i_frame]
            )
        )

    # Plot comparison results.
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3))

    axes[0].plot(n_obs, train_cce, 'bo-', label='Train CCE')
    axes[0].plot(n_obs, val_cce, 'go-', label='Val. CCE')
    axes[0].plot(n_obs, test_cce, 'ro-', label='Test CCE')
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
    fname = fp_example / Path('evolution.tiff')
    plt.savefig(
        os.fspath(fname), format='tiff', bbox_inches="tight", dpi=300
    )


def ground_truth(n_stimuli, n_dim):
    """Return a ground truth embedding."""
    stimuli = tf.keras.layers.Embedding(
        n_stimuli+1, n_dim, mask_zero=True,
        embeddings_initializer=tf.keras.initializers.RandomNormal(
            stddev=.17
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
            gamma_initializer=tf.keras.initializers.Constant(0.001),
        )
    )

    model = psiz.keras.models.Rank(stimuli=stimuli, kernel=kernel)

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
    # Create a group-agnostic stimuli layer.
    stimuli = tf.keras.layers.Embedding(
        n_stimuli+1, n_dim, mask_zero=True
    )

    # Create a group-agnostic kernel.
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
            gamma_initializer=tf.keras.initializers.Constant(0.001),
        )
    )

    model = psiz.keras.models.Rank(stimuli=stimuli, kernel=kernel)
    return model


if __name__ == "__main__":
    start_time_s = time.time()
    main()
    total_time_s = time.time() - start_time_s
    print('Total script time: {0:.0f} s'.format(total_time_s))
