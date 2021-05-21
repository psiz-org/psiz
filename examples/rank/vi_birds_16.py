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
"""Example that infers a 2D embedding using real human similarity data.

Similarity judgment data comes from an experiment using images
from 16 bird species, with 13 images per species (208 total).

Results are saved in the directory specified by `fp_example`. By
default, the beginning of this path is `~/psiz_examples` where `~`
is determined by `Path.home()`.

"""

import copy
import os
from pathlib import Path
import shutil

import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import psiz

# Uncomment the following line to force eager execution.
# tf.config.run_functions_eagerly(True)

# Uncomment and edit the following to control GPU visibility.
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    """Run script."""
    # Settings.
    fp_example = Path.home() / Path('psiz_examples', 'rank', 'vi_birds_16')
    fp_board = fp_example / Path('logs', 'fit', 'r0')
    n_dim = 2
    n_restart = 1
    epochs = 1000
    batch_size = 128

    # Directory preparation.
    fp_example.mkdir(parents=True, exist_ok=True)
    # Remove existing TensorBoard logs.
    if fp_board.exists():
        shutil.rmtree(fp_board)

    # Plot settings.
    small_size = 6
    medium_size = 8
    large_size = 10
    plt.rc('font', size=small_size)
    plt.rc('axes', titlesize=medium_size)
    plt.rc('axes', labelsize=small_size)
    plt.rc('xtick', labelsize=small_size)
    plt.rc('ytick', labelsize=small_size)
    plt.rc('legend', fontsize=small_size)
    plt.rc('figure', titlesize=large_size)

    # Import hosted rank dataset of 16 bird species.
    (obs, catalog) = psiz.datasets.load('birds-16', verbose=1)

    # Partition observations into 80% train, 10% validation and 10% test set.
    obs_train, obs_val, obs_test = psiz.utils.standard_split(obs)
    print(
        '\nData Split\n  obs_train:'
        ' {0}\n  obs_val: {1}\n  obs_test: {2}'.format(
            obs_train.n_trial, obs_val.n_trial, obs_test.n_trial
        )
    )

    # Convert observations to TF dataset.
    ds_obs_train = obs_train.as_dataset().shuffle(
        buffer_size=obs_train.n_trial, reshuffle_each_iteration=True
    ).batch(batch_size, drop_remainder=False)
    ds_obs_val = obs_val.as_dataset().batch(
        batch_size, drop_remainder=False
    )
    ds_obs_test = obs_test.as_dataset().batch(
        batch_size, drop_remainder=False
    )

    # Build VI model.
    model = build_model(catalog.n_stimuli, n_dim, obs_train.n_trial)

    # Compile settings.
    compile_kwargs = {
        'loss': tf.keras.losses.CategoricalCrossentropy(),
        'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
        'weighted_metrics': [
            tf.keras.metrics.CategoricalCrossentropy(name='cce')
        ]
    }

    # Define callbacks.
    cb_board = psiz.keras.callbacks.TensorBoardRe(
        log_dir=fp_board, histogram_freq=0,
        write_graph=False, write_images=False, update_freq='epoch',
        profile_batch=0, embeddings_freq=0, embeddings_metadata=None
    )
    cb_early = psiz.keras.callbacks.EarlyStoppingRe(
        'loss', patience=100, mode='min', restore_best_weights=False,
        verbose=1
    )
    callbacks = [cb_board, cb_early]

    # Infer embedding with restarts.
    restarter = psiz.keras.Restarter(
        model, compile_kwargs=compile_kwargs, monitor='val_loss',
        n_restart=n_restart
    )
    restart_record = restarter.fit(
        x=ds_obs_train, validation_data=ds_obs_val, epochs=epochs,
        callbacks=callbacks, verbose=0
    )
    model = restarter.model

    train_loss = restart_record.record['loss'][0]
    train_time = restart_record.record['ms_per_epoch'][0]
    val_loss = restart_record.record['val_loss'][0]

    # Evaluate test set by taking multiple samples.
    tf.keras.backend.clear_session()
    model.n_sample = 100
    model.compile(**compile_kwargs)
    test_metrics = model.evaluate(ds_obs_test, verbose=0, return_dict=True)
    test_loss = test_metrics['loss']
    print(
        '    train_loss: {0:.2f} | val_loss: {1:.2f} | '
        'test_loss: {2:.2f} | '.format(train_loss, val_loss, test_loss)
    )

    # Create visual.
    fig = plt.figure(figsize=(6.5, 4), dpi=200)
    draw_figure(
        fig, model, catalog
    )
    fname = fp_example / Path('visual.tiff')
    plt.savefig(
        os.fspath(fname), format='tiff', bbox_inches="tight", dpi=300
    )


def build_model(n_stimuli, n_dim, n_obs_train):
    """Build model.

    Arguments:
        n_stimuli: Integer indicating the number of stimuli in the
            embedding.
        n_dim: Integer indicating the dimensionality of the embedding.
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
    # little data, the posterior will be driven by an incorrect prior.
    prior_scale = .2  # Mispecified to demonstrate robustness.
    embedding_posterior = psiz.keras.layers.EmbeddingNormalDiag(
        n_stimuli+1, n_dim, mask_zero=True,
        scale_initializer=tf.keras.initializers.Constant(
            tfp.math.softplus_inverse(prior_scale).numpy()
        )
    )
    embedding_prior = psiz.keras.layers.EmbeddingShared(
        n_stimuli+1, n_dim, mask_zero=True,
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


def draw_figure(fig, model, catalog):
    """Draw figure."""
    # Settings
    s = 5
    lw = .5
    alpha = .5
    gs = fig.add_gridspec(1, 1)

    class_arr = catalog.stimuli.class_id.values
    unique_class_arr = np.unique(class_arr)

    # Define one color per class for plots.
    n_class = len(unique_class_arr)
    cmap = matplotlib.cm.get_cmap('jet')
    norm = matplotlib.colors.Normalize(vmin=0., vmax=n_class)
    class_color_array = cmap(norm(range(n_class)))

    # Plot embeddings.
    ax = fig.add_subplot(gs[0, 0])

    # Determine embedding limits.
    dist = model.stimuli.embeddings
    loc, cov = unpack_mvn(dist)
    if model.stimuli.mask_zero:
        # Drop placeholder stimulus.
        loc = loc[1:]
        cov = cov[1:]

    z_max = 1.3 * np.max(np.abs(loc))
    z_limits = [-z_max, z_max]

    # Draw stimuli 95% HDI ellipses.
    exemplar_color_array = class_color_array[squeeze_indices(class_arr)]
    psiz.mplot.hdi_bvn(
        loc, cov, ax, p=.95, edgecolor=exemplar_color_array, lw=lw,
        alpha=alpha, fill=False
    )

    # Draw stimuli modes.
    for idx_class in unique_class_arr:
        class_locs = np.equal(class_arr, idx_class)
        class_label = catalog.class_label[idx_class]
        ax.scatter(
            loc[class_locs, 0], loc[class_locs, 1], s=s,
            c=exemplar_color_array[class_locs], marker='o', edgecolors='none',
            zorder=100, label=class_label
        )

    ax.set_xlim(z_limits)
    ax.set_ylim(z_limits)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Embeddings (95% HDI)')
    ax.legend(bbox_to_anchor=(1.35, 0.9), shadow=True, title="Bird Species")

    gs.tight_layout(fig)


def squeeze_indices(idx_arr):
    """Squeeze indices of array."""
    uniq_idx_arr = np.unique(idx_arr)
    idx_arr_2 = np.zeros(idx_arr.shape, dtype=int)
    for counter, uniq_idx in enumerate(uniq_idx_arr):
        loc = np.equal(idx_arr, uniq_idx)
        idx_arr_2[loc] = counter
    return idx_arr_2


def unpack_mvn(dist):
    """Unpack multivariate normal distribution."""
    def diag_to_full_cov(v):
        """Convert diagonal variance to full covariance matrix.

        Assumes `v` represents diagonal variance elements only.
        """
        n_stimuli = v.shape[0]
        n_dim = v.shape[1]
        cov = np.zeros([n_stimuli, n_dim, n_dim])
        for i_stimulus in range(n_stimuli):
            cov[i_stimulus] = np.eye(n_dim) * v[i_stimulus]
        return cov

    loc = dist.mean().numpy()
    v = dist.variance().numpy()

    # Convert to full covariance matrix.
    cov = diag_to_full_cov(v)

    return loc, cov


if __name__ == "__main__":
    main()
