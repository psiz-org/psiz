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
# tf.config.experimental_run_functions_eagerly(True)


def main():
    """Run script."""
    # Settings.
    fp_example = Path.home() / Path('psiz_examples', 'vi_birds_16')
    fp_board = fp_example / Path('logs', 'fit', 'r0')
    n_dim = 2
    n_restart = 1
    epochs = 300
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
        '\nData Split\n  obs_train: {0}\n  obs_val: {1}\n  obs_test: {2}'.format(
            obs_train.n_trial, obs_val.n_trial, obs_test.n_trial
        )
    )

    # Build model and wrap in Proxy.
    model = build_model(catalog.n_stimuli, n_dim, obs_train.n_trial)
    proxy = psiz.models.Proxy(model=model)

    # Compile settings.
    compile_kwargs = {
        'loss': tf.keras.losses.CategoricalCrossentropy(),
        'optimizer': tf.keras.optimizers.Adam(lr=.001),
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
    callbacks = [cb_board]

    # Infer embedding.
    restart_record = proxy.fit(
        obs_train, validation_data=obs_val, epochs=epochs,
        batch_size=batch_size, callbacks=callbacks, n_restart=n_restart,
        monitor='val_loss', verbose=2, compile_kwargs=compile_kwargs
    )

    train_loss = restart_record.record['loss'][0]
    train_time = restart_record.record['ms_per_epoch'][0]
    val_loss = restart_record.record['val_loss'][0]

    model.n_sample_test = 100
    test_metrics = proxy.evaluate(
        obs_test, verbose=0, return_dict=True
    )
    test_loss = test_metrics['loss']
    print(
        '    train_loss: {0:.2f} | val_loss: {1:.2f} | '
        'test_loss: {2:.2f} | '.format(train_loss, val_loss, test_loss)
    )

    # Create visual.
    fig = plt.figure(figsize=(6.5, 4), dpi=200)
    draw_figure(
        fig, proxy.model, catalog
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
    stimuli = EmbeddingVariationalLog(
        posterior=embedding_posterior, prior=embedding_prior,
        kl_weight=kl_weight, kl_n_sample=30
    )

    kernel = psiz.keras.layers.Kernel(
        distance=psiz.keras.layers.WeightedMinkowski(
            rho_initializer=tf.keras.initializers.Constant(2.),
            trainable=False,
        ),
        similarity=psiz.keras.layers.ExponentialSimilarity(
            tau_initializer=tf.keras.initializers.Constant(1.),
            gamma_initializer=tf.keras.initializers.Constant(0.),
            trainable=False
        )
    )
    model = psiz.models.Rank(
        stimuli=stimuli, kernel=kernel, n_sample_test=10
    )
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
    dist = model.stimuli.posterior.embeddings
    loc, cov = unpack_mvn(dist)

    z_max = 1.3 * np.max(np.abs(loc))
    z_limits = [-z_max, z_max]
    
    # Draw stimuli 95% probability mass ellipses.
    exemplar_color_array = class_color_array[squeeze_indices(class_arr)]
    plot_bvn(
        ax, loc, cov, c=exemplar_color_array, r=1.96, lw=lw, alpha=alpha
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
    ax.set_title('Embeddings (95%)')
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


def plot_bvn(ax, loc, cov, c=None, r=2.576, **kwargs):
    """Plot bivariate normal embeddings.

    If covariances are supplied, ellipses are drawn to indicate regions
    of highest probability mass.

    Arguments:
        ax: A 'matplotlib' axes object.
        loc: Array denoting the means of bivariate normal
            distributions.
        cov: Array denoting the covariance matrices of
            bivariate normal distributions.
        c (optional): color array
        r (optional): The radius (specified in standard deviations) at
            which to draw the ellipse. The default value (2.576)
            corresponds to an ellipse indicating a region containing
            99% of the probability mass. Another common value is
            1.960, which indicates 95%.
        kwargs (optional): Additional key-word arguments that will be
            passed to a `matplotlib.patches.Ellipse` constructor.

    """
    n_stimuli = loc.shape[0]

    # Draw ellipsoids for each stimulus.
    for i_stimulus in range(n_stimuli):
        if c is not None:
            edgecolor = c[i_stimulus]
        ellipse = psiz.visualize.bvn_ellipse(
            loc[i_stimulus], cov[i_stimulus], r=r, fill=False,
            edgecolor=edgecolor, **kwargs
        )
        ax.add_artist(ellipse)


def unpack_mvn(dist):
    """Unpack multivariate normal distribution."""
    def scale_to_cov(scale):
        """Convert scale to covariance matrix.

        Assumes `scale` represents diagonal elements only.
        """
        n_stimuli = scale.shape[0]
        n_dim = scale.shape[1]
        cov = np.zeros([n_stimuli, n_dim, n_dim])
        for i_stimulus in range(n_stimuli):
            cov[i_stimulus] = np.eye(n_dim) * scale[i_stimulus]**2
        return cov

    if isinstance(dist, tfp.distributions.Independent):
        d = dist.distribution
    else:
        d = dist

    # Drop placeholder stimulus.
    loc = d.loc.numpy()[1:]
    scale = d.scale.numpy()[1:]

    # Convert to full covariance matrix.
    cov = scale_to_cov(scale)

    return loc, cov


@tf.keras.utils.register_keras_serializable(
    package='psiz.keras.layers', name='EmbeddingVariationalLog'
)
class EmbeddingVariationalLog(psiz.keras.layers.EmbeddingVariational):
    """Sub-class for logging weight metrics."""

    def call(self, inputs):
        """Call."""
        outputs = super().call(inputs)

        m = self.posterior.embeddings.mode()[1:]
        self.add_metric(
            tf.reduce_mean(m),
            aggregation='mean', name='po_mode_avg'
        )
        self.add_metric(
            tf.reduce_min(m),
            aggregation='mean', name='po_mode_min'
        )
        self.add_metric(
            tf.reduce_max(m),
            aggregation='mean', name='po_mode_max'
        )

        s = self.posterior.embeddings.distribution.scale[1:]
        self.add_metric(
            tf.reduce_mean(s),
            aggregation='mean', name='po_scale_avg'
        )

        s = self.prior.embeddings.distribution.distribution.distribution.scale[0, 0]
        self.add_metric(
            s, aggregation='mean', name='pr_scale'
        )

        return outputs


if __name__ == "__main__":
    main()
