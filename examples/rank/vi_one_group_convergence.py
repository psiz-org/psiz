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

NOTE: This example uses `imageio` to create frames and a GIF animation.

"""

import os
from pathlib import Path

import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
import tensorflow_probability as tfp

import psiz

# Uncomment the following line to force eager execution.
# tf.config.experimental_run_functions_eagerly(True)


def main():
    """Run script."""
    # Settings.
    fp_ani = Path.home() / Path('vi_one_group_convergence')
    n_stimuli = 30
    n_dim = 2
    n_trial = 2000
    n_restart = 3
    batch_size = 100
    n_frame = 8

    # Plot settings.
    small_size = 6
    medium_size = 8
    large_size = 10
    plt.rc('font', size=small_size)  # controls default text sizes
    plt.rc('axes', titlesize=medium_size)
    plt.rc('axes', labelsize=small_size)
    plt.rc('xtick', labelsize=small_size)
    plt.rc('ytick', labelsize=small_size)
    plt.rc('legend', fontsize=small_size)
    plt.rc('figure', titlesize=large_size)

    # Color settings.
    cmap = matplotlib.cm.get_cmap('jet')
    n_color = np.minimum(7, n_stimuli)
    norm = matplotlib.colors.Normalize(vmin=0., vmax=n_color)
    color_array = cmap(norm(range(n_color)))
    gray_array = np.ones([n_stimuli - n_color, 4])
    gray_array[:, 0:3] = .8
    color_array = np.vstack([gray_array, color_array])

    emb_true = ground_truth(n_stimuli, n_dim)
    fp_ani.mkdir(parents=True, exist_ok=True)

    # Generate a random docket of trials.
    generator = psiz.generator.RandomGenerator(
        n_stimuli, n_reference=8, n_select=2
    )
    docket = generator.generate(n_trial)

    # Simulate similarity judgments.
    agent = psiz.simulate.Agent(emb_true)
    obs = agent.simulate(docket)

    simmat_true = psiz.utils.pairwise_matrix(emb_true.similarity, emb_true.z)

    # Partition observations into train, validation and test set.
    skf = StratifiedKFold(n_splits=5)
    (train_idx, holdout_idx) = list(
        skf.split(obs.stimulus_set, obs.config_idx)
    )[0]
    obs_train = obs.subset(train_idx)
    obs_holdout = obs.subset(holdout_idx)
    skf = StratifiedKFold(n_splits=2)
    (val_idx, test_idx) = list(
        skf.split(obs_holdout.stimulus_set, obs_holdout.config_idx)
    )[0]
    obs_val = obs_holdout.subset(val_idx)
    obs_test = obs_holdout.subset(test_idx)

    # Use early stopping.
    early_stop = psiz.keras.callbacks.EarlyStoppingRe(
        'val_cce', patience=15, mode='min', restore_best_weights=True
    )
    callbacks = [early_stop]

    compile_kwargs = {
        'loss': tf.keras.losses.CategoricalCrossentropy(),
        'optimizer': tf.keras.optimizers.Adam(lr=.001),
        'weighted_metrics': [
            tf.keras.metrics.CategoricalCrossentropy(name='cce')
        ]
    }

    # Infer independent models with increasing amounts of data.
    n_obs = np.floor(
        np.linspace(15, obs_train.n_trial, n_frame)
    ).astype(np.int64)
    r2 = np.empty((n_frame)) * np.nan
    train_cce = np.empty((n_frame)) * np.nan
    val_cce = np.empty((n_frame)) * np.nan
    test_cce = np.empty((n_frame)) * np.nan
    for i_frame in range(n_frame):
        print('  Round {0}'.format(i_frame))
        include_idx = np.arange(0, n_obs[i_frame])
        obs_round_train = obs_train.subset(include_idx)

        # Define model.
        kl_weight = 1. / obs_train.n_trial
        embedding = psiz.keras.layers.EmbeddingVariational(
            n_stimuli+1, n_dim, mask_zero=True, kl_weight=kl_weight
        )
        kernel = psiz.keras.layers.Kernel(
            distance=psiz.keras.layers.WeightedMinkowskiVariational(
                kl_weight=kl_weight
            ),
            similarity=psiz.keras.layers.ExponentialSimilarityVariational(
                kl_weight=kl_weight
            )
        )
        model = psiz.models.Rank(
            embedding=embedding, kernel=kernel, n_sample_test=100
        )
        emb_inferred = psiz.models.Proxy(model=model)

        # Infer embedding.
        restart_record = emb_inferred.fit(
            obs_round_train, validation_data=obs_val, epochs=1000,
            batch_size=batch_size, callbacks=callbacks, n_restart=n_restart,
            monitor='val_cce', verbose=1, compile_kwargs=compile_kwargs
        )

        train_cce[i_frame] = restart_record.record['cce'][0]
        val_cce[i_frame] = restart_record.record['val_cce'][0]
        test_metrics = emb_inferred.evaluate(
            obs_test, verbose=0, return_dict=True
        )
        test_cce[i_frame] = test_metrics['cce']

        # Compare the inferred model with ground truth by comparing the
        # similarity matrices implied by each model.
        simmat_infer = psiz.utils.pairwise_matrix(
            emb_inferred.similarity, emb_inferred.z
        )
        r2[i_frame] = psiz.utils.matrix_comparison(
            simmat_infer, simmat_true, score='r2'
        )
        print(
            '    n_obs: {0:4d} | train_cce: {1:.2f} | '
            'val_cce: {2:.2f} | test_cce: {3:.2f} | '
            'Correlation (R^2): {4:.2f}'.format(
                n_obs[i_frame], train_cce[i_frame],
                val_cce[i_frame], test_cce[i_frame], r2[i_frame]
            )
        )

        # Create and save visual frame.
        fig0 = plt.figure(figsize=(6.5, 4), dpi=200)
        plot_frame(
            fig0, n_obs, train_cce, val_cce, test_cce, r2, emb_true,
            emb_inferred, color_array
        )
        fname = fp_ani / Path('frame_{0}.tiff'.format(i_frame))
        plt.savefig(
            os.fspath(fname), format='tiff', bbox_inches="tight", dpi=300
        )

    frames = []
    for i_frame in range(n_frame):
        fname = fp_ani / Path('frame_{0}.tiff'.format(i_frame))
        frames.append(imageio.imread(fname))
    imageio.mimwrite(fp_ani / Path('posterior.gif'), frames, fps=1)


def plot_frame(
        fig0, n_obs, train_cce, val_cce, test_cce, r2, emb_true, emb_inferred,
        color_array):
    """Plot frame."""
    gs = fig0.add_gridspec(6, 8)

    f0_ax0 = fig0.add_subplot(gs[0:3, 0:3])
    plot_loss(f0_ax0, n_obs, train_cce, val_cce, test_cce)

    f0_ax6 = fig0.add_subplot(gs[0:3, 3:6])
    plot_convergence(f0_ax6, n_obs, r2)

    f0_ax1 = fig0.add_subplot(gs[3:6, 0:3])
    plot_embeddings_distribution(f0_ax1, emb_true.z, color_array)
    f0_ax1.set_title('True Embeddings')

    # Prepare posterior for plotting.
    # Prepare covariance matrices. TODO
    # cov = None
    # z_inferred_affine, params = procprocrustean_solution(emb_true.z, emb_inferred.z)
    # Apply transformations to covariance matrices. TODO
    # Plot affine transformation of posterior.
    f0_ax2 = fig0.add_subplot(gs[3:6, 3:6])
    plot_embeddings_distribution(
        f0_ax2,
        emb_inferred.model.embedding.embeddings_posterior.distribution,
        color_array
    )
    f0_ax2.set_title('Posterior Embeddings')

    f0_ax3 = fig0.add_subplot(gs[0:2, 6:8])
    plot_univariate_distribution(
        f0_ax3,
        emb_inferred.model.kernel.distance.rho_posterior.distribution,
        landmark=emb_true.model.kernel.distance.rho.numpy(),
    )
    f0_ax3.set_title('rho')

    f0_ax4 = fig0.add_subplot(gs[2:4, 6:8])
    plot_univariate_distribution(
        f0_ax4,
        emb_inferred.model.kernel.similarity.tau_posterior.distribution,
        landmark=emb_true.model.kernel.similarity.tau.numpy(),
    )
    f0_ax4.set_title('tau')

    f0_ax5 = fig0.add_subplot(gs[4:6, 6:8])
    plot_univariate_distribution(
        f0_ax5,
        emb_inferred.model.kernel.similarity.gamma_posterior.distribution,
        landmark=np.log10(emb_true.model.kernel.similarity.gamma.numpy()),  # TODO
    )
    f0_ax5.set_title('gamma')

    gs.tight_layout(fig0)


def plot_loss(ax, n_obs, train_cce, val_cce, test_cce):
    """Plot loss."""
    # Settings
    ms = 2

    ax.plot(n_obs, train_cce, 'bo-', ms=ms, label='Train CCE')
    ax.plot(n_obs, val_cce, 'go-', ms=ms, label='Val. CCE')
    ax.plot(n_obs, test_cce, 'ro-', ms=ms, label='Test CCE')
    ax.set_title('Loss')

    ax.set_xlabel('Trials')
    limits = [0, np.max(n_obs) + 10]
    ax.set_xlim(limits)
    ticks = [np.min(n_obs), np.max(n_obs)]
    ax.set_xticks(ticks)

    ax.set_ylabel('Loss')
    ax.legend()


def plot_convergence(ax, n_obs, r2):
    """Plot convergence."""
    # Settings.
    ms = 2

    ax.plot(n_obs, r2, 'ro-',  ms=ms,)
    ax.set_title('Convergence')

    ax.set_xlabel('Trials')
    limits = [0, np.max(n_obs) + 10]
    ax.set_xlim(limits)
    ticks = [np.min(n_obs), np.max(n_obs)]
    ax.set_xticks(ticks)

    ax.set_ylabel(r'$R^2$')
    ax.set_ylim(-0.05, 1.05)


def plot_embeddings_distribution(ax, dist, c, limits=None):
    """Plot embeddings distribution.

    Assumes normal distributions.

    Arguments:
        dist: A tfp.distributions.Distribution or a 2D array of points
            representing locations.
        c: color array

    """
    # Settings.
    s = 10
    n_std = 1.96  # as ci

    if isinstance(dist, tfp.distributions.Distribution):
        loc = dist.loc.numpy()[1:]  # Drop placeholder
        scale = dist.scale.numpy()[1:]
    else:
        loc = dist
        scale = None

    if limits is None:
        z_max = 1.2 * np.max(np.abs(loc))
        limits = [-z_max, z_max]

    n_stimuli = loc.shape[0]
    n_dim = loc.shape[1]

    # Plot means.
    ax.scatter(
        loc[:, 0], loc[:, 1], s=s, c=c, marker='o', edgecolors='none'
    )

    if scale is not None:
        # Plot posterior confidence intervals.
        for i_stimulus in range(n_stimuli):
            mu = loc[i_stimulus]
            cov = np.eye(n_dim) * scale[i_stimulus]**2
            ell = error_ellipse(mu, cov, n_std)
            ell.set_facecolor('none')
            ell.set_edgecolor(c[i_stimulus])
            ax.add_artist(ell)

    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])


def error_ellipse(mu, cov, nstd):
    """Return artist of error ellipse.

    SEE: https://stackoverflow.com/questions/20126061/
    creating-a-confidence-ellipses-in-a-sccatterplot-using-matplotlib

    """
    # TODO inject kwargs
    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2 * nstd * np.sqrt(vals)
    ell = matplotlib.patches.Ellipse(
        xy=(mu[0], mu[1]), width=w, height=h, angle=theta,
        color='black'
    )
    return ell


def eigsorted(cov):
    """Sort eigenvalues."""
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:, order]


def plot_univariate_distribution(ax, dist, limits=None, landmark=None):
    """Plot univariate distribution.

    Arguments:
        ax:
        dist:
        limits:
        landmark:

    """
    x_mode = dist.mode().numpy()
    if limits is None:
        limits = [x_mode - 1, x_mode + 1]
    xg = np.linspace(limits[0], limits[1], 1000)
    y = dist.prob(xg)
    ax.plot(xg, y)
    ax.text(x_mode, .9 * np.max(y), '{0:.2f}'.format(x_mode))
    ax.set_xlabel('x')
    ax.set_ylabel('p(x)')

    if landmark is not None:
        ax.plot([landmark, landmark], [0, np.max(y)], 'r--')


def ground_truth(n_stimuli, n_dim):
    """Return a ground truth embedding."""
    embedding = psiz.keras.layers.tf.keras.layers.Embedding(
        n_stimuli+1, n_dim, mask_zero=True,
        embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=.17)
    )
    kernel = psiz.keras.layers.Kernel(
        similarity=psiz.keras.layers.ExponentialSimilarity()
    )
    model = psiz.models.Rank(embedding=embedding, kernel=kernel)

    emb = psiz.models.Proxy(model=model)
    emb.theta = {
        'rho': 2.,
        'tau': 1.,
        'beta': 10.,
        'gamma': 0.001
    }

    return emb


if __name__ == "__main__":
    main()
