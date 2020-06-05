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

    Restart Summary
    n_valid_restart 3 | total_duration: 211 s
    best | n_epoch: 112 | val_cce: 3.2553
    mean ±stddev | n_epoch: 95 ±13 | val_cce: 3.2988 ±0.0406 | 65 ±8 s | 692 ±13 ms/epoch

    R^2 Model Comparison:   0.92

"""

import copy
import shutil

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
    n_dim = 2
    n_trial = 125  # 500  # 2000
    batch_size = 100
    n_restart = 1

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

    # Partition observations into train, validation, and test set.
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

    # Use Tensorboard.
    log_dir='/tmp/psiz/tensorboard_logs'
    # Remove existing TensorBoard logs.
    shutil.rmtree(log_dir)
    cb_board = psiz.keras.callbacks.TensorBoardRe(
        log_dir=log_dir, histogram_freq=0,
        write_graph=False, write_images=False, update_freq='epoch',
        profile_batch=0, embeddings_freq=0, embeddings_metadata=None
    )
    callbacks = [cb_board]

    compile_kwargs = {
        'loss': tf.keras.losses.CategoricalCrossentropy(),
        'optimizer': tf.keras.optimizers.Adam(lr=.001),
        'weighted_metrics': [
            tf.keras.metrics.CategoricalCrossentropy(name='cce')
        ]
    }
 
    # Define model.
    kl_weight = 1. / obs_train.n_trial
    embedding = psiz.keras.layers.EmbeddingVariational(
        n_stimuli+1, n_dim, mask_zero=True, kl_weight=kl_weight, prior_scale=.17
    )
    kernel = psiz.keras.layers.Kernel(
        distance=psiz.keras.layers.WeightedMinkowski(
            fit_rho=False,
            rho_initializer=tf.keras.initializers.Constant(2.),
        ),
        similarity=psiz.keras.layers.ExponentialSimilarity(
            fit_tau=False, fit_gamma=False,
            tau_initializer=tf.keras.initializers.Constant(1.),
            gamma_initializer=tf.keras.initializers.Constant(0.),
        )
    )
    model = psiz.models.Rank(
        embedding=embedding, kernel=kernel, n_sample_test=100
    )
    emb_inferred = psiz.models.Proxy(model=model)

    # Infer embedding.
    restart_record = emb_inferred.fit(
        obs_train, validation_data=obs_val, epochs=3000, batch_size=batch_size,  # TODO 1000
        callbacks=callbacks, n_restart=n_restart, monitor='val_loss', verbose=2,
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
    fig0 = plt.figure(figsize=(6.5, 4), dpi=200)
    plot_embedding(fig0, emb_true, emb_inferred, color_array)
    plt.show()
    # fname = fp_ani / Path('frame_{0}.tiff'.format(i_frame))
    # plt.savefig(
    #     os.fspath(fname), format='tiff', bbox_inches="tight", dpi=300
    # )


def plot_embedding(fig0, emb_true, emb_inferred, color_array):
    """Plot frame."""
    # Settings.
    s = 10

    gs = fig0.add_gridspec(1, 1)

    # Plot embeddings.
    f0_ax1 = fig0.add_subplot(gs[0, 0])
    # Determine embedding limits.
    z_max = 1.3 * np.max(np.abs(emb_true.z))
    z_limits = [-z_max, z_max]

    # Apply and plot Procrustes affine transformation of posterior.
    dist = emb_inferred.model.embedding.embeddings_posterior.distribution
    loc, cov = unpack_embeddings_distribution(dist)
    r, t = psiz.utils.procrustes_2d(
        emb_true.z, loc, scale=False, n_restart=30
    )
    loc, cov = apply_affine(loc, cov, r, t)
    plot_bvn(f0_ax1, loc, cov=cov, c=color_array, show_loc=False)

    # Plot true embedding.
    f0_ax1.scatter(
        emb_true.z[:, 0], emb_true.z[:, 1], s=s, c=color_array, marker='o',
        edgecolors='none', zorder=100
    )
    f0_ax1.set_xlim(z_limits)
    f0_ax1.set_ylim(z_limits)
    f0_ax1.set_aspect('equal')
    f0_ax1.set_xticks([])
    f0_ax1.set_yticks([])
    f0_ax1.set_title('Embeddings')

    gs.tight_layout(fig0)


def unpack_embeddings_distribution(dist):
    """Unpack embeddings distribution."""
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

    loc = dist.loc.numpy()[1:]  # Drop placeholder
    scale = dist.scale.numpy()[1:]
    cov = scale_to_cov(scale)
    return loc, cov


def apply_affine(loc, cov, r, t):
    """Apply affine transformation to set of MVN."""
    n_dist = loc.shape[0]
    loc_a = copy.copy(loc)
    cov_a = copy.copy(cov)

    for i_dist in range(n_dist):
        loc_a[i_dist], cov_a[i_dist] = psiz.utils.affine_mvn(
            loc[np.newaxis, i_dist], cov[i_dist], r, t
        )
    return loc_a, cov_a


def plot_bvn(ax, loc, cov=None, c=None, r=1.96, show_loc=True):
    """Plot bivariate normal embeddings.

    If covariances are supplied, ellipses are drawn to indicate regions
    of highest probability mass.

    Arguments:
        ax: A 'matplotlib' axes object.
        loc: Array denoting the means of bivariate normal
            distributions.
        cov (optional): Array denoting the covariance matrices of
            bivariate normal distributions.
        c (optional): color array
        limits (optional): Limits of axes.
        r (optional): The radius (specified in standard deviations) at
            which to draw the ellipse. The default value corresponds to
            an ellipse indicating a region containing 95% of the
            probability mass.

    """
    # Settings.
    s = 10

    n_stimuli = loc.shape[0]

    # Plot means.
    if show_loc:
        ax.scatter(
            loc[:, 0], loc[:, 1], s=s, c=c, marker='o', edgecolors='none'
        )

    if cov is not None:
        # Draw regions of highest probability mass.
        for i_stimulus in range(n_stimuli):
            ellipse = psiz.visualize.bvn_ellipse(
                loc[i_stimulus], cov[i_stimulus], r=r, fill=False,
                edgecolor=c[i_stimulus]
            )
            ax.add_artist(ellipse)


def ground_truth(n_stimuli, n_dim):
    """Return a ground truth embedding."""
    embedding = tf.keras.layers.Embedding(
        n_stimuli+1, n_dim, mask_zero=True,
        embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=.17, seed=58)
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
