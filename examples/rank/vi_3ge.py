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
"""Example that infers three distinct stimulus embeddings.

Results are saved in the directory specified by `fp_example`. By
default, a `psiz_examples` directory is created in your home directory.

Example output:

    Model Comparison (R^2)
    ================================
      True  |        Inferred
            | Novice  Interm  Expert
    --------+-----------------------
     Novice |   0.90    0.58    0.04
     Interm |   0.43    0.67    0.31
     Expert |   0.05    0.30    0.84

"""

import copy
import os
from pathlib import Path
import shutil

import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
import tensorflow as tf
import tensorflow_probability as tfp

import psiz

# Uncomment the following line to force eager execution.
# tf.config.experimental_run_functions_eagerly(True)

# Uncomment and edit the following to control GPU visibility.
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    """Run script."""
    # Settings.
    fp_example = Path.home() / Path('psiz_examples', 'rank', 'vi_3ge')
    fp_board = fp_example / Path('logs', 'fit')
    n_stimuli = 30
    n_dim = 3
    n_dim_inferred = 2
    n_group = 3
    n_trial = 2000
    epochs = 1000
    batch_size = 128
    n_frame = 1

    # Directory preparation.
    fp_example.mkdir(parents=True, exist_ok=True)
    # Remove existing TensorBoard logs.
    if fp_board.exists():
        shutil.rmtree(fp_board)

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

    model_true = ground_truth(n_stimuli, n_group)

    # Compute ground truth similarity matrices.
    simmat_truth = (
        model_similarity(model_true, group_idx=[0]),
        model_similarity(model_true, group_idx=[1]),
        model_similarity(model_true, group_idx=[2])
    )

    # Generate a random docket of trials to show each group.
    generator = psiz.generators.RandomRank(
        n_stimuli, n_reference=8, n_select=2
    )
    docket = generator.generate(n_trial)

    # Create virtual agents for each group.
    agent_novice = psiz.agents.RankAgent(model_true, group_id=0)
    agent_interm = psiz.agents.RankAgent(model_true, group_id=1)
    agent_expert = psiz.agents.RankAgent(model_true, group_id=2)

    # Simulate similarity judgments for each group.
    obs_novice = agent_novice.simulate(docket)
    obs_interm = agent_interm.simulate(docket)
    obs_expert = agent_expert.simulate(docket)
    obs = psiz.trials.stack((obs_novice, obs_interm, obs_expert))

    # Partition observations into 80% train, 10% validation and 10% test set.
    obs_train, obs_val, obs_test = psiz.utils.standard_split(obs)

    # Convert observations to TF dataset.
    ds_obs_val = obs_val.as_dataset().batch(
        batch_size, drop_remainder=False
    )
    ds_obs_test = obs_test.as_dataset().batch(
        batch_size, drop_remainder=False
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
    r2 = np.empty([n_frame, n_group, n_group]) * np.nan
    train_loss = np.empty((n_frame)) * np.nan
    val_loss = np.empty((n_frame)) * np.nan
    test_loss = np.empty((n_frame)) * np.nan
    for i_frame in range(n_frame):
        include_idx = np.arange(0, n_obs[i_frame])
        obs_round_train = obs_train.subset(include_idx)
        ds_obs_round_train = obs_round_train.as_dataset().shuffle(
            buffer_size=obs_round_train.n_trial, reshuffle_each_iteration=True
        ).batch(batch_size, drop_remainder=False)
        print(
            '\n  Frame {0} ({1} obs)'.format(i_frame, obs_round_train.n_trial)
        )

        # Define model.
        kl_weight = 1. / obs_round_train.n_trial
        model_inferred = build_model(
            n_stimuli, n_dim_inferred, n_group, kl_weight
        )

        # Define callbacks.
        fp_board_frame = fp_board / Path('frame_{0}'.format(i_frame))
        cb_board = psiz.keras.callbacks.TensorBoardRe(
            log_dir=fp_board_frame, histogram_freq=0,
            write_graph=False, write_images=False, update_freq='epoch',
            profile_batch=0, embeddings_freq=0, embeddings_metadata=None
        )
        cb_early = psiz.keras.callbacks.EarlyStoppingRe(
            'loss', patience=100, mode='min', restore_best_weights=False,
            verbose=1
        )
        callbacks = [cb_board, cb_early]

        # Infer model.
        model_inferred.compile(**compile_kwargs)
        history = model_inferred.fit(
            ds_obs_round_train, validation_data=ds_obs_val, epochs=epochs,
            callbacks=callbacks, verbose=0
        )

        train_loss[i_frame] = history.history['loss'][-1]
        val_loss[i_frame] = history.history['val_loss'][-1]

        tf.keras.backend.clear_session()
        model_inferred.n_sample = 100
        model_inferred.compile(**compile_kwargs)
        test_metrics = model_inferred.evaluate(
            ds_obs_test, verbose=0, return_dict=True
        )
        test_loss[i_frame] = test_metrics['loss']

        # Compare the inferred model with ground truth by comparing the
        # similarity matrices implied by each model.
        simmat_inferred = (
            model_similarity(model_inferred, group_idx=[0]),
            model_similarity(model_inferred, group_idx=[1]),
            model_similarity(model_inferred, group_idx=[2])
        )

        for i_truth in range(n_group):
            for j_infer in range(n_group):
                rho, _ = pearsonr(
                    simmat_truth[i_truth], simmat_inferred[j_infer]
                )
                r2[i_frame, i_truth, j_infer] = rho**2

        # Display comparison results. A good inferred model will have a high
        # R^2 value on the diagonal elements (max is 1) and relatively low R^2
        # values on the off-diagonal elements.
        print('\n    Model Comparison (R^2)')
        print('    ================================')
        print('      True  |        Inferred')
        print('            | Novice  Interm  Expert')
        print('    --------+-----------------------')
        print('     Novice | {0: >6.2f}  {1: >6.2f}  {2: >6.2f}'.format(
            r2[i_frame, 0, 0], r2[i_frame, 0, 1], r2[i_frame, 0, 2]))
        print('     Interm | {0: >6.2f}  {1: >6.2f}  {2: >6.2f}'.format(
            r2[i_frame, 1, 0], r2[i_frame, 1, 1], r2[i_frame, 1, 2]))
        print('     Expert | {0: >6.2f}  {1: >6.2f}  {2: >6.2f}'.format(
            r2[i_frame, 2, 0], r2[i_frame, 2, 1], r2[i_frame, 2, 2]))
        print('\n')

        # Create and save visual frame.
        fig0 = plt.figure(figsize=(12, 5), dpi=200)
        plot_frame(
            fig0, n_obs, train_loss, val_loss, test_loss, r2, model_true,
            model_inferred, i_frame, color_array
        )
        fname = fp_example / Path('frame_{0}.tiff'.format(i_frame))
        plt.savefig(
            os.fspath(fname), format='tiff', bbox_inches="tight", dpi=300
        )

    # Create animation.
    if n_frame > 1:
        frames = []
        for i_frame in range(n_frame):
            fname = fp_example / Path('frame_{0}.tiff'.format(i_frame))
            frames.append(imageio.imread(fname))
        imageio.mimwrite(fp_example / Path('evolution.gif'), frames, fps=1)


def ground_truth(n_stimuli, n_group):
    """Return a ground truth embedding."""
    n_dim = 4
    embedding = psiz.keras.layers.EmbeddingDeterministic(
        n_stimuli+1, n_dim, mask_zero=True,
        embeddings_initializer=tf.keras.initializers.RandomNormal(
            stddev=.17, seed=58
        )
    )
    stimuli = psiz.keras.layers.Stimuli(embedding=embedding)
    kernel = psiz.keras.layers.AttentionKernel(
        group_level=1,
        distance=psiz.keras.layers.WeightedMinkowski(
            rho_initializer=tf.keras.initializers.Constant(2.),
            trainable=False,
        ),
        attention=psiz.keras.layers.EmbeddingDeterministic(
            n_group, n_dim, mask_zero=False,
            embeddings_initializer=tf.keras.initializers.Constant(
                np.array((
                    (1.8, 1.8, .2, .2),
                    (1., 1., 1., 1.),
                    (.2, .2, 1.8, 1.8)
                ))
            )
        ),
        similarity=psiz.keras.layers.ExponentialSimilarity(
            tau_initializer=tf.keras.initializers.Constant(1.),
            gamma_initializer=tf.keras.initializers.Constant(0.),
            trainable=False,
        )
    )

    model = psiz.models.Rank(stimuli=stimuli, kernel=kernel)
    return model


def build_model(n_stimuli, n_dim, n_group, kl_weight):
    """Build model.

    Arguments:
        n_stimuli: Integer indicating the number of stimuli in the
            embedding.
        n_dim: Integer indicating the dimensionality of the embedding.
        n_group: Integer indicating the number of groups.
        kl_weight: Float indicating the KL weight for variational
            inference.

    Returns:
        model: A TensorFlow Keras model.

    """
    prior_scale = .2

    n_source_embeddings = n_group * (n_stimuli + 1)
    embedding_posterior = psiz.keras.layers.EmbeddingNormalDiag(
        n_source_embeddings, n_dim, mask_zero=True,
        scale_initializer=tf.keras.initializers.Constant(
            tfp.math.softplus_inverse(prior_scale).numpy()
        )
    )
    embedding_prior = psiz.keras.layers.EmbeddingShared(
        n_source_embeddings, n_dim, mask_zero=True,
        embedding=psiz.keras.layers.EmbeddingNormalDiag(
            1, 1,
            loc_initializer=tf.keras.initializers.Constant(0.),
            scale_initializer=tf.keras.initializers.Constant(
                tfp.math.softplus_inverse(prior_scale).numpy()
            ),
            loc_trainable=False
        )
    )
    embedding_variational = psiz.keras.layers.EmbeddingVariational(
        posterior=embedding_posterior, prior=embedding_prior,
        kl_weight=kl_weight, kl_n_sample=30
    )
    stimuli = psiz.keras.layers.Stimuli(
        embedding=embedding_variational, group_level=1, n_group=n_group
    )

    kernel = psiz.keras.layers.Kernel(
        distance=psiz.keras.layers.WeightedMinkowski(
            rho_initializer=tf.keras.initializers.Constant(2.),
            trainable=False,
        ),
        similarity=psiz.keras.layers.ExponentialSimilarity(
            beta_initializer=tf.keras.initializers.Constant(10.),
            tau_initializer=tf.keras.initializers.Constant(1.),
            gamma_initializer=tf.keras.initializers.Constant(0.),
            trainable=False
        )
    )
    model = psiz.models.Rank(
        stimuli=stimuli, kernel=kernel, n_sample=1
    )
    return model


def plot_frame(
        fig0, n_obs, train_loss, val_loss, test_loss, r2, model_true,
        model_inferred, i_frame, color_array):
    """Plot posteriors."""
    # Settings.
    group_labels = ['Novice', 'Intermediate', 'Expert']

    n_group = model_inferred.stimuli.n_group
    n_dim = model_inferred.n_dim

    gs = fig0.add_gridspec(2, 3)

    f0_ax0 = fig0.add_subplot(gs[0, 0])
    plot_loss(f0_ax0, n_obs, train_loss, val_loss, test_loss)

    f0_ax1 = fig0.add_subplot(gs[0, 1])
    plot_convergence(fig0, f0_ax1, n_obs, r2[i_frame])

    f0_ax2 = fig0.add_subplot(gs[0, 2])
    plot_embeddings(fig0, f0_ax2, model_inferred, color_array)

    gs.tight_layout(fig0)


def plot_loss(ax, n_obs, train_loss, val_loss, test_loss):
    """Plot loss."""
    # Settings
    ms = 2

    ax.plot(n_obs, train_loss, 'bo-', ms=ms, label='Train Loss')
    ax.plot(n_obs, val_loss, 'go-', ms=ms, label='Val. Loss')
    ax.plot(n_obs, test_loss, 'ro-', ms=ms, label='Test Loss')
    ax.set_title('Optimization Objective')

    ax.set_xlabel('Trials')
    limits = [0, np.max(n_obs) + 10]
    ax.set_xlim(limits)
    ticks = [np.min(n_obs), np.max(n_obs)]
    ax.set_xticks(ticks)

    ax.set_ylabel('Loss')
    ax.legend()


def plot_convergence(fig, ax, n_obs, r2):
    """Plot convergence."""
    # Settings.
    cmap = matplotlib.cm.get_cmap('Greys')
    labels = ['Nov', 'Int', 'Exp']

    im = ax.imshow(r2, cmap=cmap, vmin=0., vmax=1.)
    fig.colorbar(im, ax=ax)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(labels)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(labels)
    ax.set_ylabel('True')
    ax.set_xlabel('Inferred')
    ax.set_title(r'$R^2$ Convergence')


def plot_embeddings(fig, ax, model_inferred, color_array):
    # Settings.
    n_group = 3
    marker_list = ['o', '<', 'p']
    markersize = 20

    # Apply and plot Procrustes affine transformation of posterior.
    loc_list = []
    cov_list = []
    z_max = 0
    for group_idx in range(n_group):
        dist = model_inferred.stimuli.embeddings
        loc, cov = unpack_mvn(dist, group_idx)
        if model_inferred.stimuli.mask_zero:
            # Drop placeholder stimulus.
            loc = loc[1:]
            cov = cov[1:]
        # Center coordinates.
        loc = loc - np.mean(loc, axis=0, keepdims=True)
        loc_list.append(loc)
        cov_list.append(cov)

        # Determine limits.
        z_max_curr = 1.1 * np.max(np.abs(loc))
        if z_max_curr > z_max:
            z_max = z_max_curr
    z_limits = [-z_max, z_max]

    # Determine rotations into group_idx=0.
    r10 = psiz.utils.procrustes_rotation(
        loc_list[1], loc_list[0], scale=False
    )

    r20 = psiz.utils.procrustes_rotation(
        loc_list[2], loc_list[0], scale=False
    )

    # Apply rotations.
    loc_list[1], cov_list[1] = apply_affine(loc_list[1], cov_list[1], r10)
    loc_list[2], cov_list[2] = apply_affine(loc_list[2], cov_list[2], r10)

    # r = 1.960  # 95%
    # r = 2.576  # 99%
    # plot_bvn(f0_ax1, loc, cov=cov, c=color_array, r=r, show_loc=False)

    # Plot inferred embeddings.
    for i_group in range(n_group):
        ax.scatter(
            loc_list[i_group][:, 0], loc_list[i_group][:, 1],
            s=markersize, c=color_array, marker=marker_list[i_group],
            edgecolors='none', zorder=100
        )
    ax.set_xlim(z_limits)
    ax.set_ylim(z_limits)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Embeddings')


def model_similarity(model, group_idx=[]):
    """Compute model similarity.

    In the deterministic case, there is one one sample and mean is
    equivalent to squeeze. In the probabilistic case, mean takes an
    average across samples.

    Arguments:
        model:
        group_idx:

    """
    ds_pairs, ds_info = psiz.utils.pairwise_index_dataset(
        model.stimuli.n_stimuli, mask_zero=True, group_idx=group_idx
    )
    simmat = np.mean(
        psiz.utils.pairwise_similarity(
            model.stimuli, model.kernel, ds_pairs
        ).numpy(),
        axis=0
    )
    return simmat


def unpack_mvn(dist, group_idx):
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

    loc = dist.mean().numpy()[group_idx]
    v = dist.variance().numpy()[group_idx]

    # Convert to full covariance matrix.
    cov = diag_to_full_cov(v)

    return loc, cov


def apply_affine(loc, cov, r=None, t=None):
    """Apply affine transformation to set of MVN."""
    n_dist = loc.shape[0]
    loc_a = copy.copy(loc)
    cov_a = copy.copy(cov)

    for i_dist in range(n_dist):
        loc_a[i_dist], cov_a[i_dist] = psiz.utils.affine_mvn(
            loc[np.newaxis, i_dist], cov[i_dist], r, t
        )
    return loc_a, cov_a


if __name__ == "__main__":
    main()
