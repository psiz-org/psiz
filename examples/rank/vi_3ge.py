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
"""Example that infers a shared embedding for three groups.

Fake data is generated from a ground truth model for three different
groups. In this example, these groups represent groups of agents with
varying levels of skill: novices, intermediates, and experts. Each group
has a different set of attention weights. An embedding model is
inferred from the simulated data and compared to the ground truth
model.

Results are saved in the directory specified by `fp_example`. By
default, a `psiz_examples` directory is created in your home directory.

Example output:

    Restart Summary
    n_valid_restart 1 | total_duration: 2104 s
    best | n_epoch: 999 | val_loss: 3.0700
    mean ±stddev | n_epoch: 999 ±0 | val_loss: 3.0700 ±0.0000 | 2088 ±0 s | 2090 ±0 ms/epoch

    Model Comparison (R^2)
    ================================
      True  |        Inferred
            | Novice  Interm  Expert
    --------+-----------------------
     Novice |   0.94    0.67    0.06
     Interm |   0.55    0.87    0.35
     Expert |   0.09    0.42    0.85
 
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
    fp_example = Path.home() / Path('psiz_examples', 'vi_3ge')
    fp_board = fp_example / Path('logs', 'fit')
    n_stimuli = 30
    n_dim = 3
    n_group = 3
    n_trial = 2000
    epochs = 1000
    n_restart = 1
    batch_size = 128
    n_frame = 4

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

    model_true = ground_truth(n_stimuli, n_group)
    emb_true = psiz.models.Proxy(model=model_true)

    # Generate a random docket of trials to show each group.
    generator = psiz.generator.RandomRank(
        n_stimuli, n_reference=8, n_select=2
    )
    docket = generator.generate(n_trial)

    # Create virtual agents for each group.
    agent_novice = psiz.generator.Agent(emb_true.model, group_id=0)
    agent_interm = psiz.generator.Agent(emb_true.model, group_id=1)
    agent_expert = psiz.generator.Agent(emb_true.model, group_id=2)

    # Simulate similarity judgments for each group.
    obs_novice = agent_novice.simulate(docket)
    obs_interm = agent_interm.simulate(docket)
    obs_expert = agent_expert.simulate(docket)
    obs = psiz.trials.stack((obs_novice, obs_interm, obs_expert))

    # Compute ground truth similarity matrices.
    def truth_sim_func0(z_q, z_ref):
        return emb_true.similarity(z_q, z_ref, group_id=0)

    def truth_sim_func1(z_q, z_ref):
        return emb_true.similarity(z_q, z_ref, group_id=1)

    def truth_sim_func2(z_q, z_ref):
        return emb_true.similarity(z_q, z_ref, group_id=2)

    simmat_truth = (
        psiz.utils.pairwise_matrix(truth_sim_func0, emb_true.z),
        psiz.utils.pairwise_matrix(truth_sim_func1, emb_true.z),
        psiz.utils.pairwise_matrix(truth_sim_func2, emb_true.z)
    )

    # Partition observations into 80% train, 10% validation and 10% test set.
    obs_train, obs_val, obs_test = psiz.utils.standard_split(obs)

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
        print(
            '\n  Frame {0} ({1} obs)'.format(i_frame, obs_round_train.n_trial)
        )

        # Define model.
        model = build_model(n_stimuli, n_dim, n_group, obs_round_train.n_trial)
        emb_inferred = psiz.models.Proxy(model=model)

        # Define callbacks.
        fp_board_frame = fp_board / Path('frame_{0}'.format(i_frame))
        cb_board = psiz.keras.callbacks.TensorBoardRe(
            log_dir=fp_board_frame, histogram_freq=0,
            write_graph=False, write_images=False, update_freq='epoch',
            profile_batch=0, embeddings_freq=0, embeddings_metadata=None
        )
        callbacks = [cb_board]

        # Infer model.
        restart_record = emb_inferred.fit(
            obs_round_train, validation_data=obs_val, epochs=epochs,
            batch_size=batch_size, callbacks=callbacks, n_restart=n_restart,
            monitor='val_loss', verbose=1, compile_kwargs=compile_kwargs
        )

        train_loss[i_frame] = restart_record.record['loss'][0]
        val_loss[i_frame] = restart_record.record['val_loss'][0]

        emb_inferred.model.n_sample_test = 100
        test_metrics = emb_inferred.evaluate(
            obs_test, verbose=0, return_dict=True
        )
        test_loss[i_frame] = test_metrics['loss']

        z = emb_inferred.model.stimuli.embeddings.mode().numpy()
        if emb_inferred.model.stimuli.mask_zero:
            z = z[:, 1:, :]
        z0 = z[0]
        z1 = z[1]
        z2 = z[2]

        # Compare the inferred model with ground truth by comparing the
        # similarity matrices implied by each model.
        def infer_sim_func(z_q, z_ref):
            return emb_inferred.similarity(z_q, z_ref)

        simmat_infer = (
            psiz.utils.pairwise_matrix(infer_sim_func, z0),
            psiz.utils.pairwise_matrix(infer_sim_func, z1),
            psiz.utils.pairwise_matrix(infer_sim_func, z2)
        )
        for i_truth in range(n_group):
            for j_infer in range(n_group):
                r2[i_frame, i_truth, j_infer] = psiz.utils.matrix_comparison(
                    simmat_truth[i_truth], simmat_infer[j_infer],
                    score='r2'
                )

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
            fig0, n_obs, train_loss, val_loss, test_loss, r2, emb_true,
            emb_inferred, i_frame
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
    stimuli = tf.keras.layers.Embedding(
        n_stimuli+1, n_dim, mask_zero=True,
        embeddings_initializer=tf.keras.initializers.RandomNormal(
            stddev=.17, seed=58
        )
    )
    kernel = psiz.keras.layers.AttentionKernel(
        distance=psiz.keras.layers.WeightedMinkowski(
            rho_initializer=tf.keras.initializers.Constant(2.),
            trainable=False,
        ),
        attention=psiz.keras.layers.GroupAttention(
            n_dim=n_dim, n_group=n_group
        ),
        similarity=psiz.keras.layers.ExponentialSimilarity(
            tau_initializer=tf.keras.initializers.Constant(1.),
            gamma_initializer=tf.keras.initializers.Constant(0.),
            trainable=False,
        )
    )
    kernel.attention.embeddings.assign(
        np.array((
            (1.8, 1.8, .2, .2),
            (1., 1., 1., 1.),
            (.2, .2, 1.8, 1.8)
        ))
    )
    model = psiz.models.Rank(stimuli=stimuli, kernel=kernel)
    return model


def build_model(n_stimuli, n_dim, n_group, n_obs_train):
    """Build model.

    Arguments:
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
    stimuli = psiz.keras.layers.EmbeddingGroup(
        n_group=n_group, embedding=embedding_variational
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
        stimuli=stimuli, kernel=kernel, n_sample_test=3
    )
    return model


def plot_frame(
        fig0, n_obs, train_loss, val_loss, test_loss, r2, emb_true,
        emb_inferred, i_frame):
    """Plot posteriors."""
    # Settings.
    group_labels = ['Novice', 'Intermediate', 'Expert']

    n_group = emb_inferred.model.n_group
    n_dim = emb_inferred.model.n_dim

    gs = fig0.add_gridspec(n_group + 1, 4)

    f0_ax0 = fig0.add_subplot(gs[0, 0:2])
    plot_loss(f0_ax0, n_obs, train_loss, val_loss, test_loss)

    f0_ax1 = fig0.add_subplot(gs[0, 2])
    plot_convergence(fig0, f0_ax1, n_obs, r2[i_frame])

    gs.tight_layout(fig0)


def plot_logitnormal(ax, dist, name=None, c=None):
    """Plot univariate distribution.

    Arguments:
        ax:
        dist:
        name:

    """
    if name is None:
        name = 'x'

    x = np.linspace(.001, .999, 1000)
    y = dist.prob(x)

    # Determine mode from samples.
    idx = np.argmax(y)
    x_mode = x[idx]

    ax.plot(x, y, c=c)
    ax.text(x_mode, .75 * np.max(y), '{0:.2f}'.format(x_mode))
    ax.set_xlabel(r'${0}$'.format(name))
    ax.set_ylabel(r'$p({0})$'.format(name))
    ax.set_xlim([0, 1])
    ax.set_xticks([0, 1])
    ax.set_xticklabels([0, 1])


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


if __name__ == "__main__":
    main()
