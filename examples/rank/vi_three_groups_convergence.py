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

"""Example that infers a shared embedding for three groups.

Fake data is generated from a ground truth model for three different
groups. In this example, these groups represent groups of agents with
varying levels of skill: novices, intermediates, and experts. Each group
has a different set of attention weights. An embedding model is
inferred from the simulated data and compared to the ground truth
model.

Example output:

    Restart Summary
    n_valid_restart 1 | total_duration: 394 s
    best | n_epoch: 233 | val_cce: 2.8810
    mean ±stddev | n_epoch: 233 ±0 | val_cce: 2.8810 ±0.0000 | 380 ±0 s | 1633 ±0 ms/epoch

    Attention weights:
          Novice | [0.58 0.46 0.07 0.04]
    Intermediate | [0.34 0.26 0.37 0.24]
          Expert | [0.06 0.06 0.68 0.39]

    Model Comparison (R^2)
    ================================
      True  |        Inferred
            | Novice  Interm  Expert
    --------+-----------------------
     Novice |   0.97    0.59    0.15
     Interm |   0.61    0.97    0.63
     Expert |   0.15    0.60    0.97

"""

import copy
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
    """Run the simulation that infers an embedding for three groups."""
    # Settings.
    fp_ani = Path.home() / Path('vi_three_groups_convergence')  # TODO
    fp_ani.mkdir(parents=True, exist_ok=True)
    n_stimuli = 30
    n_dim = 4
    n_group = 3
    n_restart = 2  # TODO
    batch_size = 200
    n_frame = 4

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

    emb_true = ground_truth(n_stimuli, n_dim, n_group)

    # Generate a random docket of trials to show each group.
    n_trial = 2000
    n_reference = 8
    n_select = 2
    generator = psiz.generator.RandomGenerator(
        n_stimuli, n_reference=n_reference, n_select=n_select
    )
    docket = generator.generate(n_trial)

    # Create virtual agents for each group.
    agent_novice = psiz.generator.Agent(emb_true, group_id=0)
    agent_interm = psiz.generator.Agent(emb_true, group_id=1)
    agent_expert = psiz.generator.Agent(emb_true, group_id=2)

    # Simulate similarity judgments for each group.
    obs_novice = agent_novice.simulate(docket)
    obs_interm = agent_interm.simulate(docket)
    obs_expert = agent_expert.simulate(docket)
    obs_all = psiz.trials.stack((obs_novice, obs_interm, obs_expert))

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

    # Partition observations into train and validation set.
    skf = StratifiedKFold(n_splits=5)
    (train_idx, holdout_idx) = list(
        skf.split(obs_all.stimulus_set, obs_all.config_idx)
    )[0]
    obs_train = obs_all.subset(train_idx)
    obs_holdout = obs_all.subset(holdout_idx)
    skf = StratifiedKFold(n_splits=2)
    (val_idx, test_idx) = list(
        skf.split(obs_holdout.stimulus_set, obs_holdout.config_idx)
    )[0]
    obs_val = obs_holdout.subset(val_idx)
    obs_test = obs_holdout.subset(test_idx)

    # Use early stopping.
    cb_early = psiz.keras.callbacks.EarlyStoppingRe(
        'val_cce', patience=20, mode='min', restore_best_weights=True  # TODO val_loss
    )
    callbacks = [cb_early]

    compile_kwargs = {
        'loss': tf.keras.losses.CategoricalCrossentropy(),
        'optimizer': tf.keras.optimizers.Adam(lr=.001),
        'weighted_metrics': [
            tf.keras.metrics.CategoricalCrossentropy(name='cce')
        ]
    }

    # Infer independent models with increasing amounts of data.
    n_obs = np.round(
        np.linspace(15, obs_train.n_trial, n_frame)
    ).astype(np.int64)
    r2 = np.empty([n_frame, n_group, n_group]) * np.nan
    train_cce = np.empty((n_frame)) * np.nan
    val_cce = np.empty((n_frame)) * np.nan
    test_cce = np.empty((n_frame)) * np.nan
    for i_frame in range(n_frame):
        print('  Round {0}'.format(i_frame))
        include_idx = np.arange(0, n_obs[i_frame])
        obs_round_train = obs_train.subset(include_idx)

        # Define model.
        kl_weight = 1. / obs_round_train.n_trial
        embedding = psiz.keras.layers.EmbeddingVariational(
            n_stimuli+1, n_dim, mask_zero=True, kl_weight=kl_weight,
            prior_scale=.17
        )
        # kernel = psiz.keras.layers.AttentionKernel(
        #     distance=psiz.keras.layers.WeightedMinkowskiVariational(
        #         kl_weight=kl_weight
        #     ),
        #     attention=psiz.keras.layers.GroupAttentionVariational(
        #         n_dim=n_dim, n_group=n_group, kl_weight=kl_weight
        #     ),
        #     similarity=psiz.keras.layers.ExponentialSimilarityVariational(
        #         kl_weight=kl_weight
        #     )
        # )
        kernel = psiz.keras.layers.AttentionKernel(
            distance=psiz.keras.layers.WeightedMinkowski(
                fit_rho=False,
                rho_initializer=tf.keras.initializers.Constant(2.),
            ),
            attention=psiz.keras.layers.GroupAttentionVariational(
                n_dim=n_dim, n_group=n_group, kl_weight=kl_weight
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

        # Infer model.
        restart_record = emb_inferred.fit(
            obs_round_train, validation_data=obs_val, epochs=1000,
            batch_size=batch_size, callbacks=callbacks, n_restart=n_restart,
            monitor='val_cce', verbose=1, compile_kwargs=compile_kwargs  # TODO
        )

        train_cce[i_frame] = restart_record.record['cce'][0]
        val_cce[i_frame] = restart_record.record['val_cce'][0]
        test_metrics = emb_inferred.evaluate(
            obs_test, verbose=0, return_dict=True
        )
        test_cce[i_frame] = test_metrics['cce']

        # Compare the inferred model with ground truth by comparing the
        # similarity matrices implied by each model.
        def infer_sim_func0(z_q, z_ref):
            return emb_inferred.similarity(z_q, z_ref, group_id=0)

        def infer_sim_func1(z_q, z_ref):
            return emb_inferred.similarity(z_q, z_ref, group_id=1)

        def infer_sim_func2(z_q, z_ref):
            return emb_inferred.similarity(z_q, z_ref, group_id=2)

        simmat_infer = (
            psiz.utils.pairwise_matrix(infer_sim_func0, emb_inferred.z),
            psiz.utils.pairwise_matrix(infer_sim_func1, emb_inferred.z),
            psiz.utils.pairwise_matrix(infer_sim_func2, emb_inferred.z)
        )
        for i_truth in range(n_group):
            for j_infer in range(n_group):
                r2[i_frame, i_truth, j_infer] = psiz.utils.matrix_comparison(
                    simmat_truth[i_truth], simmat_infer[j_infer],
                    score='r2'
                )

        # Display attention weights.
        # Permute inferred dimensions to best match ground truth.
        idx_sorted = np.argsort(-emb_inferred.w[0, :])
        attention_weight = emb_inferred.w[:, idx_sorted]
        group_labels = ["Novice", "Intermediate", "Expert"]
        print("\n    Attention weights:")
        for i_group in range(emb_inferred.n_group):
            print("    {0:>12} | {1}".format(
                group_labels[i_group],
                np.array2string(
                    attention_weight[i_group, :],
                    formatter={'float_kind': lambda x: "%.2f" % x})
                )
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
            fig0, n_obs, train_cce, val_cce, test_cce, r2, emb_true,
            emb_inferred, color_array, idx_sorted, i_frame
        )
        fname = fp_ani / Path('frame_{0}.tiff'.format(i_frame))
        plt.savefig(
            os.fspath(fname), format='tiff', bbox_inches="tight", dpi=300
        )

    # Create animation.
    frames = []
    for i_frame in range(n_frame):
        fname = fp_ani / Path('frame_{0}.tiff'.format(i_frame))
        frames.append(imageio.imread(fname))
    imageio.mimwrite(fp_ani / Path('posterior.gif'), frames, fps=1)


def ground_truth(n_stimuli, n_dim, n_group):
    """Return a ground truth embedding."""
    embedding = tf.keras.layers.Embedding(
        n_stimuli+1, n_dim, mask_zero=True,
        embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=.17)
    )
    kernel = psiz.keras.layers.AttentionKernel(
        attention=psiz.keras.layers.GroupAttention(
            n_dim=n_dim, n_group=n_group
        ),
        similarity=psiz.keras.layers.ExponentialSimilarity()
    )
    model = psiz.models.Rank(embedding=embedding, kernel=kernel)
    emb = psiz.models.Proxy(model=model)

    emb.w = np.array((
        (1.8, 1.8, .2, .2),
        (1., 1., 1., 1.),
        (.2, .2, 1.8, 1.8)
    ))
    emb.theta = {
        'rho': 2.,
        'tau': 1.,
        'beta': 10.,
        'gamma': 0.,
    }
    return emb


def plot_frame(
        fig0, n_obs, train_cce, val_cce, test_cce, r2, emb_true, emb_inferred,
        color_array, idx_sorted, i_frame):
    """Plot posteriors."""
    # Settings.
    group_labels = ['Novice', 'Intermediate', 'Expert']

    n_group = emb_inferred.model.n_group
    n_dim = emb_inferred.model.n_dim

    gs = fig0.add_gridspec(n_group + 1, n_dim)

    f0_ax0 = fig0.add_subplot(gs[0, 0:2])
    plot_loss(f0_ax0, n_obs, train_cce, val_cce, test_cce)

    f0_ax1 = fig0.add_subplot(gs[0, 2])
    plot_convergence(fig0, f0_ax1, n_obs, r2[i_frame])

    for i_group in range(n_group):
        if i_group == 0:
            c = 'r'
        elif i_group == 1:
            c = 'b'
        elif i_group == 2:
            c = 'g'
        for i_dim in range(n_dim):
            # name = 'w_{{{0},{1}}}'.format(i_group, i_dim)
            name = 'w'
            ax = fig0.add_subplot(gs[i_group + 1, i_dim])
            curr_dim = idx_sorted[i_dim]
            loc = emb_inferred.model.kernel.attention.w_posterior.distribution.loc[i_group, curr_dim]
            scale = emb_inferred.model.kernel.attention.w_posterior.distribution.scale[i_group, curr_dim]
            dist = tfp.distributions.LogitNormal(loc=loc, scale=scale)
            plot_logitnormal(ax, dist, name=name, c=c)
            if i_group == 0:
                ax.set_title('Dimension {0}'.format(i_dim))
            # if i_dim == 0:
            #     ax.set_ylabel(group_labels[i_group])

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

    # Determine mode.
    # try # x_mode = dist.mode().numpy()  TODO
    idx = np.argmax(y)
    x_mode = x[idx]

    ax.plot(x, y, c=c)
    ax.text(x_mode, .75 * np.max(y), '{0:.2f}'.format(x_mode))
    ax.set_xlabel(r'${0}$'.format(name))
    ax.set_ylabel(r'$p({0})$'.format(name))
    ax.set_xlim([0, 1])
    ax.set_xticks([0, 1])
    ax.set_xticklabels([0, 1])


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


def plot_convergence(fig, ax, n_obs, r2):
    """Plot convergence."""
    # Settings.
    # ms = 2
    # r2_diag_0 = r2[:, 0, 0]
    # r2_diag_1 = r2[:, 1, 1]
    # r2_diag_2 = r2[:, 2, 2]

    # ax.plot(n_obs, r2_diag_0, 'o-',  ms=ms)
    # ax.plot(n_obs, r2_diag_1, 'o-',  ms=ms)
    # ax.plot(n_obs, r2_diag_2, 'o-',  ms=ms)
    # ax.set_title('Convergence')

    # ax.set_xlabel('Trials')
    # limits = [0, np.max(n_obs) + 10]
    # ax.set_xlim(limits)
    # ticks = [np.min(n_obs), np.max(n_obs)]
    # ax.set_xticks(ticks)

    # ax.set_ylabel(r'$R^2$')
    # ax.set_ylim(-0.05, 1.05)
    cmap = matplotlib.cm.get_cmap('Greys')
    im = ax.imshow(r2, cmap=cmap, vmin=0., vmax=1.)
    fig.colorbar(im, ax=ax)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Nov', 'Int', 'Exp'])
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['Nov', 'Int', 'Exp'])
    ax.set_ylabel('True')
    ax.set_xlabel('Inferred')
    ax.set_title(r'$R^2$ Convergence')


if __name__ == "__main__":
    main()
