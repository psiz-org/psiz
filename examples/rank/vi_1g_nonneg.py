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
    fp_example = Path.home() / Path('psiz_examples', 'vi_1g_nonneg')
    fp_board = fp_example / Path('logs', 'fit', 'r0')
    n_stimuli = 30
    n_dim = 2
    n_dim_nonneg = 20
    n_trial = 2000
    n_restart = 1
    batch_size = 100
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

    emb_true = ground_truth(n_stimuli, n_dim)

    # Generate a random docket of trials.
    generator = psiz.generator.RandomRank(
        n_stimuli, n_reference=8, n_select=2
    )
    docket = generator.generate(n_trial)

    # Simulate similarity judgments.
    agent = psiz.simulate.Agent(emb_true.model)
    obs = agent.simulate(docket)

    simmat_true = psiz.utils.pairwise_matrix(
        emb_true.similarity, emb_true.z)

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
    r2 = np.empty((n_frame)) * np.nan
    train_loss = np.empty((n_frame)) * np.nan
    val_loss = np.empty((n_frame)) * np.nan
    test_loss = np.empty((n_frame)) * np.nan
    train_time = np.empty((n_frame)) * np.nan
    for i_frame in range(n_frame):
        include_idx = np.arange(0, n_obs[i_frame])
        obs_round_train = obs_train.subset(include_idx)
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
        callbacks = [cb_board]

        # Define model.
        kl_weight = 1. / obs_round_train.n_trial

        embedding_posterior = psiz.keras.layers.EmbeddingTruncatedNormalDiag(
            n_stimuli+1, n_dim_nonneg, mask_zero=True,
            loc_initializer=tf.keras.initializers.RandomUniform(0., .05),
            scale_initializer=psiz.keras.initializers.SoftplusUniform(
                .01, .05
            ),
        )
        embedding_prior = psiz.keras.layers.EmbeddingShared(
            n_stimuli+1, n_dim_nonneg, mask_zero=True,
            embedding=psiz.keras.layers.EmbeddingGammaDiag(
                1, 1,
                concentration_initializer=tf.keras.initializers.Constant(1.),
                rate_initializer=tf.keras.initializers.Constant(1),
                concentration_trainable=False,
            )
        )
        stimuli = EmbeddingVariationalLog(
            posterior=embedding_posterior, prior=embedding_prior,
            kl_weight=kl_weight, kl_n_sample=30,
        )

        kernel = psiz.keras.layers.Kernel(
            distance=psiz.keras.layers.WeightedMinkowski(
                rho_initializer=tf.keras.initializers.Constant(1.3),
                trainable=False,
            ),
            similarity=psiz.keras.layers.ExponentialSimilarity(
                beta_initializer=tf.keras.initializers.Constant(1.),
                tau_initializer=tf.keras.initializers.Constant(1.),
                gamma_initializer=tf.keras.initializers.Constant(0.),
                trainable=False
            )
        )
        model = psiz.models.Rank(
            stimuli=stimuli, kernel=kernel, n_sample_test=3
        )
        emb_inferred = psiz.models.Proxy(model=model)

        # Infer embedding.
        restart_record = emb_inferred.fit(
            obs_round_train, validation_data=obs_val, epochs=1000,
            batch_size=batch_size, callbacks=callbacks, n_restart=n_restart,
            monitor='val_loss', verbose=1, compile_kwargs=compile_kwargs
        )

        train_loss[i_frame] = restart_record.record['loss'][0]
        train_time[i_frame] = restart_record.record['ms_per_epoch'][0]
        val_loss[i_frame] = restart_record.record['val_loss'][0]
        
        # Test.
        emb_inferred.model.n_sample_test = 100
        test_metrics = emb_inferred.evaluate(
            obs_test, verbose=0, return_dict=True
        )
        test_loss[i_frame] = test_metrics['loss']

        # Compare the inferred model with ground truth by comparing the
        # similarity matrices implied by each model.
        simmat_infer = psiz.utils.pairwise_matrix(
            emb_inferred.similarity, emb_inferred.z
        )
        r2[i_frame] = psiz.utils.matrix_comparison(
            simmat_infer, simmat_true, score='r2'
        )
        print(
            '    n_obs: {0:4d} | train_loss: {1:.2f} | '
            'val_loss: {2:.2f} | test_loss: {3:.2f} | '
            'Correlation (R^2): {4:.2f}'.format(
                n_obs[i_frame], train_loss[i_frame],
                val_loss[i_frame], test_loss[i_frame], r2[i_frame]
            )
        )

        # Create and save visual frame.
        fig0 = plt.figure(figsize=(6.5, 4), dpi=200)
        plot_frame(
            fig0, n_obs, train_loss, val_loss, test_loss, r2, emb_true,
            emb_inferred, train_time
        )
        fname = fp_example / Path('frame_{0}.tiff'.format(i_frame))
        plt.savefig(
            os.fspath(fname), format='tiff', bbox_inches="tight", dpi=300
        )

        print(np.max(emb_inferred.z, axis=0))
        print(np.min(np.max(emb_inferred.z, axis=0)))
        print(np.mean(emb_inferred.z, axis=0))

    # Create animation.
    if n_frame > 1:
        frames = []
        for i_frame in range(n_frame):
            fname = fp_example / Path('frame_{0}.tiff'.format(i_frame))
            frames.append(imageio.imread(fname))
        imageio.mimwrite(fp_example / Path('evolution.gif'), frames, fps=1)


def plot_frame(
        fig0, n_obs, train_loss, val_loss, test_loss, r2, emb_true, emb_inferred,
        train_time):
    """Plot frame."""
    # Settings.
    s = 10

    gs = fig0.add_gridspec(2, 6)

    f0_ax0 = fig0.add_subplot(gs[0, 0:2])
    plot_loss(f0_ax0, n_obs, train_loss, val_loss, test_loss)
 
    f0_ax1 = fig0.add_subplot(gs[0, 2:4])
    plot_convergence(f0_ax1, n_obs, r2)

    f0_ax2 = fig0.add_subplot(gs[0, 4:6])
    plot_time(f0_ax2, n_obs, train_time)

    # Visualize embedding point estimates.
    f0_ax3 = fig0.add_subplot(gs[1, 0:2])
    psiz.visualize.heatmap_embeddings(
        fig0, f0_ax3, emb_inferred.model.stimuli
    )

    # Visualize embedding distributions for the first dimension.
    f0_ax4 = fig0.add_subplot(gs[1, 2:6])
    i_dim = 0
    psiz.visualize.embedding_output_dimension(
        fig0, f0_ax4, emb_inferred.model.stimuli, i_dim
    )

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


def plot_time(ax, n_obs, train_time):
    # Settings.
    ms = 2

    ax.plot(n_obs, train_time, 'o-',  ms=ms,)
    ax.set_title('Train Time')

    ax.set_xlabel('Trials')
    limits = [0, np.max(n_obs) + 10]
    ax.set_xlim(limits)
    ticks = [np.min(n_obs), np.max(n_obs)]
    ax.set_xticks(ticks)

    ax.set_ylabel('ms')


def ground_truth(n_stimuli, n_dim):
    """Return a ground truth embedding."""
    # Settings.
    scale_request = .17

    stimuli = tf.keras.layers.Embedding(
        n_stimuli+1, n_dim, mask_zero=True,
        embeddings_initializer=tf.keras.initializers.RandomNormal(
            stddev=scale_request, seed=58
        ),
        trainable=False
    )
    stimuli.build([None, None, None])
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
    model = psiz.models.Rank(stimuli=stimuli, kernel=kernel)
    emb = psiz.models.Proxy(model=model)

    scale_sample = np.std(emb.model.stimuli.embeddings.numpy())
    return emb


@tf.keras.utils.register_keras_serializable(
    package='psiz.keras.layers', name='EmbeddingVariationalLog'
)
class EmbeddingVariationalLog(psiz.keras.layers.EmbeddingVariational):
    """Sub-class for logging weight metrics."""

    def call(self, inputs):
        """Call."""
        outputs = super().call(inputs)

        self.add_metric(
            self.kl_anneal,
            aggregation='mean', name='kl_anneal'
        )

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

        l = self.posterior.embeddings.distribution.loc[1:]
        s = self.posterior.embeddings.distribution.scale[1:]
        self.add_metric(
            tf.reduce_mean(l),
            aggregation='mean', name='po_loc_avg'
        )
        self.add_metric(
            tf.reduce_mean(s),
            aggregation='mean', name='po_scale_avg'
        )

        c = self.prior.embeddings.distribution.distribution.distribution.concentration
        r = self.prior.embeddings.distribution.distribution.distribution.rate
        self.add_metric(
            tf.reduce_mean(c),
            aggregation='mean', name='pr_con'
        )
        self.add_metric(
            tf.reduce_min(r),
            aggregation='mean', name='pr_rate_min'
        )
        self.add_metric(
            tf.reduce_max(r),
            aggregation='mean', name='pr_rate_max'
        )

        return outputs


if __name__ == "__main__":
    main()
