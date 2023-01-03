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
"""Example that infers three group-specific stimulus embeddings.

The stimulus embedding is constructed in a hierarchical manner in order
to leverage the intuition that distinct-groups should have similar
embeddings.

Results are saved in the directory specified by `fp_project`. By
default, a `psiz_examples` directory is created in your home directory.

Example output:

    Model Comparison (R^2)
    ================================
      True  |        Inferred
            | Novice  Interm  Expert
    --------+-----------------------
     Novice |   0.98    0.95    0.72
     Interm |   0.92    0.94    0.83
     Expert |   0.71    0.79    0.88

"""

import copy
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # noqa
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
# tf.config.run_functions_eagerly(True)

# Uncomment and edit the following to control GPU visibility.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


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


class SimilarityModel(tf.keras.Model):
    """A similarity model."""

    def __init__(self, percept=None, kernel=None, **kwargs):
        """Initialize."""
        super(SimilarityModel, self).__init__(**kwargs)
        self.percept = percept
        self.kernel = kernel

    def call(self, inputs):
        """Call."""
        stimuli_axis = 1
        expertise_group = inputs['expertise']
        z = self.percept([inputs['rate2_stimulus_set'], expertise_group])
        z_0 = tf.gather(z, indices=tf.constant(0), axis=stimuli_axis)
        z_1 = tf.gather(z, indices=tf.constant(1), axis=stimuli_axis)
        return self.kernel([z_0, z_1])


class StochasticBehaviorModel(psiz.keras.StochasticModel):
    """A behavior model.

    No Gates.

    """

    def __init__(self, behavior=None, **kwargs):
        """Initialize."""
        super(StochasticBehaviorModel, self).__init__(**kwargs)
        self.behavior = behavior

    def call(self, inputs):
        """Call."""
        return self.behavior(inputs)


class StochasticSimilarityModel(psiz.keras.StochasticModel):
    """A similarity model."""

    def __init__(self, percept=None, kernel=None, **kwargs):
        """Initialize."""
        super(StochasticSimilarityModel, self).__init__(**kwargs)
        self.percept = percept
        self.kernel = kernel

    def call(self, inputs):
        """Call."""
        stimuli_axis = 1
        expertise_group = inputs['expertise']
        z = self.percept([inputs['rate2_stimulus_set'], expertise_group])
        z_0 = tf.gather(z, indices=tf.constant(0), axis=stimuli_axis)
        z_1 = tf.gather(z, indices=tf.constant(1), axis=stimuli_axis)
        return self.kernel([z_0, z_1])


def main():
    """Run script."""
    # Settings.
    fp_project = Path.home() / Path('psiz_examples', 'rank', 'vi_3ge')
    fp_board = fp_project / Path('logs', 'fit')
    n_stimuli = 30
    n_dim_inferred = 2
    n_group = 3
    epochs = 3000
    batch_size = 128
    n_trial = 30 * batch_size
    n_trial_train = 24 * batch_size
    n_trial_val = 3 * batch_size
    n_frame = 1  # Set n_frame > 1 to observe convergence behavior.

    # Directory preparation.
    fp_project.mkdir(parents=True, exist_ok=True)
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
    norm = matplotlib.colors.Normalize(vmin=0., vmax=n_stimuli)
    color_array = cmap(norm(range(n_stimuli)))

    # Define ground truth models.
    model_true = build_ground_truth_model()
    model_similarity_true = SimilarityModel(
        percept=model_true.behavior.percept,
        kernel=model_true.behavior.kernel
    )

    # Assemble dataset of stimuli pairs for comparing similarity matrices.
    # NOTE: We include an placeholder "target" component in dataset tuple to
    # satisfy the assumptions of `predict` method.
    content_pairs = psiz.data.Rate(
        psiz.utils.pairwise_indices(np.arange(n_stimuli) + 1, elements='upper')
    )
    group_0 = psiz.data.Group(
        np.tile(np.array([1.0, 0.0, 0.0]), (content_pairs.n_sample, 1)),
        name='expertise'
    )
    group_1 = psiz.data.Group(
        np.tile(np.array([0.0, 1.0, 0.0]), (content_pairs.n_sample, 1)),
        name='expertise'
    )
    group_2 = psiz.data.Group(
        np.tile(np.array([0.0, 0.0, 1.0]), (content_pairs.n_sample, 1)),
        name='expertise'
    )
    dummy_outcome = psiz.data.Continuous(np.ones([content_pairs.n_sample, 1]))
    tfds_pairs_group0 = psiz.data.Dataset(
        [content_pairs, group_0, dummy_outcome]
    ).export().batch(batch_size, drop_remainder=False)
    tfds_pairs_group1 = psiz.data.Dataset(
        [content_pairs, group_1, dummy_outcome]
    ).export().batch(batch_size, drop_remainder=False)
    tfds_pairs_group2 = psiz.data.Dataset(
        [content_pairs, group_2, dummy_outcome]
    ).export().batch(batch_size, drop_remainder=False)

    # Compute similarity matrix.
    simmat_truth = (
        model_similarity_true.predict(tfds_pairs_group0),
        model_similarity_true.predict(tfds_pairs_group1),
        model_similarity_true.predict(tfds_pairs_group2),
    )

    # Generate a random set of trials. Replicate for each group.
    rng = np.random.default_rng()
    eligibile_indices = np.arange(n_stimuli) + 1
    p = np.ones_like(eligibile_indices) / len(eligibile_indices)
    stimulus_set = psiz.utils.choice_wo_replace(
        eligibile_indices, (n_trial, 9), p, rng=rng
    )
    stimulus_set = np.repeat(stimulus_set, n_group, axis=0)
    content = psiz.data.Rank(stimulus_set, n_select=2)
    expertise = psiz.data.Group(
        np.tile(
            np.array(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                dtype=np.float32
            ),
            [n_trial, 1]
        ),
        name='expertise'
    )
    pds = psiz.data.Dataset([content, expertise])
    ds_content = pds.export(export_format='tfds')
    ds_content = ds_content.batch(batch_size, drop_remainder=False)

    # Simulate ranked similarity judgments and append to dataset.
    depth = content.n_outcome

    def simulate_agent(x):
        outcome_probs = model_true(x)
        outcome_distribution = tfp.distributions.Categorical(
            probs=outcome_probs
        )
        outcome_idx = outcome_distribution.sample()
        outcome_one_hot = tf.one_hot(outcome_idx, depth)
        return outcome_one_hot

    tfds_all = ds_content.map(lambda x: (x, simulate_agent(x))).unbatch()

    # Partition data into 80% train, 10% validation and 10% test set.
    tfds_train = tfds_all.take(n_trial_train)
    tfds_valtest = tfds_all.skip(n_trial_train)
    tfds_val = tfds_valtest.take(n_trial_val).cache().batch(
        batch_size, drop_remainder=False
    )
    tfds_test = tfds_valtest.skip(n_trial_val).cache().batch(
        batch_size, drop_remainder=False
    )

    # Infer independent models with increasing amounts of data.
    if n_frame == 1:
        n_trial_train_frame = np.array([n_trial_train], dtype=int)
    else:
        n_trial_train_frame = np.round(
            np.linspace(15, n_trial_train, n_frame)
        ).astype(np.int64)
    r2 = np.empty([n_frame, n_group, n_group]) * np.nan
    train_loss = np.empty((n_frame)) * np.nan
    val_loss = np.empty((n_frame)) * np.nan
    test_loss = np.empty((n_frame)) * np.nan
    for i_frame in range(n_frame):
        tfds_train_frame = tfds_train.take(
            int(n_trial_train_frame[i_frame])
        ).cache().shuffle(
            buffer_size=n_trial_train_frame[i_frame],
            reshuffle_each_iteration=True
        ).batch(
            batch_size, drop_remainder=False
        )
        print(
            '\n  Frame {0} ({1} samples)'.format(
                i_frame, n_trial_train_frame[i_frame]
            )
        )

        # Define model.
        kl_weight = 1. / n_trial_train_frame[i_frame]
        model_inferred = build_model(n_stimuli, n_dim_inferred, kl_weight)

        # Define callbacks.
        fp_board_frame = fp_board / Path('frame_{0}'.format(i_frame))
        cb_board = tf.keras.callbacks.TensorBoard(
            log_dir=fp_board_frame, histogram_freq=0,
            write_graph=False, write_images=False, update_freq='epoch',
            profile_batch=0, embeddings_freq=0, embeddings_metadata=None
        )
        cb_early = tf.keras.callbacks.EarlyStopping(
            'loss', patience=15, mode='min', restore_best_weights=False,
            verbose=1
        )
        callbacks = [cb_board, cb_early]

        # Infer model.
        history = model_inferred.fit(
            tfds_train_frame, validation_data=tfds_val, epochs=epochs,
            callbacks=callbacks, verbose=0
        )

        subnet_0 = model_inferred.behavior.percept.subnets[0]
        print(
            'avg scale posterior: {0:.4f}'.format(
                tf.reduce_mean(
                    subnet_0.posterior.embeddings.distribution.scale
                ).numpy()
            )
        )
        print(
            'avg scale intermediate: {0:.4f}'.format(
                tf.reduce_mean(
                    subnet_0.prior.embeddings.distribution.scale
                ).numpy()
            )
        )
        print(
            'avg scale prior: {0:.4f}'.format(
                subnet_0.prior.prior.embeddings.distribution.distribution.distribution.scale.numpy()[0, 0]
            )
        )

        train_loss[i_frame] = history.history['loss'][-1]
        val_loss[i_frame] = history.history['val_loss'][-1]

        tf.keras.backend.clear_session()
        model_inferred.n_sample = 100
        test_metrics = model_inferred.evaluate(
            tfds_test, verbose=0, return_dict=True
        )
        test_loss[i_frame] = test_metrics['loss']

        # Compare the inferred model with ground truth by comparing the
        # similarity matrices implied by each model.
        model_similarity_inferred = StochasticSimilarityModel(
            percept=model_inferred.behavior.percept,
            kernel=model_inferred.behavior.kernel,
            n_sample=100
        )
        simmat_inferred = (
            model_similarity_inferred.predict(tfds_pairs_group0),
            model_similarity_inferred.predict(tfds_pairs_group1),
            model_similarity_inferred.predict(tfds_pairs_group2),
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
            fig0,
            n_trial_train_frame,
            train_loss,
            val_loss,
            test_loss,
            r2,
            model_true,
            model_inferred,
            i_frame,
            color_array
        )
        fname = fp_project / Path('frame_{0}.tiff'.format(i_frame))
        plt.savefig(
            os.fspath(fname), format='tiff', bbox_inches="tight", dpi=300
        )

    # Create animation.
    if n_frame > 1:
        frames = []
        for i_frame in range(n_frame):
            fname = fp_project / Path('frame_{0}.tiff'.format(i_frame))
            frames.append(imageio.imread(fname))
        imageio.mimwrite(fp_project / Path('evolution.gif'), frames, fps=1)


def build_ground_truth_model():
    """Return a ground truth embedding."""
    # Mimicking a category learning task.
    n_class = 5
    n_exemplar_per_class = 6
    n_dim = 2
    # As task is learned, within-class exemplars become less similar via the
    # `exemplar_scale` variable.
    exemplar_scale = [0.1, .5, 1.0]

    # Draw class means.
    np.random.seed(258)
    class_mean = np.array([
        [-0., 0.],
        [-.14, .11],
        [-.14, -.12],
        [.16, .14],
        [.15, -.14],
    ])
    # Repeat class mean for each exemplar:
    class_mean_full = []
    for i_class in range(n_class):
        class_mean_full.append(
            np.repeat(
                np.expand_dims(class_mean[i_class], axis=0),
                repeats=n_exemplar_per_class, axis=0
            )
        )
    class_mean_full = np.concatenate(class_mean_full, axis=0)

    n_exemplar = (n_class * n_exemplar_per_class)
    exemplar_locs_centered = np.random.normal(
        loc=0.0, scale=.04, size=[n_exemplar, n_dim]
    )
    # Truncate exemplar draws.
    exemplar_locs_centered = np.minimum(exemplar_locs_centered, .05)

    percept_0 = build_ground_truth_percept(
        class_mean_full, exemplar_locs_centered, exemplar_scale[0]
    )
    percept_1 = build_ground_truth_percept(
        class_mean_full, exemplar_locs_centered, exemplar_scale[1]
    )
    percept_2 = build_ground_truth_percept(
        class_mean_full, exemplar_locs_centered, exemplar_scale[2]
    )
    percept = psiz.keras.layers.BraidGate(
        subnets=[percept_0, percept_1, percept_2], gating_index=-1
    )
    percept_adapter = psiz.keras.layers.GateAdapter(
        gating_keys='expertise', format_inputs_as_tuple=True
    )

    # Define shared kernel.
    kernel = psiz.keras.layers.DistanceBased(
        distance=psiz.keras.layers.Minkowski(
            trainable=False,
            rho_initializer=tf.keras.initializers.Constant(2.),
            w_initializer=tf.keras.initializers.Constant(1.)
        ),
        similarity=psiz.keras.layers.ExponentialSimilarity(
            trainable=False,
            beta_initializer=tf.keras.initializers.Constant(10.),
            tau_initializer=tf.keras.initializers.Constant(1.),
            gamma_initializer=tf.keras.initializers.Constant(0.)
        )
    )

    rank = psiz.keras.layers.RankSimilarity(
        n_reference=8,
        n_select=2,
        percept=percept,
        kernel=kernel,
        percept_adapter=percept_adapter
    )
    model = BehaviorModel(behavior=rank)
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=.001),
        weighted_metrics=[
            tf.keras.metrics.CategoricalCrossentropy(name='cce')
        ]
    )
    return model


def build_ground_truth_percept(class_mean_full, exemplar_locs_centered, scale):
    """Build a group-specific embedding."""
    n_stimuli, n_dim = np.shape(class_mean_full)
    exemplar_locs = class_mean_full + (scale * exemplar_locs_centered)
    # Add placeholder.
    exemplar_locs = np.vstack([np.array([0.0, 0.0]), exemplar_locs])
    stim = tf.keras.layers.Embedding(
        n_stimuli + 1, n_dim,
        embeddings_initializer=tf.keras.initializers.Constant(exemplar_locs),
        mask_zero=True
    )
    return stim


def build_model(n_stimuli, n_dim, kl_weight):
    """Build model.

    Args:
        n_stimuli: Integer indicating the number of stimuli in the
            embedding.
        n_dim: Integer indicating the dimensionality of the embedding.
        kl_weight: Float indicating the KL weight for variational
            inference.

    Returns:
        model: A TensorFlow Keras model.

    """
    # Start by defining population-level variational embedding.
    shared_prior = build_vi_shared_prior(n_stimuli, n_dim, kl_weight)

    # Define group-specific stimuli embeddings using the population-level
    # embedding as a shared prior.
    percept_0 = build_vi_percept(
        n_stimuli, n_dim, shared_prior, kl_weight, 'vi_percept_group_0'
    )
    percept_1 = build_vi_percept(
        n_stimuli, n_dim, shared_prior, kl_weight, 'vi_percept_group_1'
    )
    percept_2 = build_vi_percept(
        n_stimuli, n_dim, shared_prior, kl_weight, 'vi_percept_group_2'
    )
    percept = psiz.keras.layers.BraidGate(
        subnets=[percept_0, percept_1, percept_2], gating_index=-1,
    )
    percept_adapter = psiz.keras.layers.GateAdapter(
        gating_keys='expertise', format_inputs_as_tuple=True
    )

    # Define a simple, shared, non-trainable kernel.
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

    rank = psiz.keras.layers.RankSimilarity(
        n_reference=8,
        n_select=2,
        percept=percept,
        kernel=kernel,
        percept_adapter=percept_adapter
    )
    model = StochasticBehaviorModel(behavior=rank, n_sample=30)
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=.001),
        weighted_metrics=[
            tf.keras.metrics.CategoricalCrossentropy(name='cce')
        ]
    )
    return model


def build_vi_shared_prior(n_stimuli, n_dim, kl_weight):
    # Make `posterior_scale` small so that intermediate layer solutions start
    # out tight.
    posterior_scale = .0001
    prior_scale = .2
    embedding_posterior = psiz.keras.layers.EmbeddingNormalDiag(
        n_stimuli + 1, n_dim,
        scale_initializer=tf.keras.initializers.Constant(
            tfp.math.softplus_inverse(posterior_scale).numpy()
        ),
        mask_zero=True
    )
    embedding_prior = psiz.keras.layers.EmbeddingShared(
        n_stimuli+1, n_dim,
        embedding=psiz.keras.layers.EmbeddingNormalDiag(
            1, 1,
            loc_initializer=tf.keras.initializers.Constant(0.),
            scale_initializer=tf.keras.initializers.Constant(
                tfp.math.softplus_inverse(prior_scale).numpy()
            ),
            loc_trainable=False
        ),
        mask_zero=True
    )
    embedding_variational = psiz.keras.layers.EmbeddingVariational(
        posterior=embedding_posterior, prior=embedding_prior,
        kl_weight=kl_weight, kl_n_sample=30,
        name='vi_percept_shared'
    )
    return embedding_variational


def build_vi_percept(n_stimuli, n_dim, shared_prior, kl_weight, name):
    """Build VI group-specific stimuli embedding."""
    prior_scale = .0001  # Set to match posterior of shared prior.

    # NOTE: We initialize the locations to match the underlying prior.
    embedding_posterior = psiz.keras.layers.EmbeddingNormalDiag(
        n_stimuli+1, n_dim,
        loc_initializer=tf.keras.initializers.Constant(
            shared_prior.posterior.loc
        ),
        scale_initializer=tf.keras.initializers.Constant(
            tfp.math.softplus_inverse(prior_scale).numpy()
        ),
        mask_zero=True
    )
    embedding_variational = psiz.keras.layers.EmbeddingVariational(
        posterior=embedding_posterior, prior=shared_prior,
        kl_weight=kl_weight, kl_n_sample=30,
        name=name
    )
    return embedding_variational


def plot_frame(
        fig0, n_obs, train_loss, val_loss, test_loss, r2, model_true,
        model_inferred, i_frame, color_array):
    """Plot posteriors."""
    gs = fig0.add_gridspec(2, 2)

    f0_ax0 = fig0.add_subplot(gs[0, 0])
    plot_loss(f0_ax0, n_obs, train_loss, val_loss, test_loss)

    f0_ax1 = fig0.add_subplot(gs[0, 1])
    plot_convergence(fig0, f0_ax1, n_obs, r2[i_frame])

    f0_ax2 = fig0.add_subplot(gs[1, 0])
    plot_embeddings_true(fig0, f0_ax2, model_true, color_array)

    f0_ax2 = fig0.add_subplot(gs[1, 1])
    plot_embeddings(fig0, f0_ax2, model_true, model_inferred, color_array)

    gs.tight_layout(fig0)


def plot_loss(ax, n_obs, train_loss, val_loss, test_loss):
    """Plot loss."""
    # Settings
    ms = 2

    ax.plot(n_obs, train_loss, 'bo-', ms=ms, label='Train Loss')
    ax.plot(n_obs, val_loss, 'go-', ms=ms, label='Val. Loss')
    ax.plot(n_obs, test_loss, 'ro-', ms=ms, label='Test Loss')
    ax.set_title('Optimization Objective')
    ax.set_yscale('log')

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


def plot_embeddings_true(fig, ax, model_true, color_array):
    # Settings.
    n_group = 3
    marker_list = ['o', '<', 's']
    markersize = 15

    # Apply and plot Procrustes affine transformation of posterior.
    loc_list = []
    z_max = 0
    for i_group in range(n_group):
        loc = model_true.behavior.percept.subnets[i_group].embeddings
        if model_true.behavior.percept.subnets[i_group].mask_zero:
            # Drop placeholder stimulus.
            loc = loc[1:]
        # Center coordinates.
        # loc = loc - np.mean(loc, axis=0, keepdims=True)
        loc_list.append(loc)

        # Determine limits.
        z_max_curr = 1.1 * np.max(np.abs(loc))
        if z_max_curr > z_max:
            z_max = z_max_curr
    z_limits = [-z_max, z_max]

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
    ax.set_title('True')


def plot_embeddings(fig, ax, model_true, model_inferred, color_array):
    # Settings.
    n_group = 3
    marker_list = ['o', '<', 's']
    markersize = 15

    loc_true_2 = model_true.behavior.percept.subnets[2].embeddings.numpy()
    if model_true.behavior.percept.subnets[2].mask_zero:
        loc_true_2 = loc_true_2[1:]

    # Apply and plot Procrustes affine transformation of posterior.
    loc_list = []
    cov_list = []
    for i_group in range(n_group):
        dist = model_inferred.behavior.percept.subnets[i_group].embeddings
        loc, cov = unpack_mvn(dist)
        if model_inferred.behavior.percept.subnets[i_group].mask_zero:
            # Drop placeholder stimulus.
            loc = loc[1:]
            cov = cov[1:]
        # Center coordinates.
        # loc = loc - np.mean(loc, axis=0, keepdims=True)
        loc_list.append(loc)
        cov_list.append(cov)

    # Determine rotations into truth based on third group.
    rot = psiz.utils.procrustes_rotation(
        loc_list[2], loc_true_2, scale=False
    )

    # Apply rotations.
    loc_list[0], cov_list[0] = apply_affine(loc_list[0], cov_list[0], rot)
    loc_list[1], cov_list[1] = apply_affine(loc_list[1], cov_list[1], rot)
    loc_list[2], cov_list[2] = apply_affine(loc_list[2], cov_list[2], rot)

    # Plot inferred embeddings.
    z_max = 0
    for i_group in range(n_group):
        ax.scatter(
            loc_list[i_group][:, 0], loc_list[i_group][:, 1],
            s=markersize, c=color_array, marker=marker_list[i_group],
            edgecolors='none', zorder=100
        )
        # Determine limits.
        z_max_curr = 1.1 * np.max(np.abs(loc_list[i_group]))
        if z_max_curr > z_max:
            z_max = z_max_curr
    z_limits = [-z_max, z_max]

    ax.set_xlim(z_limits)
    ax.set_ylim(z_limits)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Inferred')


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
