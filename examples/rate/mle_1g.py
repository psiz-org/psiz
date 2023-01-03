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

Results are saved in the directory specified by `fp_project`. By
default, a `psiz_examples` directory is created in your home directory.

NOTE: The midpoint value has a large impact on the ability to infer a
reasonable solution. While the grid version works OK, the MVN case
is not working great. Once the above issues are resolved, still need to
experiment with noisy simulations and validation-based early stopping.

"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # noqa
from pathlib import Path
import shutil

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
import tensorflow as tf

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
        z = self.percept(inputs['rate2_stimulus_set'])
        z_0 = tf.gather(z, indices=tf.constant(0), axis=stimuli_axis)
        z_1 = tf.gather(z, indices=tf.constant(1), axis=stimuli_axis)
        return self.kernel([z_0, z_1])


def main():
    """Run script."""
    # Settings.
    fp_project = Path.home() / Path('psiz_examples', 'rate', 'mle_1g')
    fp_board = fp_project / Path('logs', 'fit')
    n_stimuli = 25
    n_dim = 2
    epochs = 1000
    lr = .001
    batch_size = 64

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

    # Directory preparation.
    fp_project.mkdir(parents=True, exist_ok=True)
    # Remove existing TensorBoard logs.
    if fp_board.exists():
        shutil.rmtree(fp_board)

    # Define ground truth models.
    # You can choose a grid arrangement of Gaussian arrangement by
    # commenting/uncommenting the following two lines.
    model_true = build_ground_truth_grid()
    # model_true = build_ground_truth_randn(n_stimuli, n_dim)
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
    dummy_outcome = psiz.data.Continuous(np.ones([content_pairs.n_sample, 1]))
    tfds_pairs = psiz.data.Dataset(
        [content_pairs, dummy_outcome]
    ).export().batch(batch_size, drop_remainder=False)

    simmat_true = model_similarity_true.predict(tfds_pairs)

    print(
        'Ground Truth Pairwise Similarity\n'
        '    min: {0:.2f}'
        '    mean: {1:.2f}'
        '    max: {2:.2f}'.format(
            np.min(simmat_true), np.mean(simmat_true), np.max(simmat_true)
        )
    )

    # Assemble an exhaustive dataset of all possible pairwise combinations.
    eligible_indices = np.arange(n_stimuli) + 1
    content = psiz.data.Rate(
        psiz.utils.pairwise_indices(eligible_indices, elements='all')
    )
    pds = psiz.data.Dataset([content])
    tfds_content = pds.export(export_format='tfds')

    # Simulate noise-free similarity judgments and append outcomes to dataset.
    tfds_all = tfds_content.map(lambda x: (x, model_true(x)))

    n_trial_train = pds.n_sample
    tfds_train = tfds_all.cache().shuffle(
        buffer_size=n_trial_train, reshuffle_each_iteration=True
    ).batch(
        batch_size=batch_size, drop_remainder=False
    )

    # Use Tensorboard callback.
    fp_board_frame = fp_board / Path('frame_0')
    cb_board = tf.keras.callbacks.TensorBoard(
        log_dir=fp_board_frame, histogram_freq=0,
        write_graph=False, write_images=False, update_freq='epoch',
        profile_batch=0, embeddings_freq=0, embeddings_metadata=None
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        'mse', patience=15, mode='min', restore_best_weights=True
    )
    callbacks = [cb_board, early_stop]

    # Infer embedding.
    model_inferred = build_model(n_stimuli, n_dim, lr)
    model_inferred.fit(
        tfds_train, epochs=epochs, callbacks=callbacks, verbose=0
    )

    # train_mse = history.history['mse'][0]
    train_metrics = model_inferred.evaluate(
        tfds_train, verbose=0, return_dict=True
    )
    train_mse = train_metrics['mse']

    # Create model that outputs similarity based on inferred model.
    model_inferred_similarity = SimilarityModel(
        percept=model_inferred.behavior.percept,
        kernel=model_inferred.behavior.kernel
    )
    # Compare the inferred model with ground truth by comparing the
    # similarity matrices implied by each model.
    simmat_infer = model_inferred_similarity.predict(tfds_pairs)

    rho, _ = pearsonr(simmat_true, simmat_infer)
    r2 = rho**2
    print(
        '    n_obs: {0:4d} | train_mse: {1:.6f} | '
        'Correlation (R^2): {2:.2f}'.format(n_trial_train, train_mse, r2)
    )
    print(
        'Ground Truth parameters\n'
        '    sigmoid lower bound: {0:.2f}'
        '    sigmoid upper bound: {1:.2f}'
        '    sigmoid midpoint: {2:.2f}'
        '    sigmoid rate: {3:.2f}'.format(
            model_true.behavior.lower.numpy(),
            model_true.behavior.upper.numpy(),
            model_true.behavior.midpoint.numpy(),
            model_true.behavior.rate.numpy()
        )
    )
    print(
        'Inferred parameters\n'
        '    sigmoid lower bound: {0:.2f}'
        '    sigmoid upper bound: {1:.2f}'
        '    sigmoid midpoint: {2:.2f}'
        '    sigmoid rate: {3:.2f}'.format(
            model_inferred.behavior.lower.numpy(),
            model_inferred.behavior.upper.numpy(),
            model_inferred.behavior.midpoint.numpy(),
            model_inferred.behavior.rate.numpy()
        )
    )

    # Create and save visual frame.
    fig = plt.figure(figsize=(6.5, 4), dpi=200)
    plot_frame(fig, model_true, model_inferred, r2)
    fname = fp_project / Path('emb.pdf')
    plt.savefig(
        os.fspath(fname), format='pdf', bbox_inches="tight", dpi=300
    )


def build_ground_truth_randn(n_stimuli, n_dim):
    """Return a ground truth embedding."""
    seed = 252
    percept = tf.keras.layers.Embedding(
        n_stimuli + 1, n_dim,
        embeddings_initializer=tf.keras.initializers.RandomNormal(
            stddev=.17, seed=seed
        ),
        mask_zero=True
    )
    kernel = psiz.keras.layers.DistanceBased(
        distance=psiz.keras.layers.Minkowski(
            rho_initializer=tf.keras.initializers.Constant(2.),
            w_initializer=tf.keras.initializers.Constant(1.),
            trainable=False
        ),
        similarity=psiz.keras.layers.ExponentialSimilarity(
            trainable=False,
            beta_initializer=tf.keras.initializers.Constant(3.),
            tau_initializer=tf.keras.initializers.Constant(1.),
            gamma_initializer=tf.keras.initializers.Constant(0.),
        )
    )
    rate = psiz.keras.layers.RateSimilarity(
        percept=percept,
        kernel=kernel,
        lower_initializer=tf.keras.initializers.Constant(0.0),
        upper_initializer=tf.keras.initializers.Constant(1.0),
        midpoint_initializer=tf.keras.initializers.Constant(.4),
        rate_initializer=tf.keras.initializers.Constant(15.),
    )
    model = BehaviorModel(behavior=rate)
    return model


def build_ground_truth_grid():
    """Create embedding points arranged on a grid."""
    x, y = np.meshgrid([-.2, -.1, 0., .1, .2], [-.2, -.1, 0., .1, .2])
    x = np.expand_dims(x.flatten(), axis=1)
    y = np.expand_dims(y.flatten(), axis=1)
    z_grid = np.hstack((x, y))
    (n_stimuli, n_dim) = z_grid.shape
    # Add placeholder.
    z_grid = np.vstack((np.ones([1, 2]), z_grid))

    percept = tf.keras.layers.Embedding(
        n_stimuli + 1, n_dim,
        embeddings_initializer=tf.keras.initializers.Constant(
            z_grid
        ),
        mask_zero=True
    )

    kernel = psiz.keras.layers.DistanceBased(
        distance=psiz.keras.layers.Minkowski(
            rho_initializer=tf.keras.initializers.Constant(2.),
            w_initializer=tf.keras.initializers.Constant(1.),
            trainable=False
        ),
        similarity=psiz.keras.layers.ExponentialSimilarity(
            trainable=False,
            beta_initializer=tf.keras.initializers.Constant(3.),
            tau_initializer=tf.keras.initializers.Constant(1.),
            gamma_initializer=tf.keras.initializers.Constant(0.),
        )
    )

    rate = psiz.keras.layers.RateSimilarity(
        percept=percept,
        kernel=kernel,
        lower_initializer=tf.keras.initializers.Constant(0.0),
        upper_initializer=tf.keras.initializers.Constant(1.0),
        midpoint_initializer=tf.keras.initializers.Constant(.5),
        rate_initializer=tf.keras.initializers.Constant(15.),
    )
    model = BehaviorModel(behavior=rate)

    return model


def build_model(n_stimuli, n_dim, lr):
    """Build a model to use for inference."""
    percept = tf.keras.layers.Embedding(
        n_stimuli + 1, n_dim, mask_zero=True
    )
    kernel = psiz.keras.layers.DistanceBased(
        distance=psiz.keras.layers.Minkowski(
            rho_initializer=tf.keras.initializers.Constant(2.),
            w_initializer=tf.keras.initializers.Constant(1.),
            trainable=False
        ),
        similarity=psiz.keras.layers.ExponentialSimilarity(
            trainable=False,
            beta_initializer=tf.keras.initializers.Constant(3.),
            tau_initializer=tf.keras.initializers.Constant(1.),
            gamma_initializer=tf.keras.initializers.Constant(0.),
        )
    )
    rate = psiz.keras.layers.RateSimilarity(percept=percept, kernel=kernel)
    model = BehaviorModel(behavior=rate)
    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        weighted_metrics=[
            tf.keras.metrics.MeanSquaredError(name='mse')
        ]
    )
    return model


def plot_frame(fig, model_true, model_inferred, r2):
    """Plot frame."""
    # Settings.
    n_stimuli = model_true.behavior.percept.input_dim
    if model_true.behavior.percept.mask_zero:
        n_stimuli = n_stimuli - 1
    cmap = matplotlib.cm.get_cmap('jet')
    norm = matplotlib.colors.Normalize(vmin=0., vmax=n_stimuli)
    color_array = cmap(norm(range(n_stimuli)))

    gs = fig.add_gridspec(1, 1)

    # Plot embeddings.
    ax = fig.add_subplot(gs[0, 0])

    # Grab percept embeddings.
    z_true = model_true.behavior.percept.embeddings.numpy()
    if model_true.behavior.percept.mask_zero:
        z_true = z_true[1:]
    z_inferred = model_inferred.behavior.percept.embeddings.numpy()
    if model_inferred.behavior.percept.mask_zero:
        z_inferred = z_inferred[1:]

    # Center coordinates.
    z_true = z_true - np.mean(z_true, axis=0, keepdims=True)
    z_inferred = z_inferred - np.mean(z_inferred, axis=0, keepdims=True)

    # Align inferred embedding with true embedding (without using scaling).
    r = psiz.utils.procrustes_rotation(z_inferred, z_true, scale=True)
    z_inferred = np.matmul(z_inferred, r)

    # Determine embedding limits.
    z_true_max = 1.3 * np.max(np.abs(z_true))
    z_infer_max = 1.3 * np.max(np.abs(z_inferred))
    z_max = np.max([z_true_max, z_infer_max])
    z_limits = [-z_max, z_max]

    # Plot true embedding.
    ax.scatter(
        z_true[:, 0], z_true[:, 1], s=15, c=color_array, marker='x'
    )

    # Plot inferred embedding.
    ax.scatter(
        z_inferred[:, 0], z_inferred[:, 1],
        s=60, marker='o', facecolors='none', edgecolors=color_array
    )

    ax.set_xlim(z_limits)
    ax.set_ylim(z_limits)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Embeddings (R^2={0:.2f})'.format(r2))

    gs.tight_layout(fig)


if __name__ == "__main__":
    main()
