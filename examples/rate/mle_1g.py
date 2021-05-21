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

NOTE: The midpoint value has a large impact on the ability to infer a
reasonable solution. While the grid version works OK, the MVN case
is not working great. Once the above issues are resolved, still need to
experiment with noisy simulations and validation-based early stopping.

"""

import itertools
import os
from pathlib import Path
import shutil

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
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    """Run script."""
    # Settings.
    fp_example = Path.home() / Path('psiz_examples', 'rate', 'mle_1g')
    fp_board = fp_example / Path('logs', 'fit')
    fp_model = fp_example / Path('inferred_model')
    n_stimuli = 25
    n_dim = 2
    n_restart = 1
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
    fp_example.mkdir(parents=True, exist_ok=True)
    # Remove existing TensorBoard logs.
    if fp_board.exists():
        shutil.rmtree(fp_board)

    # Assemble dataset of stimuli pairs for comparing similarity matrices.
    ds_pairs, ds_info = psiz.utils.pairwise_index_dataset(
        n_stimuli, mask_zero=True
    )

    # Create a ground truth model.
    # You can choose a grid arrangement of Gaussian arrangement by
    # commenting/uncommenting the following two lines.
    model_true = ground_truth_grid()
    # model_true = ground_truth_randn(n_stimuli, n_dim)

    simmat_true = np.squeeze(
        psiz.utils.pairwise_similarity(
            model_true.stimuli, model_true.kernel, ds_pairs
        ).numpy()
    )

    print(
        'Ground Truth Pairwise Similarity\n'
        '    min: {0:.2f}'
        '    mean: {1:.2f}'
        '    max: {2:.2f}'.format(
            np.min(simmat_true), np.mean(simmat_true), np.max(simmat_true)
        )
    )

    # Assemble an exhaustive docket of all possible pairwise combinations.
    docket = exhaustive_docket(n_stimuli)
    ds_docket = docket.as_dataset().batch(
        batch_size=batch_size, drop_remainder=False
    )

    # Simulate noise-free similarity judgments.
    output = model_true.predict(ds_docket)
    print(
        'Observed Ratings\n'
        '    min: {0:.2f}'
        '    mean: {1:.2f}'
        '    max: {2:.2f}'.format(
            np.min(output),
            np.mean(output),
            np.max(output)
        )
    )
    obs = psiz.trials.RateObservations(docket.stimulus_set, output)

    ds_obs_train = obs.as_dataset().shuffle(
        buffer_size=obs.n_trial, reshuffle_each_iteration=True
    ).batch(batch_size, drop_remainder=False)

    compile_kwargs = {
        'loss': tf.keras.losses.MeanSquaredError(),
        'optimizer': tf.keras.optimizers.Adam(learning_rate=lr),
        'weighted_metrics': [
            tf.keras.metrics.MeanSquaredError(name='mse')
        ]
    }

    # Infer independent models with increasing amounts of data.
    for i_restart in range(n_restart):
        # Use Tensorboard callback.
        fp_board_frame = fp_board / Path('restart_{0}'.format(i_restart))
        cb_board = psiz.keras.callbacks.TensorBoardRe(
            log_dir=fp_board_frame, histogram_freq=0,
            write_graph=False, write_images=False, update_freq='epoch',
            profile_batch=0, embeddings_freq=0, embeddings_metadata=None
        )
        callbacks = [cb_board]

        model_inferred = build_model(n_stimuli, n_dim)

        # Infer embedding.
        model_inferred.compile(**compile_kwargs)
        history = model_inferred.fit(
            ds_obs_train, epochs=epochs, callbacks=callbacks, verbose=0
        )

        # train_mse = history.history['mse'][0]
        train_metrics = model_inferred.evaluate(
            ds_obs_train, verbose=0, return_dict=True
        )
        train_mse = train_metrics['mse']

        # Compare the inferred model with ground truth by comparing the
        # similarity matrices implied by each model.
        simmat_infer = np.squeeze(
            psiz.utils.pairwise_similarity(
                model_inferred.stimuli, model_inferred.kernel, ds_pairs
            ).numpy()
        )
        rho, _ = pearsonr(simmat_true, simmat_infer)
        r2 = rho**2
        print(
            '    n_obs: {0:4d} | train_mse: {1:.6f} | '
            'Correlation (R^2): {2:.2f}'.format(obs.n_trial, train_mse, r2)
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
        plot_restart(fig, model_true, model_inferred, r2)
        fname = fp_example / Path('restart_{0}.pdf'.format(i_restart))
        plt.savefig(
            os.fspath(fname), format='pdf', bbox_inches="tight", dpi=300
        )


def exhaustive_docket(n_stimuli):
    """Assemble an exhausitive docket.

    Arguments:
        n_stimuli: The number of stimuli.

    Returns:
        A psiz.trials.RateDocket object.

    """
    stimulus_set_self = np.stack(
        (np.arange(n_stimuli), np.arange(n_stimuli)), axis=1
    )
    stimulus_set_diff = np.asarray(
        list(itertools.combinations(np.arange(n_stimuli), 2))
    )
    stimulus_set = np.vstack((stimulus_set_self, stimulus_set_diff))
    return psiz.trials.RateDocket(stimulus_set)


def ground_truth_randn(n_stimuli, n_dim):
    """Return a ground truth embedding."""
    seed = 252
    stimuli = tf.keras.layers.Embedding(
        n_stimuli+1, n_dim, mask_zero=True,
        embeddings_initializer=tf.keras.initializers.RandomNormal(
            stddev=.17, seed=seed
        )
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

    behavior = psiz.keras.layers.RateBehavior(
        lower_initializer=tf.keras.initializers.Constant(0.0),
        upper_initializer=tf.keras.initializers.Constant(1.0),
        midpoint_initializer=tf.keras.initializers.Constant(.4),
        rate_initializer=tf.keras.initializers.Constant(15.),
    )
    print('Ground truth rate: {0:.2f}'.format(behavior.rate.numpy()))
    return psiz.keras.models.Rate(
        stimuli=stimuli, kernel=kernel, behavior=behavior
    )


def ground_truth_grid():
    """Create embedding points arranged on a grid."""
    x, y = np.meshgrid([-.2, -.1, 0., .1, .2], [-.2, -.1, 0., .1, .2])
    x = np.expand_dims(x.flatten(), axis=1)
    y = np.expand_dims(y.flatten(), axis=1)
    z_grid = np.hstack((x, y))
    (n_stimuli, n_dim) = z_grid.shape
    # Add placeholder.
    z_grid = np.vstack((np.ones([1, 2]), z_grid))

    stimuli = tf.keras.layers.Embedding(
        n_stimuli+1, n_dim, mask_zero=True,
        embeddings_initializer=tf.keras.initializers.Constant(
            z_grid
        )
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

    behavior = psiz.keras.layers.RateBehavior(
        lower_initializer=tf.keras.initializers.Constant(0.0),
        upper_initializer=tf.keras.initializers.Constant(1.0),
        midpoint_initializer=tf.keras.initializers.Constant(.5),
        rate_initializer=tf.keras.initializers.Constant(15.),
    )
    model = psiz.keras.models.Rate(
        stimuli=stimuli, kernel=kernel, behavior=behavior
    )

    return model


def build_model(n_stimuli, n_dim):
    """Build a model to use for inference."""
    stimuli = tf.keras.layers.Embedding(
        n_stimuli+1, n_dim, mask_zero=True
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

    behavior = BehaviorLog()
    model = psiz.keras.models.Rate(
        stimuli=stimuli, kernel=kernel, behavior=behavior
    )
    return model


def plot_restart(fig, model_true, model_inferred, r2):
    """Plot frame."""
    # Settings.
    cmap = matplotlib.cm.get_cmap('jet')
    norm = matplotlib.colors.Normalize(vmin=0., vmax=model_true.n_stimuli)
    color_array = cmap(norm(range(model_true.n_stimuli)))

    gs = fig.add_gridspec(1, 1)

    # Plot embeddings.
    ax = fig.add_subplot(gs[0, 0])

    # Grab stimuli embeddings.
    z_true = model_true.stimuli.embeddings.numpy()
    if model_true.stimuli.mask_zero:
        z_true = z_true[1:]
    z_inferred = model_inferred.stimuli.embeddings.numpy()
    if model_inferred.stimuli.mask_zero:
        z_inferred = z_inferred[1:]

    # Center coordinates.
    z_true = z_true - np.mean(z_true, axis=0, keepdims=True)
    z_inferred = z_inferred - np.mean(z_inferred, axis=0, keepdims=True)

    # Determine embedding limits.
    z_true_max = 1.3 * np.max(np.abs(z_true))
    z_infer_max = 1.3 * np.max(np.abs(z_inferred))
    z_max = np.max([z_true_max, z_infer_max])
    z_limits = [-z_max, z_max]

    # Align inferred embedding with true embedding (without using scaling).
    r = psiz.utils.procrustes_rotation(z_inferred, z_true, scale=False)
    z_inferred = np.matmul(z_inferred, r)

    # Plot true embedding.
    ax.scatter(
        z_true[:, 0], z_true[:, 1],
        s=15, c=color_array, marker='x', edgecolors='none'
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


class BehaviorLog(psiz.keras.layers.RateBehavior):
    """Sub-class for logging weight metrics."""

    def call(self, inputs):
        """Call."""
        outputs = super().call(inputs)

        # self.add_metric(
        #     self.lower,
        #     aggregation='mean', name='lower'
        # )
        # self.add_metric(
        #     self.upper,
        #     aggregation='mean', name='upper'
        # )
        self.add_metric(
            self.midpoint,
            aggregation='mean', name='midpoint'
        )
        self.add_metric(
            self.rate,
            aggregation='mean', name='rate'
        )

        return outputs


if __name__ == "__main__":
    main()
