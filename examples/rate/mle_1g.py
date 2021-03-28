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
import tensorflow as tf
import tensorflow_probability as tfp

import psiz

# Uncomment the following line to force eager execution.
# tf.config.experimental_run_functions_eagerly(True)

# Modify the following to control GPU visibility.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def main():
    """Run script."""
    # Settings.
    fp_example = Path.home() / Path('psiz_examples', 'rate', 'mle_1g')
    fp_board = fp_example / Path('logs', 'fit')
    n_stimuli = 25
    n_dim = 2
    n_restart = 1
    epochs = 10000
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

    # Create a ground truth model.
    model_true = ground_truth_grid()
    # model_true = ground_truth_randn(n_stimuli, n_dim)

    proxy_true = psiz.models.Proxy(model_true)
    simmat_true = psiz.utils.pairwise_matrix(
        proxy_true.similarity, proxy_true.z[0]
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
    output = np.random.binomial(1, output)
    print(output.shape)
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
        'optimizer': tf.keras.optimizers.Adam(lr=lr),
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

        model = build_model(n_stimuli, n_dim)

        # Infer embedding.
        model.compile(**compile_kwargs)
        history = model.fit(
            ds_obs_train, epochs=epochs, callbacks=callbacks, verbose=0
        )

        # train_mse = history.history['mse'][0]
        train_metrics = model.evaluate(
            ds_obs_train, verbose=0, return_dict=True
        )
        train_mse = train_metrics['mse']

        # Compare the inferred model with ground truth by comparing the
        # similarity matrices implied by each model.
        proxy_inferred = psiz.models.Proxy(model)
        simmat_infer = psiz.utils.pairwise_matrix(
            proxy_inferred.similarity, proxy_inferred.z[0]
        )
        r2 = psiz.utils.matrix_comparison(
            simmat_infer, simmat_true, score='r2'
        )
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
                model.behavior.lower.numpy(),
                model.behavior.upper.numpy(),
                model.behavior.midpoint.numpy(),
                model.behavior.rate.numpy()
            )
        )

        # Create and save visual frame.
        fig = plt.figure(figsize=(6.5, 4), dpi=200)
        plot_restart(fig, proxy_true, proxy_inferred, r2)
        fname = fp_example / Path('restart_{0}.pdf'.format(i_restart))
        plt.savefig(
            os.fspath(fname), format='pdf', bbox_inches="tight", dpi=300
        )
        
    print(model)
    return model

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
    stimulus_set_temp = np.vstack((stimulus_set_self, stimulus_set_diff))
    stimulus_set = np.vstack([stimulus_set_temp for i in range(100)])
    return psiz.trials.RateDocket(stimulus_set)


def ground_truth_randn(n_stimuli, n_dim):
    """Return a ground truth embedding."""
    seed = 252
    stimuli = psiz.keras.layers.Stimuli(
        embedding=tf.keras.layers.Embedding(
            n_stimuli+1, n_dim, mask_zero=True,
            embeddings_initializer=tf.keras.initializers.RandomNormal(
                stddev=.17, seed=seed
            )
        )
    )
    stimuli.build([None, None, None])
    kernel = psiz.keras.layers.Kernel(
        distance=psiz.keras.layers.WeightedMinkowski(
            rho_initializer=tf.keras.initializers.Constant(2.),
            trainable=False
        ),
        similarity=psiz.keras.layers.ExponentialSimilarity(
            beta_initializer=tf.keras.initializers.Constant(3.),
            tau_initializer=tf.keras.initializers.Constant(1.),
            gamma_initializer=tf.keras.initializers.Constant(0.),
            trainable=False,
        )
    )
    behavior = psiz.keras.layers.RateBehavior(
        lower_initializer=tf.keras.initializers.Constant(0.0),
        upper_initializer=tf.keras.initializers.Constant(1.0),
        midpoint_initializer=tf.keras.initializers.Constant(.4),
        rate_initializer=tf.keras.initializers.Constant(15.),
    )
    print('Ground truth rate: {0:.2f}'.format(behavior.rate.numpy()))
    return psiz.models.Rate(
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

    stimuli = psiz.keras.layers.Stimuli(
        embedding=tf.keras.layers.Embedding(
            n_stimuli+1, n_dim, mask_zero=True,
        )
    )
    kernel = psiz.keras.layers.Kernel(
        distance=psiz.keras.layers.WeightedMinkowski(
            rho_initializer=tf.keras.initializers.Constant(2.),
            trainable=False,
        ),
        similarity=psiz.keras.layers.ExponentialSimilarity(
            beta_initializer=tf.keras.initializers.Constant(3.),
            tau_initializer=tf.keras.initializers.Constant(1.),
            gamma_initializer=tf.keras.initializers.Constant(0.),
            trainable=False
        )
    )
    behavior = psiz.keras.layers.RateBehavior(
        lower_initializer=tf.keras.initializers.Constant(0.0),
        upper_initializer=tf.keras.initializers.Constant(1.0),
        midpoint_initializer=tf.keras.initializers.Constant(.5),
        rate_initializer=tf.keras.initializers.Constant(15.),
    )
    model = psiz.models.Rate(
        stimuli=stimuli, kernel=kernel, behavior=behavior
    )
    model.stimuli.build([None, None, None])
    model.stimuli.embedding.embeddings.assign(z_grid)
    return model


def build_model(n_stimuli, n_dim):
    """Build a model to use for inference."""
    stimuli = psiz.keras.layers.Stimuli(
        embedding=tf.keras.layers.Embedding(
            n_stimuli+1, n_dim, mask_zero=True
        )
    )

    kernel = psiz.keras.layers.Kernel(
        distance=psiz.keras.layers.WeightedMinkowski(
            rho_initializer=tf.keras.initializers.Constant(2.),
            trainable=False,
        ),
        similarity=psiz.keras.layers.ExponentialSimilarity(
            fit_tau=False, fit_gamma=False, fit_beta=False,
            beta_initializer=tf.keras.initializers.Constant(3.),
            tau_initializer=tf.keras.initializers.Constant(1.),
            gamma_initializer=tf.keras.initializers.Constant(0.),
        )
    )

    behavior = BehaviorLog()
    model = psiz.models.Rate(
        stimuli=stimuli, kernel=kernel, behavior=behavior
    )
    return model


def plot_restart(fig, proxy_true, proxy_inferred, r2):
    """Plot frame."""
    # Settings.
    cmap = matplotlib.cm.get_cmap('jet')
    norm = matplotlib.colors.Normalize(vmin=0., vmax=proxy_true.n_stimuli)
    color_array = cmap(norm(range(proxy_true.n_stimuli)))

    gs = fig.add_gridspec(1, 1)

    # Plot embeddings.
    ax = fig.add_subplot(gs[0, 0])

    # Determine embedding limits.
    z_true_max = 1.3 * np.max(np.abs(proxy_true.z[0]))
    z_infer_max = 1.3 * np.max(np.abs(proxy_inferred.z[0]))
    z_max = np.max([z_true_max, z_infer_max])
    z_limits = [-z_max, z_max]

    # Align inferred embedding with true embedding (without using scaling).
    r, t = psiz.utils.procrustes_2d(
        proxy_true.z[0], proxy_inferred.z[0], scale=False, n_restart=30
    )
    z_affine = np.matmul(proxy_inferred.z[0], r) + t

    # Plot true embedding.
    ax.scatter(
        proxy_true.z[0, :, 0], proxy_true.z[0, :, 1],
        s=15, c=color_array, marker='x', edgecolors='none'
    )

    # Plot inferred embedding.
    ax.scatter(
        z_affine[:, 0], z_affine[:, 1],
        s=60, marker='o', facecolors='none', edgecolors=color_array
    )

    ax.set_xlim(z_limits)
    ax.set_ylim(z_limits)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Embeddings (R^2={0:.2f})'.format(r2))

    gs.tight_layout(fig)


@tf.keras.utils.register_keras_serializable(
    package='psiz.keras.layers', name='BehaviorLog'
)
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
