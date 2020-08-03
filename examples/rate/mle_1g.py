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

import itertools
import os
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import psiz

# Uncomment the following line to force eager execution.
# tf.config.experimental_run_functions_eagerly(True)


def main():
    """Run script."""
    # Settings.
    fp_example = Path.home() / Path('psiz_examples', 'rate', 'mle_1g')
    fp_board = fp_example / Path('logs', 'fit')
    n_stimuli = 30
    n_dim = 2
    n_restart = 3
    n_trial = 1000
    epochs = 1000
    batch_size = 128
    n_frame = 1

    # Directory preparation.
    fp_example.mkdir(parents=True, exist_ok=True)
    # Remove existing TensorBoard logs.
    if fp_board.exists():
        shutil.rmtree(fp_board)

    model_true = ground_truth(n_stimuli, n_dim)

    # Generate a random docket of trials.
    generator = psiz.generator.RandomRate(n_stimuli)
    
    stimulus_set_0 = np.asarray(list(itertools.combinations(np.arange(n_stimuli), 2)))
    stimulus_set_1 = generator.generate(20)
    stimulus_set = np.vstack([stimulus_set_0, stimulus_set_1])
    n_trial = stimulus_set.shape[0]
    group = np.ones([n_trial], dtype=int)

    # Simulate similarity judgments.
    # agent = psiz.simulate.Agent(model_true.model)
    # obs = agent.simulate(docket)
    inputs = {
        'stimulus_set': tf.constant(stimulus_set + 1, dtype=tf.int32),
        'group': tf.constant(group, dtype=tf.int32)
    }
    output = model_true(inputs).numpy()

    proxy = psiz.models.Proxy(model_true)
    simmat_true = psiz.utils.pairwise_matrix(proxy.similarity, proxy.z)

    # Partition observations into 80% train, 10% validation and 10% test set.
    # obs_train, obs_val, obs_test = psiz.utils.standard_split(obs)
    w = np.ones(n_trial)
    locs_train = np.zeros([n_trial], dtype=bool)
    locs_train[0:-20] = True
    locs_val = np.logical_not(locs_train)

    # Create dataset.
    inputs_train = {
        'stimulus_set': tf.constant(stimulus_set[locs_train] + 1, dtype=tf.int32),
        'group': tf.constant(group[locs_train], dtype=tf.int32)
    }
    w_train = tf.constant(w[locs_train])
    y_train = tf.constant(output[locs_train])
    ds_obs_train = tf.data.Dataset.from_tensor_slices(
        (inputs_train, y_train, w_train)
    ).batch(batch_size, drop_remainder=False)

    inputs_val = {
        'stimulus_set': tf.constant(stimulus_set[locs_val] + 1, dtype=tf.int32),
        'group': tf.constant(group[locs_val], dtype=tf.int32)
    }
    w_val = tf.constant(w[locs_val])
    y_val = tf.constant(output[locs_val])
    ds_obs_val = tf.data.Dataset.from_tensor_slices(
        (inputs_val, y_val, w_val)
    ).batch(batch_size, drop_remainder=False)

    # Use early stopping.
    early_stop = psiz.keras.callbacks.EarlyStoppingRe(
        'val_mse', patience=30, mode='min', restore_best_weights=True
    )

    compile_kwargs = {
        'loss': tf.keras.losses.MeanSquaredError(),
        'optimizer': tf.keras.optimizers.Adam(lr=.001),
        'weighted_metrics': [
            tf.keras.metrics.MeanSquaredError(name='mse')
        ]
    }

    # Infer independent models with increasing amounts of data.
    n_obs = n_trial * np.ones([n_trial], dtype=int)  # TODO
    # if n_frame == 1:
    #     n_obs = np.array([obs_train.n_trial], dtype=int)
    # else:
    #     n_obs = np.round(
    #         np.linspace(15, obs_train.n_trial, n_frame)
    #     ).astype(np.int64)
    r2 = np.empty((n_frame))
    train_mse = np.empty((n_frame))
    val_mse = np.empty((n_frame))
    test_mse = np.empty((n_frame))
    for i_frame in range(n_frame):
        # include_idx = np.arange(0, n_obs[i_frame])
        # obs_round_train = obs_train.subset(include_idx)
        # print(
        #     '\n  Frame {0} ({1} obs)'.format(i_frame, obs_round_train.n_trial)
        # )
        # ds_obs_train = obs_train.as_dataset()
        # ds_obs_val = obs_val.as_dataset()

        # Use Tensorboard callback.
        fp_board_frame = fp_board / Path('frame_{0}'.format(i_frame))
        cb_board = psiz.keras.callbacks.TensorBoardRe(
            log_dir=fp_board_frame, histogram_freq=0,
            write_graph=False, write_images=False, update_freq='epoch',
            profile_batch=0, embeddings_freq=0, embeddings_metadata=None
        )
        callbacks = [early_stop, cb_board]

        model = build_model(n_stimuli, n_dim)

        # Infer embedding.
        model.compile(**compile_kwargs)
        history = model.fit(
            ds_obs_train, validation_data=ds_obs_val, epochs=epochs,
            callbacks=callbacks, verbose=1
        )

        train_mse[i_frame] = history.history['mse'][0]
        val_mse[i_frame] = history.history['val_mse'][0]
        test_metrics = model.evaluate(
            ds_obs_val, verbose=0, return_dict=True
        )
        test_mse[i_frame] = test_metrics['mse']

        # Compare the inferred model with ground truth by comparing the
        # similarity matrices implied by each model.
        proxy_inferred = psiz.models.Proxy(model)
        simmat_infer = psiz.utils.pairwise_matrix(
            proxy_inferred.similarity, proxy_inferred.z
        )
        r2[i_frame] = psiz.utils.matrix_comparison(
            simmat_infer, simmat_true, score='r2'
        )
        print(
            '    n_obs: {0:4d} | train_mse: {1:.2f} | '
            'val_mse: {2:.2f} | test_mse: {3:.2f} | '
            'Correlation (R^2): {4:.2f}'.format(
                n_obs[i_frame], train_mse[i_frame],
                val_mse[i_frame], test_mse[i_frame], r2[i_frame]
            )
        )

    # Plot comparison results.
    # fig, axes = plt.subplots(1, 2, figsize=(6.5, 3))

    # axes[0].plot(n_obs, train_mse, 'bo-', label='Train CCE')
    # axes[0].plot(n_obs, val_mse, 'go-', label='Val. CCE')
    # axes[0].plot(n_obs, test_mse, 'ro-', label='Test CCE')
    # axes[0].set_title('Model Loss')
    # axes[0].set_xlabel('Number of Judged Trials')
    # axes[0].set_ylabel('Loss')
    # axes[0].legend()

    # axes[1].plot(n_obs, r2, 'ro-')
    # axes[1].set_title('Model Convergence to Ground Truth')
    # axes[1].set_xlabel('Number of Judged Trials')
    # axes[1].set_ylabel(r'Squared Pearson Correlation ($R^2$)')
    # axes[1].set_ylim(-0.05, 1.05)

    # plt.tight_layout()
    # fname = fp_example / Path('evolution.tiff')
    # plt.savefig(
    #     os.fspath(fname), format='tiff', bbox_inches="tight", dpi=300
    # )


def ground_truth(n_stimuli, n_dim):
    """Return a ground truth embedding."""
    stimuli = tf.keras.layers.Embedding(
        n_stimuli+1, n_dim, mask_zero=True,
        embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=.17)
    )
    kernel = psiz.keras.layers.Kernel(
        distance=psiz.keras.layers.WeightedMinkowski(
            rho_initializer=tf.keras.initializers.Constant(2.),
            trainable=False
        ),
        similarity=psiz.keras.layers.ExponentialSimilarity(
            fit_tau=False, fit_gamma=False,
            tau_initializer=tf.keras.initializers.Constant(1.),
            gamma_initializer=tf.keras.initializers.Constant(0.001),
        )
    )
    behavior = psiz.keras.layers.RateBehavior(
        lower_initializer=tf.keras.initializers.Constant(0.0),
        upper_initializer=tf.keras.initializers.Constant(1.0),
        midpoint_initializer=tf.keras.initializers.Constant(.5),
        rate_initializer=tf.keras.initializers.Constant(5.),
    )
    print('Ground truth rate: {0:.2f}'.format(behavior.rate.numpy()))
    return psiz.models.Rate(
        stimuli=stimuli, kernel=kernel, behavior=behavior
    )


def build_model(n_stimuli, n_dim):
    stimuli = tf.keras.layers.Embedding(
        n_stimuli+1, n_dim, mask_zero=True
    )
    kernel = psiz.keras.layers.Kernel(
        similarity=psiz.keras.layers.ExponentialSimilarity()
    )
    behavior = BehaviorLog()
    model = psiz.models.Rate(
        stimuli=stimuli, kernel=kernel, behavior=behavior
    )
    return model


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
