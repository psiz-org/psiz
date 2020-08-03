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
tf.config.experimental_run_functions_eagerly(True)


def main():
    """Run script."""
    # Settings.
    fp_example = Path.home() / Path('psiz_examples', 'rate', 'mle_1g')
    fp_board = fp_example / Path('logs', 'fit')
    n_stimuli = 30
    n_dim = 2
    n_restart = 3
    epochs = 300
    batch_size = 32
    n_frame = 1

    # Directory preparation.
    fp_example.mkdir(parents=True, exist_ok=True)
    # Remove existing TensorBoard logs.
    if fp_board.exists():
        shutil.rmtree(fp_board)

    model_true = ground_truth(n_stimuli, n_dim)

    proxy = psiz.models.Proxy(model_true)
    simmat_true = psiz.utils.pairwise_matrix(proxy.similarity, proxy.z[0])

    # Generate a random docket of trials.
    # generator = psiz.generator.RandomRate(n_stimuli)
    # rate_docket_1 = generator.generate(20)

    stimulus_set_all = np.asarray(list(itertools.combinations(np.arange(n_stimuli), 2)))
    n_trial = stimulus_set_all.shape[0]
    rate_docket_all = psiz.trials.RateDocket(stimulus_set_all)
    group = np.ones([n_trial], dtype=int)
    ds_docket = rate_docket_all.as_dataset(group)
    
    # Simulate noiseless similarity judgments.
    output = np.mean(model_true(ds_docket).numpy(), axis=0)

    obs = psiz.trials.RateObservations(stimulus_set_all, output)
    
    ds_obs_train = obs.as_dataset().shuffle(
        buffer_size=obs.n_trial, reshuffle_each_iteration=True
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
    for i_restart in range(30):
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
            '    n_obs: {0:4d} | train_mse: {1:.2f} | '
            'Correlation (R^2): {2:.2f}'.format(n_trial, train_mse, r2)
        )


def ground_truth(n_stimuli, n_dim):
    """Return a ground truth embedding."""
    stimuli = psiz.keras.layers.Stimuli(
        embedding=tf.keras.layers.Embedding(
            n_stimuli+1, n_dim, mask_zero=True,
            embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=.17)
        )
    )
    stimuli.build([None, None, None])
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
    stimuli = psiz.keras.layers.Stimuli(
        embedding=tf.keras.layers.Embedding(
            n_stimuli+1, n_dim, mask_zero=True
        )
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
