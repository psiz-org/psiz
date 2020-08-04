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
import tensorflow_probability as tfp

import psiz

# Uncomment the following line to force eager execution.
# tf.config.experimental_run_functions_eagerly(True)


def main():
    """Run script."""
    # Settings.
    fp_example = Path.home() / Path('psiz_examples', 'rate', 'mle_1g')
    fp_board = fp_example / Path('logs', 'fit')
    n_stimuli = 25  # 30
    n_dim = 2
    n_restart = 10
    epochs = 1000
    lr = .001
    batch_size = 64
    n_frame = 1

    # randn 30 stim
    # epochs = 1000 lr=.001 =>  128: .83, .76 | 64: .94, .90 | 32: .88, .86
    # epochs = 3000 lr=.001 =>  128: .85, .82 | 64: .70 .63 | 32: .85 .79
    # epochs = 1000 lr=.0001 => 128: .32, .30 | 64:  | 32: 

    # MLE vs VI
    # epochs = 1000 lr=.001 bs=64 =>
    # mle: .94 .88

    # Directory preparation.
    fp_example.mkdir(parents=True, exist_ok=True)
    # Remove existing TensorBoard logs.
    if fp_board.exists():
        shutil.rmtree(fp_board)

    model_true = ground_truth_randn(n_stimuli, n_dim)
    # model_true = ground_truth_grid()

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

        
        # model = build_model(n_stimuli, n_dim)
        model = build_model_vi(n_stimuli, n_dim, obs.n_trial)

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


def ground_truth_grid():
    #  Create embedding points arranged on a grid.
    x, y = np.meshgrid([.1, .2, .3, .4, .5], [.1, .2, .3, .4, .5])
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
            beta_initializer=tf.keras.initializers.Constant(10.),
            tau_initializer=tf.keras.initializers.Constant(1.),
            gamma_initializer=tf.keras.initializers.Constant(0.),
            trainable=False
        )
    )
    behavior = psiz.keras.layers.RateBehavior(
        lower_initializer=tf.keras.initializers.Constant(0.0),
        upper_initializer=tf.keras.initializers.Constant(1.0),
        midpoint_initializer=tf.keras.initializers.Constant(.5),
        rate_initializer=tf.keras.initializers.Constant(5.),
    )
    model = psiz.models.Rate(
        stimuli=stimuli, kernel=kernel, behavior=behavior
    )
    model.stimuli.build([None, None, None])
    model.stimuli.embedding.embeddings.assign(z_grid)
    return model


def build_model(n_stimuli, n_dim):
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
            fit_tau=False, fit_gamma=False,
            tau_initializer=tf.keras.initializers.Constant(1.),
            gamma_initializer=tf.keras.initializers.Constant(0.),
        )
    )

    behavior = BehaviorLog()
    model = psiz.models.Rate(
        stimuli=stimuli, kernel=kernel, behavior=behavior
    )
    return model


def build_model_vi(n_stimuli, n_dim, n_obs_train):
    kl_weight = 1. / n_obs_train

    embedding_posterior = psiz.keras.layers.EmbeddingNormalDiag(
        n_stimuli+1, n_dim, mask_zero=True,
        scale_initializer=tf.keras.initializers.Constant(
            tfp.math.softplus_inverse(.01).numpy()
        )
    )

    prior_scale = .2
    embedding_prior = psiz.keras.layers.EmbeddingShared(
        n_stimuli+1, n_dim, mask_zero=True,
        embedding=psiz.keras.layers.EmbeddingNormalDiag(
            1, 1,
            loc_initializer=tf.keras.initializers.Constant(0.),
            scale_initializer=tf.keras.initializers.Constant(
                tfp.math.softplus_inverse(prior_scale).numpy()
            ),
            loc_trainable=False,
        )
    )
    embedding_variational = psiz.keras.layers.EmbeddingVariational(
        posterior=embedding_posterior, prior=embedding_prior,
        kl_weight=kl_weight, kl_n_sample=30
    )
    stimuli = psiz.keras.layers.Stimuli(embedding=embedding_variational)

    kernel = psiz.keras.layers.Kernel(
        distance=psiz.keras.layers.WeightedMinkowski(
            rho_initializer=tf.keras.initializers.Constant(2.),
            trainable=False,
        ),
        similarity=psiz.keras.layers.ExponentialSimilarity(
            fit_tau=False, fit_gamma=False,
            tau_initializer=tf.keras.initializers.Constant(1.),
            gamma_initializer=tf.keras.initializers.Constant(0.),
        )
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
