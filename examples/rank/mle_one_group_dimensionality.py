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
"""Example that infers an embedding with dimensionality regularizer.

Fake data is generated from a ground truth model assuming one group.
The Squeeze regularizer encourages solutions that use a relatively low
number of dimensions.

"""
import numpy as np
import tensorflow as tf

import psiz

# Uncomment the following line to force eager execution.
# tf.config.experimental_run_functions_eagerly(True)


def main():
    """Run simulation."""
    # Settings.
    n_group = 1
    n_stimuli = 30
    n_dim_true = 3
    n_restart = 3
    n_dim_max = 30
    squeeze_rate = 1.
    n_trial = 5000
    batch_size = 100

    emb_true = ground_truth(n_stimuli, n_dim_true)

    # Generate a random docket of trials.
    generator = psiz.generator.RandomGenerator(
        n_stimuli, n_reference=8, n_select=2
    )
    docket = generator.generate(n_trial)

    # Simulate similarity judgments.
    agent = psiz.simulate.Agent(emb_true)
    obs = agent.simulate(docket)

    simmat_true = psiz.utils.similarity_matrix(emb_true.similarity, emb_true.z)

    # Partition observations into 80% train, 10% validation and 10% test set.
    obs_train, obs_val, obs_test = psiz.utils.standard_split(obs)

    # Use early stopping.
    early_stop = psiz.keras.callbacks.EarlyStoppingRe(
        'val_cce', patience=15, mode='min', restore_best_weights=True
    )
    callbacks = [early_stop]

    compile_kwargs = {
        'loss': tf.keras.losses.CategoricalCrossentropy(),
        'optimizer': tf.keras.optimizers.Adam(lr=.001),
        'weighted_metrics': [
            tf.keras.metrics.CategoricalCrossentropy(name='cce')
        ]
    }

    # Define model.
    embedding = tf.keras.layers.Embedding(
        n_stimuli+1, n_dim_max, mask_zero=True,
        embeddings_regularizer=psiz.keras.regularizers.Squeeze(
            rate=squeeze_rate
        )
    )
    kernel = psiz.keras.layers.Kernel(
        similarity=psiz.keras.layers.ExponentialSimilarity()
    )
    rank_model = psiz.models.Rank(embedding=embedding, kernel=kernel)
    emb_inferred = psiz.models.Proxy(model=rank_model)

    # Infer embedding.
    restart_record = emb_inferred.fit(
        obs_train, validation_data=obs_val, epochs=1000, batch_size=batch_size,
        callbacks=callbacks, n_restart=n_restart, monitor='val_cce', verbose=2,
        compile_kwargs=compile_kwargs
    )

    # Compare the inferred model with ground truth by comparing the
    # similarity matrices implied by each model.
    simmat_infer = psiz.utils.similarity_matrix(
        emb_inferred.similarity, emb_inferred.z
    )
    r_squared = psiz.utils.matrix_comparison(
        simmat_true, simmat_infer, score='r2'
    )
    print("    r2: {0:.2f}\n".format(r_squared))

    # Analyze effective dimensionality.
    dimension_usage_mean = np.mean(np.abs(emb_inferred.z), axis=0)
    dimension_usage_max = np.max(np.abs(emb_inferred.z), axis=0)
    max_val = np.max(dimension_usage_max)
    idx_sorted = np.argsort(-dimension_usage_max)
    dimension_usage_mean = dimension_usage_mean[idx_sorted] / max_val
    dimension_usage_max = dimension_usage_max[idx_sorted] / max_val
    print(
        "    Proportion mean: {0}".format(
            np.array2string(
                dimension_usage_mean,
                formatter={'float_kind': lambda x: "%.3f" % x}
            )
        )
    )
    print(
        "    Proportion max: {0}".format(
            np.array2string(
                dimension_usage_max,
                formatter={'float_kind': lambda x: "%.3f" % x}
            )
        )
    )


def ground_truth(n_stimuli, n_dim):
    """Return a ground truth embedding."""
    embedding = tf.keras.layers.Embedding(
        n_stimuli+1, n_dim, mask_zero=True,
        embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=.17)
    )
    kernel = psiz.keras.layers.Kernel(
        distance=psiz.keras.layers.WeightedMinkowski(
            rho_initializer=tf.keras.initializers.Constant(2.),
            trainable=False,
        ),
        similarity=psiz.keras.layers.ExponentialSimilarity(
            fit_tau=False, fit_gamma=False,
            tau_initializer=tf.keras.initializers.Constant(1.),
            gamma_initializer=tf.keras.initializers.Constant(0.001),
        )
    )
    rank_model = psiz.models.Rank(
        embedding=embedding, kernel=kernel
    )
    emb = psiz.models.Proxy(rank_model)
    return emb


if __name__ == "__main__":
    main()
