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

"""Example that infers an embedding from two trial configurations.

Fake data is generated from a ground truth model for two different
trial configurations: 2-choose-1 and 8-choose-2. This example
demonstrates how one can use data collected in a variety of formats to
infer a single embedding.

Example output:

    R^2 Model Comparison:   0.95

"""

import edward2 as ed
import numpy as np
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf

import psiz

# Uncomment the following line to force eager execution.
tf.config.experimental_run_functions_eagerly(True)


def main():
    """Run script."""
    # Settings.
    n_stimuli = 25
    n_dim = 3
    n_restart = 3

    # Ground truth embedding.
    emb_true = ground_truth(n_stimuli, n_dim)

    # Generate a random docket of 8-choose-2 trials.
    n_reference = 8
    n_select = 2
    gen_8c2 = psiz.generator.RandomGenerator(
        n_stimuli, n_reference=n_reference, n_select=n_select
    )
    n_trial = 2000
    docket = gen_8c2.generate(n_trial)

    # Simulate similarity judgments.
    agent = psiz.simulate.Agent(emb_true)
    obs = agent.simulate(docket)

    # Partition observations into train and validation set.
    skf = StratifiedKFold(n_splits=10)
    (train_idx, val_idx) = list(
        skf.split(obs.stimulus_set, obs.config_idx)
    )[0]
    obs_train = obs.subset(train_idx)
    obs_val = obs.subset(val_idx)

    # Use early stopping.
    cb_early = psiz.keras.callbacks.EarlyStoppingRe(
        'val_nll', patience=10, mode='min', restore_best_weights=True
    )
    # Visualize using TensorBoard.
    cb_board = psiz.keras.callbacks.TensorBoardRe(
        log_dir='/tmp/psiz/tensorboard_logs', histogram_freq=0,
        write_graph=False, write_images=False, update_freq='epoch',
        profile_batch=0, embeddings_freq=0, embeddings_metadata=None
    )
    callbacks = [cb_early, cb_board]

    compile_kwargs = {
        'loss': tf.keras.losses.CategoricalCrossentropy(),
        'optimizer': tf.keras.optimizers.RMSprop(lr=.001),
        'weighted_metrics': [
            tf.keras.metrics.CategoricalCrossentropy(name='nll'),
            tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_k_acc')
        ]
    }

    # Infer embedding.
    embedding = tf.keras.layers.Embedding(
        n_stimuli+1, n_dim, mask_zero=True
    )
    similarity = psiz.keras.layers.ExponentialSimilarity()
    rankModel = psiz.models.Rank(
        embedding=embedding, similarity=similarity
    )
    emb_inferred = psiz.models.Proxy(model=rankModel)
    restart_record = emb_inferred.fit(
        obs_train, validation_data=obs_val, epochs=1000, verbose=2,
        callbacks=callbacks, n_restart=n_restart, monitor='val_nll',
        compile_kwargs=compile_kwargs
    )

    # Compare the inferred model with ground truth by comparing the
    # similarity matrices implied by each model.
    simmat_truth = psiz.utils.pairwise_matrix(emb_true.similarity, emb_true.z)
    simmat_infer = psiz.utils.pairwise_matrix(
        emb_inferred.similarity, emb_inferred.z
    )
    r_squared = psiz.utils.matrix_comparison(
        simmat_truth, simmat_infer, score='r2'
    )

    # Display comparison results. A good inferred model will have a high
    # R^2 value on the diagonal elements (max is 1) and relatively low R^2
    # values on the off-diagonal elements.
    print(
        '\n    R^2 Model Comparison: {0: >6.2f}\n'.format(r_squared)
    )


def ground_truth(n_stimuli, n_dim):
    """Return a ground truth embedding."""
    embedding = tf.keras.layers.Embedding(
        n_stimuli+1, n_dim, mask_zero=True,
        embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=.17)
    )
    similarity = psiz.keras.layers.ExponentialSimilarity()
    rankModel = psiz.models.Rank(embedding=embedding, similarity=similarity)
    emb = psiz.models.Proxy(rankModel)

    emb.theta = {
        'rho': 2.,
        'tau': 1.,
        'beta': 10.,
        'gamma': 0.001
    }

    return emb


if __name__ == "__main__":
    main()
