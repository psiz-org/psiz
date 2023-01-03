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

"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # noqa
from pathlib import Path
import shutil
import time

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
        z = self.percept(inputs['rate2_stimulus_set'])
        z_0 = tf.gather(z, indices=tf.constant(0), axis=stimuli_axis)
        z_1 = tf.gather(z, indices=tf.constant(1), axis=stimuli_axis)
        return self.kernel([z_0, z_1])


def main():
    """Run script."""
    # Settings.
    fp_project = Path.home() / Path('psiz_examples', 'rank', 'mle_1g')
    fp_board = fp_project / Path('logs', 'fit')
    n_stimuli = 30
    n_dim = 3
    epochs = 100
    batch_size = 128
    n_trial = 30 * batch_size
    n_trial_train = 24 * batch_size
    n_trial_val = 3 * batch_size
    n_frame = 1  # Set to 8 to observe convergence behavior.

    # Directory preparation.
    fp_project.mkdir(parents=True, exist_ok=True)
    # Remove existing TensorBoard logs.
    if fp_board.exists():
        shutil.rmtree(fp_board)

    # Define ground truth models.
    model_true = build_ground_truth_model(n_stimuli, n_dim)
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

    # Compute similarity matrix.
    simmat_true = model_similarity_true.predict(tfds_pairs)

    # Generate a random set of trials.
    rng = np.random.default_rng()
    eligibile_indices = np.arange(n_stimuli) + 1
    p = np.ones_like(eligibile_indices) / len(eligibile_indices)
    stimulus_set = psiz.utils.choice_wo_replace(
        eligibile_indices, (n_trial, 9), p, rng=rng
    )
    content = psiz.data.Rank(stimulus_set, n_select=2)
    pds = psiz.data.Dataset([content])
    tfds_content = pds.export(export_format='tfds')

    # Simulate similarity judgments and append outcomes to dataset.
    depth = content.n_outcome

    def simulate_agent(x):
        outcome_probs = model_true(x)
        outcome_distribution = tfp.distributions.Categorical(
            probs=outcome_probs
        )
        outcome_idx = outcome_distribution.sample()
        outcome_one_hot = tf.one_hot(outcome_idx, depth)
        return outcome_one_hot

    tfds_all = tfds_content.map(lambda x: (x, simulate_agent(x))).cache()

    # Partition data into 80% train, 10% validation and 10% test set.
    tfds_train = tfds_all.take(n_trial_train)
    tfds_valtest = tfds_all.skip(n_trial_train)
    tfds_val = tfds_valtest.take(n_trial_val).cache().batch(
        batch_size, drop_remainder=False
    )
    tfds_test = tfds_valtest.skip(n_trial_val).cache().batch(
        batch_size, drop_remainder=False
    )

    # Use early stopping.
    early_stop = tf.keras.callbacks.EarlyStopping(
        'val_cce', patience=30, mode='min', restore_best_weights=True
    )

    # Infer independent models with increasing amounts of data.
    if n_frame == 1:
        n_trial_train_frame = np.array([n_trial_train], dtype=int)
    else:
        n_trial_train_frame = np.round(
            np.linspace(15, n_trial_train, n_frame)
        ).astype(np.int64)
    r2 = np.empty((n_frame))
    train_cce = np.empty((n_frame))
    val_cce = np.empty((n_frame))
    test_cce = np.empty((n_frame))
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

        # Use Tensorboard callback.
        fp_board_frame = fp_board / Path('frame_{0}'.format(i_frame))
        cb_board = tf.keras.callbacks.TensorBoard(
            log_dir=fp_board_frame, histogram_freq=0,
            write_graph=False, write_images=False, update_freq='epoch',
            profile_batch=0, embeddings_freq=0, embeddings_metadata=None
        )
        callbacks = [early_stop, cb_board]

        # Infer embedding.
        model_inferred = build_model(n_stimuli, n_dim)
        history = model_inferred.fit(
            x=tfds_train_frame, validation_data=tfds_val, epochs=epochs,
            callbacks=callbacks, verbose=0
        )
        train_cce[i_frame] = history.history['cce'][-1]
        val_cce[i_frame] = history.history['val_cce'][-1]
        test_metrics = model_inferred.evaluate(
            tfds_test, verbose=0, return_dict=True
        )
        test_cce[i_frame] = test_metrics['cce']

        # Define model that outputs similarity based on inferred model.
        model_inferred_similarity = SimilarityModel(
            percept=model_inferred.behavior.percept,
            kernel=model_inferred.behavior.kernel
        )
        # Compare the inferred model with ground truth by comparing the
        # similarity matrices implied by each model.
        simmat_infer = model_inferred_similarity.predict(tfds_pairs)

        rho, _ = pearsonr(simmat_true, simmat_infer)
        r2[i_frame] = rho**2

        print(
            '    n_trial_train_frame: {0:4d} | train_cce: {1:.2f} | '
            'val_cce: {2:.2f} | test_cce: {3:.2f} | '
            'Correlation (R^2): {4:.2f}'.format(
                n_trial_train_frame[i_frame], train_cce[i_frame],
                val_cce[i_frame], test_cce[i_frame], r2[i_frame]
            )
        )

    # Plot comparison results.
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3))

    axes[0].plot(n_trial_train_frame, train_cce, 'bo-', label='Train CCE')
    axes[0].plot(n_trial_train_frame, val_cce, 'go-', label='Val. CCE')
    axes[0].plot(n_trial_train_frame, test_cce, 'ro-', label='Test CCE')
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Number of Judged Trials')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    axes[1].plot(n_trial_train_frame, r2, 'ro-')
    axes[1].set_title('Model Convergence to Ground Truth')
    axes[1].set_xlabel('Number of Judged Trials')
    axes[1].set_ylabel(r'Squared Pearson Correlation ($R^2$)')
    axes[1].set_ylim(-0.05, 1.05)

    plt.tight_layout()
    fname = fp_project / Path('evolution.tiff')
    plt.savefig(
        os.fspath(fname), format='tiff', bbox_inches="tight", dpi=300
    )


def build_ground_truth_model(n_stimuli, n_dim):
    """Return a ground truth embedding."""
    percept = tf.keras.layers.Embedding(
        n_stimuli + 1,
        n_dim,
        embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=.17),
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
            beta_initializer=tf.keras.initializers.Constant(10.),
            tau_initializer=tf.keras.initializers.Constant(1.),
            gamma_initializer=tf.keras.initializers.Constant(0.001),
        )
    )
    rank = psiz.keras.layers.RankSimilarity(
        n_reference=8, n_select=2, percept=percept, kernel=kernel
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


def build_model(n_stimuli, n_dim):
    """Build model.

    Args:
        n_stimuli: Integer indicating the number of stimuli in the
            embedding.
        n_dim: Integer indicating the dimensionality of the embedding.

    Returns:
        model: A TensorFlow Keras model.

    """
    # Create a group-agnostic percept layer.
    percept = tf.keras.layers.Embedding(n_stimuli + 1, n_dim, mask_zero=True)
    # Create a group-agnostic kernel.
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
            gamma_initializer=tf.keras.initializers.Constant(0.001),
        )
    )
    rank = psiz.keras.layers.RankSimilarity(
        n_reference=8, n_select=2, percept=percept, kernel=kernel
    )
    model = BehaviorModel(behavior=rank)
    compile_kwargs = {
        'loss': tf.keras.losses.CategoricalCrossentropy(),
        'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
        'weighted_metrics': [
            tf.keras.metrics.CategoricalCrossentropy(name='cce')
        ]
    }
    model.compile(**compile_kwargs)
    return model


if __name__ == "__main__":
    start_time_s = time.time()
    main()
    total_time_s = time.time() - start_time_s
    print('Total script time: {0:.0f} s'.format(total_time_s))
