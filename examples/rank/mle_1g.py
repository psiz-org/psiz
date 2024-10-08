# -*- coding: utf-8 -*-
# Copyright 2024 The PsiZ Authors. All Rights Reserved.
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

import keras
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
import tensorflow_probability as tfp

import psiz

# NOTE: Uncomment the following lines to force eager execution.
# import tensorflow as tf
# tf.config.run_functions_eagerly(True)

# NOTE: Uncomment and edit the following to control GPU visibility.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class RankModel(keras.Model):
    """A soft rank model.

    No Gates.

    """

    def __init__(self, percept=None, proximity=None, soft_8rank2=None, **kwargs):
        """Initialize."""
        super(RankModel, self).__init__(**kwargs)
        self.percept = percept
        self.proximity = proximity
        self.soft_8rank2 = soft_8rank2
        self.stimuli_axis = 1

    def call(self, inputs):
        """Call."""
        z = self.percept(inputs["given8rank2_stimulus_set"])
        z_q, z_r = keras.ops.split(z, [1], self.stimuli_axis)
        s = self.proximity([z_q, z_r])
        return self.soft_8rank2(s)


class SimilarityModel(keras.Model):
    """A similarity model."""

    def __init__(self, percept=None, proximity=None, **kwargs):
        """Initialize."""
        super(SimilarityModel, self).__init__(**kwargs)
        self.percept = percept
        self.proximity = proximity

    def call(self, inputs):
        """Call."""
        stimuli_axis = 1
        z = self.percept(inputs["rate2_stimulus_set"])
        z_0 = keras.ops.take(z, indices=0, axis=stimuli_axis)
        z_1 = keras.ops.take(z, indices=1, axis=stimuli_axis)
        return self.proximity([z_0, z_1])


def main():
    """Run script."""
    # Settings.
    fp_project = Path.home() / Path("psiz_examples", "rank", "mle_1g")
    fp_board = fp_project / Path("logs", "fit")
    n_stimuli = 100
    n_dim = 3
    epochs = 1000
    batch_size = 512
    n_trial = 20 * batch_size
    n_trial_train = 16 * batch_size
    n_trial_val = 2 * batch_size
    # NOTE: Set `n_frame = 1` to see fit with all data. Set to greater than
    # 1 (e..g, `n_frame = 8`) to observe convergence behavior.
    n_frame = 8
    patience = 10

    # Directory preparation.
    fp_project.mkdir(parents=True, exist_ok=True)
    # Remove existing TensorBoard logs.
    if fp_board.exists():
        shutil.rmtree(fp_board)

    # Define ground truth models.
    model_true = build_ground_truth_model(n_stimuli, n_dim)
    model_similarity_true = SimilarityModel(
        percept=model_true.percept, proximity=model_true.proximity
    )

    # Assemble dataset of stimuli pairs for comparing similarity matrices.
    # NOTE: We include an placeholder "target" component in dataset tuple to
    # satisfy the assumptions of `predict` method.
    content_pairs = psiz.data.Rate(
        psiz.utils.pairwise_indices(np.arange(n_stimuli) + 1, elements="upper")
    )
    dummy_outcome = psiz.data.Continuous(np.ones([content_pairs.n_sample, 1]))
    ds_pairs = (
        psiz.data.Dataset([content_pairs, dummy_outcome])
        .export()
        .batch(batch_size, drop_remainder=False)
    )

    # Compute similarity matrix.
    simmat_true = model_similarity_true.predict(ds_pairs, verbose=0)

    # Generate a random set of trials.
    rng = np.random.default_rng()
    eligibile_indices = np.arange(n_stimuli) + 1
    p = np.ones_like(eligibile_indices) / len(eligibile_indices)
    stimulus_set = psiz.utils.choice_wo_replace(
        eligibile_indices, (n_trial, 9), p, rng=rng
    )
    content = psiz.data.Rank(stimulus_set, n_select=2)
    pds = psiz.data.Dataset([content])
    ds_content = pds.export(export_format="tfds")

    # Simulate similarity judgments and append outcomes to dataset.
    ds_content = ds_content.batch(batch_size, drop_remainder=False)
    depth = content.n_outcome

    def simulate_agent(x):
        outcome_probs = model_true(x)
        outcome_distribution = tfp.distributions.Categorical(probs=outcome_probs)
        outcome_idx = outcome_distribution.sample()
        outcome_one_hot = keras.ops.one_hot(outcome_idx, depth)
        return outcome_one_hot

    ds_all = ds_content.map(lambda x: (x, simulate_agent(x))).cache()
    ds_all = ds_all.unbatch()

    # Partition data into 80% train, 10% validation and 10% test set.
    ds_train = ds_all.take(n_trial_train)
    ds_valtest = ds_all.skip(n_trial_train)
    ds_val = (
        ds_valtest.take(n_trial_val).cache().batch(batch_size, drop_remainder=False)
    )
    ds_test = (
        ds_valtest.skip(n_trial_val).cache().batch(batch_size, drop_remainder=False)
    )

    # Infer independent models with increasing amounts of data.
    if n_frame == 1:
        n_trial_train_frame = np.array([n_trial_train], dtype=int)
    else:
        n_trial_train_frame = np.round(np.linspace(15, n_trial_train, n_frame)).astype(
            np.int64
        )
    r2 = np.empty((n_frame))
    train_cce = np.empty((n_frame))
    val_cce = np.empty((n_frame))
    test_cce = np.empty((n_frame))
    epochs_used = np.empty((n_frame), dtype=int)
    for i_frame in range(n_frame):
        ds_train_frame = (
            ds_train.take(int(n_trial_train_frame[i_frame]))
            .cache()
            .shuffle(
                buffer_size=n_trial_train_frame[i_frame], reshuffle_each_iteration=True
            )
            .batch(batch_size, drop_remainder=False)
        )
        print(
            "\n  Frame {0} ({1} samples)".format(i_frame, n_trial_train_frame[i_frame])
        )

        # Use early stopping.
        cb_early_stop = keras.callbacks.EarlyStopping(
            "val_cce",
            patience=patience,
            mode="min",
            restore_best_weights=True,
            verbose=0,
        )
        # Use Tensorboard callback.
        fp_board_frame = fp_board / Path("frame_{0}".format(i_frame))
        cb_board = keras.callbacks.TensorBoard(
            log_dir=fp_board_frame,
            histogram_freq=0,
            write_graph=False,
            write_images=False,
            update_freq="epoch",
            profile_batch=0,
            embeddings_freq=0,
            embeddings_metadata=None,
        )
        callbacks = [cb_early_stop, cb_board]

        # Infer embedding.
        model_inferred = build_model(n_stimuli, n_dim)
        history = model_inferred.fit(
            x=ds_train_frame,
            validation_data=ds_val,
            epochs=epochs,
            callbacks=callbacks,
            verbose=0,
        )
        train_metrics = model_inferred.evaluate(
            ds_train_frame, verbose=0, return_dict=True
        )
        val_metrics = model_inferred.evaluate(ds_val, verbose=0, return_dict=True)
        test_metrics = model_inferred.evaluate(ds_test, verbose=0, return_dict=True)
        train_cce[i_frame] = train_metrics["cce"]
        val_cce[i_frame] = val_metrics["cce"]
        test_cce[i_frame] = test_metrics["cce"]
        epochs_used[i_frame] = len(history.history["cce"]) - patience

        # Define model that outputs similarity based on inferred model.
        model_inferred_similarity = SimilarityModel(
            percept=model_inferred.percept,
            proximity=model_inferred.proximity,
        )
        # Compare the inferred model with ground truth by comparing the
        # similarity matrices implied by each model.
        simmat_infer = model_inferred_similarity.predict(ds_pairs, verbose=0)

        rho, _ = pearsonr(simmat_true, simmat_infer)
        r2[i_frame] = rho**2

        print(
            f"    n_obs: {n_trial_train_frame[i_frame]:4d} "
            f"| epochs: {epochs_used[i_frame]:4d} "
            f"| train_cce: {train_cce[i_frame]:.2f} | "
            f"val_cce: {val_cce[i_frame]:.2f} | "
            f"test_cce: {test_cce[i_frame]:.2f} | "
            f"correlation (R^2): {r2[i_frame]:.2f}"
        )

    # Plot comparison results.
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3))

    axes[0].plot(n_trial_train_frame, train_cce, "bo-", label="Train CCE")
    axes[0].plot(n_trial_train_frame, val_cce, "go-", label="Val. CCE")
    axes[0].plot(n_trial_train_frame, test_cce, "ro-", label="Test CCE")
    axes[0].set_title("Model Loss")
    axes[0].set_xlabel("Number of Samples")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(n_trial_train_frame, r2, "ro-")
    axes[1].set_title("Model Convergence to Ground Truth")
    axes[1].set_xlabel("Number of Samples")
    axes[1].set_ylabel(r"Squared Pearson Correlation ($R^2$)")
    axes[1].set_ylim(-0.05, 1.05)

    plt.tight_layout()
    fname = fp_project / Path("evolution.tiff")
    plt.savefig(os.fspath(fname), format="tiff", bbox_inches="tight", dpi=300)


def build_ground_truth_model(n_stimuli, n_dim):
    """Return a ground truth embedding."""
    percept = keras.layers.Embedding(
        n_stimuli + 1,
        n_dim,
        embeddings_initializer=keras.initializers.RandomNormal(stddev=0.17),
        mask_zero=True,
    )
    proximity = psiz.keras.layers.Minkowski(
        rho_initializer=keras.initializers.Constant(2.0),
        w_initializer=keras.initializers.Constant(1.0),
        activation=psiz.keras.layers.ExponentialSimilarity(
            trainable=False,
            beta_initializer=keras.initializers.Constant(10.0),
            tau_initializer=keras.initializers.Constant(1.0),
            gamma_initializer=keras.initializers.Constant(0.001),
        ),
        trainable=False,
    )
    soft_8rank2 = psiz.keras.layers.SoftRank(n_select=2, trainable=False)
    model = RankModel(percept=percept, proximity=proximity, soft_8rank2=soft_8rank2)
    model.compile(
        loss=keras.losses.CategoricalCrossentropy(),
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        weighted_metrics=[keras.metrics.CategoricalCrossentropy(name="cce")],
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
    percept = keras.layers.Embedding(
        n_stimuli + 1,
        n_dim,
        embeddings_initializer=keras.initializers.RandomNormal(stddev=0.0001),
        mask_zero=True,
    )
    # Create a group-agnostic proximity layer.
    proximity = psiz.keras.layers.Minkowski(
        rho_initializer=keras.initializers.Constant(2.0),
        w_initializer=keras.initializers.Constant(1.0),
        activation=psiz.keras.layers.ExponentialSimilarity(
            trainable=False,
            beta_initializer=keras.initializers.Constant(10.0),
            tau_initializer=keras.initializers.Constant(1.0),
            gamma_initializer=keras.initializers.Constant(0.001),
        ),
        trainable=False,
    )
    soft_8rank2 = psiz.keras.layers.SoftRank(n_select=2, trainable=False)
    model = RankModel(percept=percept, proximity=proximity, soft_8rank2=soft_8rank2)
    compile_kwargs = {
        "loss": keras.losses.CategoricalCrossentropy(),
        "optimizer": keras.optimizers.Adam(learning_rate=0.001),
        "weighted_metrics": [keras.metrics.CategoricalCrossentropy(name="cce")],
    }
    model.compile(**compile_kwargs)
    return model


if __name__ == "__main__":
    start_time_s = time.time()
    main()
    total_time_s = time.time() - start_time_s
    print("Total script time: {0:.0f} s".format(total_time_s))
