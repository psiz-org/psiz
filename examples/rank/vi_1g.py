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

import copy
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # noqa
from pathlib import Path
import shutil

import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
import tensorflow as tf
import tensorflow_probability as tfp

import psiz
from psiz.tfp import unpack_mvn

# Uncomment the following line to force eager execution.
# tf.config.run_functions_eagerly(True)

# Modify the following to control GPU visibility.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class RankModel(tf.keras.Model):
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
        z_q, z_r = tf.split(z, [1, 8], self.stimuli_axis)
        s = self.proximity([z_q, z_r])
        return self.soft_8rank2(s)


class SimilarityModel(tf.keras.Model):
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
        z_0 = tf.gather(z, indices=tf.constant(0), axis=stimuli_axis)
        z_1 = tf.gather(z, indices=tf.constant(1), axis=stimuli_axis)
        return self.proximity([z_0, z_1])


class StochasticRankModel(psiz.keras.StochasticModel):
    """A behavior model.

    No Gates.

    """

    def __init__(self, percept=None, proximity=None, soft_8rank2=None, **kwargs):
        """Initialize."""
        super(StochasticRankModel, self).__init__(**kwargs)
        self.percept = percept
        self.proximity = proximity
        self.soft_8rank2 = soft_8rank2
        self.stimuli_axis = 1

    def call(self, inputs):
        """Call."""
        z = self.percept(inputs["given8rank2_stimulus_set"])
        z_q, z_r = tf.split(z, [1, 8], self.stimuli_axis)
        s = self.proximity([z_q, z_r])
        return self.soft_8rank2(s)


class StochasticSimilarityModel(psiz.keras.StochasticModel):
    """A similarity model."""

    def __init__(self, percept=None, proximity=None, **kwargs):
        """Initialize."""
        super(StochasticSimilarityModel, self).__init__(**kwargs)
        self.percept = percept
        self.proximity = proximity

    def call(self, inputs):
        """Call."""
        stimuli_axis = 1
        z = self.percept(inputs["rate2_stimulus_set"])
        z_0 = tf.gather(z, indices=tf.constant(0), axis=stimuli_axis)
        z_1 = tf.gather(z, indices=tf.constant(1), axis=stimuli_axis)
        return self.proximity([z_0, z_1])


def main():
    """Run script."""
    # Settings.
    fp_project = Path.home() / Path("psiz_examples", "rank", "vi_1g")
    fp_board = fp_project / Path("logs", "fit")
    n_stimuli = 30
    n_dim = 2
    epochs = 1000
    batch_size = 128
    n_trial = 30 * batch_size
    n_trial_train = 24 * batch_size
    n_trial_val = 3 * batch_size
    n_frame = 1  # Set to 7 to observe convergence behavior.

    # Directory preparation.
    fp_project.mkdir(parents=True, exist_ok=True)
    # Remove existing TensorBoard logs.
    if fp_board.exists():
        shutil.rmtree(fp_board)

    # Plot settings.
    small_size = 6
    medium_size = 8
    large_size = 10
    plt.rc("font", size=small_size)  # controls default text sizes
    plt.rc("axes", titlesize=medium_size)
    plt.rc("axes", labelsize=small_size)
    plt.rc("xtick", labelsize=small_size)
    plt.rc("ytick", labelsize=small_size)
    plt.rc("legend", fontsize=small_size)
    plt.rc("figure", titlesize=large_size)

    # Color settings.
    cmap = matplotlib.cm.get_cmap("jet")
    n_color = np.minimum(7, n_stimuli)
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=n_color)
    color_array = cmap(norm(range(n_color)))
    gray_array = np.ones([n_stimuli - n_color, 4])
    gray_array[:, 0:3] = 0.8
    color_array = np.vstack([gray_array, color_array])

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
    tfds_pairs = (
        psiz.data.Dataset([content_pairs, dummy_outcome])
        .export()
        .batch(batch_size, drop_remainder=False)
    )

    # Compute similarity matrix.
    simmat_true = model_similarity_true.predict(tfds_pairs)

    # Generate a random docket of trials.
    rng = np.random.default_rng()
    eligibile_indices = np.arange(n_stimuli) + 1
    p = np.ones_like(eligibile_indices) / len(eligibile_indices)
    stimulus_set = psiz.utils.choice_wo_replace(
        eligibile_indices, (n_trial, 9), p, rng=rng
    )
    content = psiz.data.Rank(stimulus_set, n_select=2)
    pds = psiz.data.Dataset([content])
    tfds_content = pds.export(export_format="tfds")

    # Simulate similarity judgments and append outcomes to dataset.
    tfds_content = tfds_content.batch(batch_size=batch_size, drop_remainder=False)

    def simulate_agent(x):
        depth = content.n_outcome
        outcome_probs = model_true(x)
        outcome_distribution = tfp.distributions.Categorical(probs=outcome_probs)
        outcome_idx = outcome_distribution.sample()
        outcome_one_hot = tf.one_hot(outcome_idx, depth)
        return outcome_one_hot

    tfds_all = tfds_content.map(lambda x: (x, simulate_agent(x))).cache()
    tfds_all = tfds_all.unbatch()

    # Partition data into 80% train, 10% validation and 10% test set.
    tfds_train = tfds_all.take(n_trial_train)
    tfds_valtest = tfds_all.skip(n_trial_train)
    tfds_val = (
        tfds_valtest.take(n_trial_val).cache().batch(batch_size, drop_remainder=False)
    )
    tfds_test = (
        tfds_valtest.skip(n_trial_val).cache().batch(batch_size, drop_remainder=False)
    )

    # Infer independent models with increasing amounts of data.
    if n_frame == 1:
        n_trial_train_frame = np.array([n_trial_train], dtype=int)
    else:
        n_trial_train_frame = np.round(np.linspace(15, n_trial_train, n_frame)).astype(
            np.int64
        )
    r2 = np.empty((n_frame)) * np.nan
    train_loss = np.empty((n_frame)) * np.nan
    val_loss = np.empty((n_frame)) * np.nan
    test_loss = np.empty((n_frame)) * np.nan
    for i_frame in range(n_frame):
        tfds_train_frame = (
            tfds_train.take(int(n_trial_train_frame[i_frame]))
            .cache()
            .shuffle(
                buffer_size=n_trial_train_frame[i_frame], reshuffle_each_iteration=True
            )
            .batch(batch_size, drop_remainder=False)
        )
        print(
            "\n  Frame {0} ({1} samples)".format(i_frame, n_trial_train_frame[i_frame])
        )

        # Use Tensorboard callback.
        fp_board_frame = fp_board / Path("frame_{0}".format(i_frame))
        cb_board = tf.keras.callbacks.TensorBoard(
            log_dir=fp_board_frame,
            histogram_freq=0,
            write_graph=False,
            write_images=False,
            update_freq="epoch",
            profile_batch=0,
            embeddings_freq=0,
            embeddings_metadata=None,
        )
        cb_early = tf.keras.callbacks.EarlyStopping(
            "loss", patience=15, mode="min", restore_best_weights=False, verbose=1
        )
        callbacks = [cb_board, cb_early]

        # Define model.
        model_inferred = build_model(n_stimuli, n_dim, n_trial_train_frame[i_frame])

        # Infer embedding.
        history = model_inferred.fit(
            x=tfds_train_frame,
            validation_data=tfds_val,
            epochs=epochs,
            callbacks=callbacks,
            verbose=0,
        )

        dist = model_inferred.percept.prior.embeddings.distribution
        print(
            "    Inferred prior scale: {0:.4f}".format(
                dist.distribution.distribution.scale[0, 0]
            )
        )

        # NOTE: The following are noisy estimates of final train/val loss.
        # Less noisy estimates could be obatined by running `evaluate` on
        # the train and validation set like test in the next block.
        train_loss[i_frame] = history.history["loss"][-1]
        val_loss[i_frame] = history.history["val_loss"][-1]

        tf.keras.backend.clear_session()
        model_inferred.n_sample = 100
        test_metrics = model_inferred.evaluate(tfds_test, verbose=0, return_dict=True)
        test_loss[i_frame] = test_metrics["loss"]

        # Compare the inferred model with ground truth by comparing the
        # similarity matrices implied by each model.
        model_similarity = StochasticSimilarityModel(
            percept=model_inferred.percept,
            proximity=model_inferred.proximity,
            n_sample=100,
        )
        simmat_infer = model_similarity.predict(tfds_pairs)

        rho, _ = pearsonr(simmat_true, simmat_infer)
        r2[i_frame] = rho**2

        print(
            "    n_obs: {0:4d} | train_loss: {1:.2f} | "
            "val_loss: {2:.2f} | test_loss: {3:.2f} | "
            "Correlation (R^2): {4:.2f}".format(
                n_trial_train_frame[i_frame],
                train_loss[i_frame],
                val_loss[i_frame],
                test_loss[i_frame],
                r2[i_frame],
            )
        )

        # Create and save visual frame.
        fig0 = plt.figure(figsize=(6.5, 4), dpi=200)
        plot_frame(
            fig0,
            n_trial_train_frame,
            train_loss,
            val_loss,
            test_loss,
            r2,
            model_true,
            model_inferred,
            color_array,
        )
        fname = fp_project / Path("frame_{0}.tiff".format(i_frame))
        plt.savefig(os.fspath(fname), format="tiff", bbox_inches="tight", dpi=300)

    # Create animation.
    if n_frame > 1:
        frames = []
        for i_frame in range(n_frame):
            fname = fp_project / Path("frame_{0}.tiff".format(i_frame))
            frames.append(imageio.imread(fname))
        imageio.mimwrite(fp_project / Path("evolution.gif"), frames, fps=1)


def plot_frame(
    fig0,
    n_obs,
    train_loss,
    val_loss,
    test_loss,
    r2,
    model_true,
    model_inferred,
    color_array,
):
    """Plot frame."""
    # Settings.
    s = 10

    z_true = model_true.percept.embeddings.numpy()
    if model_true.percept.mask_zero:
        z_true = z_true[1:]

    gs = fig0.add_gridspec(2, 2)

    f0_ax0 = fig0.add_subplot(gs[0, 0])
    plot_loss(f0_ax0, n_obs, train_loss, val_loss, test_loss)

    f0_ax2 = fig0.add_subplot(gs[1, 0])
    plot_convergence(f0_ax2, n_obs, r2)

    # Plot embeddings.
    f0_ax1 = fig0.add_subplot(gs[0:2, 1])
    # Determine embedding limits.
    z_max = 1.3 * np.max(np.abs(z_true))
    z_limits = [-z_max, z_max]

    # Apply and plot Procrustes affine transformation of posterior.
    dist = model_inferred.percept.embeddings
    loc, cov = unpack_mvn(dist)
    if model_inferred.percept.mask_zero:
        # Drop placeholder stimulus.
        loc = loc[1:]
        cov = cov[1:]

    # Center points.
    loc = loc - np.mean(loc, axis=0, keepdims=True)
    z_true = z_true - np.mean(z_true, axis=0, keepdims=True)

    r = psiz.utils.procrustes_rotation(loc, z_true, scale=False)

    loc, cov = apply_affine(loc, cov, r)
    psiz.mplot.hdi_bvn(loc, cov, f0_ax1, p=0.99, edgecolor=color_array, fill=False)

    # Plot true embedding.
    f0_ax1.scatter(
        z_true[:, 0],
        z_true[:, 1],
        s=s,
        c=color_array,
        marker="o",
        edgecolors="none",
        zorder=100,
    )
    f0_ax1.set_xlim(z_limits)
    f0_ax1.set_ylim(z_limits)
    f0_ax1.set_aspect("equal")
    f0_ax1.set_xticks([])
    f0_ax1.set_yticks([])
    f0_ax1.set_title("Embeddings (99% HDI)")

    gs.tight_layout(fig0)


def plot_loss(ax, n_obs, train_loss, val_loss, test_loss):
    """Plot loss."""
    # Settings
    ms = 2

    ax.plot(n_obs, train_loss, "bo-", ms=ms, label="Train Loss")
    ax.plot(n_obs, val_loss, "go-", ms=ms, label="Val. Loss")
    ax.plot(n_obs, test_loss, "ro-", ms=ms, label="Test Loss")
    ax.set_title("Optimization Objective")

    ax.set_xlabel("Trials")
    limits = [0, np.max(n_obs) + 10]
    ax.set_xlim(limits)
    ticks = [np.min(n_obs), np.max(n_obs)]
    ax.set_xticks(ticks)

    ax.set_ylabel("Loss")
    ax.legend()


def plot_convergence(ax, n_obs, r2):
    """Plot convergence."""
    # Settings.
    ms = 2

    ax.plot(n_obs, r2, "ro-", ms=ms)
    ax.set_title("Model Convergence")

    ax.set_xlabel("Trials")
    limits = [0, np.max(n_obs) + 10]
    ax.set_xlim(limits)
    ticks = [np.min(n_obs), np.max(n_obs)]
    ax.set_xticks(ticks)

    ax.set_ylabel(r"$R^2$")
    ax.set_ylim(-0.05, 1.05)


def build_ground_truth_model(n_stimuli, n_dim):
    """Return a ground truth embedding."""
    # Settings.
    scale_request = 0.17

    percept = tf.keras.layers.Embedding(
        (n_stimuli + 1),
        n_dim,
        embeddings_initializer=tf.keras.initializers.RandomNormal(
            stddev=scale_request, seed=58
        ),
        mask_zero=True,
    )
    proximity = psiz.keras.layers.Minkowski(
        rho_initializer=tf.keras.initializers.Constant(2.0),
        w_initializer=tf.keras.initializers.Constant(1.0),
        activation=psiz.keras.layers.ExponentialSimilarity(
            trainable=False,
            beta_initializer=tf.keras.initializers.Constant(10.0),
            tau_initializer=tf.keras.initializers.Constant(1.0),
            gamma_initializer=tf.keras.initializers.Constant(0.0),
        ),
        trainable=False,
    )
    soft_8rank2 = psiz.keras.layers.SoftRank(n_select=2)
    model = RankModel(percept=percept, proximity=proximity, soft_8rank2=soft_8rank2)
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        weighted_metrics=[tf.keras.metrics.CategoricalCrossentropy(name="cce")],
    )
    return model


def build_model(n_stimuli, n_dim, n_obs_train):
    """Build model.

    Args:
        n_stimuli: Integer indicating the number of stimuli in the
            embedding.
        n_dim: Integer indicating the dimensionality of the embedding.
        n_obs_train: Integer indicating the number of training
            observations. Used to determine KL weight for variational
            inference.

    Returns:
        model: A TensorFlow Keras model.

    """
    kl_weight = 1.0 / n_obs_train

    # Note that scale of the prior can be misspecified. The true scale
    # is .17, but halving (.085) or doubling (.34) still works well. When
    # the prior scale is much smaller than appropriate and there is
    # little data, the posterior *will* be driven by the incorrect prior.
    prior_scale = 0.2  # Mispecified to demonstrate robustness.

    # Create variational stimuli layer.
    embedding_posterior = psiz.keras.layers.EmbeddingNormalDiag(
        (n_stimuli + 1),
        n_dim,
        scale_initializer=tf.keras.initializers.Constant(
            tfp.math.softplus_inverse(prior_scale).numpy()
        ),
        mask_zero=True,
    )
    embedding_prior = psiz.keras.layers.EmbeddingShared(
        (n_stimuli + 1),
        n_dim,
        embedding=psiz.keras.layers.EmbeddingNormalDiag(
            1,
            1,
            loc_initializer=tf.keras.initializers.Constant(0.0),
            scale_initializer=tf.keras.initializers.Constant(
                tfp.math.softplus_inverse(prior_scale).numpy()
            ),
            loc_trainable=False,
        ),
        mask_zero=True,
    )
    percept = psiz.keras.layers.EmbeddingVariational(
        posterior=embedding_posterior,
        prior=embedding_prior,
        kl_weight=kl_weight,
        kl_n_sample=30,
    )
    proximity = psiz.keras.layers.Minkowski(
        rho_initializer=tf.keras.initializers.Constant(2.0),
        w_initializer=tf.keras.initializers.Constant(1.0),
        activation=psiz.keras.layers.ExponentialSimilarity(
            trainable=False,
            beta_initializer=tf.keras.initializers.Constant(10.0),
            tau_initializer=tf.keras.initializers.Constant(1.0),
            gamma_initializer=tf.keras.initializers.Constant(0.0),
        ),
        trainable=False,
    )
    soft_8rank2 = psiz.keras.layers.SoftRank(n_select=2)
    model = StochasticRankModel(
        percept=percept, proximity=proximity, soft_8rank2=soft_8rank2, n_sample=30
    )
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        weighted_metrics=[tf.keras.metrics.CategoricalCrossentropy(name="cce")],
    )
    return model


def apply_affine(loc, cov, r=None, t=None):
    """Apply affine transformation to set of MVN."""
    n_dist = loc.shape[0]
    loc_a = copy.copy(loc)
    cov_a = copy.copy(cov)

    for i_dist in range(n_dist):
        loc_a[i_dist], cov_a[i_dist] = psiz.utils.affine_mvn(
            loc[np.newaxis, i_dist], cov[i_dist], r, t
        )
    return loc_a, cov_a


if __name__ == "__main__":
    main()
