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
"""Example that infers a shared embedding for three groups.

Fake data is generated from a ground truth model for three different
groups. In this example, these groups represent groups of agents with
varying levels of skill: novices, intermediates, and experts. Each group
has a different set of attention weights. An embedding model is
inferred from the simulated data and compared to the ground truth
model.

Example output:
    Attention weights:
          Novice | [1.86 1.73 0.24 0.17]
    Intermediate | [1.03 1.02 1.01 0.95]
          Expert | [0.22 0.17 1.90 1.71]

    Model Comparison (R^2)
    ================================
      True  |        Inferred
            | Novice  Interm  Expert
    --------+-----------------------
     Novice |   0.98    0.67    0.17
     Interm |   0.63    0.99    0.62
     Expert |   0.15    0.59    0.99

"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # noqa

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


class RankModel(tf.keras.Model):
    """A soft rank model.

    Proximity BraidGate.

    """

    def __init__(
        self, percept=None, braided_proximity=None, soft_8rank2=None, **kwargs
    ):
        """Initialize."""
        super(RankModel, self).__init__(**kwargs)
        self.percept = percept
        self.braided_proximity = braided_proximity
        self.soft_8rank2 = soft_8rank2
        self.stimuli_axis = 1

    def call(self, inputs):
        """Call."""
        z = self.percept(inputs["given8rank2_stimulus_set"])
        z_q, z_r = tf.split(z, [1, 8], self.stimuli_axis)
        s = self.braided_proximity([z_q, z_r, inputs["expertise"]])
        return self.soft_8rank2(s)


class SimilarityModel(tf.keras.Model):
    """A similarity model."""

    def __init__(self, percept=None, braided_proximity=None, **kwargs):
        """Initialize."""
        super(SimilarityModel, self).__init__(**kwargs)
        self.percept = percept
        self.braided_proximity = braided_proximity

    def call(self, inputs):
        """Call."""
        stimuli_axis = 1
        expertise_group = inputs["expertise"]
        z = self.percept(inputs["rate2_stimulus_set"])
        z_0 = tf.gather(z, indices=tf.constant(0), axis=stimuli_axis)
        z_1 = tf.gather(z, indices=tf.constant(1), axis=stimuli_axis)
        return self.braided_proximity([z_0, z_1, expertise_group])


def main():
    """Run the simulation that infers an embedding for three groups."""
    # Settings.
    n_stimuli = 30
    n_dim = 4
    n_group = 3
    epochs = 30
    batch_size = 128
    n_trial = 30 * batch_size
    n_trial_train = 24 * batch_size

    # Define ground truth models.
    model_true = build_ground_truth_model(n_stimuli, n_dim)
    model_similarity_true = SimilarityModel(
        percept=model_true.percept, braided_proximity=model_true.braided_proximity
    )

    # Assemble dataset of stimuli pairs for comparing similarity matrices.
    # NOTE: We include an placeholder "target" component in dataset tuple to
    # satisfy the assumptions of `predict` method.
    content_pairs = psiz.data.Rate(
        psiz.utils.pairwise_indices(np.arange(n_stimuli) + 1, elements="upper")
    )
    group_0 = psiz.data.Group(
        np.tile(np.array([1.0, 0.0, 0.0]), (content_pairs.n_sample, 1)),
        name="expertise",
    )
    group_1 = psiz.data.Group(
        np.tile(np.array([0.0, 1.0, 0.0]), (content_pairs.n_sample, 1)),
        name="expertise",
    )
    group_2 = psiz.data.Group(
        np.tile(np.array([0.0, 0.0, 1.0]), (content_pairs.n_sample, 1)),
        name="expertise",
    )
    dummy_outcome = psiz.data.Continuous(np.ones([content_pairs.n_sample, 1]))
    tfds_pairs_group0 = (
        psiz.data.Dataset([content_pairs, group_0, dummy_outcome])
        .export()
        .batch(batch_size, drop_remainder=False)
    )
    tfds_pairs_group1 = (
        psiz.data.Dataset([content_pairs, group_1, dummy_outcome])
        .export()
        .batch(batch_size, drop_remainder=False)
    )
    tfds_pairs_group2 = (
        psiz.data.Dataset([content_pairs, group_2, dummy_outcome])
        .export()
        .batch(batch_size, drop_remainder=False)
    )

    # Compute similarity matrix.
    simmat_truth = (
        model_similarity_true.predict(tfds_pairs_group0),
        model_similarity_true.predict(tfds_pairs_group1),
        model_similarity_true.predict(tfds_pairs_group2),
    )

    # Generate a random set of trials. Replicate for each group.
    rng = np.random.default_rng()
    eligibile_indices = np.arange(n_stimuli) + 1
    p = np.ones_like(eligibile_indices) / len(eligibile_indices)
    stimulus_set = psiz.utils.choice_wo_replace(
        eligibile_indices, (n_trial, 9), p, rng=rng
    )
    stimulus_set = np.repeat(stimulus_set, n_group, axis=0)
    content = psiz.data.Rank(stimulus_set, n_select=2)
    expertise = psiz.data.Group(
        np.tile(
            np.array(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
            ),
            [n_trial, 1],
        ),
        name="expertise",
    )
    pds = psiz.data.Dataset([content, expertise])
    tfds_content = pds.export(export_format="tfds")

    # Simulate ranked similarity judgments and append outcomes to dataset.
    tfds_content = tfds_content.batch(batch_size, drop_remainder=False)
    depth = content.n_outcome

    def simulate_agent(x):
        outcome_probs = model_true(x)
        outcome_distribution = tfp.distributions.Categorical(probs=outcome_probs)
        outcome_idx = outcome_distribution.sample()
        outcome_one_hot = tf.one_hot(outcome_idx, depth)
        return outcome_one_hot

    tfds_all = tfds_content.map(lambda x: (x, simulate_agent(x))).unbatch()

    # Partition data into 80% train and 20% validation.
    tfds_train = (
        tfds_all.take(n_trial_train * 3)
        .cache()
        .shuffle(buffer_size=n_trial_train * 3, reshuffle_each_iteration=True)
        .batch(batch_size, drop_remainder=False)
    )
    tfds_val = (
        tfds_all.skip(n_trial_train * 3).cache().batch(batch_size, drop_remainder=False)
    )

    # Use early stopping.
    early_stop = tf.keras.callbacks.EarlyStopping(
        "val_cce", patience=15, mode="min", restore_best_weights=True
    )
    callbacks = [early_stop]

    model_inferred = build_model(n_stimuli, n_dim)

    # Infer embedding.
    model_inferred.fit(
        x=tfds_train,
        validation_data=tfds_val,
        epochs=epochs,
        callbacks=callbacks,
        verbose=0,
    )

    # Create a model that computes similarity based on inferred model.
    model_similarity_inferred = SimilarityModel(
        percept=model_inferred.percept,
        braided_proximity=model_inferred.braided_proximity,
    )
    # Compute inferred similarity matrix.
    simmat_inferred = (
        model_similarity_inferred.predict(tfds_pairs_group0),
        model_similarity_inferred.predict(tfds_pairs_group1),
        model_similarity_inferred.predict(tfds_pairs_group2),
    )

    # Compare the inferred model with ground truth by comparing the
    # similarity matrices implied by each model.
    r_squared = np.empty((n_group, n_group))
    for i_truth in range(n_group):
        for j_infer in range(n_group):
            rho, _ = pearsonr(simmat_truth[i_truth], simmat_inferred[j_infer])
            r_squared[i_truth, j_infer] = rho**2

    # Display attention weights.
    # Permute inferred dimensions to best match ground truth.
    attention_weight = tf.stack(
        [
            model_inferred.braided_proximity.subnets[0].w,
            model_inferred.braided_proximity.subnets[1].w,
            model_inferred.braided_proximity.subnets[2].w,
        ],
        axis=0,
    ).numpy()
    idx_sorted = np.argsort(-attention_weight[0, :])
    attention_weight = attention_weight[:, idx_sorted]
    group_labels = ["Novice", "Intermediate", "Expert"]
    print("\n    Attention weights:")
    for i_group in range(attention_weight.shape[0]):
        print(
            "    {0:>12} | {1}".format(
                group_labels[i_group],
                np.array2string(
                    attention_weight[i_group, :],
                    formatter={"float_kind": lambda x: "%.2f" % x},
                ),
            )
        )

    # Display comparison results. A good inferred model will have a high
    # R^2 value on the diagonal elements (max is 1) and relatively low R^2
    # values on the off-diagonal elements.
    print("\n    Model Comparison (R^2)")
    print("    ================================")
    print("      True  |        Inferred")
    print("            | Novice  Interm  Expert")
    print("    --------+-----------------------")
    print(
        "     Novice | {0: >6.2f}  {1: >6.2f}  {2: >6.2f}".format(
            r_squared[0, 0], r_squared[0, 1], r_squared[0, 2]
        )
    )
    print(
        "     Interm | {0: >6.2f}  {1: >6.2f}  {2: >6.2f}".format(
            r_squared[1, 0], r_squared[1, 1], r_squared[1, 2]
        )
    )
    print(
        "     Expert | {0: >6.2f}  {1: >6.2f}  {2: >6.2f}".format(
            r_squared[2, 0], r_squared[2, 1], r_squared[2, 2]
        )
    )
    print("\n")


def build_ground_truth_model(n_stimuli, n_dim):
    """Return a ground truth embedding."""
    percept = tf.keras.layers.Embedding(
        n_stimuli + 1,
        n_dim,
        embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=0.17),
        mask_zero=True,
    )
    # Define group-specific proximity layers.
    shared_activation = psiz.keras.layers.ExponentialSimilarity(
        trainable=False,
        beta_initializer=tf.keras.initializers.Constant(10.0),
        tau_initializer=tf.keras.initializers.Constant(1.0),
        gamma_initializer=tf.keras.initializers.Constant(0.0),
    )
    proximity_0 = psiz.keras.layers.Minkowski(
        rho_trainable=False,
        rho_initializer=tf.keras.initializers.Constant(2.0),
        w_initializer=tf.keras.initializers.Constant([1.8, 1.8, 0.2, 0.2]),
        w_constraint=psiz.keras.constraints.NonNegNorm(scale=n_dim, p=1.0),
        activation=shared_activation,
    )
    proximity_1 = psiz.keras.layers.Minkowski(
        rho_trainable=False,
        rho_initializer=tf.keras.initializers.Constant(2.0),
        w_initializer=tf.keras.initializers.Constant([1.0, 1.0, 1.0, 1.0]),
        w_constraint=psiz.keras.constraints.NonNegNorm(scale=n_dim, p=1.0),
        activation=shared_activation,
    )
    proximity_2 = psiz.keras.layers.Minkowski(
        rho_trainable=False,
        rho_initializer=tf.keras.initializers.Constant(2.0),
        w_initializer=tf.keras.initializers.Constant([0.2, 0.2, 1.8, 1.8]),
        w_constraint=psiz.keras.constraints.NonNegNorm(scale=n_dim, p=1.0),
        activation=shared_activation,
    )
    braided_proximity = psiz.keras.layers.BraidGate(
        subnets=[proximity_0, proximity_1, proximity_2], gating_index=-1
    )
    soft_8rank2 = psiz.keras.layers.SoftRank(n_select=2)
    model = RankModel(
        percept=percept, braided_proximity=braided_proximity, soft_8rank2=soft_8rank2
    )
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        weighted_metrics=[tf.keras.metrics.CategoricalCrossentropy(name="cce")],
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
    percept = tf.keras.layers.Embedding(n_stimuli + 1, n_dim, mask_zero=True)
    # Define group-specific proximity layers.
    shared_activation = psiz.keras.layers.ExponentialSimilarity(
        trainable=False,
        beta_initializer=tf.keras.initializers.Constant(10.0),
        tau_initializer=tf.keras.initializers.Constant(1.0),
        gamma_initializer=tf.keras.initializers.Constant(0.0),
    )
    proximity_0 = build_proximity(shared_activation, n_dim)
    proximity_1 = build_proximity(shared_activation, n_dim)
    proximity_2 = build_proximity(shared_activation, n_dim)
    braided_proximity = psiz.keras.layers.BraidGate(
        subnets=[proximity_0, proximity_1, proximity_2], gating_index=-1
    )
    soft_8rank2 = psiz.keras.layers.SoftRank(n_select=2)
    model = RankModel(
        percept=percept, braided_proximity=braided_proximity, soft_8rank2=soft_8rank2
    )
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        weighted_metrics=[tf.keras.metrics.CategoricalCrossentropy(name="cce")],
    )
    return model


def build_proximity(similarity, n_dim):
    """Build proximity layer with learnable 'attention weights'."""
    proximity = psiz.keras.layers.Minkowski(
        rho_trainable=False,
        rho_initializer=tf.keras.initializers.Constant(2.0),
        w_constraint=psiz.keras.constraints.NonNegNorm(scale=n_dim, p=1.0),
        activation=similarity,
    )
    return proximity


if __name__ == "__main__":
    main()
