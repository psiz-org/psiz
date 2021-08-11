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
"""Module for testing generator.py.

Todo:
    - more tests for ActiveGenerator
    - more information gain tests

"""

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

import psiz


@pytest.fixture(scope="module")
def rank_1g_mle_rand():
    n_stimuli = 10
    n_dim = 2

    stimuli = psiz.keras.layers.EmbeddingNormalDiag(
        n_stimuli+1, n_dim, mask_zero=True,
        scale_initializer=tf.keras.initializers.Constant(
            tfp.math.softplus_inverse(.01).numpy()
        )
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
            gamma_initializer=tf.keras.initializers.Constant(0.),
        )
    )

    behavior = psiz.keras.layers.RankBehavior()
    model = psiz.keras.models.Rank(
        stimuli=stimuli, kernel=kernel, behavior=behavior
    )
    return model


@pytest.fixture(scope="module")
def rank_1g_mle_det():
    n_stimuli = 10
    n_dim = 2

    z = np.array(
        [
            [0.0, 0.0],
            [0.12737487, 1.3211997],
            [0.8335809, 1.5255479],
            [0.8801151, 0.6451549],
            [0.55950886, 1.8086979],
            [1.9089336, -0.15246096],
            [2.8184545, 2.6377177],
            [0.00032808, 0.94420123],
            [0.21504205, 0.92544436],
            [2.0352089, 0.84319389],
            [-0.04342473, 1.4128358]
        ], dtype=np.float32
    )

    stimuli = tf.keras.layers.Embedding(
        n_stimuli+1, n_dim, mask_zero=True,
        embeddings_initializer=tf.keras.initializers.Constant(z)
    )

    kernel = psiz.keras.layers.DistanceBased(
        distance=psiz.keras.layers.Minkowski(
            rho_initializer=tf.keras.initializers.Constant(2.),
            w_initializer=tf.keras.initializers.Constant(1.),
            trainable=False
        ),
        similarity=psiz.keras.layers.ExponentialSimilarity(
            trainable=False,
            beta_initializer=tf.keras.initializers.Constant(1.),
            tau_initializer=tf.keras.initializers.Constant(1.),
            gamma_initializer=tf.keras.initializers.Constant(0.),
        )
    )

    behavior = psiz.keras.layers.RankBehavior()
    model = psiz.keras.models.Rank(
        stimuli=stimuli, kernel=kernel, behavior=behavior
    )
    return model


def simulated_samples(z):
    """Simulate posterior samples for a set of embedding points."""
    n_stimuli = z.shape[0]
    n_dim = z.shape[1]
    n_sample = 1000

    stim_cov = np.array((.08, .01, .01, .01, .01, .01, .01, .01, .01, .01))
    # Draw samples
    z_samples = np.empty((n_sample, n_stimuli, n_dim))
    np.random.seed(34895)
    for i_stimulus in range(n_stimuli):
        z_samples[:, i_stimulus, :] = np.random.multivariate_normal(
            z[i_stimulus], stim_cov[i_stimulus] * np.identity(n_dim),
            (n_sample)
        )
    z_samples = np.transpose(z_samples, axes=[1, 2, 0])
    return {'z': z_samples}


def test_random_generator():
    """Test random generator."""
    n_stimuli_desired = 10
    n_trial_desired = 50
    n_reference_desired = 4
    n_select_desired = 2
    is_ranked_desired = True
    gen = psiz.trials.RandomRank(
        n_stimuli_desired, n_reference=n_reference_desired,
        n_select=n_select_desired
    )
    docket = gen.generate(n_trial_desired)

    assert docket.n_trial == n_trial_desired
    assert sum(docket.n_reference == n_reference_desired) == n_trial_desired
    assert docket.stimulus_set.shape[0] == n_trial_desired
    assert docket.stimulus_set.shape[1] == n_reference_desired + 1
    min_actual = np.min(docket.stimulus_set)
    max_actual = np.max(docket.stimulus_set)
    assert min_actual >= -1  # Need -1 for padding.
    assert max_actual < n_stimuli_desired
    n_unique_desired = n_reference_desired + 1
    for i_trial in range(n_trial_desired):
        # NOTE: The padding padding values (-1)'s are counted as unique, thus
        # the indexing into stimulus set.
        assert (
            len(np.unique(
                docket.stimulus_set[i_trial, 0:n_reference_desired+1])
                ) == n_unique_desired
        )
    assert sum(docket.n_select == n_select_desired) == n_trial_desired
    assert sum(docket.is_ranked == is_ranked_desired) == n_trial_desired


# def test_information_gain(rank_1g_mle_rand):
#     """Test expected information gain computation."""
#     z = rank_1g_mle_rand.stimuli.embeddings.numpy()
#     if rank_1g_mle_rand.stimuli.mask_zero:
#         z = z[1:]
#     samples = simulated_samples(z)

#     gen = generator.ActiveGenerator(rank_1g_mle_rand.n_stimuli)
#     docket_0 = psiz.trials.RankDocket(
#         np.array([[0, 1, 2]], dtype=np.int32),
#         np.array([1], dtype=np.int32)
#         )
#     docket_1 = psiz.trials.RankDocket(
#         np.array([[3, 1, 2]], dtype=np.int32),
#         np.array([1], dtype=np.int32)
#         )
#     # Compute expected information gain.
#     ig_0 = psiz.trials.information_gain(
#         rank_1g_mle_rand, samples, docket_0
#     )
#     ig_1 = psiz.trials.information_gain(
#         rank_1g_mle_rand, samples, docket_1
#     )

#     assert ig_0 > ig_1

#     # Check case for multiple candidate trials.
#     docket_01 = psiz.trials.RankDocket(
#         np.array([[0, 1, 2], [3, 1, 2]], dtype=np.int32),
#         np.array([1, 1], dtype=np.int32)
#         )
#     ig_01 = psiz.trials.information_gain(
#         rank_1g_mle_rand, samples, docket_01
#     )
#     assert ig_01[0] == ig_0
#     assert ig_01[1] == ig_1

#     # Check case for multiple candidate trials with different configurations.
#     docket_23 = psiz.trials.RankDocket(
#         np.array([[0, 1, 2, 4, 9], [3, 1, 5, 6, 8]], dtype=np.int32),
#         np.array([2, 2], dtype=np.int32)
#         )
    # ig_23 = psiz.trials.information_gain(
    #     rank_1g_mle_rand, samples, docket_23
    # )

#     docket_0123 = psiz.trials.RankDocket(
#         np.array([
#             [0, 1, 2, -1, -1],
#             [3, 1, 2, -1, -1],
#             [0, 1, 2, 4, 9],
#             [3, 1, 5, 6, 8]
#         ], dtype=np.int32),
#         np.array([1, 1, 2, 2], dtype=np.int32)
#         )
#     ig_0123 = psiz.trials.information_gain(
#         rank_1g_mle_rand, samples, docket_0123
#     )
#     np.testing.assert_almost_equal(ig_0123[0], ig_0)
#     np.testing.assert_almost_equal(ig_0123[1], ig_1)
#     np.testing.assert_almost_equal(ig_0123[2], ig_23[0])
#     np.testing.assert_almost_equal(ig_0123[3], ig_23[1])


# TODO should I TF approach?
# def test_kl_divergence():
#     """Test computation of KL divergence."""
#     mu_left = np.array([0, 0])
#     mu_right = np.array([10, 0])

#     cov_sm = np.eye(2)
#     cov_lg = 10 * np.eye(2)

#     kl_sm_sm = psiz.trials.normal_kl_divergence(
#         mu_left, cov_sm, mu_right, cov_sm)
#     kl_sm_lg = psiz.trials.normal_kl_divergence(
#         mu_left, cov_sm, mu_right, cov_lg)
#     kl_lg_sm = psiz.trials.normal_kl_divergence(
#         mu_left, cov_lg, mu_right, cov_sm)
#     kl_lg_lg = psiz.trials.normal_kl_divergence(
#         mu_left, cov_lg, mu_right, cov_lg)

#     np.testing.assert_almost_equal(kl_sm_sm, 50.00000, decimal=5)
#     np.testing.assert_almost_equal(kl_sm_lg, 6.402585, decimal=5)
#     np.testing.assert_almost_equal(kl_lg_sm, 56.697415, decimal=5)
#     np.testing.assert_almost_equal(kl_lg_lg, 5.000000, decimal=5)


# def test_select_query_references(rank_1g_mle_det):
#     """Test select_query_references."""
#     max_candidate = 2000
#     max_neighbor = rank_1g_mle_det.n_stimuli

#     query_idx_list = np.array([7, 1])
#     n_trial_per_query_list = np.array([max_candidate, max_candidate])

#     samples = simulated_samples(rank_1g_mle_det.z)

#     n_reference = 8
#     n_select = 2
#     is_ranked = True

#     i_query = 0
#     docket, ig_top = psiz.trials._select_query_references(
#         i_query, rank_1g_mle_det, samples, query_idx_list,
#         n_trial_per_query_list, n_reference, n_select, is_ranked,
#         max_candidate, max_neighbor
#     )

#     # Check query indices in assembled docket.
#     n_mismatch = np.sum(
#         np.not_equal(
#             docket.stimulus_set[0:max_candidate, 0], query_idx_list[0]
#         )
#     )
#     assert n_mismatch == 0

#     # Check that all stimuli used in a trial are unique. For example check
#     # that they query stimulus is not also used as a reference.
#     for i_trial in range(docket.n_trial):
#         n_unique = len(np.unique(docket.stimulus_set[i_trial]))
#         assert n_unique == (n_reference + 1)


# @pytest.fixture(scope="module")
# def unbalanced_trials():
#     """Return a set of unbalanced trials.

#     In this context, unbalanced means trials in which a stimulus
#     occurs rarely.
#     """

#     n_stimuli_desired = 9
#     n_trial_desired = 200
#     n_reference_desired = 2
#     n_select_desired = 1
#     gen = psiz.trials.RandomRank(
#         n_reference=n_reference_desired,
#         n_select=n_select_desired)
#     unjudged_trials_0 = gen.generate(
#         n_trial_desired, n_stimuli_desired)
#     n_stimuli_desired = 10
#     n_trial_desired = 50
#     gen = psiz.trials.RandomRank(
#         n_reference=n_reference_desired,
#         n_select=n_select_desired)
#     unjudged_trials_1 = gen.generate(
#         n_trial_desired, n_stimuli_desired)
    # unjudged_trials = psiz.trials.stack(
    #     (unjudged_trials_0, unjudged_trials_1)
    # )
#     return unjudged_trials
