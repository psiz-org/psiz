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
"""Root pytest setup."""

import numpy as np
import pytest
import tensorflow as tf

import psiz.keras.models
import psiz.keras.layers


@pytest.fixture(scope="module")
def rank_1g_mle_determ():
    n_stimuli = 3
    n_dim = 2
    stimuli = tf.keras.layers.Embedding(
        n_stimuli + 1, n_dim, mask_zero=True,
        embeddings_initializer=tf.keras.initializers.Constant(
            np.array(
                [
                    [0.0, 0.0], [.1, .1], [.15, .2], [.4, .5]
                ], dtype=np.float32
            )
        )
    )

    shared_similarity = psiz.keras.layers.ExponentialSimilarity(
        trainable=False,
        beta_initializer=tf.keras.initializers.Constant(10.),
        tau_initializer=tf.keras.initializers.Constant(1.),
        gamma_initializer=tf.keras.initializers.Constant(0.)
    )

    kernel = psiz.keras.layers.DistanceBased(
        distance=psiz.keras.layers.Minkowski(
            rho_trainable=False,
            rho_initializer=tf.keras.initializers.Constant(2.),
            w_initializer=tf.keras.initializers.Constant(
                [1.2, .8]
            ),
            w_constraint=psiz.keras.constraints.NonNegNorm(
                scale=n_dim, p=1.
            ),
        ),
        similarity=shared_similarity
    )

    behavior = psiz.keras.layers.RankBehavior()

    model = psiz.keras.models.Rank(
        stimuli=stimuli, kernel=kernel, behavior=behavior
    )
    return model


@pytest.fixture(scope="module")
def rank_1g_mle_random():
    n_stimuli = 10
    n_dim = 2
    stimuli = tf.keras.layers.Embedding(
        n_stimuli + 1, n_dim, mask_zero=True
    )

    shared_similarity = psiz.keras.layers.ExponentialSimilarity(
        trainable=False,
        beta_initializer=tf.keras.initializers.Constant(10.),
        tau_initializer=tf.keras.initializers.Constant(1.),
        gamma_initializer=tf.keras.initializers.Constant(0.)
    )

    kernel = psiz.keras.layers.DistanceBased(
        distance=psiz.keras.layers.Minkowski(
            rho_trainable=False,
            rho_initializer=tf.keras.initializers.Constant(2.),
            w_initializer=tf.keras.initializers.Constant(
                [1.2, .8]
            ),
            w_constraint=psiz.keras.constraints.NonNegNorm(
                scale=n_dim, p=1.
            ),
        ),
        similarity=shared_similarity
    )

    behavior = psiz.keras.layers.RankBehavior()

    model = psiz.keras.models.Rank(
        stimuli=stimuli, kernel=kernel, behavior=behavior
    )
    return model


@pytest.fixture(scope="module")
def rank_2g_mle_determ():
    n_stimuli = 3
    n_dim = 2
    stimuli = tf.keras.layers.Embedding(
        n_stimuli + 1, n_dim, mask_zero=True,
        embeddings_initializer=tf.keras.initializers.Constant(
            np.array(
                [
                    [0.0, 0.0], [.1, .1], [.15, .2], [.4, .5]
                ], dtype=np.float32
            )
        )
    )

    shared_similarity = psiz.keras.layers.ExponentialSimilarity(
        trainable=False,
        beta_initializer=tf.keras.initializers.Constant(10.),
        tau_initializer=tf.keras.initializers.Constant(1.),
        gamma_initializer=tf.keras.initializers.Constant(0.)
    )

    # Define group-specific kernels.
    kernel_0 = psiz.keras.layers.DistanceBased(
        distance=psiz.keras.layers.Minkowski(
            rho_trainable=False,
            rho_initializer=tf.keras.initializers.Constant(2.),
            w_initializer=tf.keras.initializers.Constant(
                [1.2, .8]
            ),
            w_constraint=psiz.keras.constraints.NonNegNorm(
                scale=n_dim, p=1.
            ),
        ),
        similarity=shared_similarity
    )

    kernel_1 = psiz.keras.layers.DistanceBased(
        distance=psiz.keras.layers.Minkowski(
            rho_trainable=False,
            rho_initializer=tf.keras.initializers.Constant(2.),
            w_initializer=tf.keras.initializers.Constant(
                [.7, 1.3]
            ),
            w_constraint=psiz.keras.constraints.NonNegNorm(
                scale=n_dim, p=1.
            ),
        ),
        similarity=shared_similarity
    )

    kernel_group = psiz.keras.layers.BraidGate(
        subnets=[kernel_0, kernel_1], gating_index=-1
    )

    behavior = psiz.keras.layers.RankBehavior()

    model = psiz.keras.models.Rank(
        stimuli=stimuli, kernel=kernel_group, behavior=behavior,
        use_group_kernel=True
    )
    return model


@pytest.fixture(scope="module")
def rank_2stim_2kern_determ():
    n_stimuli = 3
    n_dim = 2
    stimuli_0 = tf.keras.layers.Embedding(
        n_stimuli + 1, n_dim, mask_zero=True,
        embeddings_initializer=tf.keras.initializers.Constant(
            np.array(
                [
                    [0.0, 0.0], [.1, .1], [.15, .2], [.4, .5]
                ], dtype=np.float32
            )
        )
    )

    stimuli_1 = tf.keras.layers.Embedding(
        n_stimuli + 1, n_dim, mask_zero=True,
        embeddings_initializer=tf.keras.initializers.Constant(
            np.array(
                [
                    [0.0, 0.0], [.15, .2], [.4, .5], [.1, .1]
                ], dtype=np.float32
            )
        )
    )

    stimuli_group = psiz.keras.layers.BraidGate(
        subnets=[stimuli_0, stimuli_1], gating_index=-1
    )

    shared_similarity = psiz.keras.layers.ExponentialSimilarity(
        trainable=False,
        beta_initializer=tf.keras.initializers.Constant(10.),
        tau_initializer=tf.keras.initializers.Constant(1.),
        gamma_initializer=tf.keras.initializers.Constant(0.)
    )

    # Define group-specific kernels.
    kernel_0 = psiz.keras.layers.DistanceBased(
        distance=psiz.keras.layers.Minkowski(
            rho_trainable=False,
            rho_initializer=tf.keras.initializers.Constant(2.),
            w_initializer=tf.keras.initializers.Constant(
                [1.2, .8]
            ),
            w_constraint=psiz.keras.constraints.NonNegNorm(
                scale=n_dim, p=1.
            ),
        ),
        similarity=shared_similarity
    )

    kernel_1 = psiz.keras.layers.DistanceBased(
        distance=psiz.keras.layers.Minkowski(
            rho_trainable=False,
            rho_initializer=tf.keras.initializers.Constant(2.),
            w_initializer=tf.keras.initializers.Constant(
                [.7, 1.3]
            ),
            w_constraint=psiz.keras.constraints.NonNegNorm(
                scale=n_dim, p=1.
            ),
        ),
        similarity=shared_similarity
    )

    kernel_group = psiz.keras.layers.BraidGate(
        subnets=[kernel_0, kernel_1], gating_index=-1
    )

    behavior = psiz.keras.layers.RankBehavior()

    model = psiz.keras.models.Rank(
        stimuli=stimuli_group, kernel=kernel_group, behavior=behavior,
        use_group_kernel=True, use_group_stimuli=True
    )
    return model


@pytest.fixture(scope="module")
def rank_2stim_2kern_nomask_determ():
    n_stimuli = 3
    n_dim = 2
    stimuli_0 = tf.keras.layers.Embedding(
        n_stimuli, n_dim, mask_zero=False,
        embeddings_initializer=tf.keras.initializers.Constant(
            np.array(
                [
                    [0.0, 0.0], [.1, .1], [.15, .2], [.4, .5]
                ], dtype=np.float32
            )
        )
    )

    stimuli_1 = tf.keras.layers.Embedding(
        n_stimuli, n_dim, mask_zero=False,
        embeddings_initializer=tf.keras.initializers.Constant(
            np.array(
                [
                    [0.0, 0.0], [.15, .2], [.4, .5], [.1, .1]
                ], dtype=np.float32
            )
        )
    )

    stimuli_group = psiz.keras.layers.BraidGate(
        subnets=[stimuli_0, stimuli_1], gating_index=-1
    )

    shared_similarity = psiz.keras.layers.ExponentialSimilarity(
        trainable=False,
        beta_initializer=tf.keras.initializers.Constant(10.),
        tau_initializer=tf.keras.initializers.Constant(1.),
        gamma_initializer=tf.keras.initializers.Constant(0.)
    )

    # Define group-specific kernels.
    kernel_0 = psiz.keras.layers.DistanceBased(
        distance=psiz.keras.layers.Minkowski(
            rho_trainable=False,
            rho_initializer=tf.keras.initializers.Constant(2.),
            w_initializer=tf.keras.initializers.Constant(
                [1.2, .8]
            ),
            w_constraint=psiz.keras.constraints.NonNegNorm(
                scale=n_dim, p=1.
            ),
        ),
        similarity=shared_similarity
    )

    kernel_1 = psiz.keras.layers.DistanceBased(
        distance=psiz.keras.layers.Minkowski(
            rho_trainable=False,
            rho_initializer=tf.keras.initializers.Constant(2.),
            w_initializer=tf.keras.initializers.Constant(
                [.7, 1.3]
            ),
            w_constraint=psiz.keras.constraints.NonNegNorm(
                scale=n_dim, p=1.
            ),
        ),
        similarity=shared_similarity
    )

    kernel_group = psiz.keras.layers.BraidGate(
        subnets=[kernel_0, kernel_1], gating_index=-1
    )

    behavior = psiz.keras.layers.RankBehavior()

    model = psiz.keras.models.Rank(
        stimuli=stimuli_group, kernel=kernel_group, behavior=behavior,
        use_group_kernel=True, use_group_stimuli=True
    )
    return model


@pytest.fixture(scope="module")
def rank_2stim_2kern_2behav():
    n_stimuli = 20
    n_dim = 2
    stimuli_0 = tf.keras.layers.Embedding(
        n_stimuli + 1, n_dim, mask_zero=True
    )

    stimuli_1 = tf.keras.layers.Embedding(
        n_stimuli + 1, n_dim, mask_zero=True
    )

    stimuli_group = psiz.keras.layers.BraidGate(
        subnets=[stimuli_0, stimuli_1], gating_index=-1
    )

    shared_similarity = psiz.keras.layers.ExponentialSimilarity(
        trainable=False,
        beta_initializer=tf.keras.initializers.Constant(10.),
        tau_initializer=tf.keras.initializers.Constant(1.),
        gamma_initializer=tf.keras.initializers.Constant(0.)
    )

    # Define group-specific kernels.
    kernel_0 = psiz.keras.layers.DistanceBased(
        distance=psiz.keras.layers.Minkowski(
            rho_trainable=False,
            rho_initializer=tf.keras.initializers.Constant(2.),
            w_initializer=tf.keras.initializers.Constant(
                [1.2, .8]
            ),
            w_constraint=psiz.keras.constraints.NonNegNorm(
                scale=n_dim, p=1.
            ),
        ),
        similarity=shared_similarity
    )

    kernel_1 = psiz.keras.layers.DistanceBased(
        distance=psiz.keras.layers.Minkowski(
            rho_trainable=False,
            rho_initializer=tf.keras.initializers.Constant(2.),
            w_initializer=tf.keras.initializers.Constant(
                [.7, 1.3]
            ),
            w_constraint=psiz.keras.constraints.NonNegNorm(
                scale=n_dim, p=1.
            ),
        ),
        similarity=shared_similarity
    )

    kernel_group = psiz.keras.layers.BraidGate(
        subnets=[kernel_0, kernel_1], gating_index=-1
    )

    behavior_0 = psiz.keras.layers.RankBehavior()
    behavior_1 = psiz.keras.layers.RankBehavior()
    behavior_group = psiz.keras.layers.BraidGate(
        subnets=[behavior_0, behavior_1], gating_index=-1
    )

    model = psiz.keras.models.Rank(
        stimuli=stimuli_group, kernel=kernel_group, behavior=behavior_group,
        use_group_kernel=True, use_group_stimuli=True, use_group_behavior=True
    )
    return model
