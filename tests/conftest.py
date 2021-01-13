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

import psiz.models
import psiz.keras.layers


@pytest.fixture(scope="module")
def rank_2g_mle_determ():
    n_stimuli = 3
    n_dim = 2
    n_group = 2
    embedding = psiz.keras.layers.EmbeddingDeterministic(
        n_stimuli+1, n_dim, mask_zero=True
    )
    embedding.build([None, None, None])
    z = np.array(
        [
            [0.0, 0.0], [.1, .1], [.15, .2], [.4, .5]
        ], dtype=np.float32
    )
    embedding.embeddings.assign(z)

    stimuli = psiz.keras.layers.Stimuli(embedding=embedding)

    kernel = psiz.keras.layers.AttentionKernel(
        group_level=1,
        distance=psiz.keras.layers.WeightedMinkowski(
            rho_initializer=tf.keras.initializers.Constant(2.),
            trainable=False,
        ),
        attention=psiz.keras.layers.GroupAttention(
            n_dim=n_dim, n_group=n_group
        ),
        similarity=psiz.keras.layers.ExponentialSimilarity(
            fit_tau=False, fit_gamma=False, fit_beta=False,
            tau_initializer=tf.keras.initializers.Constant(1.),
            gamma_initializer=tf.keras.initializers.Constant(0.),
            beta_initializer=tf.keras.initializers.Constant(10.),
        )
    )
    kernel.attention.embeddings.assign(
        np.array((
            (1.2, .8),
            (.7, 1.3)
        ))
    )

    behavior = psiz.keras.layers.RankBehavior()

    model = psiz.models.Rank(
        stimuli=stimuli, kernel=kernel, behavior=behavior
    )
    return model
