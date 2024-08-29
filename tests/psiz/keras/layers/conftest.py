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
"""keras.layers pytest setup."""


import keras
import numpy as np
import pytest
import tensorflow as tf

from psiz.keras.layers import Minkowski, ExponentialSimilarity


@pytest.fixture
def paired_inputs_v0():
    """A minibatch of non-gate inputs."""
    # Create a simple batch (batch_size=5).

    inputs_0 = tf.constant(
        np.array(
            [
                [0.0, 0.1, 0.2],
                [1.0, 1.1, 1.2],
                [2.0, 2.1, 2.2],
                [3.0, 3.1, 3.2],
                [4.0, 4.1, 4.2],
            ],
            dtype=np.float32,
        )
    )

    inputs_1 = tf.constant(
        np.array(
            [
                [5.0, 5.1, 5.2],
                [6.0, 6.1, 6.2],
                [7.0, 7.1, 7.2],
                [8.0, 8.1, 8.2],
                [9.0, 9.1, 9.2],
            ],
            dtype=np.float32,
        )
    )

    return [inputs_0, inputs_1]


@pytest.fixture
def paired_inputs_v1():
    """A minibatch of embedding coordinate inputs."""
    # Create a simple batch (batch_size=5).

    inputs_0 = tf.constant(
        np.array(
            [
                [0.0, 0.1, 0.2],
                [1.0, 1.1, 1.2],
                [2.0, 2.1, 2.2],
                [3.0, 3.1, 3.2],
                [4.0, 4.1, 4.2],
            ],
            dtype=np.float32,
        )
    )

    inputs_1 = tf.constant(
        np.array(
            [
                [1.0, 1.1, 1.2],
                [2.1, 2.2, 2.3],
                [3.2, 3.3, 3.4],
                [4.4, 4.3, 4.2],
                [4.0, 4.1, 4.2],
            ],
            dtype=np.float32,
        )
    )

    return [inputs_0, inputs_1]


@pytest.fixture
def groups_v0():
    """A minibatch of group indices."""
    # Create a simple batch (batch_size=5).
    groups = tf.constant(
        [[0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 1, 1], [0, 2, 1]], dtype=tf.int32
    )
    return groups


@pytest.fixture
def kernel_v0():
    kernel = Minkowski(
        rho_initializer=keras.initializers.Constant(2.0),
        w_initializer=keras.initializers.Constant(1.0),
        activation=ExponentialSimilarity(
            fit_tau=False,
            fit_gamma=False,
            fit_beta=False,
            tau_initializer=keras.initializers.Constant(1.0),
            gamma_initializer=keras.initializers.Constant(0.0),
            beta_initializer=keras.initializers.Constant(0.1),
        ),
        trainable=False,
    )
    return kernel


@pytest.fixture
def group_3g_empty_v0():
    """A minibatch of group indices."""
    # Create a simple batch (batch_size=5).
    groups = tf.constant(
        [[0, 1, 0], [0, 1, 0], [0, 2, 0], [0, 1, 1], [0, 2, 1]], dtype=np.int32
    )
    return groups


@pytest.fixture
def category_learning_inputs_v0():
    """A minibatch of category learning inputs.

    No timestep axis.
    No groups.

    """
    # Create a simple batch (batch_size=4).
    stimulus_set = tf.constant(
        np.array(
            [
                [1],
                [11],
                [1],
                [2],
            ],
            dtype=np.int32,
        )
    )
    objective_query_label = tf.constant(
        np.array(
            [0, 1, 0, 0],
            dtype=np.int32,
        )
    )
    objective_query_label = keras.utils.to_categorical(
        objective_query_label, num_classes=3
    )
    inputs = {
        "categorize_stimulus_set": stimulus_set,
        "categorize_objective_query_label": objective_query_label,
    }
    return inputs


@pytest.fixture
def category_learning_inputs_v1():
    """A minibatch of category learning inputs."""
    # Create a simple batch (batch_size=5).
    stimulus_set = tf.constant(
        np.array(
            [
                [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]],
                [[11], [12], [13], [14], [15], [16], [17], [18], [19], [20]],
                [[1], [3], [5], [7], [9], [11], [13], [15], [17], [19]],
                # NOTE: 2 masked trials
                [[2], [4], [6], [8], [10], [12], [14], [16], [0], [0]],
            ],
            dtype=np.int32,
        )
    )
    objective_query_label = tf.constant(
        np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
                [0, 0, 0, 0, 0, 1, 1, 1, 2, 2],
                [0, 0, 0, 0, 0, 1, 1, 2, 0, 0],
            ],
            dtype=np.int32,
        )
    )
    objective_query_label = keras.utils.to_categorical(
        objective_query_label, num_classes=3
    )
    inputs = {
        "categorize_stimulus_set": stimulus_set,
        "categorize_objective_query_label": objective_query_label,
    }
    return inputs
