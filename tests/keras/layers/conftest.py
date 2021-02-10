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
"""keras.layers pytest setup."""

import numpy as np
import pytest
import tensorflow as tf


@pytest.fixture
def paired_inputs_v0():
    """A minibatch of non-gate inupts."""
    # Create a simple batch (batch_size=5).

    inputs_0 = tf.constant(
        np.array(
            [
                [0.0, 0.1, 0.2],
                [1.0, 1.1, 1.2],
                [2.0, 2.1, 2.2],
                [3.0, 3.1, 3.2],
                [4.0, 4.1, 4.2]
            ], dtype=np.float32
        )
    )

    inputs_1 = tf.constant(
        np.array(
            [
                [5.0, 5.1, 5.2],
                [6.0, 6.1, 6.2],
                [7.0, 7.1, 7.2],
                [8.0, 8.1, 8.2],
                [9.0, 9.1, 9.2]
            ], dtype=np.float32
        )
    )

    return [inputs_0, inputs_1]


# TODO remove
@pytest.fixture
def pw_inputs_v0():
    """A minibatch of non-gate inupts."""
    # Create a simple batch (batch_size=5).

    inputs_0 = tf.constant(
        np.array(
            [
                [0.0, 0.1, 0.2],
                [1.0, 1.1, 1.2],
                [2.0, 2.1, 2.2],
                [3.0, 3.1, 3.2],
                [4.0, 4.1, 4.2]
            ], dtype=np.float32
        )
    )

    inputs_1 = tf.constant(
        np.array(
            [
                [5.0, 5.1, 5.2],
                [6.0, 6.1, 6.2],
                [7.0, 7.1, 7.2],
                [8.0, 8.1, 8.2],
                [9.0, 9.1, 9.2]
            ], dtype=np.float32
        )
    )

    inputs = tf.stack([inputs_0, inputs_1], axis=-1)
    return inputs


@pytest.fixture
def pw_inputs_v1():
    """A minibatch of non-gate inupts."""
    # Create a simple batch (batch_size=5).

    inputs_0 = tf.constant(
        np.array(
            [
                [0.0, 0.1, 0.2],
                [1.0, 1.1, 1.2],
                [2.0, 2.1, 2.2],
                [3.0, 3.1, 3.2],
                [4.0, 4.1, 4.2]
            ], dtype=np.float32
        )
    )

    inputs_1 = tf.constant(
        np.array(
            [
                [5.0, 5.1, 5.2],
                [6.0, 6.1, 6.2],
                [7.0, 7.1, 7.2],
                [8.0, 8.1, 8.2],
                [9.0, 9.1, 9.2]
            ], dtype=np.float32
        )
    )

    inputs = tf.stack([inputs_0, inputs_1], axis=-1)
    return inputs


@pytest.fixture
def group_v0():
    """A minibatch of group indices."""
    # Create a simple batch (batch_size=5).
    group = tf.constant(
        np.array(
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 2, 0],
                [0, 1, 1],
                [0, 2, 1]
            ], dtype=np.int32
        )
    )
    return group


@pytest.fixture
def group_3g_empty_v0():
    """A minibatch of group indices."""
    # Create a simple batch (batch_size=5).
    group = tf.constant(
        np.array(
            [
                [0, 1, 0],
                [0, 1, 0],
                [0, 2, 0],
                [0, 1, 1],
                [0, 2, 1]
            ], dtype=np.int32
        )
    )
    return group
