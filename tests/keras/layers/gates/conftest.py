# -*- coding: utf-8 -*-
# Copyright 2022 The PsiZ Authors. All Rights Reserved.
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


# TODO taken from conftest.py:group_v0, note no "s".
@pytest.fixture
def groups_v0():
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
def groups_v1():
    """A minibatch of group indices.

    Col-1 and col-2 define disjoint groups.

    """
    # Create a simple batch (batch_size=5).
    group = tf.constant(
        np.array(
            [
                [0, 1, 0],
                [0, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, 0, 1]
            ], dtype=np.int32
        )
    )
    return group


@pytest.fixture
def groups_v2():
    """A minibatch of group indices.

    Col-1 and col-2 define intersecting groups.

    """
    # Create a simple batch (batch_size=5).
    group = tf.constant(
        np.array(
            [
                [0, 1, 0],
                [0, 1, 0],
                [0, 1, 1],
                [0, 0, 1],
                [0, 0, 1]
            ], dtype=np.int32
        )
    )
    return group


@pytest.fixture
def groups_v3():
    """A minibatch of group indices.

    Col-1 and col-2 define intersecting groups.

    """
    # Create a simple batch (batch_size=5).
    group = tf.constant(
        np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.8, 0.2],
                [0.0, 0.5, 0.5],
                [0.0, 0.2, 0.8],
                [0.0, 0.0, 1.0]
            ], dtype=np.float32
        )
    )
    return group
