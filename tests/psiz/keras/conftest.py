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

import numpy as np
import pytest


@pytest.fixture
def gates_v0():
    """A minibatch of gates."""
    # Create a simple batch (batch_size=5).
    gates = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.3, 0.7],
            [0.5, 0.0, 0.0],
        ],
        dtype="float32",
    )
    return gates


@pytest.fixture
def groups_v0_0():
    """A minibatch of group indices."""
    # Create a simple batch (batch_size=5).
    groups = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype="float32",
    )
    return groups


@pytest.fixture
def groups_v0_1():
    """A minibatch of group weights without timestep axis.."""
    # Create a simple batch (batch_size=5).
    groups = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype="float32",
    )
    return groups


@pytest.fixture
def groups_v0_2():
    """A minibatch of group indices."""
    # Create a simple batch (batch_size=5).
    groups = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype="float32",
    )
    return groups


@pytest.fixture
def groups_v1_12():
    """A minibatch of group indices.

    Disjoint gate weights.

    """
    # Create a simple batch (batch_size=5).
    groups = np.array(
        [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype="float32"
    )
    return groups


@pytest.fixture
def groups_v2_12():
    """A minibatch of group indices.

    Overlapping gate weights.

    """
    # Create a simple batch (batch_size=5).
    groups = np.array(
        [[1.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 1.0]], dtype="float32"
    )
    return groups


@pytest.fixture
def groups_v3_12():
    """A minibatch of gate weights."""
    # Create a simple batch (batch_size=5).
    groups = np.array(
        [[1.0, 0.0], [0.8, 0.2], [0.5, 0.5], [0.2, 0.8], [0.0, 1.0]], dtype="float32"
    )
    return groups


@pytest.fixture
def group_3g_empty_v0_0():
    """A minibatch of group indices."""
    # Create a simple batch (batch_size=5).
    groups = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype="float32",
    )
    return groups


@pytest.fixture
def group_3g_empty_v0_1():
    """A minibatch of group indices."""
    # Create a simple batch (batch_size=5).
    groups = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype="float32",
    )
    return groups


@pytest.fixture
def groups_5x3x3_index_v0_2():
    """A minibatch of group indices.

    * 5 batches
    * index "gate weight" columns
    * 3 timesteps

    """
    # Create a simple batch (batch_size=5).
    groups = np.array(
        [
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
            # Last two batches intentionally have different groups for
            # each timestep.
            [[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]],
            [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
        ],
        dtype="float32",
    )
    return groups


@pytest.fixture
def inputs_single():
    """A minibatch of inputs."""
    # Create a simple batch (batch_size=5).
    inputs = np.array(
        [
            [0.0, 0.1, 0.2],
            [1.0, 1.1, 1.2],
            [2.0, 2.1, 2.2],
            [3.0, 3.1, 3.2],
            [4.0, 4.1, 4.2],
        ],
        dtype="float32",
    )
    return inputs


@pytest.fixture
def inputs_list():
    """A minibatch of a list of inputs."""
    # Create a simple batch (batch_size=5).
    inputs_0 = np.array(
        [
            [0.0, 0.1, 0.2],
            [1.0, 1.1, 1.2],
            [2.0, 2.1, 2.2],
            [3.0, 3.1, 3.2],
            [4.0, 4.1, 4.2],
        ],
        dtype="float32",
    )

    inputs_1 = np.array(
        [
            [10.0, 10.1, 10.2],
            [11.0, 11.1, 11.2],
            [12.0, 12.1, 12.2],
            [13.0, 13.1, 13.2],
            [14.0, 14.1, 14.2],
        ],
        dtype="float32",
    )

    inputs = [inputs_0, inputs_1]
    return inputs


@pytest.fixture
def gates_v0_timestep():
    """A minibatch of gates."""
    # Create a batch with timesteps (batch_size=5, sequence_length=2).
    gates = np.array(
        [
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
            [[0.0, 0.3, 0.7], [0.0, 0.3, 0.7]],
            # NOTE: The last batch is intentionally different for the
            # two timesteps.
            [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]],
        ],
        dtype="float32",
    )
    return gates


@pytest.fixture
def inputs_list_timestep():
    """A minibatch of list inputs that have a timestep axis."""
    # Create a simple batch (batch_size=5).
    inputs_0 = np.array(
        [
            [[0.0, 0.1, 0.2], [0.01, 0.11, 0.21]],
            [[1.0, 1.1, 1.2], [1.01, 1.11, 1.21]],
            [[2.0, 2.1, 2.2], [2.01, 2.11, 2.21]],
            [[3.0, 3.1, 3.2], [3.01, 3.11, 3.21]],
            [[4.0, 4.1, 4.2], [4.01, 4.11, 4.21]],
        ],
        dtype="float32",
    )

    inputs_1 = np.array(
        [
            [[10.0, 10.1, 10.2], [10.01, 10.11, 10.21]],
            [[11.0, 11.1, 11.2], [11.01, 11.11, 11.21]],
            [[12.0, 12.1, 12.2], [12.01, 12.11, 12.21]],
            [[13.0, 13.1, 13.2], [13.01, 13.11, 13.21]],
            [[14.0, 14.1, 14.2], [14.01, 14.11, 14.21]],
        ],
        dtype="float32",
    )

    inputs = [inputs_0, inputs_1]
    return inputs


@pytest.fixture
def inputs_dict():
    """A minibatch of inputs formated as a dictionary."""
    # Create a simple batch (batch_size=5).
    inputs_0 = np.array(
        [
            [0.0, 0.1, 0.2],
            [1.0, 1.1, 1.2],
            [2.0, 2.1, 2.2],
            [3.0, 3.1, 3.2],
            [4.0, 4.1, 4.2],
        ],
        dtype="float32",
    )

    inputs_1 = np.array(
        [
            [10.0, 10.1, 10.2],
            [11.0, 11.1, 11.2],
            [12.0, 12.1, 12.2],
            [13.0, 13.1, 13.2],
            [14.0, 14.1, 14.2],
        ],
        dtype="float32",
    )

    inputs = {"inputs_0": inputs_0, "inputs_1": inputs_1}
    return inputs


@pytest.fixture
def inputs_dict_timestep():
    """A minibatch of dictionary inputs that have a timestep axis."""
    # Create a simple batch (batch_size=5).
    inputs_0 = np.array(
        [
            [[0.0, 0.1, 0.2], [0.01, 0.11, 0.21]],
            [[1.0, 1.1, 1.2], [1.01, 1.11, 1.21]],
            [[2.0, 2.1, 2.2], [2.01, 2.11, 2.21]],
            [[3.0, 3.1, 3.2], [3.01, 3.11, 3.21]],
            [[4.0, 4.1, 4.2], [4.01, 4.11, 4.21]],
        ],
        dtype="float32",
    )

    inputs_1 = np.array(
        [
            [[10.0, 10.1, 10.2], [10.01, 10.11, 10.21]],
            [[11.0, 11.1, 11.2], [11.01, 11.11, 11.21]],
            [[12.0, 12.1, 12.2], [12.01, 12.11, 12.21]],
            [[13.0, 13.1, 13.2], [13.01, 13.11, 13.21]],
            [[14.0, 14.1, 14.2], [14.01, 14.11, 14.21]],
        ],
        dtype="float32",
    )

    inputs = {"inputs_0": inputs_0, "inputs_1": inputs_1}
    return inputs
