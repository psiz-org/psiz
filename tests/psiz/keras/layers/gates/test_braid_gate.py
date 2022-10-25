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
"""Test BraidGate."""

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from psiz.keras.layers import DistanceBased
from psiz.keras.layers import EmbeddingNormalDiag
from psiz.keras.layers import ExponentialSimilarity
from psiz.keras.layers import BraidGate
from psiz.keras.layers import Minkowski
from psiz.keras.layers import MinkowskiStochastic
from psiz.keras.layers import MinkowskiVariational


def build_vi_kernel(similarity, n_dim, kl_weight):
    """Build kernel for single group."""
    mink_prior = MinkowskiStochastic(
        rho_loc_trainable=False, rho_scale_trainable=True,
        w_loc_trainable=False, w_scale_trainable=False,
        w_scale_initializer=tf.keras.initializers.Constant(.1)
    )

    mink_posterior = MinkowskiStochastic(
        rho_loc_trainable=False, rho_scale_trainable=True,
        w_loc_trainable=True, w_scale_trainable=True,
        w_scale_initializer=tf.keras.initializers.Constant(.1)
    )

    mink = MinkowskiVariational(
        prior=mink_prior, posterior=mink_posterior,
        kl_weight=kl_weight, kl_n_sample=30
    )

    kernel = DistanceBased(
        distance=mink,
        similarity=similarity
    )
    return kernel


# Copied from test_sparse_dispatcher:Increment.
class Increment(tf.keras.layers.Layer):
    """A simple layer that increments input by a value."""

    def __init__(self, v, **kwargs):
        """Initialize."""
        super(Increment, self).__init__(**kwargs)
        self.v = tf.constant(v)

    def call(self, inputs):
        """Call."""
        return inputs + self.v


# Copied from test_sparse_dispatcher:AddPairs.
class AddPairs(tf.keras.layers.Layer):
    """A simple layer that increments input by a value."""

    def __init__(self, v, **kwargs):
        """Initialize."""
        super(AddPairs, self).__init__(**kwargs)
        self.v = tf.constant(v)

    def call(self, inputs):
        """Call."""
        return inputs[0] + inputs[1] + self.v


# Copied from test_sparse_dispatcher:AddPairsDict.
class AddPairsDict(tf.keras.layers.Layer):
    """A simple layer that increments input by a value."""

    def __init__(self, v, **kwargs):
        """Initialize."""
        super(AddPairsDict, self).__init__(**kwargs)
        self.v = tf.constant(v)

    def call(self, inputs):
        """Call."""
        return inputs['inputs_0'] + inputs['inputs_1'] + self.v


@pytest.fixture
def groups_v0_0():
    """A minibatch of group indices."""
    # Create a simple batch (batch_size=5).
    groups = tf.constant(
        [
            [0],
            [0],
            [0],
            [0],
            [0]
        ], dtype=tf.int32
    )
    return groups


@pytest.fixture
def groups_v0_1():
    """A minibatch of group indices."""
    # Create a simple batch (batch_size=5).
    groups = tf.constant(
        [
            [0],
            [1],
            [2],
            [1],
            [2]
        ], dtype=tf.int32
    )
    return groups


@pytest.fixture
def groups_v0_2():
    """A minibatch of group indices."""
    # Create a simple batch (batch_size=5).
    groups = tf.constant(
        [
            [0],
            [0],
            [0],
            [1],
            [1]
        ], dtype=tf.int32
    )
    return groups


@pytest.fixture
def groups_v3_12():
    """A minibatch of gate weights."""
    # Create a simple batch (batch_size=5).
    groups = tf.constant(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [0.5, 0.5],
            [0.2, 0.8],
            [0.0, 1.0]
        ], dtype=np.float32
    )
    return groups


@pytest.fixture
def group_3g_empty_v0_0():
    """A minibatch of group indices."""
    # Create a simple batch (batch_size=5).
    groups = tf.constant(
        [
            [0],
            [0],
            [0],
            [0],
            [0]
        ], dtype=np.int32
    )
    return groups


@pytest.fixture
def group_3g_empty_v0_1():
    """A minibatch of group indices."""
    # Create a simple batch (batch_size=5).
    groups = tf.constant(
        [
            [1],
            [1],
            [2],
            [1],
            [2]
        ], dtype=np.int32
    )
    return groups


@pytest.fixture
def inputs_emb_v0():
    """A minibatch of non-gate inupts."""
    # Create a simple batch (batch_size=5).
    inputs = tf.constant(
        np.array(
            [
                [0, 1, 2],
                [3, 4, 5],
                [6, 7, 8],
                [9, 10, 11],
                [9, 10, 11]
            ], dtype=np.int32
        )
    )
    return inputs


@pytest.fixture
def inputs_emb_v1():
    """A minibatch of non-gate inupts."""
    # Create a simple batch (batch_size=5).
    inputs = tf.constant(
        np.array(
            [
                [[0, 1, 2], [3, 4, 5]],
                [[0, 1, 2], [3, 4, 5]],
                [[0, 1, 2], [3, 4, 5]],
                [[6, 7, 8], [9, 10, 11]],
                [[6, 7, 8], [9, 10, 11]]
            ], dtype=np.int32
        )
    )
    return inputs


@pytest.fixture
def emb_subnets_determ():
    """A list of subnets"""
    base_array = np.array([
        [0.0, 0.1, 0.2],
        [1.0, 1.1, 1.2],
        [2.0, 2.1, 2.2],
        [3.0, 3.1, 3.2],
        [4.0, 4.1, 4.2],
        [5.0, 5.1, 5.2],
        [6.0, 6.1, 6.2],
        [7.0, 7.1, 7.2],
        [8.0, 8.1, 8.2],
        [9.0, 9.1, 9.2],
        [10.0, 10.1, 10.2],
        [11.0, 11.1, 11.2],
    ], dtype=np.float32)

    emb_0 = tf.keras.layers.Embedding(
        input_dim=12, output_dim=3,
        embeddings_initializer=tf.keras.initializers.Constant(base_array)
    )
    emb_1 = tf.keras.layers.Embedding(
        input_dim=12, output_dim=3,
        embeddings_initializer=tf.keras.initializers.Constant(base_array + 100)
    )
    emb_2 = tf.keras.layers.Embedding(
        input_dim=12, output_dim=3,
        embeddings_initializer=tf.keras.initializers.Constant(base_array + 200)
    )
    subnets = [emb_0, emb_1, emb_2]
    return subnets


@pytest.fixture
def emb_subnets_stoch_rank0():
    """A list of subnets"""
    # Settings.
    prior_scale = .0000000001
    n_stimuli = 12
    n_dim = 3

    base_array = np.array([
        [0.0, 0.1, 0.2],
        [1.0, 1.1, 1.2],
        [2.0, 2.1, 2.2],
        [3.0, 3.1, 3.2],
        [4.0, 4.1, 4.2],
        [5.0, 5.1, 5.2],
        [6.0, 6.1, 6.2],
        [7.0, 7.1, 7.2],
        [8.0, 8.1, 8.2],
        [9.0, 9.1, 9.2],
        [10.0, 10.1, 10.2],
        [11.0, 11.1, 11.2],
    ], dtype=np.float32)

    emb_0 = EmbeddingNormalDiag(
        n_stimuli, n_dim, mask_zero=True,
        loc_initializer=tf.keras.initializers.Constant(
            base_array
        ),
        scale_initializer=tf.keras.initializers.Constant(
            tfp.math.softplus_inverse(prior_scale).numpy()
        )
    )

    emb_1 = EmbeddingNormalDiag(
        n_stimuli, n_dim, mask_zero=True,
        loc_initializer=tf.keras.initializers.Constant(
            base_array + 100
        ),
        scale_initializer=tf.keras.initializers.Constant(
            tfp.math.softplus_inverse(prior_scale).numpy()
        )
    )

    emb_2 = EmbeddingNormalDiag(
        n_stimuli, n_dim, mask_zero=True,
        loc_initializer=tf.keras.initializers.Constant(
            base_array + 200
        ),
        scale_initializer=tf.keras.initializers.Constant(
            tfp.math.softplus_inverse(prior_scale).numpy()
        )
    )

    subnets = [emb_0, emb_1, emb_2]
    return subnets


@pytest.fixture
def emb_subnets_stoch_rank1():
    """A list of subnets"""
    # Settings.
    prior_scale = .0000000001
    n_stimuli = 12
    n_dim = 3

    base_array = np.array([
        [0.0, 0.1, 0.2],
        [1.0, 1.1, 1.2],
        [2.0, 2.1, 2.2],
        [3.0, 3.1, 3.2],
        [4.0, 4.1, 4.2],
        [5.0, 5.1, 5.2],
        [6.0, 6.1, 6.2],
        [7.0, 7.1, 7.2],
        [8.0, 8.1, 8.2],
        [9.0, 9.1, 9.2],
        [10.0, 10.1, 10.2],
        [11.0, 11.1, 11.2],
    ], dtype=np.float32)

    emb_0 = EmbeddingNormalDiag(
        n_stimuli, n_dim, mask_zero=True,
        loc_initializer=tf.keras.initializers.Constant(
            base_array
        ),
        scale_initializer=tf.keras.initializers.Constant(
            tfp.math.softplus_inverse(prior_scale).numpy()
        )
    )

    emb_1 = EmbeddingNormalDiag(
        n_stimuli, n_dim, mask_zero=True,
        loc_initializer=tf.keras.initializers.Constant(
            base_array + 100
        ),
        scale_initializer=tf.keras.initializers.Constant(
            tfp.math.softplus_inverse(prior_scale).numpy()
        )
    )

    emb_2 = EmbeddingNormalDiag(
        n_stimuli, n_dim, mask_zero=True,
        loc_initializer=tf.keras.initializers.Constant(
            base_array + 200
        ),
        scale_initializer=tf.keras.initializers.Constant(
            tfp.math.softplus_inverse(prior_scale).numpy()
        )
    )

    subnets = [emb_0, emb_1, emb_2]
    return subnets


@pytest.fixture
def kernel_subnets():
    """A list of subnets"""
    pw_0 = DistanceBased(
        distance=Minkowski(
            rho_initializer=tf.keras.initializers.Constant(2.),
            w_initializer=tf.keras.initializers.Constant(1.),
            trainable=False
        ),
        similarity=ExponentialSimilarity(
            fit_tau=False, fit_gamma=False, fit_beta=False,
            tau_initializer=tf.keras.initializers.Constant(1.),
            gamma_initializer=tf.keras.initializers.Constant(0.0),
            beta_initializer=tf.keras.initializers.Constant(.1),
        )
    )

    pw_1 = DistanceBased(
        distance=Minkowski(
            rho_initializer=tf.keras.initializers.Constant(2.),
            w_initializer=tf.keras.initializers.Constant(1.),
            trainable=False
        ),
        similarity=ExponentialSimilarity(
            fit_tau=False, fit_gamma=False, fit_beta=False,
            tau_initializer=tf.keras.initializers.Constant(1.),
            gamma_initializer=tf.keras.initializers.Constant(0.0),
            beta_initializer=tf.keras.initializers.Constant(.1),
        )
    )

    pw_2 = DistanceBased(
        distance=Minkowski(
            rho_initializer=tf.keras.initializers.Constant(2.),
            w_initializer=tf.keras.initializers.Constant(1.),
            trainable=False
        ),
        similarity=ExponentialSimilarity(
            fit_tau=False, fit_gamma=False, fit_beta=False,
            tau_initializer=tf.keras.initializers.Constant(1.),
            gamma_initializer=tf.keras.initializers.Constant(0.0),
            beta_initializer=tf.keras.initializers.Constant(.1),
        )
    )

    subnets = [pw_0, pw_1, pw_2]
    return subnets


@pytest.fixture
def nested_subnet_gate_v0():
    """A list of subnets"""
    subnet_0 = AddPairs(0.00)
    subnet_1 = AddPairs(0.01)
    subnet_2 = AddPairs(0.02)
    subnet_3 = AddPairs(0.03)

    group_inner = BraidGate(
        subnets=[subnet_2, subnet_3], gating_index=-1
    )
    group_outer = BraidGate(
        subnets=[subnet_0, subnet_1, group_inner], gating_index=-2,
        pass_gate_weights=[False, False, False]
    )
    return group_outer


@pytest.fixture
def inputs_5x3_v1():
    inputs = tf.constant(
        [
            [0.0, 0.1, 0.2],
            [1.0, 1.1, 1.2],
            [2.0, 2.1, 2.2],
            [3.0, 3.1, 3.2],
            [4.0, 4.1, 4.2]
        ], dtype=tf.float32
    )
    return inputs


def test_build_subnet_1input(emb_subnets_determ):
    braid_layer = BraidGate(subnets=emb_subnets_determ, gating_index=-1)
    braid_layer.build(
        [tf.TensorShape([None, None]), tf.TensorShape([None, None])]
    )

    subnet_0 = braid_layer.subnets[0]
    subnet_1 = braid_layer.subnets[1]
    subnet_2 = braid_layer.subnets[2]

    tf.debugging.assert_equal(
        subnet_0.embeddings, emb_subnets_determ[0].embeddings
    )
    tf.debugging.assert_equal(
        subnet_1.embeddings, emb_subnets_determ[1].embeddings
    )
    tf.debugging.assert_equal(
        subnet_2.embeddings, emb_subnets_determ[2].embeddings
    )


def test_call_1input_emb_determ_2d_input(
        emb_subnets_determ, inputs_emb_v0, groups_v0_1):
    """Test call BraidGate call.

    Does not have timestep axis.

    """
    groups = groups_v0_1
    braid_layer = BraidGate(subnets=emb_subnets_determ, gating_index=-1)

    outputs = braid_layer([inputs_emb_v0, groups])

    desired_outputs = tf.constant([
        [[0.0, 0.1, 0.2], [1.0, 1.1, 1.2], [2.0, 2.1, 2.2]],
        [[103.0, 103.1, 103.2], [104.0, 104.1, 104.2], [105.0, 105.1, 105.2]],
        [[206.0, 206.1, 206.2], [207.0, 207.1, 207.2], [208.0, 208.1, 208.2]],
        [[109.0, 109.1, 109.2], [110.0, 110.1, 110.2], [111.0, 111.1, 111.2]],
        [[209.0, 209.1, 209.2], [210.0, 210.1, 210.2], [211.0, 211.1, 211.2]],
    ], dtype=tf.float32)
    tf.debugging.assert_equal(outputs, desired_outputs)


def test_call_1input_emb_determ_3d_input(
        emb_subnets_determ, inputs_emb_v1, groups_v0_1):
    """Test call with data inputs larger than 2D."""
    groups = groups_v0_1

    braid_layer = BraidGate(subnets=emb_subnets_determ, gating_index=-1)

    outputs = braid_layer([inputs_emb_v1, groups])

    desired_outputs = np.array([
        [
            [[0.0, 0.1, 0.2], [1.0, 1.1, 1.2], [2.0, 2.1, 2.2]],
            [[3.0, 3.1, 3.2], [4.0, 4.1, 4.2], [5.0, 5.1, 5.2]]
        ],
        [
            [
                [100.0, 100.1, 100.2],
                [101.0, 101.1, 101.2],
                [102.0, 102.1, 102.2]
            ],
            [
                [103.0, 103.1, 103.2],
                [104.0, 104.1, 104.2],
                [105.0, 105.1, 105.2]]
        ],
        [
            [
                [200.0, 200.1, 200.2],
                [201.0, 201.1, 201.2],
                [202.0, 202.1, 202.2]
            ],
            [
                [203.0, 203.1, 203.2],
                [204.0, 204.1, 204.2],
                [205.0, 205.1, 205.2]
            ]
        ],
        [
            [
                [106.0, 106.1, 106.2],
                [107.0, 107.1, 107.2],
                [108.0, 108.1, 108.2]
            ],
            [
                [109.0, 109.1, 109.2],
                [110.0, 110.1, 110.2],
                [111.0, 111.1, 111.2]
            ]
        ],
        [
            [
                [206.0, 206.1, 206.2],
                [207.0, 207.1, 207.2],
                [208.0, 208.1, 208.2]
            ],
            [
                [209.0, 209.1, 209.2],
                [210.0, 210.1, 210.2],
                [211.0, 211.1, 211.2]
            ]
        ]
    ], dtype=np.float32)
    np.testing.assert_array_almost_equal(outputs.numpy(), desired_outputs)


def test_call_1input_emb_stoch_2d_input_rank0(
        emb_subnets_stoch_rank0, inputs_emb_v0, groups_v0_1):
    """Test call that does not require an internal reshape."""
    groups = groups_v0_1

    braid_layer = BraidGate(
        subnets=emb_subnets_stoch_rank0, gating_index=-1
    )

    outputs = braid_layer([inputs_emb_v0, groups])

    desired_outputs = np.array([
        [[0.0, 0.1, 0.2], [1.0, 1.1, 1.2], [2.0, 2.1, 2.2]],
        [[103.0, 103.1, 103.2], [104.0, 104.1, 104.2], [105.0, 105.1, 105.2]],
        [[206.0, 206.1, 206.2], [207.0, 207.1, 207.2], [208.0, 208.1, 208.2]],
        [[109.0, 109.1, 109.2], [110.0, 110.1, 110.2], [111.0, 111.1, 111.2]],
        [[209.0, 209.1, 209.2], [210.0, 210.1, 210.2], [211.0, 211.1, 211.2]],
    ], dtype=np.float32)
    np.testing.assert_array_almost_equal(
        outputs.numpy(), desired_outputs, decimal=1
    )


def test_call_1input_emb_stoch_2d_input_rank1(
        emb_subnets_stoch_rank1, inputs_emb_v0, groups_v0_1):
    """Test call that does not require an internal reshape."""
    groups = groups_v0_1

    braid_layer = BraidGate(
        subnets=emb_subnets_stoch_rank1, gating_index=-1
    )

    outputs = braid_layer([inputs_emb_v0, groups])

    desired_outputs = np.array([
        [[0.0, 0.1, 0.2], [1.0, 1.1, 1.2], [2.0, 2.1, 2.2]],
        [[103.0, 103.1, 103.2], [104.0, 104.1, 104.2], [105.0, 105.1, 105.2]],
        [[206.0, 206.1, 206.2], [207.0, 207.1, 207.2], [208.0, 208.1, 208.2]],
        [[109.0, 109.1, 109.2], [110.0, 110.1, 110.2], [111.0, 111.1, 111.2]],
        [[209.0, 209.1, 209.2], [210.0, 210.1, 210.2], [211.0, 211.1, 211.2]],
    ], dtype=np.float32)
    np.testing.assert_array_almost_equal(
        outputs.numpy(), desired_outputs, decimal=1
    )


def test_call_listinput_timestep(gates_v0_timestep, inputs_list_timestep):
    """Test list input with timestep axis.

    This test expands on a test in `test_sparse_dispatcher.py`:
    `test_list_dispatch_timestep`.

    """
    groups = gates_v0_timestep

    subnets = [AddPairs(0.00), AddPairs(0.01), AddPairs(0.02)]
    braid_layer = BraidGate(subnets=subnets, gating_index=-1)

    outputs = braid_layer(
        [inputs_list_timestep[0], inputs_list_timestep[1], groups]
    )

    outputs_desired = tf.constant([
        [[10.00, 10.20, 10.40], [10.02, 10.22, 10.42]],
        [[12.01, 12.21, 12.41], [12.03, 12.23, 12.43]],
        [[14.02, 14.22, 14.42], [14.04, 14.24, 14.44]],
        # NOTE: The fourth batch is a superposition of subnet[1] and subnet[2].
        #  .3 * [[16.01, 16.21, 16.41], [16.03, 16.23, 16.43]] +
        #  .7 * [[16.02, 16.22, 16.42], [16.04, 16.24, 16.44]]
        [[16.017, 16.217, 16.417], [16.037, 16.237, 16.437]],
        [[9.00, 9.10, 9.20], [9.015, 9.115, 9.215]],
    ])
    tf.debugging.assert_near(outputs, outputs_desired)


def test_call_1input_emb_empty_branch(
        emb_subnets_stoch_rank0, inputs_emb_v0, group_3g_empty_v0_1):
    """Test call that does not require an internal reshape."""
    braid_layer = BraidGate(subnets=emb_subnets_stoch_rank0, gating_index=-1)

    outputs = braid_layer([inputs_emb_v0, group_3g_empty_v0_1])

    desired_outputs = np.array([
        [[100.0, 100.1, 100.2], [101.0, 101.1, 101.2], [102.0, 102.1, 102.2]],
        [[103.0, 103.1, 103.2], [104.0, 104.1, 104.2], [105.0, 105.1, 105.2]],
        [[206.0, 206.1, 206.2], [207.0, 207.1, 207.2], [208.0, 208.1, 208.2]],
        [[109.0, 109.1, 109.2], [110.0, 110.1, 110.2], [111.0, 111.1, 111.2]],
        [[209.0, 209.1, 209.2], [210.0, 210.1, 210.2], [211.0, 211.1, 211.2]],
    ], dtype=np.float32)
    np.testing.assert_array_almost_equal(
        outputs.numpy(), desired_outputs, decimal=1
    )


def test_serialization_1input_emb(
        emb_subnets_stoch_rank1, inputs_emb_v0, groups_v0_0):
    groups = groups_v0_0

    # Build group-specific layer.
    braid_layer = BraidGate(
        subnets=emb_subnets_stoch_rank1, gating_index=-1
    )
    _ = braid_layer([inputs_emb_v0, groups])

    # Get configuration.
    config = braid_layer.get_config()

    # Reconstruct layer from configuration.
    group_layer_recon = BraidGate.from_config(config)

    # Assert reconstructed attributes are the same as original.
    assert braid_layer.n_subnet == group_layer_recon.n_subnet
    assert (
        braid_layer.gating_index ==
        group_layer_recon.gating_index
    )

    outputs = braid_layer([inputs_emb_v0, groups])

    outputs_recon = group_layer_recon([inputs_emb_v0, groups])

    # Assert calls are roughly equal (given constant initializer).
    np.testing.assert_array_almost_equal(
        outputs.numpy(), outputs_recon.numpy(), decimal=1
    )


def test_compute_output_shape_1input_deterministic_embedding(
        emb_subnets_determ, inputs_emb_v0, groups_v0_1):
    """Test `compute_output_shape`."""
    groups = groups_v0_1

    braid_layer = BraidGate(
        subnets=emb_subnets_determ, gating_index=-1
    )
    input_shape = [inputs_emb_v0.shape, groups.shape]
    braid_layer.build(input_shape)
    output_shape = braid_layer.compute_output_shape(input_shape)
    assert output_shape == tf.TensorShape([5, 3, 3])


def test_compute_output_shape_1input_stochastic_embedding(
        emb_subnets_stoch_rank0, inputs_emb_v0, groups_v0_1):
    """Test `compute_output_shape`."""
    groups = groups_v0_1

    braid_layer = BraidGate(
        subnets=emb_subnets_stoch_rank0, gating_index=-1
    )
    input_shape = [inputs_emb_v0.shape, groups.shape]
    braid_layer.build(input_shape)
    output_shape = braid_layer.compute_output_shape(input_shape)
    assert output_shape == tf.TensorShape([5, 3, 3])


def test_subnet_listinput(kernel_subnets):
    braid_layer = BraidGate(subnets=kernel_subnets, gating_index=-1)
    braid_layer.build(
        [
            tf.TensorShape([None, 3]), tf.TensorShape([None, 3]),
            tf.TensorShape([None, 3])
        ]
    )

    subnet_0 = braid_layer.subnets[0]
    subnet_1 = braid_layer.subnets[1]
    subnet_2 = braid_layer.subnets[2]

    tf.debugging.assert_equal(
        subnet_0.distance.rho, kernel_subnets[0].distance.rho
    )
    tf.debugging.assert_equal(
        subnet_0.distance.w, kernel_subnets[0].distance.w
    )

    tf.debugging.assert_equal(
        subnet_1.distance.rho, kernel_subnets[1].distance.rho
    )
    tf.debugging.assert_equal(
        subnet_1.distance.w, kernel_subnets[1].distance.w
    )

    tf.debugging.assert_equal(
        subnet_2.distance.rho, kernel_subnets[2].distance.rho
    )
    tf.debugging.assert_equal(
        subnet_2.distance.w, kernel_subnets[2].distance.w
    )


def test_bad_instantiation_listinput(
        kernel_subnets, paired_inputs_v0, groups_v0_0):
    """Test bad instantiation of layer."""
    inputs = [paired_inputs_v0[0], paired_inputs_v0[1], groups_v0_0]

    # Test bad instantiation that is missing `gating_index`.
    braid_layer = BraidGate(subnets=kernel_subnets)
    with pytest.raises(Exception) as e_info:
        _ = braid_layer(inputs)
    assert e_info.type == ValueError


def test_call_listinput_kernel(kernel_subnets, paired_inputs_v0, groups_v0_0):
    inputs = [paired_inputs_v0[0], paired_inputs_v0[1], groups_v0_0]

    braid_layer = BraidGate(subnets=kernel_subnets, gating_index=-1)
    outputs = braid_layer(inputs)

    # x = np.exp(-.1 * np.sqrt(3*(5**2)))
    desired_outputs = np.array([
        0.4206200260541147,
        0.4206200260541147,
        0.4206200260541147,
        0.4206200260541147,
        0.4206200260541147
    ])

    np.testing.assert_array_almost_equal(desired_outputs, outputs.numpy())


def test_call_listinput_kernel_empty_branch(
        paired_inputs_v0, group_3g_empty_v0_0):
    """Test call with empty branch.

    This test does not have an explicit assert, but tests that such a
    call does not raise a runtime error.

    """
    n_dim = 3
    kl_weight = .1

    shared_similarity = ExponentialSimilarity(
        beta_initializer=tf.keras.initializers.Constant(10.),
        tau_initializer=tf.keras.initializers.Constant(1.),
        gamma_initializer=tf.keras.initializers.Constant(0.),
        trainable=False
    )

    # Define group-specific kernels.
    kernel_0 = build_vi_kernel(shared_similarity, n_dim, kl_weight)
    kernel_1 = build_vi_kernel(shared_similarity, n_dim, kl_weight)
    kernel_2 = build_vi_kernel(shared_similarity, n_dim, kl_weight)
    kernel_group = BraidGate(
        subnets=[kernel_0, kernel_1, kernel_2], gating_index=-1
    )

    _ = kernel_group(
        [paired_inputs_v0[0], paired_inputs_v0[1], group_3g_empty_v0_0]
    )


def test_compute_output_shape_listinput_kernel(
        kernel_subnets, paired_inputs_v0, groups_v0_0):
    """Test output_shape method."""
    groups = groups_v0_0

    braid_layer = BraidGate(subnets=kernel_subnets, gating_index=-1)

    input_shape = [
        tf.TensorShape(tf.shape(paired_inputs_v0[0])),
        tf.TensorShape(tf.shape(paired_inputs_v0[1])),
        tf.TensorShape(tf.shape(groups))
    ]
    braid_layer.build(input_shape)
    output_shape = braid_layer.compute_output_shape(input_shape)
    desired_output_shape = tf.TensorShape([5])
    tf.debugging.assert_equal(output_shape, desired_output_shape)


def test_serialization_listinput(
        kernel_subnets, paired_inputs_v0, groups_v0_0):
    """Test serialization."""
    groups = groups_v0_0

    group_layer_0 = BraidGate(
        subnets=kernel_subnets, gating_index=-1
    )
    outputs_0 = group_layer_0(
        [paired_inputs_v0[0], paired_inputs_v0[1], groups]
    )
    config = group_layer_0.get_config()
    assert config['gating_index'] == -1

    group_layer_1 = BraidGate.from_config(config)
    outputs_1 = group_layer_1(
        [paired_inputs_v0[0], paired_inputs_v0[1], groups]
    )

    tf.debugging.assert_equal(outputs_0, outputs_1)


def test_bad_instantiation_dictinput(
        gates_v0_timestep, inputs_dict_timestep):
    """Test bad instantiation."""
    inputs = inputs_dict_timestep
    inputs['groups'] = gates_v0_timestep

    subnets = [AddPairsDict(0.00), AddPairsDict(0.01), AddPairsDict(0.02)]

    # Check bad instantiation that is missing `gating_key` argument.
    braid_layer = BraidGate(subnets=subnets)
    with pytest.raises(Exception) as e_info:
        _ = braid_layer(inputs)
    assert e_info.type == ValueError


def test_call_dictinput_timestep(gates_v0_timestep, inputs_dict_timestep):
    """Test dictionary input with timestep axis.

    This test expands on a test in `test_sparse_dispatcher.py`:
    `test_dict_dispatch_timestep`.

    """
    inputs = inputs_dict_timestep
    inputs['groups'] = gates_v0_timestep

    subnets = [AddPairsDict(0.00), AddPairsDict(0.01), AddPairsDict(0.02)]

    braid_layer = BraidGate(subnets=subnets, gating_key='groups')
    outputs = braid_layer(inputs)

    outputs_desired = tf.constant([
        [[10.00, 10.20, 10.40], [10.02, 10.22, 10.42]],
        [[12.01, 12.21, 12.41], [12.03, 12.23, 12.43]],
        [[14.02, 14.22, 14.42], [14.04, 14.24, 14.44]],
        # NOTE: The fourth batch is a superposition of subnet[1] and subnet[2].
        #  .3 * [[16.01, 16.21, 16.41], [16.03, 16.23, 16.43]] +
        #  .7 * [[16.02, 16.22, 16.42], [16.04, 16.24, 16.44]]
        [[16.017, 16.217, 16.417], [16.037, 16.237, 16.437]],
        [[9.00, 9.10, 9.20], [9.015, 9.115, 9.215]],
    ])
    tf.debugging.assert_near(outputs, outputs_desired)


def test_call_nested(
    inputs_5x3_v1, nested_subnet_gate_v0, paired_inputs_v0, groups_v0_1,
    groups_v0_2
):
    """Test two-level nested BraidGate."""
    inputs_0 = inputs_5x3_v1
    groups_1 = groups_v0_1
    groups_2 = groups_v0_2

    braid_layer = nested_subnet_gate_v0

    # Desired values.
    inputs_1 = tf.constant(
        [
            [5.0, 5.1, 5.2],
            [6.0, 6.1, 6.2],
            [7.0, 7.1, 7.2],
            [8.0, 8.1, 8.2],
            [9.0, 9.1, 9.2]
        ], dtype=tf.float32
    )
    gate_increment = tf.expand_dims(
        tf.constant([0., 0.01, 0.02, 0.01, 0.03]), axis=1
    )
    desired_outputs = inputs_0 + inputs_1 + gate_increment

    outputs = braid_layer(
        [paired_inputs_v0[0], paired_inputs_v0[1], groups_1, groups_2]
    )

    tf.debugging.assert_equal(desired_outputs, outputs.numpy())


def test_call_mixture(inputs_5x3_v1, groups_v3_12):
    """Test BraidedGate when mixing branches."""
    inputs_0 = inputs_5x3_v1
    groups = groups_v3_12

    braid_layer = BraidGate(
        subnets=[Increment(1.), Increment(3.)], gating_index=-1
    )

    desired_increment = tf.constant(
        [
            [1.0, 1.0, 1.0],
            [1.4, 1.4, 1.4],
            [2.0, 2.0, 2.0],
            [2.6, 2.6, 2.6],
            [3.0, 3.0, 3.0]
        ], dtype=tf.float32
    )
    desired_outputs = inputs_0 + desired_increment

    outputs = braid_layer([inputs_0, groups])

    np.testing.assert_array_almost_equal(desired_outputs, outputs.numpy())


def test_call_mixture_w_emb(groups_v3_12):
    """Test BraidedGate when mixing branches."""
    # inputs = paired_inputs_v0
    groups = groups_v3_12

    # Define BraidGate layer
    emb_loc_0 = np.array(
        [
            [0.1, 0.2],
            [1.1, 1.2],
            [2.1, 2.2],
            [3.1, 3.2],
            [4.1, 4.2],
            [5.1, 5.2],
            [6.1, 6.2],
            [7.1, 7.2],
            [8.1, 8.2],
            [9.1, 9.2],
            [10.1, 10.2],
            [11.1, 11.2],
            [12.1, 12.2],
            [13.1, 13.2],
            [14.1, 14.2],
        ], dtype=np.float32
    )
    emb_loc_1 = np.array(
        [
            [20.1, -0.2],
            [21.1, -1.2],
            [22.1, -2.2],
            [23.1, -3.2],
            [24.1, -4.2],
            [25.1, -5.2],
            [26.1, -6.2],
            [27.1, -7.2],
            [28.1, -8.2],
            [29.1, -9.2],
            [30.1, -10.2],
            [31.1, -11.2],
            [32.1, -12.2],
            [33.1, -13.2],
            [34.1, -14.2],
        ], dtype=np.float32
    )
    emb_0 = tf.keras.layers.Embedding(
        15, 2, mask_zero=False,
        embeddings_initializer=tf.keras.initializers.Constant(emb_loc_0)
    )
    emb_1 = tf.keras.layers.Embedding(
        15, 2, mask_zero=False,
        embeddings_initializer=tf.keras.initializers.Constant(emb_loc_1)
    )
    braid_layer = BraidGate(subnets=[emb_0, emb_1], gating_index=-1)

    inputs_0 = np.array(
        [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [9, 10, 11],
            [12, 13, 14]
        ], dtype=np.float32
    )
    outputs = braid_layer([inputs_0, groups])

    # The desired "linear mixture" of embedding outputs.
    desired_outputs = np.array(
        [
            # 1.0, 0.0 mix
            [
                [0.1, 0.2],
                [1.1, 1.2],
                [2.1, 2.2],
            ],
            # 0.8, 0.2 mix
            [
                [7.1, 1.92],
                [8.1, 2.52],
                [9.1, 3.12],
            ],
            # 0.5, 0.5 mix
            [
                [16.1, 0.0],
                [17.1, 0.0],
                [18.1, 0.0]
            ],
            # 0.2, 0.8 mix
            [
                [25.1, -5.52],
                [26.1, -6.12],
                [27.1, -6.72]
            ],
            # 0.0, 1.0 mix
            [
                [32.1, -12.2],
                [33.1, -13.2],
                [34.1, -14.2],
            ],
        ], dtype=np.float32
    )
    np.testing.assert_array_almost_equal(desired_outputs, outputs.numpy())
