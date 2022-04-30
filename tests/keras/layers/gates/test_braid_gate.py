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


@pytest.fixture
def emb_inputs_v0():
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
def emb_inputs_v1():
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

    group_inner = BraidGate(subnets=[subnet_2, subnet_3], groups_subset=2)
    group_outer = BraidGate(
        subnets=[subnet_0, subnet_1, group_inner], groups_subset=1,
        pass_groups=[False, False, True]
    )
    return group_outer


def test_inp1_subnet_method(emb_subnets_determ):
    group_layer = BraidGate(subnets=emb_subnets_determ, groups_subset=0)
    group_layer.build([[None, None], [None, None]])

    subnet_0 = group_layer.subnets[0]
    subnet_1 = group_layer.subnets[1]
    subnet_2 = group_layer.subnets[2]

    tf.debugging.assert_equal(
        subnet_0.embeddings, emb_subnets_determ[0].embeddings
    )
    tf.debugging.assert_equal(
        subnet_1.embeddings, emb_subnets_determ[1].embeddings
    )
    tf.debugging.assert_equal(
        subnet_2.embeddings, emb_subnets_determ[2].embeddings
    )


def test_inp1_emb_call_determ_2d_input(
        emb_subnets_determ, emb_inputs_v0, groups_v0):
    """Test call that does not require an internal reshape."""
    groups = groups_v0
    group_layer = BraidGate(subnets=emb_subnets_determ, groups_subset=1)

    outputs = group_layer([emb_inputs_v0, groups])

    desired_outputs = np.array([
        [[0.0, 0.1, 0.2], [1.0, 1.1, 1.2], [2.0, 2.1, 2.2]],
        [[103.0, 103.1, 103.2], [104.0, 104.1, 104.2], [105.0, 105.1, 105.2]],
        [[206.0, 206.1, 206.2], [207.0, 207.1, 207.2], [208.0, 208.1, 208.2]],
        [[109.0, 109.1, 109.2], [110.0, 110.1, 110.2], [111.0, 111.1, 111.2]],
        [[209.0, 209.1, 209.2], [210.0, 210.1, 210.2], [211.0, 211.1, 211.2]],
    ], dtype=np.float32)
    np.testing.assert_array_almost_equal(outputs.numpy(), desired_outputs)


def test_inp1_emb_call_determ_3d_input(
        emb_subnets_determ, emb_inputs_v1, groups_v0):
    """Test call with data inputs larger than 2D."""
    groups = groups_v0

    group_layer = BraidGate(subnets=emb_subnets_determ, groups_subset=1)

    outputs = group_layer([emb_inputs_v1, groups])

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


def test_inp1_emb_call_stoch_2d_input_rank0(
        emb_subnets_stoch_rank0, emb_inputs_v0, groups_v0):
    """Test call that does not require an internal reshape."""
    groups = groups_v0

    group_layer = BraidGate(subnets=emb_subnets_stoch_rank0, groups_subset=1)

    outputs = group_layer([emb_inputs_v0, groups])

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


def test_inp1_emb_call_stoch_2d_input_rank1(
        emb_subnets_stoch_rank1, emb_inputs_v0, groups_v0):
    """Test call that does not require an internal reshape."""
    groups = groups_v0

    group_layer = BraidGate(subnets=emb_subnets_stoch_rank1, groups_subset=1)

    outputs = group_layer([emb_inputs_v0, groups])

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


def test_inp1_emb_call_empty_branch(
        emb_subnets_stoch_rank0, emb_inputs_v0, group_3g_empty_v0):
    """Test call that does not require an internal reshape."""
    group_layer = BraidGate(subnets=emb_subnets_stoch_rank0, groups_subset=1)

    outputs = group_layer([emb_inputs_v0, group_3g_empty_v0])

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


def test_inp1_emb_serialization(
        emb_subnets_stoch_rank1, emb_inputs_v0, groups_v0):
    groups = groups_v0

    # Build group-specific layer.
    group_layer = BraidGate(subnets=emb_subnets_stoch_rank1, groups_subset=0)

    # Get configuration.
    config = group_layer.get_config()

    # Reconstruct layer from configuration.
    group_layer_recon = BraidGate.from_config(config)

    # Assert reconstructed attributes are the same as original.
    assert group_layer.n_subnet == group_layer_recon.n_subnet
    assert group_layer.groups_subset == group_layer_recon.groups_subset

    outputs = group_layer([emb_inputs_v0, groups])

    outputs_recon = group_layer_recon([emb_inputs_v0, groups])

    # Assert calls are roughly equal (given constant initializer).
    np.testing.assert_array_almost_equal(
        outputs.numpy(), outputs_recon.numpy(), decimal=1
    )


def test_inp1_compute_deterministic_embedding_output_shape(
        emb_subnets_determ, emb_inputs_v0, groups_v0):
    """Test `compute_output_shape`."""
    groups = groups_v0

    group_layer = BraidGate(subnets=emb_subnets_determ, groups_subset=1)
    output_shape = group_layer.compute_output_shape(
        [emb_inputs_v0.shape, groups.shape]
    )
    assert output_shape == tf.TensorShape([5, 3, 3])


def test_inp1_compute_stochastic_embedding_output_shape(
        emb_subnets_stoch_rank0, emb_inputs_v0, groups_v0):
    """Test `compute_output_shape`."""
    groups = groups_v0

    group_layer = BraidGate(subnets=emb_subnets_stoch_rank0, groups_subset=1)
    output_shape = group_layer.compute_output_shape(
        [emb_inputs_v0.shape, groups.shape]
    )
    assert output_shape == tf.TensorShape([5, 3, 3])


def test_inpmulti_subnet_method(kernel_subnets):
    group_layer = BraidGate(subnets=kernel_subnets, groups_subset=0)
    group_layer.build([[None, 3], [None, 3], [None, 3]])

    subnet_0 = group_layer.subnets[0]
    subnet_1 = group_layer.subnets[1]
    subnet_2 = group_layer.subnets[2]

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


def test_inpmulti_kernel_call(kernel_subnets, paired_inputs_v0, groups_v0):
    groups = groups_v0

    group_layer = BraidGate(subnets=kernel_subnets, groups_subset=0)
    outputs = group_layer(
        [paired_inputs_v0[0], paired_inputs_v0[1], groups]
    )

    # x = np.exp(-.1 * np.sqrt(3*(5**2)))
    desired_outputs = np.array([
        0.4206200260541147,
        0.4206200260541147,
        0.4206200260541147,
        0.4206200260541147,
        0.4206200260541147
    ])

    np.testing.assert_array_almost_equal(desired_outputs, outputs.numpy())


def test_inpmulti_call_kernel_empty_branch(
        paired_inputs_v0, group_3g_empty_v0):
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
        subnets=[kernel_0, kernel_1, kernel_2], groups_subset=0
    )

    _ = kernel_group(
        [paired_inputs_v0[0], paired_inputs_v0[1], group_3g_empty_v0]
    )


def test_inpmulti_kernel_output_shape(
        kernel_subnets, paired_inputs_v0, groups_v0):
    """Test output_shape method."""
    groups = groups_v0

    group_layer = BraidGate(subnets=kernel_subnets, groups_subset=0)

    input_shape = [
        tf.TensorShape(tf.shape(paired_inputs_v0[0])),
        tf.TensorShape(tf.shape(paired_inputs_v0[1])),
        tf.TensorShape(tf.shape(groups))
    ]
    output_shape = group_layer.compute_output_shape(input_shape)
    desired_output_shape = tf.TensorShape([5])
    tf.debugging.assert_equal(output_shape, desired_output_shape)


def test_inpmulti_serialization(kernel_subnets, paired_inputs_v0, groups_v0):
    """Test serialization."""
    groups = groups_v0

    group_layer_0 = BraidGate(subnets=kernel_subnets, groups_subset=0)
    outputs_0 = group_layer_0(
        [paired_inputs_v0[0], paired_inputs_v0[1], groups]
    )
    config = group_layer_0.get_config()
    assert config['groups_subset'] == 0

    group_layer_1 = BraidGate.from_config(config)
    outputs_1 = group_layer_1(
        [paired_inputs_v0[0], paired_inputs_v0[1], groups]
    )

    tf.debugging.assert_equal(outputs_0, outputs_1)


def test_nested_call(nested_subnet_gate_v0, paired_inputs_v0, groups_v0):
    """Test two-level nested BraidGate."""
    groups = groups_v0

    group_layer = nested_subnet_gate_v0

    # Desired values.
    inputs_0 = np.array(
        [
            [0.0, 0.1, 0.2],
            [1.0, 1.1, 1.2],
            [2.0, 2.1, 2.2],
            [3.0, 3.1, 3.2],
            [4.0, 4.1, 4.2]
        ], dtype=np.float32
    )
    inputs_1 = np.array(
        [
            [5.0, 5.1, 5.2],
            [6.0, 6.1, 6.2],
            [7.0, 7.1, 7.2],
            [8.0, 8.1, 8.2],
            [9.0, 9.1, 9.2]
        ], dtype=np.float32
    )
    gate_increment = np.expand_dims(
        np.array([0., 0.01, 0.02, 0.01, 0.03]), axis=1
    )
    desired_outputs = inputs_0 + inputs_1 + gate_increment

    outputs = group_layer(
        [paired_inputs_v0[0], paired_inputs_v0[1], groups]
    )

    np.testing.assert_array_almost_equal(desired_outputs, outputs.numpy())


# TODO abstract out inputs_0
def test_mixture_call(groups_v3):
    """Test BraidedGate when mixing branches."""
    # inputs = paired_inputs_v0
    groups = groups_v3

    group_layer = BraidGate(
        subnets=[Increment(1.), Increment(3.)], groups_subset=[1, 2]
    )

    inputs_0 = np.array(
        [
            [0.0, 0.1, 0.2],
            [1.0, 1.1, 1.2],
            [2.0, 2.1, 2.2],
            [3.0, 3.1, 3.2],
            [4.0, 4.1, 4.2]
        ], dtype=np.float32
    )

    desired_increment = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.4, 1.4, 1.4],
            [2.0, 2.0, 2.0],
            [2.6, 2.6, 2.6],
            [3.0, 3.0, 3.0]
        ], dtype=np.float32
    )
    desired_outputs = inputs_0 + desired_increment

    outputs = group_layer([inputs_0, groups])

    np.testing.assert_array_almost_equal(desired_outputs, outputs.numpy())


# TODO abstract out inputs_0
def test_mixture_call_w_emb(groups_v3):
    """Test BraidedGate when mixing branches."""
    # inputs = paired_inputs_v0
    groups = groups_v3

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
    group_layer = BraidGate(
        subnets=[emb_0, emb_1], groups_subset=[1, 2]
    )

    inputs_0 = np.array(
        [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [9, 10, 11],
            [12, 13, 14]
        ], dtype=np.float32
    )
    outputs = group_layer([inputs_0, groups])

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
