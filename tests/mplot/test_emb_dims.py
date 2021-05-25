# -*- coding: utf-8 -*-
# Copyright 2021 The PsiZ Authors. All Rights Reserved.
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

import pytest
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import psiz.mplot


def emb_0(mask_zero, is_dist):
    """Location parameters for embedding."""
    n_stimuli = 5
    n_dim = 3
    if mask_zero:
        locs = np.array([
            [0., 0., 0.],
            [-.14, .11, .11],
            [-.14, -.12, .13],
            [.16, .14, .12],
            [.15, -.14, .05],
            [.2, .2, .2],
        ])
        n_stimuli += 1
    else:
        locs = np.array([
            [-.14, .11, .11],
            [-.14, -.12, .13],
            [.16, .14, .12],
            [.15, -.14, .05],
            [.2, .2, .2],
        ])

    if is_dist:
        emb = psiz.keras.layers.EmbeddingNormalDiag(
            n_stimuli, n_dim, mask_zero=mask_zero,
            loc_initializer=tf.keras.initializers.Constant(locs),
            scale_initializer=tf.keras.initializers.Constant(
                tfp.math.softplus_inverse(.2).numpy()
            )
        )
    else:
        emb = tf.keras.layers.Embedding(
            n_stimuli, n_dim, mask_zero=mask_zero,
            embeddings_initializer=tf.keras.initializers.Constant(locs)
        )
    emb.build([None])

    return emb


@pytest.mark.parametrize("mask_zero", [False, True])
def test_deterministic_emb_input(mask_zero):
    """Basic test of deterministic embedding."""
    is_dist = False
    emb = emb_0(mask_zero, is_dist)

    num_figures_before = plt.gcf().number

    fig = plt.figure(figsize=(6.5, 4), dpi=200)
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0, 0])
    idx = 1
    psiz.mplot.embedding_input_dimension(fig, ax, emb, idx, c='b')
    gs.tight_layout(fig)

    num_figures_after = plt.gcf().number
    assert num_figures_before < num_figures_after

    # Check scatter of point estimates.
    cs = ax.collections[0]
    arr = cs.get_offsets()

    desired_arr = np.array(
        [
            [0., -0.14],
            [1., -0.12],
            [2.,  0.13]
        ]
    )
    np.testing.assert_array_almost_equal(
        arr.data, desired_arr
    )


@pytest.mark.parametrize("mask_zero", [False, True])
def test_stochastic_emb_input(mask_zero):
    """Basic test of deterministic embedding."""
    is_dist = True
    emb = emb_0(mask_zero, is_dist)

    num_figures_before = plt.gcf().number

    fig = plt.figure(figsize=(6.5, 4), dpi=200)
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0, 0])
    idx = 1
    psiz.mplot.embedding_input_dimension(fig, ax, emb, idx, c='b')
    gs.tight_layout(fig)

    num_figures_after = plt.gcf().number
    assert num_figures_before < num_figures_after

    # Check scatter of mode.
    cs = ax.collections[0]
    arr = cs.get_offsets()

    desired_arr = np.array(
        [
            [0., -0.14],
            [1., -0.12],
            [2.,  0.13]
        ]
    )
    np.testing.assert_array_almost_equal(
        arr.data, desired_arr
    )

    # Spot check middle density intervals.
    np.testing.assert_array_almost_equal(ax.lines[0]._x, np.array([0., 0.]))
    np.testing.assert_array_almost_equal(
        ax.lines[0]._y, np.array([-0.65516615,  0.37516624])
    )

    np.testing.assert_array_almost_equal(ax.lines[-1]._x, np.array([2., 2.]))
    np.testing.assert_array_almost_equal(
        ax.lines[-1]._y, np.array([-0.00489804,  0.26489803])
    )


@pytest.mark.parametrize("mask_zero", [False, True])
def test_deterministic_emb_output(mask_zero):
    """Basic test of deterministic embedding."""
    is_dist = False
    emb = emb_0(mask_zero, is_dist)

    num_figures_before = plt.gcf().number

    fig = plt.figure(figsize=(6.5, 4), dpi=200)
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0, 0])
    idx = 1
    psiz.mplot.embedding_output_dimension(fig, ax, emb, idx, c='b')
    gs.tight_layout(fig)

    num_figures_after = plt.gcf().number
    assert num_figures_before < num_figures_after

    # Check scatter of point estimates.
    cs = ax.collections[0]
    arr = cs.get_offsets()

    desired_arr = np.array([
        [0.,  0.11],
        [1., -0.12],
        [2.,  0.14],
        [3., -0.14],
        [4.,  0.2]
    ])
    np.testing.assert_array_almost_equal(
        arr.data, desired_arr
    )


@pytest.mark.parametrize("mask_zero", [False, True])
def test_stochastic_emb_output(mask_zero):
    """Basic test of deterministic embedding."""
    is_dist = True
    emb = emb_0(mask_zero, is_dist)

    num_figures_before = plt.gcf().number

    fig = plt.figure(figsize=(6.5, 4), dpi=200)
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0, 0])
    idx = 1
    psiz.mplot.embedding_output_dimension(fig, ax, emb, idx, c='b')
    gs.tight_layout(fig)

    num_figures_after = plt.gcf().number
    assert num_figures_before < num_figures_after

    # Check scatter of mode.
    cs = ax.collections[0]
    arr = cs.get_offsets()

    desired_arr = np.array([
        [0.,  0.11],
        [1., -0.12],
        [2.,  0.14],
        [3., -0.14],
        [4.,  0.2]
    ])
    np.testing.assert_array_almost_equal(
        arr.data, desired_arr
    )

    # Spot check middle density intervals.
    np.testing.assert_array_almost_equal(ax.lines[0]._x, np.array([0., 0.]))
    np.testing.assert_array_almost_equal(
        ax.lines[0]._y, np.array([-0.40516615,  0.62516624])
    )

    np.testing.assert_array_almost_equal(ax.lines[-1]._x, np.array([4., 4.]))
    np.testing.assert_array_almost_equal(
        ax.lines[-1]._y, np.array([0.06510197, 0.33489805])
    )


@pytest.mark.parametrize("mask_zero", [False, True])
@pytest.mark.parametrize("is_dist", [False, True])
@pytest.mark.parametrize("cmap", [False, True])
def test_emb_heatmap(is_dist, mask_zero, cmap):
    """Test plotter `heatmap_embedding`."""
    emb = emb_0(mask_zero, is_dist)

    num_figures_before = plt.gcf().number

    fig = plt.figure(figsize=(6.5, 4), dpi=200)
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0, 0])

    if cmap:
        cmap = matplotlib.cm.get_cmap('Blues')
        psiz.mplot.heatmap_embeddings(fig, ax, emb, cmap=cmap)
    else:
        psiz.mplot.heatmap_embeddings(fig, ax, emb)

    num_figures_after = plt.gcf().number
    assert num_figures_before < num_figures_after

    desired_array = np.array([
        [-.14, .11, .11],
        [-.14, -.12, .13],
        [.16, .14, .12],
        [.15, -.14, .05],
        [.2, .2, .2],
    ])
    np.testing.assert_array_almost_equal(
        desired_array, ax.images[0].get_array().data,
    )
