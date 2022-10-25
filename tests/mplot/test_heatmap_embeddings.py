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

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from psiz.keras.layers import EmbeddingNormalDiag
from psiz.mplot import heatmap_embeddings


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
        emb = EmbeddingNormalDiag(
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
@pytest.mark.parametrize("is_dist", [False, True])
@pytest.mark.parametrize("cmap", [False, True])
@pytest.mark.parametrize("ax_present", [False, True])
def test_emb_heatmap(is_dist, mask_zero, cmap, ax_present):
    """Test plotter `heatmap_embedding`."""
    emb = emb_0(mask_zero, is_dist)

    num_figures_before = plt.gcf().number

    fig = plt.figure(figsize=(6.5, 4), dpi=200)
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0, 0])

    if cmap:
        cmap = matplotlib.colormaps['Blues']
        heatmap_embeddings(emb, ax=ax, cmap=cmap)
    else:
        if ax_present:
            heatmap_embeddings(emb, ax=ax)
        else:
            heatmap_embeddings(emb)

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
    plt.close(fig)
