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
"""Module of visualization tools.

Functions:
    embedding_output_dimension: Visualize embedding values for a
        requested output dimension.

"""

import numpy as np
import tensorflow_probability as tfp


def embedding_output_dimension(fig, ax, embedding, idx, c='b'):
    """Visualize all embedding values for a requested output dimension.

    Plots point estimates of embedding values for the requested
    output dimension.

    If the embedding layer is a distribution, also attempts to draw a
    thick linewidth interval indicating the middle 50% probability mass
    and a thin linewidth interval indicating the middle 99% probability
    mass via the inverse CDF function.

    Intended to handle rank 2 embeddings.

    Arguments:
        fig: A Matplotlib Figure object.
        ax: A Matplotlib Axes object.
        embedding: An embedding layer.
        idx: Index of requested output dimension to visualize.
        c (optional): Color of interval marks.

    """
    if isinstance(embedding.embeddings, tfp.distributions.Distribution):
        z_mode = embedding.embeddings.mode().numpy()
        is_distribution = True
    else:
        z_mode = embedding.embeddings.numpy()
        is_distribution = False

    # Handle masking.
    if embedding.mask_zero:
        z_mode = z_mode[1:, :]
    n_input_dim = z_mode.shape[0]
    # n_output_dim = z_mode.shape[1]

    z_mode = z_mode[:, idx]
    y_min = np.min(z_mode)
    y_max = np.max(z_mode)

    # Scatter point estimates.
    xg = np.arange(n_input_dim)
    ax.scatter(xg, z_mode, c=c, marker='_', linewidth=1)

    if is_distribution:
        dist = embedding.embeddings.distribution

        # Middle density interval: 99% probability mass.
        p = .99
        v = (1 - p) / 2
        mdi99_lower = dist.quantile(v).numpy()[:, idx]
        mdi99_upper = dist.quantile(1 - v).numpy()[:, idx]
        # Override ymin and ymax based on 99quant.
        y_min = np.min(mdi99_lower)
        y_max = np.max(mdi99_upper)

        # Middle density interval: 50% probability mass.
        p = .5
        v = (1 - p) / 2
        mdi50_lower = dist.quantile(v).numpy()[:, idx]
        mdi50_upper = dist.quantile(1 - v).numpy()[:, idx]

        if embedding.mask_zero:
            mdi99_lower = mdi99_lower[1:]
            mdi99_upper = mdi99_upper[1:]
            mdi50_lower = mdi50_lower[1:]
            mdi50_upper = mdi50_upper[1:]

        for i_dim in range(n_input_dim):
            xg = np.array([i_dim, i_dim])
            # Draw thin line for MDI 99%.
            yg = np.array(
                [mdi99_lower[i_dim], mdi99_upper[i_dim]]
            )
            ax.plot(xg, yg, c=c, linewidth=1)

            # Draw thick line for MDI 50%.
            yg = np.array(
                [mdi50_lower[i_dim], mdi50_upper[i_dim]]
            )
            ax.plot(xg, yg, c=c, linewidth=3)

    ax.set_xlabel('Input Dimension')
    ax.set_xlim([-.5, n_input_dim - .5])

    ax.set_ylabel(r'$z$')
    ax.set_ylim([1.05 * y_min, 1.05 * y_max])
