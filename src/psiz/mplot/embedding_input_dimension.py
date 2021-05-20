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
    embedding_input_dimension: Visualize embedding values for a
        requested input dimension.

"""

import numpy as np
import tensorflow_probability as tfp


def embedding_input_dimension(fig, ax, embedding, idx, c='b'):
    """Visualize embedding values for a requested input dimension.

    Plots point estimates of embedding values for the requested
    input dimension.

    Intended to handle rank 2 and rank 3 embeddings.

    If the embedding layer is a distribution, also attempts to draw a
    thick linewidth interval indicating the middle 50% probability mass
    and a thin linewidth interval indicating the middle 99% probability
    mass via the inverse CDF function.

    Arguments:
        fig: A Matplotlib Figure object.
        ax: A Matplotlib Axes object.
        embedding: An embedding layer.
        idx: Index of requested input dimension to visualize.
        c (optional): Color of interval marks.

    """
    if isinstance(embedding.embeddings, tfp.distributions.Distribution):
        z_mode = embedding.embeddings.mode().numpy()
    else:
        z_mode = embedding.embeddings.numpy()

    rank = z_mode.ndim

    y_max = np.max(z_mode)
    z_mode = z_mode[idx, :]

    # Handle masking.
    if embedding.mask_zero:
        z_mode = z_mode[1:]

    n_output_dim = z_mode.shape[0]

    # Scatter point estimate.
    xg = np.arange(n_output_dim)
    ax.scatter(xg, z_mode, c=c, marker='_')

    # Add posterior quantiles if available.
    if hasattr(embedding, 'posterior'):
        dist = embedding.posterior.embeddings.distribution

        # Middle 99% of probability mass.
        p = .99
        v = (1 - p) / 2
        quant_lower = dist.quantile(v).numpy()[idx, :]
        quant_upper = dist.quantile(1-v).numpy()
        y_max = np.max(quant_upper)
        quant_upper = quant_upper[idx, :]

        # Middle 50% probability mass.
        p = .5
        v = (1 - p) / 2
        mid_lower = dist.quantile(v).numpy()[idx, :]
        mid_upper = dist.quantile(1-v).numpy()[idx, :]

        if embedding.posterior.mask_zero:
            quant_lower = quant_lower[1:]
            quant_upper = quant_upper[1:]
            mid_lower = mid_lower[1:]
            mid_upper = mid_upper[1:]

        for i_dim in range(n_output_dim):
            xg = np.array([i_dim, i_dim])
            yg = np.array(
                [quant_lower[i_dim], quant_upper[i_dim]]
            )
            ax.plot(xg, yg, c=c, linewidth=1)

            yg = np.array(
                [mid_lower[i_dim], mid_upper[i_dim]]
            )
            ax.plot(xg, yg, c=c, linewidth=3)

    ax.set_xlabel('Output Dimension')
    ax.set_xlim([-.5, n_output_dim-.5])

    ax.set_ylabel(r'$x$')
    ax.set_ylim([0, 1.05 * y_max])
    ax.set_yticks([0, 1.05 * y_max])
    ax.set_yticklabels(['0', '{0:.1f}'.format(1.05 * y_max)])
