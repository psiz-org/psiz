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
    heatmap_embeddings: Create a heatmap of embeddings.

"""

import matplotlib
import numpy as np
import tensorflow_probability as tfp


def heatmap_embeddings(fig, ax, embedding, cmap=None):
    """Visualize embeddings as a heatmap.

    Intended to handle rank 2 embeddings.

    Arguments:
        fig: A Matplotlib Figure object.
        ax: A Matplotlib Axes object.
        embedding: An embedding layer.
        cmap (optional): A Matplotlib compatible colormap.

    """
    if cmap is None:
        cmap = matplotlib.cm.get_cmap('Greys')

    if isinstance(embedding.embeddings, tfp.distributions.Distribution):
        # Handle distribution.
        z_mode = embedding.embeddings.mode().numpy()
    else:
        # Handle point estimate.
        z_mode = embedding.embeddings.numpy()

    if embedding.mask_zero:
        z_mode = z_mode[1:]

    n_output_dim = z_mode.shape[-1]
    z_mode_min = np.min(z_mode)
    z_mode_max = np.max(z_mode)
    # MAYBE use `matshow` instead of imshow to fix interpolation issue?
    im = ax.imshow(
        z_mode, cmap=cmap, interpolation='none',
        vmin=z_mode_min, vmax=z_mode_max
    )

    # NOTE: `imshow` displays different rows as different values of y and
    # different columns as different values of x.
    ax.set_xlabel('Output')
    ax.set_xticks([0, n_output_dim - 1])
    ax.set_xticklabels([0, n_output_dim - 1])
    ax.set_ylabel('Input')
    # TODO use fig = plt.gcf() instead of using passed in `fig` argument.
    fig.colorbar(im, ax=ax)
