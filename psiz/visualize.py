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
    bvn_ellipse: Create an ellipse representing a bivariate normal
        distribution.
    heatmap_embeddings: Create a heatmap of embeddings.
    embedding_output_dimension: Visualize embedding values for a
        requested output dimension.
    embedding_input_dimension: Visualize embedding values for a
        requested input dimension.

"""

import os

from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow_probability as tfp


def heatmap_embeddings(fig, ax, embedding, group_idx=0, cmap=None):
    """Visualize embeddings as a heatmap.

    Intended to handle rank 2 and rank 3 embeddings.

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

    rank = z_mode.ndim
    if rank == 3:
        z_mode = z_mode[group_idx]
    if embedding.mask_zero:
        z_mode = z_mode[1:]

    n_dim = z_mode.shape[-1]
    z_mode_max = np.max(z_mode)
    im = ax.imshow(
        z_mode, cmap=cmap, interpolation='none', vmin=0., vmax=z_mode_max
    )
    # TODO use matshow to fix interpolation issue.

    # Note: imshow displays different rows as different values of y and
    # different columns as different values of x.
    ax.set_xticks([0, n_dim-1])
    ax.set_xticklabels([0, n_dim-1])
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Stimulus')
    fig.colorbar(im, ax=ax)


def embedding_output_dimension(fig, ax, embedding, idx, group_idx=0, c='b'):
    """Visualize embedding values for a requested output dimension.

    Plots point estimates of embedding values for the requested
    output dimension.

    If the embedding layer is a distribution, also attempts to draw a
    thick linewidth interval indicating the middle 50% probability mass
    and a thin linewidth interval indicating the middle 99% probability
    mass via the inverse CDF function.

    Intended to handle rank 2 and rank 3 embeddings.

    Arguments:
        fig: A Matplotlib Figure object.
        ax: A Matplotlib Axes object.
        embedding: An embedding layer.
        idx: Index of requested output dimension to visualize.
        c (optional): Color of interval marks.

    """
    if isinstance(embedding.embeddings, tfp.distributions.Distribution):
        z_mode = embedding.embeddings.mode().numpy()
    else:
        z_mode = embedding.embeddings.numpy()

    rank = z_mode.ndim
    if rank == 3:
        z_mode = z_mode[group_idx]

    y_max = np.max(z_mode)
    z_mode = z_mode[:, idx]

    # Handle masking.
    if embedding.mask_zero:
        z_mode = z_mode[1:]

    n_input_dim = z_mode.shape[0]

    # Scatter point estimate.
    xg = np.arange(n_input_dim)
    ax.scatter(xg, z_mode, c=c, marker='_', linewidth=1)

    if hasattr(embedding, 'posterior'):
        dist = embedding.posterior.embeddings.distribution

        # Middle 99% probability mass.
        p = .99
        v = (1 - p) / 2
        quant_lower = dist.quantile(v).numpy()[:, idx]
        quant_upper = dist.quantile(1-v).numpy()
        y_max = np.max(quant_upper)
        quant_upper = quant_upper[:, idx]

        # Middle 50% probability mass.
        p = .5
        v = (1 - p) / 2
        mid_lower = dist.quantile(v).numpy()[:, idx]
        mid_upper = dist.quantile(1-v).numpy()[:, idx]

        if embedding.posterior.mask_zero:
            quant_lower = quant_lower[1:]
            quant_upper = quant_upper[1:]
            mid_lower = mid_lower[1:]
            mid_upper = mid_upper[1:]

        for i_dim in range(n_input_dim):
            xg = np.array([i_dim, i_dim])
            yg = np.array(
                [quant_lower[i_dim], quant_upper[i_dim]]
            )
            ax.plot(xg, yg, c=c, linewidth=1)

            yg = np.array(
                [mid_lower[i_dim], mid_upper[i_dim]]
            )
            ax.plot(xg, yg, c=c, linewidth=3)

    ax.set_xlabel('Input Dimension')
    ax.set_xlim([-.5, n_input_dim-.5])

    ax.set_ylabel(r'$x$')
    ax.set_ylim([0, 1.05 * y_max])
    ax.set_yticks([0, 1.05 * y_max])
    ax.set_yticklabels(['0', '{0:.1f}'.format(1.05 * y_max)])


def embedding_input_dimension(fig, ax, embedding, idx, group_idx=0, c='b'):
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
    if rank == 3:
        z_mode = z_mode[group_idx]

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


def visualize_embedding_static(
        z, class_id=None, classes=None, fname=None):
    """Generate a static scatter plot of the supplied embedding points.

    Arguments:
        z: A real-valued two-dimensional array representing the embedding.
            shape = (n_stimuli, n_dim)
        class_id: (optional) An integer array contianing class IDs
            that indicate the class membership of each stimulus.
            shape = (n_stimuli, 1)
        classes: (optional) A dictionary mapping class IDs to strings.
        fname (optional): The pdf filename to save the figure,
            otherwise the figure is displayed. Can be either a path
            string or a pathlib Path object.

    """
    # Settings
    dot_size = 20
    cmap = matplotlib.cm.get_cmap('jet')

    [n_stimuli, n_dim] = z.shape

    n_class = 1
    if class_id is None:
        use_legend = False

        class_id = np.ones((n_stimuli))
        unique_class_list = np.unique(class_id)
        n_class = 1
        color_array = cmap((0, 1))
        color_array = color_array[np.newaxis, 0, :]

        class_legend = ['all']
    else:
        use_legend = True
        unique_class_list = np.unique(class_id)
        n_class = len(unique_class_list)
        norm = matplotlib.colors.Normalize(vmin=0., vmax=n_class)
        color_array = cmap(norm(range(n_class)))

        if classes is not None:
            class_legend = infer_legend(unique_class_list, classes)
        else:
            class_legend = unique_class_list

    if n_dim == 2:
        fig, ax = plt.subplots()
        # Plot each class separately in order to use legend.
        for i_class in range(n_class):
            locs = class_id == unique_class_list[i_class]
            ax.scatter(
                z[locs, 0], z[locs, 1], c=color_array[np.newaxis, i_class, :],
                s=dot_size, label=class_legend[i_class], edgecolors='none')

        if use_legend:
            ax.legend(bbox_to_anchor=(-.05, 1), loc=1, borderaxespad=0.)

        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

    if n_dim >= 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot each class separately in order to use legend.
        for i_class in range(n_class):
            locs = class_id == unique_class_list[i_class]
            ax.scatter(
                z[locs, 0], z[locs, 1], z[locs, 2],
                c=color_array[np.newaxis, i_class, :], s=dot_size,
                # edgecolors='none',
                label=class_legend[i_class]
            )

        if use_legend:
            ax.legend(bbox_to_anchor=(-.05, 1), loc=1, borderaxespad=0.)

        ax.set_xticks([], [])
        ax.set_yticks([], [])
        ax.set_zticks([], [])

    if fname is None:
        # plt.tight_layout()
        plt.show()
    else:
        # Note: The dpi must be supplied otherwise the aspect ratio will be
        # changed when savefig is called.
        plt.savefig(
            os.fspath(fname), format='pdf', bbox_inches="tight", dpi=300
        )


def visualize_embedding_images(
        ax, z, class_id=None, classes=None, filepaths=None, image_size=.2,
        dot_size=10, show_dot=False):
    """Generate a static scatter plot of the supplied embedding points.

    Arguments:
        ax: TODO
        z: A real-valued two-dimensional array representing the embedding.
            shape = (n_stimuli, n_dim)
        class_id: (optional) An integer array contianing class IDs
            that indicate the class membership of each stimulus.
            shape = (n_stimuli, 1)
        classes: (optional) A dictionary mapping class IDs to strings.
        filepaths: TODO should not be optional

    """
    # Settings
    cmap = matplotlib.cm.get_cmap('jet')

    [n_stimuli, n_dim] = z.shape

    n_class = 1
    if class_id is None:
        use_legend = False

        class_id = np.ones((n_stimuli))
        unique_class_list = np.unique(class_id)
        n_class = 1
        color_array = cmap((0, 1))
        color_array = color_array[np.newaxis, 0, :]

        class_legend = ['all']
    else:
        use_legend = True
        unique_class_list = np.unique(class_id)
        n_class = len(unique_class_list)
        norm = matplotlib.colors.Normalize(vmin=0., vmax=n_class)
        color_array = cmap(norm(range(n_class)))

        if classes is not None:
            class_legend = infer_legend(unique_class_list, classes)
        else:
            class_legend = unique_class_list

    if show_dot:
        alpha = 1.0
    else:
        use_legend = False
        alpha = 0.0

    # Plot each class separately in order to use legend.
    for i_class in range(n_class):
        locs = class_id == unique_class_list[i_class]
        ax.scatter(
            z[locs, 0], z[locs, 1], c=color_array[np.newaxis, i_class, :],
            s=dot_size, label=class_legend[i_class], edgecolors='none',
            alpha=alpha
        )

    # Plot images.
    for idx, i_file in enumerate(filepaths):
        img = Image.open(i_file)
        w, h = img.size
        ar = w/h
        dx = ar * image_size
        dy = image_size
        x = z[idx, 0]
        y = z[idx, 1]
        ax.imshow(img, extent=(x - dx/2, x + dx/2, y - dy/2, y + dy/2))

    if use_legend:
        lgnd = ax.legend(bbox_to_anchor=(-.05, 1), loc=1, borderaxespad=0.)
        # Hack to make dots a readable size in legend regardless of size
        # in plot.
        for i_handle in lgnd.legendHandles:
            i_handle._sizes = [30]

    z_max = 1.2 * np.max(np.abs(z))
    ax.set_xlim([-z_max, z_max])
    ax.set_ylim([-z_max, z_max])
    ax.set_aspect('equal')

    ax.set_xticks([])
    ax.set_yticks([])


# def visualize_embedding_movie(
#       Z3, class_id=None, classes=None, fname=None):
#     """
#     """
# TODO


def visualize_convergence(data, fname=None):
    """Visualize convergence analysis.

    Arguments:
        data: The output of calling psiz.utils.assess_convergence. A
            dictionary having the fields, `n_trial_array`, `val`, and
            `measure`.
        fname (optional): The pdf filename to save the figure,
            otherwise the figure is displayed. Can be either a path
            string or a pathlib Path object.

    """
    n_trial_array = data["n_trial_array"]
    val = data["val"]
    val_mean = np.mean(val, axis=0)
    val_std = np.std(val, axis=0)

    _, ax = plt.subplots()
    ax.plot(
        n_trial_array, val_mean,
        linestyle='-', marker="o"
    )
    ax.fill_between(
        n_trial_array, val_mean - val_std, val_mean + val_std,
        alpha=.5
    )
    ax.set_xlabel('Number of Trials')
    ax.set_ylabel('Convergence Measure ({0})'.format(data["measure"]))
    ax.set_ylim([0, 1])

    str_rho = "{0} = {1:.2f}".format(data['measure'], val_mean[-1])
    ax.text(
        n_trial_array[-1], val_mean[-1] + .03, str_rho,
        horizontalalignment='center', verticalalignment='center'
    )

    if fname is None:
        plt.show()
    else:
        # Note: The dpi must be supplied otherwise the aspect ratio will be
        # changed when savefig is called.
        plt.savefig(
            os.fspath(fname), format='pdf', bbox_inches="tight", dpi=300
        )


def infer_legend(unique_class_list, classes):
    """Infer text for legend entries."""
    n_class = len(unique_class_list)
    legend_entries = []
    for i_class in range(n_class):
        class_label = classes[unique_class_list[i_class]]
        legend_entries.append(class_label)
    return legend_entries


def dot_2d(
        ax, z, class_id=None, classes=None):
    """Generate a static scatter plot of the supplied embedding points.

    Arguments:
        z: A real-valued two-dimensional array representing the embedding.
            shape = (n_stimuli, n_dim)
        class_id: (optional) An integer array contianing class IDs
            that indicate the class membership of each stimulus.
            shape = (n_stimuli, 1)
        classes: (optional) A dictionary mapping class IDs to strings.
        fname (optional): The pdf filename to save the figure,
            otherwise the figure is displayed. Can be either a path
            string or a pathlib Path object.

    """
    # Settings
    dot_size = 20
    cmap = matplotlib.cm.get_cmap('jet')

    [n_stimuli, n_dim] = z.shape

    n_class = 1
    if class_id is None:
        use_legend = False

        class_id = np.ones((n_stimuli))
        unique_class_list = np.unique(class_id)
        n_class = 1
        color_array = cmap((0, 1))
        color_array = color_array[np.newaxis, 0, :]

        class_legend = ['all']
    else:
        use_legend = True
        unique_class_list = np.unique(class_id)
        n_class = len(unique_class_list)
        norm = matplotlib.colors.Normalize(vmin=0., vmax=n_class)
        color_array = cmap(norm(range(n_class)))

        if classes is not None:
            class_legend = infer_legend(unique_class_list, classes)
        else:
            class_legend = unique_class_list

    # Plot each class separately in order to use legend.
    for i_class in range(n_class):
        locs = class_id == unique_class_list[i_class]
        ax.scatter(
            z[locs, 0], z[locs, 1], c=color_array[np.newaxis, i_class, :],
            s=dot_size, label=class_legend[i_class], edgecolors='none')

    if use_legend:
        ax.legend(bbox_to_anchor=(-.05, 1), loc=1, borderaxespad=0.)

    # ax.set_aspect('equal')
    # ax.set_xticks([])
    # ax.set_yticks([])


def image_2d(
        ax, z, class_id=None, classes=None, filepaths=None, image_size=.2,
        dot_size=10, show_dot=False):
    """Generate a static scatter plot of the supplied embedding points.

    Arguments:
        ax: TODO
        z: A real-valued two-dimensional array representing the embedding.
            shape = (n_stimuli, n_dim)
        class_id: (optional) An integer array contianing class IDs
            that indicate the class membership of each stimulus.
            shape = (n_stimuli, 1)
        classes: (optional) A dictionary mapping class IDs to strings.
        filepaths: TODO should not be optional

    """
    # Settings
    cmap = matplotlib.cm.get_cmap('jet')

    [n_stimuli, n_dim] = z.shape

    n_class = 1
    if class_id is None:
        use_legend = False

        class_id = np.ones((n_stimuli))
        unique_class_list = np.unique(class_id)
        n_class = 1
        color_array = cmap((0, 1))
        color_array = color_array[np.newaxis, 0, :]

        class_legend = ['all']
    else:
        use_legend = True
        unique_class_list = np.unique(class_id)
        n_class = len(unique_class_list)
        norm = matplotlib.colors.Normalize(vmin=0., vmax=n_class)
        color_array = cmap(norm(range(n_class)))

        if classes is not None:
            class_legend = infer_legend(unique_class_list, classes)
        else:
            class_legend = unique_class_list

    if show_dot:
        alpha = 1.0
    else:
        use_legend = False
        alpha = 0.0

    # Plot each class separately in order to use legend.
    for i_class in range(n_class):
        locs = class_id == unique_class_list[i_class]
        ax.scatter(
            z[locs, 0], z[locs, 1], c=color_array[np.newaxis, i_class, :],
            s=dot_size, label=class_legend[i_class], edgecolors='none',
            alpha=alpha
        )

    # Plot images.
    for idx, i_file in enumerate(filepaths):
        img = Image.open(i_file)
        w, h = img.size
        ar = w/h
        dx = ar * image_size
        dy = image_size
        x = z[idx, 0]
        y = z[idx, 1]
        ax.imshow(img, extent=(x - dx/2, x + dx/2, y - dy/2, y + dy/2))

    if use_legend:
        lgnd = ax.legend(bbox_to_anchor=(-.05, 1), loc=1, borderaxespad=0.)
        # Hack to make dots a readable size in legend regardless of size
        # in plot.
        for i_handle in lgnd.legendHandles:
            i_handle._sizes = [30]


def bvn_ellipse(loc, cov, r=1.96, **kwargs):
    """Return ellipse object representing bivariate normal.

    This code was inspired by a solution posted on Stack Overflow:
    https://stackoverflow.com/a/25022642/1860294

    Arguments:
        loc: A 1D array denoting the mean.
        cov: A 2D array denoting the covariance matrix.
        r (optional): The radius (specified in standard deviations) at
            which to draw the ellipse. The default value corresponds to
            an ellipse indicating a region containing 95% of the
            probability mass.
        kwargs (optional): Additional key-word arguments to pass to
            `matplotlib.patches.Ellipse` constructor.

    Returns:
        ellipse: A `matplotlib.patches.Ellipse` artist object.

    """
    def eig_sorted(cov):
        """Sort eigenvalues."""
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    vals, vecs = eig_sorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2 * r * np.sqrt(vals)
    ellipse = matplotlib.patches.Ellipse(
        xy=(loc[0], loc[1]), width=w, height=h, angle=theta, **kwargs
    )
    return ellipse
