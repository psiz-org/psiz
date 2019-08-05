# -*- coding: utf-8 -*-
# Copyright 2019 The PsiZ Authors. All Rights Reserved.
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
# ==============================================================================

"""Module for visualizing embeddings.

Todo:
    classes -> class_dict
"""

import os

from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


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
        dot_size=10, show_dot=True):
    """Generate a static scatter plot of the supplied embedding points.

    Arguments:
        ax:
        z: A real-valued two-dimensional array representing the embedding.
            shape = (n_stimuli, n_dim)
        class_id: (optional) An integer array contianing class IDs
            that indicate the class membership of each stimulus.
            shape = (n_stimuli, 1)
        classes: (optional) A dictionary mapping class IDs to strings.
        filename (optional): The pdf filename to save the figure,
            otherwise the figure is displayed.

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
            class_legend = visualize.infer_legend(unique_class_list, classes)
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
