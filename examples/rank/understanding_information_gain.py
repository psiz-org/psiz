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
"""Example that explores expected information gain.

Results are saved in the directory specified by `fp_project`. By
default, a `psiz_examples` directory is created in your home directory.

"""

import itertools
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # noqa
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import psiz
from psiz.tf.information_theory import ig_model_categorical


# Uncomment the following line to force eager execution.
# tf.config.run_functions_eagerly(True)

# Uncomment and edit the following to control GPU visibility.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class BehaviorModel(psiz.keras.StochasticModel):
    """A behavior model.

    No Gates.

    """

    def __init__(self, behavior=None, **kwargs):
        """Initialize."""
        super(BehaviorModel, self).__init__(**kwargs)
        self.behavior = behavior

    def call(self, inputs):
        """Call."""
        return self.behavior(inputs)


def main():
    """Run script."""
    # Settings.
    fp_project = Path.home() / Path(
        'psiz_examples', 'rank', 'information_gain'
    )
    n_reference = 2
    n_select = 1
    n_col = 7
    batch_size = 512
    n_sample = 10

    # Directory preparation.
    fp_project.mkdir(parents=True, exist_ok=True)

    # Plot settings.
    small_size = 6
    medium_size = 8
    large_size = 10
    plt.rc('font', size=small_size)
    plt.rc('axes', titlesize=medium_size)
    plt.rc('axes', labelsize=small_size)
    plt.rc('xtick', labelsize=small_size)
    plt.rc('ytick', labelsize=small_size)
    plt.rc('legend', fontsize=small_size)
    plt.rc('figure', titlesize=large_size)

    n_case = 5
    case_list = []
    for i_case in range(n_case):
        model = build_model(n_reference, n_select, i_case)
        model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=.001),
        )

        # Create exhaustive list of candidate trials.
        n_stimuli = model.behavior.percept.input_dim
        eligable_list = np.arange(1, n_stimuli, dtype=np.int32)
        stimulus_set = candidate_list(eligable_list, n_reference)
        content = psiz.data.Rank(stimulus_set, n_select=n_select)
        tfds_all = psiz.data.Dataset([content]).export(
            export_format='tfds'
        ).batch(batch_size, drop_remainder=False)

        # Compute expected information gain for candidate trials in mini
        # batches.
        expected_ig = []
        for data in tfds_all:
            expected_ig.append(ig_model_categorical([model], data, n_sample))
        expected_ig = tf.concat(expected_ig, 0).numpy()

        # Select data to represent case in visualization.
        case_data = package_case_data(model, stimulus_set, expected_ig)
        case_list.append(case_data)

    # Visualize.
    draw_figure(case_list, n_col)
    fname = fp_project / Path('visual.pdf')
    plt.savefig(
        os.fspath(fname), format='pdf', bbox_inches="tight", dpi=300
    )


def draw_figure(case_list, n_col):
    """Draw figure."""
    fig = plt.figure(figsize=(6.5, 5))
    n_case = len(case_list)
    gs = fig.add_gridspec(n_case, n_col)
    for idx in range(n_case):
        draw_scenario(fig, gs, idx, case_list[idx])

    plt.suptitle('Candidate Trials', x=.58)
    gs.tight_layout(fig)


def draw_scenario(fig, gs, row, case_data):
    """Draw sceanrio."""
    model = case_data['model']
    stimulus_set = case_data['stimulus_set']
    expected_ig = case_data['expected_ig']
    standardized_ig = case_data['standardized_ig']

    # Settings.
    lw = .5
    fontdict = {
        'fontsize': 4,
        'verticalalignment': 'baseline',
        'horizontalalignment': 'center'
    }

    n_trial = stimulus_set.shape[0]

    # Define one color per class for plots.
    n_stimuli = model.behavior.percept.input_dim
    cmap = matplotlib.cm.get_cmap('jet')
    norm = matplotlib.colors.Normalize(vmin=0., vmax=n_stimuli)
    color_arr = cmap(norm(range(n_stimuli)))

    ax = fig.add_subplot(gs[row, 0])
    # Draw model state.
    dist = model.behavior.percept.embeddings
    loc, cov = unpack_mvn(dist)
    if model.behavior.percept.mask_zero:
        # Drop placeholder stimulus.
        plot_bivariate_normal(
            ax, loc[1:], cov[1:], c=color_arr, r=1.96, lw=lw
        )
    else:
        plot_bivariate_normal(
            ax, loc, cov, c=color_arr, r=1.96, lw=lw
        )

    ax.set_title('Posterior')
    ax.set_aspect('equal')
    ax.set_xlim(.05, .45)
    ax.set_xticks([])
    ax.set_ylim(.05, .45)
    ax.set_yticks([])

    for i_subplot in range(n_trial):
        ax = fig.add_subplot(gs[row, i_subplot + 1])
        candidate_subplot(
            fig, ax, loc, stimulus_set[i_subplot], expected_ig[i_subplot],
            standardized_ig[i_subplot], color_arr, fontdict
        )


def candidate_subplot(
        fig, ax, z, stimulus_set, expected_ig, v, color_array, fontdict):
    """Plot subplot of candidate trial."""
    ax.scatter(
        z[stimulus_set[0], 0],
        z[stimulus_set[0], 1],
        s=15, c=color_array[np.newaxis, stimulus_set[0]], marker=r'$q$')
    ax.scatter(
        z[stimulus_set[1:], 0],
        z[stimulus_set[1:], 1],
        s=15, c=color_array[stimulus_set[1:]], marker=r'$r$')
    ax.set_aspect('equal')
    ax.set_xlim(.05, .45)
    ax.set_xticks([])
    ax.set_ylim(.05, .45)
    ax.set_yticks([])
    # plt.text(
    #     .25, .47, "{0:.4f}".format(expected_ig),
    #     fontdict=fontdict)
    rect_back = matplotlib.patches.Rectangle(
        (.45, .05), .06, .4, clip_on=False, color=[.9, .9, .9])
    ax.add_patch(rect_back)
    rect_val = matplotlib.patches.Rectangle(
        (.45, .05), .06, v * .4, clip_on=False, color=[.3, .3, .3])
    ax.add_patch(rect_val)


def build_model(n_reference, n_select, case):
    """Return a ground truth embedding.

    Args:
        case: Integer indicating the model case.

    Case 0:
    Case 1:
    Case 2:
    Case 3:
    Case 4:

    """
    # Settings.
    n_sample = 1000

    if case < 4:
        # Create embedding points arranged on a grid.
        x, y = np.meshgrid([.1, .2, .3, .4], [.1, .2, .3, .4])
        x = np.expand_dims(x.flatten(), axis=1)
        y = np.expand_dims(y.flatten(), axis=1)
        z_grid = np.hstack((x, y))
        (n_stimuli, n_dim) = z_grid.shape
        # Add placeholder.
        z_grid = np.vstack((np.ones([1, 2]), z_grid))
    else:
        # Create embedding points arranged in a circle.
        origin = np.array([[0.25, 0.25]])
        theta = np.linspace(0, 2 * np.pi, 9)
        theta = theta[0:-1]

        r = .15
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        x = np.expand_dims(x, axis=1)
        y = np.expand_dims(y, axis=1)
        z = np.hstack((x, y)) + origin
        z_circle = np.concatenate((origin, z), axis=0)
        (n_stimuli, n_dim) = z_circle.shape
        # Add placeholder.
        z_circle = np.vstack((np.ones([1, 2]), z_circle))

    prior_scale = .17
    percept = psiz.keras.layers.EmbeddingNormalDiag(
        n_stimuli + 1, n_dim, mask_zero=True,
        scale_initializer=tf.keras.initializers.Constant(
            tfp.math.softplus_inverse(prior_scale).numpy()
        )
    )
    kernel = psiz.keras.layers.DistanceBased(
        distance=psiz.keras.layers.Minkowski(
            rho_initializer=tf.keras.initializers.Constant(2.),
            w_initializer=tf.keras.initializers.Constant(1.),
            trainable=False
        ),
        similarity=psiz.keras.layers.ExponentialSimilarity(
            trainable=False,
            beta_initializer=tf.keras.initializers.Constant(10.),
            tau_initializer=tf.keras.initializers.Constant(1.),
            gamma_initializer=tf.keras.initializers.Constant(0.),
        )
    )
    rank = psiz.keras.layers.RankSimilarity(
        n_reference=n_reference,
        n_select=n_select,
        percept=percept,
        kernel=kernel
    )
    model = BehaviorModel(behavior=rank, n_sample=n_sample)
    model.behavior.percept.build([None, None, None])

    if case == 0:
        # One stimulus with relatively high uncertainty.
        loc = z_grid
        scale = .01 * np.ones([n_stimuli + 1, n_dim], dtype=np.float32)
        scale[5, :] = .05
    elif case == 1:
        # One stimulus with relatively high uncertainty.
        loc = z_grid
        scale = .01 * np.ones([n_stimuli + 1, n_dim], dtype=np.float32)
        scale[10, :] = .05
    elif case == 2:
        loc = z_grid
        scale = .01 * np.ones([n_stimuli + 1, n_dim], dtype=np.float32)
        scale[5, 0] = .05
        scale[5, 1] = .01
    elif case == 3:
        loc = z_grid
        scale = .01 * np.ones([n_stimuli + 1, n_dim], dtype=np.float32)
        scale[6, :] = .03
        scale[8, :] = .03
    elif case == 4:
        loc = z_circle
        scale = .01 * np.ones([n_stimuli + 1, n_dim], dtype=np.float32)
        scale[4, :] = .03

    # Assign scenario variables.
    model.behavior.percept.loc.assign(loc)
    model.behavior.percept.untransformed_scale.assign(
        tfp.math.softplus_inverse(scale)
    )
    return model


def package_case_data(model, stimulus_set, expected_ig):
    """Package data for case visualization."""
    n_candidate = stimulus_set.shape[0]

    # Sort by expected information gain.
    sorted_indices = np.argsort(-expected_ig)
    expected_ig = expected_ig[sorted_indices]
    stimulus_set = stimulus_set[sorted_indices]
    standardized_ig = expected_ig - np.min(expected_ig)
    standardized_ig = standardized_ig / np.max(standardized_ig)

    # Select the N best trials and a M trials evenly spaced from best to
    # worst.
    n_best = 3
    n_total = 6
    intermediate_trial_idx = np.linspace(
        n_best, n_candidate - 1, n_total - n_best, dtype=np.int32
    )
    trial_idx = np.concatenate(
        (np.arange(n_best), intermediate_trial_idx), axis=0
    )

    # Filter trials.
    stimulus_set_viz = stimulus_set[trial_idx]
    expected_ig_viz = expected_ig[trial_idx]
    standardized_ig_viz = standardized_ig[trial_idx]

    case_data = {
        'model': model,
        'stimulus_set': stimulus_set_viz,
        'expected_ig': expected_ig_viz,
        'standardized_ig': standardized_ig_viz
    }
    return case_data


def candidate_list(eligable_list, n_reference):
    """Determine all possible trials."""
    stimulus_set = np.empty([0, n_reference + 1], dtype=np.int32)
    for i_stim in eligable_list:
        locs = np.not_equal(eligable_list, i_stim)
        sub_list = itertools.combinations(eligable_list[locs], n_reference)
        for item in sub_list:
            item = np.hstack((i_stim * np.ones(1), item))
            stimulus_set = np.vstack((stimulus_set, item))
    stimulus_set = stimulus_set.astype(dtype=np.int32)
    return stimulus_set


def unpack_mvn(dist):
    """Unpack multivariate normal distribution."""
    def diag_to_full_cov(v):
        """Convert diagonal variance to full covariance matrix.

        Assumes `v` represents diagonal variance elements only.
        """
        n_stimuli = v.shape[0]
        n_dim = v.shape[1]
        cov = np.zeros([n_stimuli, n_dim, n_dim])
        for i_stimulus in range(n_stimuli):
            cov[i_stimulus] = np.eye(n_dim) * v[i_stimulus]
        return cov

    loc = dist.mean().numpy()
    v = dist.variance().numpy()

    # Convert to full covariance matrix.
    cov = diag_to_full_cov(v)

    return loc, cov


def plot_bivariate_normal(ax, loc, cov, c=None, r=2.576, **kwargs):
    """Plot set of bivariate normals.

    If covariances are supplied, ellipses are drawn to indicate regions
    of highest probability mass.

    Args:
        ax: A 'matplotlib' axes object.
        loc: Array denoting the means of bivariate normal
            distributions.
        cov: Array denoting the covariance matrices of
            bivariate normal distributions.
        c (optional): color array
        r (optional): The radius (specified in standard deviations) at
            which to draw the ellipse. The default value (2.576)
            corresponds to an ellipse indicating a region containing
            99% of the probability mass. Another common value is
            1.960, which indicates 95%.
        kwargs (optional): Additional key-word arguments that will be
            passed to a `matplotlib.patches.Ellipse` constructor.

    """
    n_stimuli = loc.shape[0]

    # Draw ellipsoids for each stimulus.
    for i_stimulus in range(n_stimuli):
        if c is not None:
            edgecolor = c[i_stimulus]
        ellipse = psiz.mplot.bvn_ellipse(
            loc[i_stimulus], cov[i_stimulus], r=r, fill=False,
            edgecolor=edgecolor, **kwargs
        )
        ax.add_artist(ellipse)


if __name__ == "__main__":
    main()
