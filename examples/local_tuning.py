# -*- coding: utf-8 -*-
# Copyright 2018 The PsiZ Authors. All Rights Reserved.
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

"""Example of local tuning.

An example that illustrates how dimension-wide tuning with additional
dimensions can accomplish a similar effect as local tunings with the
same number of dimensions.

Todo:
    - Clean up example.

Model Comparison (R^2)
================================
 Novice            |   0.99
 Expert            |   0.98
 Novice (joint 2D) |   0.72
 Expert (joint 2D) |   0.85
 Novice (joint 3D) |   0.99
 Expert (joint 3D) |   0.92

"""

import copy
import numpy as np
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf

from psiz.trials import stack
from psiz.models import Exponential, HeavyTailed, StudentsT
from psiz.simulate import Agent
from psiz.generator import RandomGenerator
from psiz import utils


def main():
    """Example of local tuning."""
    # Ground truth models.
    # model_nov_tru = novice_ground_truth()
    # model_exp_tru = expert_ground_truth()
    np.random.seed(123)
    model_nov_tru, model_exp_tru, l = ground_truth()
    z_nov_tru = model_nov_tru.z['value']
    z_exp_tru = model_exp_tru.z['value']
    # plot_figure(z_nov_tru, l, z_exp_tru, l)

    # z_nov_inf = z_nov_tru
    # z_exp_inf = z_exp_tru
    # z_jinf = z_nov_tru
    # z_nov_jinf = z_nov_tru
    # z_exp_jinf = z_exp_tru
    # plot_figure(
    #     z_nov_tru, z_exp_tru,
    #     z_nov_inf, z_exp_inf,
    #     z_jinf, z_nov_jinf, z_exp_jinf)

    # Generate random docket of trials.
    n_stimuli = model_nov_tru.n_stimuli
    n_dim = model_nov_tru.n_dim
    n_trial = 10000
    n_reference = 8
    n_selected = 2
    generator = RandomGenerator(n_stimuli)
    docket = generator.generate(n_trial, n_reference, n_selected)

    # Simulate observations.
    agent_novice = Agent(model_nov_tru)
    agent_expert = Agent(model_exp_tru)
    obs_novice = agent_novice.simulate(docket)
    obs_novice.set_group_id(0)
    obs_expert_1 = agent_expert.simulate(docket)
    obs_expert_1.set_group_id(0)
    obs_expert_2 = agent_expert.simulate(docket)
    obs_expert_2.set_group_id(1)

    # Infer separate models.
    model_nov_inf = Exponential(n_stimuli, n_dim)
    model_nov_inf.freeze({'theta': {'beta': 10, 'rho': 2, 'tau': 1}})
    model_nov_inf.fit(obs_novice, n_restart=3, verbose=3)
    z_nov_inf = model_nov_inf.z['value']
    (z_nov_inf, _) = utils.procrustean_solution(z_nov_tru, z_nov_inf, n_restart=200)
    r_squared_nov = utils.compare_models(model_nov_tru, model_nov_inf)
    
    model_exp_inf = Exponential(n_stimuli, n_dim)
    model_exp_inf.freeze({'theta': {'beta': 10, 'rho': 2, 'tau': 1}})
    model_exp_inf.fit(obs_expert_1, n_restart=3, verbose=3)
    z_exp_inf = model_exp_inf.z['value']
    (z_exp_inf, _) = utils.procrustean_solution(z_exp_tru, z_exp_inf, n_restart=200)
    r_squared_exp = utils.compare_models(model_exp_tru, model_exp_inf)

    # Infer joint 2D model.
    obs_all = stack((obs_novice, obs_expert_2))
    n_group = 2
    model_inferred_2d = Exponential(n_stimuli, n_dim, n_group)
    model_inferred_2d.freeze({'theta': {'beta': 10, 'rho': 2, 'tau': 1}})
    model_inferred_2d.fit(obs_all, n_restart=3, verbose=3)
    r_squared_nov_j2d = utils.compare_models(
        model_nov_tru, model_inferred_2d, group_id_b=0)
    r_squared_exp_j2d = utils.compare_models(
        model_exp_tru, model_inferred_2d, group_id_b=1)

    # Infer joint 3D model.
    model_inferred_3d = Exponential(n_stimuli, 3, n_group)
    model_inferred_3d.freeze({'theta': {'beta': 10, 'rho': 2, 'tau': 1}})
    model_inferred_3d.fit(obs_all, n_restart=3, verbose=3)
    r_squared_nov_j3d = utils.compare_models(
        model_nov_tru, model_inferred_3d, group_id_b=0)
    r_squared_exp_j3d = utils.compare_models(
        model_exp_tru, model_inferred_3d, group_id_b=1)

    print('\nModel Comparison (R^2)')
    print('================================')
    print(' Novice            | {0: >6.2f}'.format(r_squared_nov))
    print(' Expert            | {0: >6.2f}'.format(r_squared_exp))
    print(' Novice (joint 2D) | {0: >6.2f}'.format(r_squared_nov_j2d))
    print(' Expert (joint 2D) | {0: >6.2f}'.format(r_squared_exp_j2d))
    print(' Novice (joint 3D) | {0: >6.2f}'.format(r_squared_nov_j3d))
    print(' Expert (joint 3D) | {0: >6.2f}'.format(r_squared_exp_j3d))
    print('\n')

    # print('Tunings:')
    # print(model_inferred_2d.phi['phi_1']['value'])
    # (z_jinf, _) = utils.procrustean_solution(
    #     z_nov_tru,
    #     model_inferred_2d.z['value'],
    #     n_restart=200)
    # (z_nov_jinf, _) = utils.procrustean_solution(
    #     z_nov_tru,
    #     model_inferred_2d.z['value'] * model_inferred_2d.phi['phi_1']['value'][0, :],
    #     n_restart=200)
    # (z_exp_jinf, _) = utils.procrustean_solution(
    #     z_exp_tru,
    #     model_inferred_2d.z['value'] * model_inferred_2d.phi['phi_1']['value'][1, :],
    #     n_restart=200)

    # plot_figure(
    #     z_nov_tru, z_exp_tru,
    #     z_nov_inf, z_exp_inf,
    #     z_jinf, z_nov_jinf, z_exp_jinf)


def ground_truth():
    n_dim = 2
    n_stimuli = 50
    n_stimuli_half = int(n_stimuli / 2)
    # mean = np.zeros((n_dim))
    # cov = .1 * np.identity(n_dim)
    # z = np.random.multivariate_normal(mean, cov, (n_stimuli))
    z_0 = np.random.rand(n_stimuli_half, n_dim)
    z_0 = z_0 + np.array((-.85, -.5))
    z_0 = np.array((1., 2.)) * z_0
    l_0 = np.zeros(n_stimuli_half)

    z_1 = np.random.rand(n_stimuli_half, n_dim)
    z_1 = z_1 + np.array((-.15, -.5))
    z_1 = np.array((1., 2.)) * z_1
    l_1 = np.ones(n_stimuli_half)

    z_nov = np.concatenate((z_0, z_1), axis=0)
    # z_nov = z_nov * [.9, .9]
    l = np.concatenate((l_0, l_1), axis=0)

    m = .2  # The orignal border of stretching.
    d = .35  # How much stretching displaces outside points.
    c = .25  # Contraction of outside points

    z_exp = copy.copy(z_nov)
    # Left side outside.
    locs = np.less(z_exp[:, 0], -m)
    # z_exp[locs, 0] = z_exp[locs, 0] - d
    z_exp[locs, 0] = (c * (z_exp[locs, 0] + m)) - (m + d)

    # Right-side outside.
    locs = np.greater(z_exp[:, 0], m)
    # z_exp[locs, 0] = z_exp[locs, 0] + d
    z_exp[locs, 0] = (c * (z_exp[locs, 0] - m)) + (m + d)

    # Left side inside.
    locs = np.logical_and(
        np.less(z_exp[:, 0], 0.),
        np.greater(z_exp[:, 0], -m)
    )
    z_exp[locs, 0] = z_exp[locs, 0] + (z_exp[locs, 0] / m * d)

    # Right-side inside.
    locs = np.logical_and(
        np.greater(z_exp[:, 0], 0.),
        np.less(z_exp[:, 0], m)
    )
    z_exp[locs, 0] = z_exp[locs, 0] + (z_exp[locs, 0] / m * d)

    model_nov = Exponential(
        n_stimuli, n_dim)
    model_nov.freeze({
        'z': z_nov,
        'theta': {'beta': 10, 'rho': 2, 'tau': 1}
    })

    model_exp = Exponential(
        n_stimuli, n_dim)
    model_exp.freeze({
        'z': z_exp,
        'theta': {'beta': 10, 'rho': 2, 'tau': 1}
    })
    return model_nov, model_exp, l


def novice_ground_truth():
    """Novice ground truth model."""
    n_dim = 2
    x, y = np.meshgrid(
        np.array([-.3, -.2, -.1, 0., .1, .2, .3]),
        np.array([-.2, -.1, 0., .1, .2])
    )
    x = np.expand_dims(x.flatten(), axis=1)
    y = np.expand_dims(y.flatten(), axis=1)
    z = np.hstack((x, y))
    n_stimuli = z.shape[0]
    model = Exponential(
        n_stimuli, n_dim)
    model.freeze({
        'z': z,
        'theta': {'beta': 10, 'rho': 2, 'tau': 1}
    })
    return model


def expert_ground_truth():
    """Expert ground truth model."""
    n_dim = 2
    x, y = np.meshgrid(
        np.array([-.5, -.4, -.25, .0, .25, .4, .5]),
        np.array([-.2, -.1, 0., .1, .2])
    )
    x = np.expand_dims(x.flatten(), axis=1)
    y = np.expand_dims(y.flatten(), axis=1)
    z = np.hstack((x, y))
    n_stimuli = z.shape[0]
    model = Exponential(
        n_stimuli, n_dim)
    model.freeze({
        'z': z,
        'theta': {'beta': 10, 'rho': 2, 'tau': 1}
    })
    return model


def plot_figure(z_nov_tru, l_nov, z_exp_tru, l_exp):
    """Plot figure."""
    # Figure settings.
    # lim_x = np.array((-.55, .55))
    # lim_y = np.array((-.25, .25))
    lim_x = np.array((-1.1, 1.1))
    lim_y = np.array((-1.05, 1.05))
    n_stimuli = z_nov_tru.shape[0]
    cmap = matplotlib.cm.get_cmap('jet')
    norm = matplotlib.colors.Normalize(vmin=0., vmax=len(np.unique(l_nov)))
    # color_array = cmap(norm(range(n_stimuli)))
    color_array = cmap(norm(l_nov))

    fig = plt.figure(figsize=(5.5, 2), dpi=200)

    ax2 = fig.add_subplot(121)
    ax2.scatter(
        z_nov_tru[:, 0], z_nov_tru[:, 1], s=4, c=color_array, marker='o')
    ax2.set_title('Novice')
    ax2.set_aspect('equal')
    ax2.set_xlim(lim_x[0], lim_x[1])
    ax2.set_xticks([])
    ax2.set_ylim(lim_y[0], lim_y[1])
    ax2.set_yticks([])

    ax3 = fig.add_subplot(122)
    ax3.scatter(
        z_exp_tru[:, 0], z_exp_tru[:, 1], s=4, c=color_array, marker='o')
    ax3.set_title('Expert')
    ax3.set_aspect('equal')
    ax3.set_xlim(lim_x[0], lim_x[1])
    ax3.set_xticks([])
    ax3.set_ylim(lim_y[0], lim_y[1])
    ax3.set_yticks([])

    plt.show()


# def plot_figure(
#         z_nov_tru, z_exp_tru, z_nov_inf, z_exp_inf, z_jinf, z_nov_jinf,
#         z_exp_jinf):
#     """Plot figure."""
#     # Figure settings.
#     # lim_x = np.array((-.55, .55))
#     # lim_y = np.array((-.25, .25))
#     lim_x = np.array((-1.05, 1.05))
#     lim_y = np.array((-1.05, 1.05))
#     n_stimuli = z_nov_tru.shape[0]
#     cmap = matplotlib.cm.get_cmap('jet')
#     norm = matplotlib.colors.Normalize(vmin=0., vmax=n_stimuli)
#     color_array = cmap(norm(range(n_stimuli)))

#     fig = plt.figure(figsize=(5.5, 2), dpi=200)

#     ax2 = fig.add_subplot(332)
#     ax2.scatter(
#         z_nov_tru[:, 0], z_nov_tru[:, 1], s=4, c=color_array, marker='o')
#     ax2.set_title('Novice')
#     ax2.set_aspect('equal')
#     ax2.set_xlim(lim_x[0], lim_x[1])
#     ax2.set_xticks([])
#     ax2.set_ylim(lim_y[0], lim_y[1])
#     ax2.set_yticks([])

#     ax3 = fig.add_subplot(333)
#     ax3.scatter(
#         z_exp_tru[:, 0], z_exp_tru[:, 1], s=4, c=color_array, marker='o')
#     ax3.set_title('Expert')
#     ax3.set_aspect('equal')
#     ax3.set_xlim(lim_x[0], lim_x[1])
#     ax3.set_xticks([])
#     ax3.set_ylim(lim_y[0], lim_y[1])
#     ax3.set_yticks([])

#     ax5 = fig.add_subplot(335)
#     ax5.scatter(
#         z_nov_inf[:, 0], z_nov_inf[:, 1], s=4, c=color_array, marker='o')
#     ax5.set_aspect('equal')
#     ax5.set_xlim(lim_x[0], lim_x[1])
#     ax5.set_xticks([])
#     ax5.set_ylim(lim_y[0], lim_y[1])
#     ax5.set_yticks([])

#     ax6 = fig.add_subplot(336)
#     ax6.scatter(
#         z_exp_inf[:, 0], z_exp_inf[:, 1], s=4, c=color_array, marker='o')
#     ax6.set_aspect('equal')
#     ax6.set_xlim(lim_x[0], lim_x[1])
#     ax6.set_xticks([])
#     ax6.set_ylim(lim_y[0], lim_y[1])
#     ax6.set_yticks([])

#     ax7 = fig.add_subplot(337)
#     ax7.scatter(
#         z_jinf[:, 0], z_jinf[:, 1], s=4, c=color_array, marker='o')
#     ax7.set_aspect('equal')
#     ax7.set_xlim(lim_x[0], lim_x[1])
#     ax7.set_xticks([])
#     ax7.set_ylim(lim_y[0], lim_y[1])
#     ax7.set_yticks([])

#     ax8 = fig.add_subplot(338)
#     ax8.scatter(
#         z_nov_jinf[:, 0], z_nov_jinf[:, 1], s=4, c=color_array, marker='o')
#     ax8.set_aspect('equal')
#     ax8.set_xlim(lim_x[0], lim_x[1])
#     ax8.set_xticks([])
#     ax8.set_ylim(lim_y[0], lim_y[1])
#     ax8.set_yticks([])

#     ax9 = fig.add_subplot(339)
#     ax9.scatter(
#         z_exp_jinf[:, 0], z_exp_jinf[:, 1], s=4, c=color_array, marker='o')
#     ax9.set_aspect('equal')
#     ax9.set_xlim(lim_x[0], lim_x[1])
#     ax9.set_xticks([])
#     ax9.set_ylim(lim_y[0], lim_y[1])
#     ax9.set_yticks([])

#     plt.show()


if __name__ == "__main__":
    main()
