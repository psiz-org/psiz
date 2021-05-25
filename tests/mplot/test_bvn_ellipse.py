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
import matplotlib.pyplot as plt
import numpy as np

import psiz.mplot


@pytest.mark.parametrize("ax", [False, True])
def test_hdi_bvn(ax):
    """Basic test of `hdi_bvn` plotter.

    NOTE: Does not test if HDI is computed correctly.

    """

    # Create data for two normal distributions.
    loc = np.array([
        [0, 0],
        [0, 0.1]
    ])
    cov = np.array([
        [[1, 0], [0, 1]],
        [[1, 0], [0, 2]],
    ])
    edgecolor = np.array([
        [.9, .1, .1],
        [.1, .1, .9]
    ])
    lw = 1
    alpha = .5
    fill = False

    num_figures_before = plt.gcf().number

    fig = plt.figure(figsize=(6.5, 4), dpi=200)
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0, 0])
    if ax:
        psiz.mplot.hdi_bvn(
            loc, cov, ax, p=.95, edgecolor=edgecolor, lw=lw,
            alpha=alpha, fill=fill
        )
    else:
        psiz.mplot.hdi_bvn(
            loc, cov, p=.95, edgecolor=edgecolor, lw=lw,
            alpha=alpha, fill=fill
        )

    num_figures_after = plt.gcf().number
    assert num_figures_before < num_figures_after

    assert len(ax.artists) == 2
    np.testing.assert_array_equal(ax.artists[0].center, loc[0])
    np.testing.assert_array_equal(ax.artists[1].center, loc[1])
