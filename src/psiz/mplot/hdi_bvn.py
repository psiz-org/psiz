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

import matplotlib.pyplot as plt
import scipy.stats as st

import psiz


def hdi_bvn(loc, cov, ax=None, p=.99, **kwargs):
    """Plot HDI of bivariate normal distributions.

    Ellipses are drawn to indicate the higest density interval.

    Arguments:
        loc: Array denoting the means of bivariate normal
            distributions.
            shape=(n_distr, 2)
        cov: Array denoting the covariance matrices of
            bivariate normal distributions.
            shape=(n_distr, 2, 2)
        ax (optional): A 'matplotlib' `AxesSubplot` object.
        p (optional): The amount of probability that the HDI should
            indicate. This must be a float ]0, 1[.
        kwargs (optional): Additional key-word arguments that will be
            passed to a `matplotlib.patches.Ellipse` constructor. NOTE:
            Any values that have a `__len__` property with be inspected
            to see if the length matches the number of distributions.
            If so, it will be assumed that these are values that should
            be applied on a per-distribution bases. For example,
            if edgecolor has shape=(n_distr, 3), it will be assumed
            that the `edgecolor[i_distr]` should be applied to the ith
            distribution.

    """
    if ax is None:
        ax = plt.gca()

    # Determine number of distributions.
    n_distr = loc.shape[0]

    # Convert proability to standard deviations (z-score) using percent point
    # function of normal distribution.
    outside_prob = 1 - p
    r = st.norm.ppf(1 - outside_prob / 2)

    # Intercept `kwargs` and isolate distribute specific arguments.
    k_pop_list = []
    distr_kwargs = {}
    for k, v in kwargs.items():
        if hasattr(v, "__len__"):
            if len(v) == n_distr:
                # This will grab the first dim of numpy arrays.
                # Assume argument for `bvn_ellipse`.
                k_pop_list.append(k)
                distr_kwargs.update({k: v})
    for k in k_pop_list:
        kwargs.pop(k)

    # Draw BVN HDI ellipses for each distribution.
    for i_distr in range(n_distr):
        # Assemble key-word arguments for single distribution.
        curr_distr_kwargs = {}
        for k, v in distr_kwargs.items():
            curr_distr_kwargs.update({k: v[i_distr]})
        curr_distr_kwargs.update(kwargs)

        # Create distribution ellipse.
        ellipse = psiz.mplot.bvn_ellipse(
            loc[i_distr], cov[i_distr], r=r, **curr_distr_kwargs
        )

        # Add ellipse to `AxesSubplot`.
        ax.add_artist(ellipse)
