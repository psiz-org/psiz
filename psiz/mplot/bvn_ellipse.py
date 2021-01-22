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
"""Module of Matplotlib tools.

Functions:
    bvn_ellipse: Create an ellipse representing a bivariate normal
        distribution.

"""

import numpy as np
from matplotlib.patches import Ellipse


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
            highest probability mass (i.e, HDI). Another common value
            is 2.576, which indicates 99%.
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
    ellipse = Ellipse(
        xy=(loc[0], loc[1]), width=w, height=h, angle=theta, **kwargs
    )
    return ellipse
