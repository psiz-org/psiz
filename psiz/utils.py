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

"""Module of helpful utility functions."""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from itertools import permutations
import math


def matrix_correlation(mat_a, mat_b):
    """Return the R^2 score between two square matrices.

    Args:
        mat_a: A square matrix.
        mat_b: A square matrix the same size as mat_a

    Returns:
        The R^2 score between the two matrices.

    Notes:
        When computing R^2 values of two similarity matrices, it is
        assumed, by definition, that the corresponding diagonal
        elements are the same between the two matrices being compared.
        Therefore, the diagonal elements are not included in the R^2
        computation to prevent inflating the R^2 value. On a similar
        note, including both the upper and lower triangular portion
        does not artificially inflate R^2 values for symmetric
        matrices.

    """
    n_row = mat_a.shape[0]
    idx_upper = np.triu_indices(n_row, 1)
    idx_lower = np.triu_indices(n_row, 1)
    idx = (
        np.hstack((idx_upper[0], idx_lower[0])),
        np.hstack((idx_upper[1], idx_lower[1])),
    )
    # Explained variance score.
    return r2_score(mat_a[idx], mat_b[idx])


def possible_outcomes(trial_configuration):
    """Return the possible outcomes of a trial configuration.

    Args:
        trial_configuration: A trial configuration Pandas Series.

    Returns:
        An 2D array indicating all possible outcomes where the values
            indicate indices of the reference stimuli. Each row
            corresponds to one outcome. Note the indices refer to
            references only and does not include an index for the
            query.

    """
    n_reference = trial_configuration['n_reference']
    n_selected = int(trial_configuration['n_selected'])

    reference_list = range(n_reference)

    # Get all permutations of length n_selected.
    perm = permutations(reference_list, n_selected)

    selection = list(perm)
    n_outcome = len(selection)

    outcomes = np.empty((n_outcome, n_reference), dtype=np.int32)
    for i_outcome in range(n_outcome):
        # Fill in selections.
        outcomes[i_outcome, 0:n_selected] = selection[i_outcome]
        # Fill in unselected.
        dummy_idx = np.arange(n_reference)
        for i_selected in range(n_selected):
            loc = dummy_idx != outcomes[i_outcome, i_selected]
            dummy_idx = dummy_idx[loc]

        outcomes[i_outcome, n_selected:] = dummy_idx

    return outcomes


def elliptical_slice(
        initial_theta, prior, lnpdf, pdf_params=(), cur_lnpdf=None, 
        angle_range=None):
    """Return samples from elliptical slice sampler.

    Markov chain update for a distribution with a Gaussian "prior"
    factored out.

    Args:
        initial_theta: initial vector
        prior: cholesky decomposition of the covariance matrix (like
            what numpy.linalg.cholesky returns), or a sample from the
            prior
        lnpdf: function evaluating the log of the pdf to be sampled
        pdf_params: parameters to pass to the pdf
        cur_lnpdf (optional): value of lnpdf at initial_theta
        angle_range: Default 0: explore whole ellipse with break point
            at first rejection. Set in (0,2*pi] to explore a bracket of
            the specified width centred uniformly at random.

    Returns:
        new_theta, new_lnpdf

    History:
        Originally written in MATLAB by Iain Murray
        (http://homepages.inf.ed.ac.uk/imurray2/pub/10ess/elliptical_slice.m)
        2012-02-24 - Written - Bovy (IAS)

    """
    D = len(initial_theta)
    if cur_lnpdf is None:
        cur_lnpdf = lnpdf(initial_theta, *pdf_params)

    # Set up the ellipse and the slice threshold
    if len(prior.shape) == 1:  # prior = prior sample
        nu = prior
    else:  # prior = cholesky decomp
        if not prior.shape[0] == D or not prior.shape[1] == D:
            raise IOError(
                "Prior must be given by a D-element sample or DxD chol(Sigma)")
        nu = np.dot(prior, np.random.normal(size=D))
    hh = math.log(np.random.uniform()) + cur_lnpdf

    # Set up a bracket of angles and pick a first proposal.
    # "phi = (theta'-theta)" is a change in angle.
    if angle_range is None or angle_range == 0.:
        # Bracket whole ellipse with both edges at first proposed point
        phi = np.random.uniform() * 2. * math.pi
        phi_min = phi - 2. * math.pi
        phi_max = phi
    else:
        # Randomly center bracket on current point
        phi_min = -angle_range * np.random.uniform()
        phi_max = phi_min + angle_range
        phi = np.random.uniform() * (phi_max - phi_min) + phi_min

    # Slice sampling loop
    while True:
        # Compute xx for proposed angle difference and check if it's on the slice
        xx_prop = initial_theta * math.cos(phi) + nu * math.sin(phi)
        cur_lnpdf = lnpdf(xx_prop, *pdf_params)
        if cur_lnpdf > hh:
            # New point is on slice, ** EXIT LOOP **
            break
        # Shrink slice to rejected point
        if phi > 0:
            phi_max = phi
        elif phi < 0:
            phi_min = phi
        else:
            raise RuntimeError('BUG DETECTED: Shrunk to current position and still not acceptable.')
        # Propose new angle difference
        phi = np.random.uniform() * (phi_max - phi_min) + phi_min
    return (xx_prop, cur_lnpdf)
