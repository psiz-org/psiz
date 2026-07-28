# -*- coding: utf-8 -*-
# Copyright 2026 The PsiZ Authors. All Rights Reserved.
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
"""Tests for backend-agnostic multivariate normal unpacking."""

import numpy as np

from psiz.stochastic import unpack_mvn


class _DiagonalMvn:
    def __init__(self, loc, variance):
        self._loc = np.asarray(loc)
        self._variance = np.asarray(variance)

    def mean(self):
        return self._loc

    def covariance(self):
        raise NotImplementedError()

    def variance(self):
        return self._variance


class _FullCovarianceMvn:
    def __init__(self, loc, covariance):
        self._loc = np.asarray(loc)
        self._covariance = np.asarray(covariance)

    def mean(self):
        return self._loc

    def covariance(self):
        return self._covariance


def test_unpack_diagonal_covariance_mvn():
    loc_desired = np.array(
        [
            [0.1, 0.2],
            [1.1, 1.2],
            [2.1, 2.2],
        ],
        dtype=np.float32,
    )
    variance_desired = np.array(
        [
            [0.028900036588311195, 0.028900036588311195],
            [0.028900036588311195, 0.028900036588311195],
            [0.028900036588311195, 0.028900036588311195],
        ],
        dtype=np.float32,
    )
    cov_desired = np.array(
        [
            [[0.028900036588311195, 0.0], [0.0, 0.028900036588311195]],
            [[0.028900036588311195, 0.0], [0.0, 0.028900036588311195]],
            [[0.028900036588311195, 0.0], [0.0, 0.028900036588311195]],
        ],
        dtype=np.float32,
    )

    dist = _DiagonalMvn(loc=loc_desired, variance=variance_desired)
    loc, cov = unpack_mvn(dist)

    assert loc.shape == (3, 2)
    assert cov.shape == (3, 2, 2)
    np.testing.assert_allclose(loc, loc_desired)
    np.testing.assert_allclose(cov, cov_desired)


def test_unpack_full_covariance_mvn():
    loc_desired = np.array(
        [
            [0.1, 0.2],
            [1.1, 1.2],
            [2.1, 2.2],
        ],
        dtype=np.float32,
    )
    cov_desired = np.array(
        [
            [[0.028900036588311195, 0.0], [0.0, 0.028900036588311195]],
            [[0.028900036588311195, 0.0], [0.0, 0.028900036588311195]],
            [[0.028900036588311195, 0.0], [0.0, 0.028900036588311195]],
        ],
        dtype=np.float32,
    )

    dist = _FullCovarianceMvn(loc=loc_desired, covariance=cov_desired)
    loc, cov = unpack_mvn(dist)

    assert loc.shape == (3, 2)
    assert cov.shape == (3, 2, 2)
    np.testing.assert_allclose(loc, loc_desired)
    np.testing.assert_allclose(cov, cov_desired)