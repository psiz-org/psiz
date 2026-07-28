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
"""Backend-agnostic helpers for stochastic distributions."""

from __future__ import annotations

import keras
import numpy as np


def unpack_mvn(dist):
    """Unpack a multivariate normal distribution into loc and covariance arrays."""
    loc = keras.ops.convert_to_numpy(dist.mean())

    try:
        cov = keras.ops.convert_to_numpy(dist.covariance())
    except NotImplementedError:
        variance = keras.ops.convert_to_numpy(dist.variance())
        cov = _diag_to_full_cov(variance)

    return loc, cov


def _diag_to_full_cov(v):
    """Convert diagonal variance to full covariance matrices."""
    v = np.asarray(v)
    n_stimuli = v.shape[0]
    n_dim = v.shape[1]
    cov = np.zeros([n_stimuli, n_dim, n_dim], dtype=v.dtype)
    for i_stimulus in range(n_stimuli):
        cov[i_stimulus] = np.eye(n_dim, dtype=v.dtype) * v[i_stimulus]
    return cov


__all__ = ["unpack_mvn"]