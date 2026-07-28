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
"""KL divergence API with exact and fallback behavior."""

from __future__ import annotations

import keras

from psiz.stochastic.adapters import get_stochastic_adapter


def kl_divergence(
    posterior,
    prior,
    *,
    backend_override=None,
    fallback="monte_carlo",
    n_sample=100,
):
    """Compute KL divergence with backend adapter and fallback policy."""
    adapter = get_stochastic_adapter(backend_override)
    try:
        return adapter.kl_divergence(posterior, prior)
    except NotImplementedError:
        if fallback != "monte_carlo":
            raise
        samples = posterior.sample(sample_shape=n_sample)
        return keras.ops.mean(posterior.log_prob(samples) - prior.log_prob(samples))
