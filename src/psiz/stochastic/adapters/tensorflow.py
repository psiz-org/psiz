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
"""TensorFlow stochastic adapter backed by TensorFlow Probability."""

from __future__ import annotations

import tensorflow_probability as tfp
from tensorflow_probability.python.distributions import kullback_leibler as kl_lib


class TensorFlowStochasticAdapter:
    """Stochastic adapter using TensorFlow Probability distributions."""

    distribution_type = tfp.distributions.Distribution

    def normal(self, loc, scale):
        return tfp.distributions.Normal(loc=loc, scale=scale)

    def laplace(self, loc, scale):
        return tfp.distributions.Laplace(loc=loc, scale=scale)

    def log_normal(self, loc, scale):
        return tfp.distributions.LogNormal(loc=loc, scale=scale)

    def logit_normal(self, loc, scale):
        return tfp.distributions.LogitNormal(loc=loc, scale=scale)

    def gamma(self, concentration, rate):
        return tfp.distributions.Gamma(concentration=concentration, rate=rate)

    def truncated_normal(self, loc, scale, low, high):
        return tfp.distributions.TruncatedNormal(loc=loc, scale=scale, low=low, high=high)

    def dirichlet(self, concentration):
        return tfp.distributions.Dirichlet(concentration=concentration)

    def independent(self, distribution, reinterpreted_batch_ndims):
        return tfp.distributions.Independent(
            distribution,
            reinterpreted_batch_ndims=reinterpreted_batch_ndims,
        )

    def shared_sample_distribution(self, distribution, sample_shape):
        reshape = tfp.bijectors.Reshape(event_shape_out=[], event_shape_in=[1, 1])
        return tfp.distributions.TransformedDistribution(
            distribution=tfp.distributions.Sample(
                distribution,
                sample_shape=sample_shape,
            ),
            bijector=reshape,
        )

    def kl_divergence(self, posterior, prior):
        return kl_lib.kl_divergence(posterior, prior)
