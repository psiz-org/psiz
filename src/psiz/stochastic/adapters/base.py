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
"""Base classes and Keras-ops stochastic distributions."""

from __future__ import annotations

import math
from dataclasses import dataclass

import keras


SQRT_TWO = math.sqrt(2.0)
HALF_LOG_TWO_PI = 0.5 * math.log(2.0 * math.pi)


def _to_shape_tensor(value):
    """Return a rank-1 int32 tensor representing a shape."""
    if value is None or value == ():
        return keras.ops.convert_to_tensor([], dtype="int32")
    if isinstance(value, int):
        return keras.ops.convert_to_tensor([value], dtype="int32")
    if isinstance(value, (tuple, list)):
        return keras.ops.convert_to_tensor(list(value), dtype="int32")
    return keras.ops.cast(keras.ops.reshape(value, [-1]), "int32")


def _as_int(value):
    """Best-effort conversion to Python int."""
    try:
        return int(value)
    except TypeError:
        return int(keras.ops.convert_to_numpy(value))


def _concat_sample_and_batch_shape(sample_shape, batch_shape):
    sample_shape = _to_shape_tensor(sample_shape)
    batch_shape = _to_shape_tensor(batch_shape)
    shape = keras.ops.concatenate([sample_shape, batch_shape], axis=0)
    return tuple(keras.ops.convert_to_numpy(shape).tolist())


@dataclass(frozen=True)
class DistributionProtocol:
    """Marker protocol for PsiZ stochastic distributions."""


class KerasDistribution(DistributionProtocol):
    """Backend-agnostic distribution implemented with keras.ops."""

    def sample(self, sample_shape=(), seed=None):
        raise NotImplementedError()

    def log_prob(self, value):
        raise NotImplementedError()

    def mean(self):
        raise NotImplementedError()

    def variance(self):
        raise NotImplementedError()

    def mode(self):
        return self.mean()

    def stddev(self):
        return keras.ops.sqrt(self.variance())

    def quantile(self, p):
        raise NotImplementedError("Quantile is not implemented for this distribution.")

    def batch_shape_tensor(self):
        return keras.ops.shape(self.mean())

    @property
    def batch_shape(self):
        return tuple(keras.ops.convert_to_numpy(self.batch_shape_tensor()).tolist())

    @property
    def event_shape(self):
        return ()


class NormalDistribution(KerasDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

    def sample(self, sample_shape=(), seed=None):
        batch_shape = keras.ops.shape(self.loc)
        shape = _concat_sample_and_batch_shape(sample_shape, batch_shape)
        eps = keras.random.normal(shape=shape, dtype=self.loc.dtype, seed=seed)
        return self.loc + self.scale * eps

    def log_prob(self, value):
        z = (value - self.loc) / self.scale
        return -(HALF_LOG_TWO_PI + keras.ops.log(self.scale) + 0.5 * keras.ops.square(z))

    def mean(self):
        return self.loc

    def variance(self):
        return keras.ops.square(self.scale)

    def quantile(self, p):
        p = keras.ops.clip(p, keras.backend.epsilon(), 1.0 - keras.backend.epsilon())
        return self.loc + self.scale * SQRT_TWO * keras.ops.erfinv(2.0 * p - 1.0)


class LaplaceDistribution(KerasDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

    def sample(self, sample_shape=(), seed=None):
        batch_shape = keras.ops.shape(self.loc)
        shape = _concat_sample_and_batch_shape(sample_shape, batch_shape)
        u = keras.random.uniform(
            shape=shape,
            minval=-0.5,
            maxval=0.5,
            dtype=self.loc.dtype,
            seed=seed,
        )
        return self.loc - self.scale * keras.ops.sign(u) * keras.ops.log1p(
            -2.0 * keras.ops.abs(u)
        )

    def log_prob(self, value):
        return -keras.ops.abs(value - self.loc) / self.scale - keras.ops.log(
            2.0 * self.scale
        )

    def mean(self):
        return self.loc

    def variance(self):
        return 2.0 * keras.ops.square(self.scale)

    def quantile(self, p):
        p = keras.ops.clip(p, keras.backend.epsilon(), 1.0 - keras.backend.epsilon())
        return keras.ops.where(
            p < 0.5,
            self.loc + self.scale * keras.ops.log(2.0 * p),
            self.loc - self.scale * keras.ops.log(2.0 * (1.0 - p)),
        )


class LogNormalDistribution(KerasDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale
        self._normal = NormalDistribution(loc, scale)

    def sample(self, sample_shape=(), seed=None):
        return keras.ops.exp(self._normal.sample(sample_shape=sample_shape, seed=seed))

    def log_prob(self, value):
        safe_value = keras.ops.maximum(value, keras.backend.epsilon())
        return self._normal.log_prob(keras.ops.log(safe_value)) - keras.ops.log(safe_value)

    def mean(self):
        return keras.ops.exp(self.loc + 0.5 * keras.ops.square(self.scale))

    def variance(self):
        s2 = keras.ops.square(self.scale)
        return (keras.ops.exp(s2) - 1.0) * keras.ops.exp(2.0 * self.loc + s2)

    def mode(self):
        return keras.ops.exp(self.loc - keras.ops.square(self.scale))

    def quantile(self, p):
        return keras.ops.exp(self._normal.quantile(p))


class LogitNormalDistribution(KerasDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale
        self._normal = NormalDistribution(loc, scale)

    def sample(self, sample_shape=(), seed=None):
        return keras.ops.sigmoid(self._normal.sample(sample_shape=sample_shape, seed=seed))

    def log_prob(self, value):
        eps = keras.backend.epsilon()
        safe_value = keras.ops.clip(value, eps, 1.0 - eps)
        logit_value = keras.ops.log(safe_value) - keras.ops.log(1.0 - safe_value)
        jac = keras.ops.log(safe_value) + keras.ops.log(1.0 - safe_value)
        return self._normal.log_prob(logit_value) - jac

    def mean(self):
        return keras.ops.sigmoid(self.loc)

    def variance(self):
        samples = self.sample(sample_shape=128)
        return keras.ops.var(samples, axis=0)

    def mode(self):
        return keras.ops.sigmoid(self.loc)

    def quantile(self, p):
        return keras.ops.sigmoid(self._normal.quantile(p))


class GammaDistribution(KerasDistribution):
    def __init__(self, concentration, rate):
        self.concentration = concentration
        self.rate = rate

    def sample(self, sample_shape=(), seed=None):
        batch_shape = keras.ops.shape(self.concentration)
        shape = _concat_sample_and_batch_shape(sample_shape, batch_shape)
        draws = keras.random.gamma(
            shape=shape,
            alpha=self.concentration,
            dtype=self.concentration.dtype,
            seed=seed,
        )
        return draws / self.rate

    def log_prob(self, value):
        raise NotImplementedError("Gamma log_prob is not implemented for keras-ops.")

    def mean(self):
        return self.concentration / self.rate

    def variance(self):
        return self.concentration / keras.ops.square(self.rate)

    def mode(self):
        return keras.ops.where(
            self.concentration >= 1.0,
            (self.concentration - 1.0) / self.rate,
            keras.ops.zeros_like(self.concentration),
        )


class TruncatedNormalDistribution(KerasDistribution):
    def __init__(self, loc, scale, low, high):
        self.loc = loc
        self.scale = scale
        self.low = low
        self.high = high
        self._normal = NormalDistribution(loc, scale)

    def _z_low(self):
        return (self.low - self.loc) / self.scale

    def _z_high(self):
        return (self.high - self.loc) / self.scale

    def _cdf(self, z):
        return 0.5 * (1.0 + keras.ops.erf(z / SQRT_TWO))

    def _cdf_range(self):
        cdf_low = self._cdf(self._z_low())
        cdf_high = self._cdf(self._z_high())
        return cdf_low, cdf_high

    def sample(self, sample_shape=(), seed=None):
        cdf_low, cdf_high = self._cdf_range()
        batch_shape = keras.ops.shape(self.loc)
        shape = _concat_sample_and_batch_shape(sample_shape, batch_shape)
        u = keras.random.uniform(
            shape=shape,
            minval=0.0,
            maxval=1.0,
            dtype=self.loc.dtype,
            seed=seed,
        )
        u = cdf_low + (cdf_high - cdf_low) * u
        z = SQRT_TWO * keras.ops.erfinv(2.0 * u - 1.0)
        return self.loc + self.scale * z

    def log_prob(self, value):
        cdf_low, cdf_high = self._cdf_range()
        z = (value - self.loc) / self.scale
        log_unnormalized = -(
            HALF_LOG_TWO_PI + keras.ops.log(self.scale) + 0.5 * keras.ops.square(z)
        )
        normalizer = keras.ops.log(cdf_high - cdf_low + keras.backend.epsilon())
        in_support = keras.ops.logical_and(value >= self.low, value <= self.high)
        neg_inf = keras.ops.full_like(log_unnormalized, -1e9)
        return keras.ops.where(in_support, log_unnormalized - normalizer, neg_inf)

    def mean(self):
        # Approximate mean with deterministic midpoint in standardized support.
        return 0.5 * (self.low + self.high)

    def variance(self):
        width = keras.ops.maximum(self.high - self.low, keras.backend.epsilon())
        return keras.ops.square(width) / 12.0

    def mode(self):
        return keras.ops.clip(self.loc, self.low, self.high)

    def quantile(self, p):
        p = keras.ops.clip(p, keras.backend.epsilon(), 1.0 - keras.backend.epsilon())
        cdf_low, cdf_high = self._cdf_range()
        u = cdf_low + p * (cdf_high - cdf_low)
        z = SQRT_TWO * keras.ops.erfinv(2.0 * u - 1.0)
        return self.loc + self.scale * z


class DirichletDistribution(KerasDistribution):
    def __init__(self, concentration):
        self.concentration = concentration

    def sample(self, sample_shape=(), seed=None):
        batch_shape = keras.ops.shape(self.concentration)
        shape = _concat_sample_and_batch_shape(sample_shape, batch_shape)
        draws = keras.random.gamma(
            shape=shape,
            alpha=self.concentration,
            dtype=self.concentration.dtype,
            seed=seed,
        )
        norm = keras.ops.sum(draws, axis=-1, keepdims=True)
        return draws / keras.ops.maximum(norm, keras.backend.epsilon())

    def log_prob(self, value):
        raise NotImplementedError("Dirichlet log_prob is not implemented for keras-ops.")

    def mean(self):
        total = keras.ops.sum(self.concentration, axis=-1, keepdims=True)
        return self.concentration / total

    def variance(self):
        total = keras.ops.sum(self.concentration, axis=-1, keepdims=True)
        numer = self.concentration * (total - self.concentration)
        denom = keras.ops.square(total) * (total + 1.0)
        return numer / denom

    def mode(self):
        total = keras.ops.sum(self.concentration, axis=-1, keepdims=True)
        k = keras.ops.cast(keras.ops.shape(self.concentration)[-1], self.concentration.dtype)
        numer = self.concentration - 1.0
        denom = total - k
        return numer / keras.ops.maximum(denom, keras.backend.epsilon())


class IndependentDistribution(KerasDistribution):
    """Wrap a base distribution and reinterpret trailing batch dims as event dims."""

    def __init__(self, distribution, reinterpreted_batch_ndims):
        self.distribution = distribution
        self.reinterpreted_batch_ndims = _as_int(reinterpreted_batch_ndims)

    def sample(self, sample_shape=(), seed=None):
        return self.distribution.sample(sample_shape=sample_shape, seed=seed)

    def log_prob(self, value):
        lp = self.distribution.log_prob(value)
        if self.reinterpreted_batch_ndims <= 0:
            return lp
        rank = _as_int(keras.ops.ndim(lp))
        start = rank - self.reinterpreted_batch_ndims
        axes = list(range(start, rank))
        return keras.ops.sum(lp, axis=axes)

    def mean(self):
        return self.distribution.mean()

    def variance(self):
        return self.distribution.variance()

    def mode(self):
        return self.distribution.mode()

    def stddev(self):
        return self.distribution.stddev()

    def quantile(self, p):
        return self.distribution.quantile(p)

    def batch_shape_tensor(self):
        shape = self.distribution.batch_shape_tensor()
        rank = _as_int(keras.ops.size(shape))
        if self.reinterpreted_batch_ndims <= 0:
            return shape
        return shape[: rank - self.reinterpreted_batch_ndims]

    @property
    def event_shape(self):
        shape = self.distribution.batch_shape_tensor()
        rank = _as_int(keras.ops.size(shape))
        if self.reinterpreted_batch_ndims <= 0:
            return ()
        event = shape[rank - self.reinterpreted_batch_ndims :]
        return tuple(keras.ops.convert_to_numpy(event).tolist())


class SharedSampleDistribution(KerasDistribution):
    """Expand a source distribution across a target sample/event shape."""

    def __init__(self, distribution, sample_shape):
        self.distribution = distribution
        self.sample_shape = tuple(sample_shape)

    def _squeeze_scalar_tail(self, value):
        rank = _as_int(keras.ops.ndim(value))
        if rank >= 2:
            value = keras.ops.squeeze(value, axis=rank - 1)
        rank = _as_int(keras.ops.ndim(value))
        if rank >= 1:
            value = keras.ops.squeeze(value, axis=rank - 1)
        return value

    def sample(self, sample_shape=(), seed=None):
        combined = tuple(_as_int(x) for x in self.sample_shape)
        if sample_shape not in (None, ()):
            prefix = tuple(_as_int(x) for x in _to_shape_tensor(sample_shape))
            combined = prefix + combined
        value = self.distribution.sample(sample_shape=combined, seed=seed)
        return self._squeeze_scalar_tail(value)

    def log_prob(self, value):
        lp = self.distribution.log_prob(value)
        ndim = _as_int(keras.ops.ndim(lp))
        axes = list(range(ndim - len(self.sample_shape), ndim))
        return keras.ops.sum(lp, axis=axes)

    def mean(self):
        value = self.distribution.mean()
        return keras.ops.broadcast_to(value, list(self.sample_shape) + [1, 1])

    def variance(self):
        value = self.distribution.variance()
        return keras.ops.broadcast_to(value, list(self.sample_shape) + [1, 1])

    def mode(self):
        value = self.distribution.mode()
        return keras.ops.broadcast_to(value, list(self.sample_shape) + [1, 1])

    @property
    def event_shape(self):
        return self.sample_shape


class KerasOpsStochasticAdapter:
    """Stochastic adapter implemented with backend-agnostic keras.ops."""

    distribution_type = DistributionProtocol

    def normal(self, loc, scale):
        return NormalDistribution(loc=loc, scale=scale)

    def laplace(self, loc, scale):
        return LaplaceDistribution(loc=loc, scale=scale)

    def log_normal(self, loc, scale):
        return LogNormalDistribution(loc=loc, scale=scale)

    def logit_normal(self, loc, scale):
        return LogitNormalDistribution(loc=loc, scale=scale)

    def gamma(self, concentration, rate):
        return GammaDistribution(concentration=concentration, rate=rate)

    def truncated_normal(self, loc, scale, low, high):
        return TruncatedNormalDistribution(loc=loc, scale=scale, low=low, high=high)

    def dirichlet(self, concentration):
        return DirichletDistribution(concentration=concentration)

    def independent(self, distribution, reinterpreted_batch_ndims):
        return IndependentDistribution(
            distribution=distribution,
            reinterpreted_batch_ndims=reinterpreted_batch_ndims,
        )

    def shared_sample_distribution(self, distribution, sample_shape):
        return SharedSampleDistribution(distribution=distribution, sample_shape=sample_shape)

    def kl_divergence(self, posterior, prior):
        if isinstance(posterior, NormalDistribution) and isinstance(prior, NormalDistribution):
            posterior_scale = posterior.scale
            prior_scale = prior.scale
            mean_delta = posterior.loc - prior.loc
            variance_ratio = (keras.ops.square(posterior_scale) + keras.ops.square(mean_delta)) / keras.ops.square(prior_scale)
            log_scale_ratio = keras.ops.log(prior_scale) - keras.ops.log(posterior_scale)
            return 0.5 * (variance_ratio - 1.0 + 2.0 * log_scale_ratio)
        raise NotImplementedError(
            "Exact KL is not implemented for keras-ops adapter distributions."
        )
