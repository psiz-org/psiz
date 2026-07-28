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
"""Stochastic adapter resolution and canonical parameter helpers."""

from __future__ import annotations

from psiz.backend import resolve_backend
from psiz.stochastic.adapters.base import DistributionProtocol


_CANONICAL_PARAMETER_ALIASES = {
    "loc": ("loc", "mean"),
    "scale": ("scale", "stddev", "sigma"),
    "concentration": ("concentration", "alpha"),
    "rate": ("rate", "beta"),
    "low": ("low", "lower", "a"),
    "high": ("high", "upper", "b"),
}


def get_stochastic_adapter(backend_override=None):
    """Return stochastic adapter for resolved backend."""
    backend = resolve_backend(backend_override)
    if backend == "tensorflow":
        from psiz.stochastic.adapters.tensorflow import TensorFlowStochasticAdapter

        return TensorFlowStochasticAdapter()
    if backend == "torch":
        from psiz.stochastic.adapters.torch import TorchStochasticAdapter

        return TorchStochasticAdapter()
    if backend == "jax":
        from psiz.stochastic.adapters.jax import JaxStochasticAdapter

        return JaxStochasticAdapter()
    raise ValueError(f"Unsupported backend '{backend}'.")


def canonicalize_parameters(params):
    """Canonicalize distribution constructor parameter aliases."""
    canonical = {}
    for canonical_name, aliases in _CANONICAL_PARAMETER_ALIASES.items():
        alias_hits = [alias for alias in aliases if alias in params]
        if len(alias_hits) > 1:
            raise ValueError(
                f"Multiple aliases provided for '{canonical_name}': {alias_hits}."
            )
        if len(alias_hits) == 1:
            canonical[canonical_name] = params[alias_hits[0]]
    for key, value in params.items():
        if not any(key in aliases for aliases in _CANONICAL_PARAMETER_ALIASES.values()):
            canonical[key] = value
    return canonical


def is_distribution(obj):
    """Return true if object is a stochastic distribution instance."""
    if isinstance(obj, DistributionProtocol):
        return True
    try:
        import tensorflow_probability as tfp

        return isinstance(obj, tfp.distributions.Distribution)
    except Exception:
        return False


__all__ = [
    "canonicalize_parameters",
    "get_stochastic_adapter",
    "is_distribution",
]
