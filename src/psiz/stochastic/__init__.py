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
"""Stochastic adapter layer for backend-agnostic PsiZ stochastic code."""

from psiz.stochastic.adapters import canonicalize_parameters
from psiz.stochastic.adapters import get_stochastic_adapter
from psiz.stochastic.adapters import is_distribution
from psiz.stochastic.kl import kl_divergence
from psiz.stochastic.transforms import softplus_inverse

__all__ = [
    "canonicalize_parameters",
    "get_stochastic_adapter",
    "is_distribution",
    "kl_divergence",
    "softplus_inverse",
]
