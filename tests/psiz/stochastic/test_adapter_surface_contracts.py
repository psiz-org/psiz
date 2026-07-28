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
"""Fast adapter-surface tests that do not require native backend installs."""

import pytest

from psiz.stochastic.adapters import canonicalize_parameters


pytestmark = pytest.mark.adapter_surface


def test_adapter_surface_parameter_aliases():
    canonical = canonicalize_parameters(
        {
            "mean": 0.25,
            "sigma": 0.5,
            "alpha": 3.0,
            "unused": "passthrough",
        }
    )

    assert canonical["loc"] == 0.25
    assert canonical["scale"] == 0.5
    assert canonical["concentration"] == 3.0
    assert canonical["unused"] == "passthrough"


def test_adapter_surface_rejects_duplicate_aliases():
    with pytest.raises(ValueError, match="Multiple aliases provided"):
        _ = canonicalize_parameters({"loc": 1.0, "mean": 1.0})
