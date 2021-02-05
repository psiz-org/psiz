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
"""Test Stochastic."""

import pytest

from psiz.keras.layers.stochastic import Stochastic


def test_init():
    stoch_0 = Stochastic()
    assert stoch_0.sample_shape == ()

    stoch_1 = Stochastic(sample_shape=1)
    assert stoch_1.sample_shape == 1

    stoch_2 = Stochastic(sample_shape=3)
    assert stoch_2.sample_shape == 3
