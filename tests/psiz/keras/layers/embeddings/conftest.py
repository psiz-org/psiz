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
"""psiz.keras.layers.embeddings pytest setup."""

import pytest

import numpy as np


@pytest.fixture
def flat_embeddings():
    z = np.array([
        [0.0, 0.1, 0.2],  # 0, 0, 0
        [1.0, 1.1, 1.2],  # 1, 0, 0
        [2.0, 2.1, 2.2],  # 2, 0, 0
        [3.0, 3.1, 3.2],  # 0, 1, 0
        [4.0, 4.1, 4.2],  # 1, 1, 0
        [5.0, 5.1, 5.2],  # 2, 1, 0
        [6.0, 6.1, 6.2],  # 0, 0, 1
        [7.0, 7.1, 7.2],  # 1, 0, 1
        [8.0, 8.1, 8.2],  # 2, 0, 1
        [9.0, 9.1, 9.2],  # 0, 1, 1
        [10.0, 10.1, 10.2],  # 1, 1, 1
        [11.0, 11.1, 11.2],  # 2, 1, 1
    ])
    return z
