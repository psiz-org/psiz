# -*- coding: utf-8 -*-
# Copyright 2024 The PsiZ Authors. All Rights Reserved.
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
"""Test constraints module."""


import keras
import numpy as np

from psiz.keras.constraints import LessEqualThan


def test_all():
    """Test all methods."""
    # Initialize.
    con = LessEqualThan(max_value=0.1)
    assert con.max_value == 0.1

    # Check get_config.
    config = con.get_config()
    assert config["max_value"] == 0.1

    # Check call.
    w0 = np.array([[1.36, -0.35], [1.40, -0.41]], dtype="float32")
    w1 = con(w0)
    w_desired = np.array([[0.1, -0.35], [0.1, -0.41]], dtype="float32")
    np.testing.assert_array_almost_equal(keras.ops.convert_to_numpy(w1), w_desired)
