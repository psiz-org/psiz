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

from psiz.keras.regularizers import Squeeze


def test_all():
    """Test all methods."""
    # Initialize.
    rate = 0.1
    reg = Squeeze(rate=rate)

    # Check get_config.
    config = reg.get_config()
    assert config["rate"] == rate

    # Check call.
    z = np.array([[0.5, 0.6, 0.7], [-0.1, 1.2, 0.2], [-1.0, 1.3, 0.3]])
    output = reg(z)
    output = keras.ops.convert_to_numpy(output)

    np.testing.assert_array_almost_equal(output, np.array(0.3))
