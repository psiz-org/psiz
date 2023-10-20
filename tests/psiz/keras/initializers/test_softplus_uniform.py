# -*- coding: utf-8 -*-
# Copyright 2021 The PsiZ Authors. All Rights Reserved.
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

import tensorflow as tf
from tensorflow.keras import backend

from psiz.keras.initializers import SoftplusUniform


def test_all():
    """Test all methods."""
    # Initialize.
    minval = 0.01
    maxval = 0.05
    hinge_softness = 1.0
    seed = 252
    initializer = SoftplusUniform(
        minval, maxval, hinge_softness=hinge_softness, seed=seed
    )
    assert initializer.minval == 0.01
    assert initializer.maxval == 0.05
    assert initializer.hinge_softness == 1.0
    assert initializer.seed == 252

    # Check `get_config`.
    config = initializer.get_config()
    assert config["minval"] == 0.01
    assert config["maxval"] == 0.05
    assert config["hinge_softness"] == 1.0
    assert config["seed"] == 252

    # Check call does not raise error.
    tf_shape = tf.TensorShape([2, 4])
    _ = initializer(tf_shape)

    _ = initializer(tf_shape, dtype=backend.floatx())
