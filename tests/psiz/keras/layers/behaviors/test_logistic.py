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


import keras
import numpy as np

import psiz


class TestLogistic:

    def test_v0(self):
        """Test logistic layer init and call."""
        logistic = psiz.keras.layers.Logistic()

        x = np.array(
            [
                [0.0, 0.1, 0.3],
                [1.0, 2.0, 3.0],
                [0.0, -0.1, -0.3],
                [-1.0, -2.0, -3.0],
            ],
            dtype="float32",
        )
        y = logistic(x)
        y = keras.ops.convert_to_numpy(y)

        # Check default initialization.
        np.testing.assert_allclose(logistic.upper, 1.0)
        np.testing.assert_allclose(logistic.midpoint, 0.0)
        np.testing.assert_allclose(logistic.rate, 1.0)

        # Desired outcome.
        y_desired = np.array(
            [
                [0.5, 0.5249792, 0.5744425],
                [0.7310586, 0.880797, 0.95257413],
                [0.5, 0.4750208, 0.4255575],
                [0.26894143, 0.11920292, 0.04742587],
            ],
            dtype="float32",
        )

        np.testing.assert_allclose(y, y_desired)

    def test_v1(self):
        """Test logistic layer init and call."""
        logistic = psiz.keras.layers.Logistic(
            upper_initializer=keras.initializers.Constant(2.0),
            midpoint_initializer=keras.initializers.Constant(1.0),
            rate_initializer=keras.initializers.Constant(0.5),
        )

        x = np.array(
            [
                [0.0, 0.1, 0.3],
                [1.0, 2.0, 3.0],
                [0.0, -0.1, -0.3],
                [-1.0, -2.0, -3.0],
            ],
            dtype="float32",
        )
        y = logistic(x)
        y = keras.ops.convert_to_numpy(y)

        # Check default initialization.
        np.testing.assert_allclose(logistic.upper, 2.0)
        np.testing.assert_allclose(logistic.midpoint, 1.0)
        np.testing.assert_allclose(logistic.rate, 0.5)

        # Desired outcome.
        y_desired = np.array(
            [
                [0.75508136, 0.7787215, 0.8267648],
                [1.0, 1.2449187, 1.4621172],
                [0.75508136, 0.73172885, 0.6859791],
                [0.53788286, 0.36485106, 0.23840584],
            ],
            dtype="float32",
        )

        np.testing.assert_allclose(y, y_desired)

    def test_serialization_v0(self):
        """Test serialization."""
        logistic = psiz.keras.layers.Logistic(
            upper_initializer=keras.initializers.Constant(2.0),
            midpoint_initializer=keras.initializers.Constant(1.0),
            rate_initializer=keras.initializers.Constant(0.5),
            midpoint_constraint=keras.constraints.NonNeg(),
            rate_constraint=keras.constraints.NonNeg(),
            name="my_logistic",
        )

        x = np.array(
            [
                [0.0, 0.1, 0.3],
                [1.0, 2.0, 3.0],
                [0.0, -0.1, -0.3],
                [-1.0, -2.0, -3.0],
            ],
            dtype="float32",
        )
        _ = logistic(x)

        cfg = logistic.get_config()
        # Verify.
        assert cfg["name"] == "my_logistic"
        assert cfg["trainable"]
        assert cfg["upper_initializer"]["class_name"] == "Constant"
        assert cfg["upper_initializer"]["config"]["value"] == 2.0
        assert cfg["midpoint_initializer"]["class_name"] == "Constant"
        assert cfg["midpoint_initializer"]["config"]["value"] == 1.0
        assert cfg["rate_initializer"]["class_name"] == "Constant"
        assert cfg["rate_initializer"]["config"]["value"] == 0.5

        logistic_2 = psiz.keras.layers.Logistic.from_config(cfg)
        assert logistic_2.name == "my_logistic"
        assert logistic_2.trainable
        assert logistic_2.dtype == "float32"
        assert logistic_2.upper_initializer.value == 2.0
        assert logistic_2.midpoint_initializer.value == 1.0
        assert logistic_2.rate_initializer.value == 0.5
        assert isinstance(logistic_2.upper_constraint, keras.constraints.NonNeg)
        assert isinstance(logistic_2.midpoint_constraint, keras.constraints.NonNeg)
        assert isinstance(logistic_2.rate_constraint, keras.constraints.NonNeg)
