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
"""Test similarities module."""


import keras
import numpy as np

from psiz.keras.layers import StudentsTSimilarity


class TestStudentsT:

    def test_init_default(self):
        """Test default initialization."""
        similarity = StudentsTSimilarity()

        assert similarity.fit_tau
        assert similarity.fit_alpha

    def test_init_options_0(self):
        """Test initialization with optional arguments."""
        similarity = StudentsTSimilarity(
            fit_tau=False,
            fit_alpha=False,
            tau_initializer=keras.initializers.Constant(1.0),
            alpha_initializer=keras.initializers.Constant(1.2),
        )

        assert not similarity.fit_tau
        assert not similarity.fit_alpha

    def test_call(self):
        """Test call."""
        similarity = StudentsTSimilarity(
            tau_initializer=keras.initializers.Constant(2.0),
            alpha_initializer=keras.initializers.Constant(1.0),
        )

        d = np.array(
            [[0.68166146, 1.394038], [0.81919687, 1.25966185]], dtype="float32"
        )
        s_actual = similarity(d)
        s_actual = keras.ops.convert_to_numpy(s_actual)

        s_desired = np.array(
            [[0.68275124, 0.33974987], [0.5984142, 0.3865858]], dtype="float32"
        )
        np.testing.assert_allclose(s_actual, s_desired, rtol=1e-5)

    def test_get_config(self):
        similarity = StudentsTSimilarity()
        config = similarity.get_config()

        assert config["fit_tau"]
        assert config["fit_alpha"]

    def test_serialization(self):
        """Test serialization with weights."""
        similarity = StudentsTSimilarity()

        # Call to ensure built.
        d = np.array(
            [[0.68166146, 1.394038], [0.81919687, 1.25966185]], dtype="float32"
        )
        s0 = similarity(d)
        s0 = keras.ops.convert_to_numpy(s0)

        config = similarity.get_config()
        # OR config = keras.layers.serialize(similarity)
        weights = similarity.get_weights()

        recon_layer = StudentsTSimilarity.from_config(config)
        # OR recon_layer = keras.layers.deserialize(config)
        recon_layer.build([[None, 2], [None, 2]])
        recon_layer.set_weights(weights)
        s1 = recon_layer(d)
        s1 = keras.ops.convert_to_numpy(s1)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(similarity.tau),
            keras.ops.convert_to_numpy(recon_layer.tau),
        )
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(similarity.alpha),
            keras.ops.convert_to_numpy(recon_layer.alpha),
        )
        np.testing.assert_allclose(s0, s1)
