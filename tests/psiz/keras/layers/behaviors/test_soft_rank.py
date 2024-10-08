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
import pytest

import psiz


class TestSoftRank:

    def test_outcome_probability_v0(self):
        """Test outcome probabilty computation."""
        n_stimuli = 4
        n_dim = 2
        beta = 10.0

        percept = keras.layers.Embedding(
            n_stimuli + 1,
            n_dim,
            mask_zero=True,
            embeddings_initializer=keras.initializers.Constant(
                np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.1], [0.3, 0.1], [0.4, 0.1]])
            ),
        )
        proximity = psiz.keras.layers.Minkowski(
            rho_initializer=keras.initializers.Constant(2.0),
            w_initializer=keras.initializers.Constant(1.0),
            trainable=False,
            activation=psiz.keras.layers.ExponentialSimilarity(
                beta_initializer=keras.initializers.Constant(beta),
                tau_initializer=keras.initializers.Constant(1.0),
                gamma_initializer=keras.initializers.Constant(0.0),
                trainable=False,
            ),
        )
        softrank_2_1 = psiz.keras.layers.SoftRank(n_select=1)

        stimulus_set = np.array(
            [
                [1, 2, 3],
                [2, 3, 4],
                [2, 1, 4],
                [4, 3, 1],
            ]
        )
        z = percept(stimulus_set)
        stimuli_axis = 1
        z_q, z_r = keras.ops.split(z, [1], stimuli_axis)
        s = proximity([z_q, z_r])
        outcome_prob = softrank_2_1(s)

        # Desired outcome.
        coords_x = 0.1 * keras.ops.cast(stimulus_set, dtype="float32")
        z_q = keras.ops.take(coords_x, indices=[0], axis=1)
        z_r = keras.ops.take(coords_x, indices=[1, 2], axis=1)
        d_qr = keras.ops.abs(z_q - z_r)
        s_qr = keras.ops.exp(-beta * d_qr)
        total_s = keras.ops.sum(s_qr, axis=1, keepdims=True)
        outcome_prob_desired = s_qr / total_s
        outcome_prob_desired = keras.ops.convert_to_numpy(outcome_prob_desired)

        np.testing.assert_allclose(outcome_prob, outcome_prob_desired, rtol=1e-5)

    def test_outcome_probability_v1(self):
        """Test outcome probabilty computation.

        Use deterministic temperature.

        """
        n_stimuli = 4
        n_dim = 2
        beta = 10.0

        percept = keras.layers.Embedding(
            n_stimuli + 1,
            n_dim,
            mask_zero=True,
            embeddings_initializer=keras.initializers.Constant(
                np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.1], [0.3, 0.1], [0.4, 0.1]])
            ),
        )
        proximity = psiz.keras.layers.Minkowski(
            rho_initializer=keras.initializers.Constant(2.0),
            w_initializer=keras.initializers.Constant(1.0),
            trainable=False,
            activation=psiz.keras.layers.ExponentialSimilarity(
                beta_initializer=keras.initializers.Constant(beta),
                tau_initializer=keras.initializers.Constant(1.0),
                gamma_initializer=keras.initializers.Constant(0.0),
                trainable=False,
            ),
        )
        softrank_2_1 = psiz.keras.layers.SoftRank(
            n_select=1, temperature_initializer=keras.initializers.Constant(value=0.001)
        )

        stimulus_set = np.array(
            [
                [1, 2, 3],
                [2, 3, 4],
                [2, 1, 4],
                [4, 3, 1],
            ]
        )
        z = percept(stimulus_set)
        stimuli_axis = 1
        z_q, z_r = keras.ops.split(z, [1], stimuli_axis)
        s = proximity([z_q, z_r])
        outcome_prob = softrank_2_1(s)

        # Desired outcome.
        outcome_prob_desired = np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
            ],
            dtype="float32",
        )
        outcome_prob_desired = keras.ops.convert_to_numpy(outcome_prob_desired)
        np.testing.assert_allclose(outcome_prob, outcome_prob_desired, rtol=1e-5)

    def test_outcome_probability_v2(self):
        """Test outcome probabilty computation.

        Use rank-3 input tensor.

        """
        n_stimuli = 4
        n_dim = 2
        beta = 10.0

        percept = keras.layers.Embedding(
            n_stimuli + 1,
            n_dim,
            mask_zero=True,
            embeddings_initializer=keras.initializers.Constant(
                np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.1], [0.3, 0.1], [0.4, 0.1]])
            ),
        )
        proximity = psiz.keras.layers.Minkowski(
            rho_initializer=keras.initializers.Constant(2.0),
            w_initializer=keras.initializers.Constant(1.0),
            trainable=False,
            activation=psiz.keras.layers.ExponentialSimilarity(
                beta_initializer=keras.initializers.Constant(beta),
                tau_initializer=keras.initializers.Constant(1.0),
                gamma_initializer=keras.initializers.Constant(0.0),
                trainable=False,
            ),
        )
        softrank_2_1 = psiz.keras.layers.SoftRank(n_select=1)

        stimulus_set = np.array(
            [
                [[1, 2, 3], [1, 2, 3]],
                [[2, 3, 4], [2, 3, 4]],
                [[2, 1, 4], [2, 1, 4]],
                [[4, 3, 1], [4, 3, 1]],
            ]
        )
        z = percept(stimulus_set)
        stimuli_axis = 2
        z_q, z_r = keras.ops.split(z, [1], stimuli_axis)
        s = proximity([z_q, z_r])
        outcome_prob = softrank_2_1(s)

        # Desired outcome.
        coords_x = 0.1 * keras.ops.cast(stimulus_set, dtype="float32")
        z_q = keras.ops.take(coords_x, indices=[0], axis=stimuli_axis)
        z_r = keras.ops.take(coords_x, indices=[1, 2], axis=stimuli_axis)
        d_qr = keras.ops.abs(z_q - z_r)
        s_qr = keras.ops.exp(-beta * d_qr)
        total_s = keras.ops.sum(s_qr, axis=stimuli_axis, keepdims=True)
        outcome_prob_desired = s_qr / total_s
        outcome_prob_desired = keras.ops.convert_to_numpy(outcome_prob_desired)

        np.testing.assert_allclose(outcome_prob, outcome_prob_desired, rtol=1e-5)

    def test_outcome_probability_v4(self):
        """Test outcome probabilty computation.

        Verify placeholder behavior.

        """
        softrank_3_1 = psiz.keras.layers.SoftRank(n_select=1)

        strengths = np.array(
            [
                [1, 2, 3],
                [0, 0, 0],
                [2, 1, 4],
                [4, 3, 1],
            ],
            dtype="float32",
        )
        outcome_prob = softrank_3_1(strengths)
        outcome_prob = keras.ops.convert_to_numpy(outcome_prob)

        # Desired outcome.
        # NOTE: The key feature in this test is that we want uniform output
        # probabilites for placeholder trials because we want to avoid nans in
        # the loss computation.
        outcome_prob_desired = np.array(
            [
                [0.16666666, 0.3333333, 0.5],
                [0.33333334, 0.33333334, 0.33333334],
                [0.2857143, 0.14285715, 0.5714286],
                [0.5, 0.375, 0.125],
            ],
            dtype="float32",
        )

        np.testing.assert_allclose(outcome_prob, outcome_prob_desired, rtol=1e-5)

    def test_serialization_v0(self):
        """Test serialization."""
        # Default settings.
        rank = psiz.keras.layers.SoftRank(n_select=1, name="rs1")
        cfg = rank.get_config()
        # Verify.
        assert cfg["name"] == "rs1"
        assert cfg["trainable"]
        assert cfg["n_select"] == 1
        assert cfg["temperature_initializer"]["class_name"] == "Constant"
        assert cfg["temperature_initializer"]["config"]["value"] == 1.0

        rank2 = psiz.keras.layers.SoftRank.from_config(cfg)
        assert rank2.name == "rs1"
        assert rank2.trainable is True
        assert rank2.dtype == "float32"
        assert rank2.n_select == 1
        assert rank2.trainable
        assert rank2.temperature_initializer.value == 1.0

        # Some optional settings.
        rank = psiz.keras.layers.SoftRank(
            n_select=2,
            name="rs2",
            trainable=False,
            temperature_initializer=keras.initializers.Constant(0.01),
            temperature_constraint=keras.constraints.NonNeg(),
            temperature_regularizer=keras.regularizers.L1(l1=0.03),
        )
        cfg = rank.get_config()
        # Verify.
        assert cfg["name"] == "rs2"
        assert not cfg["trainable"]
        assert cfg["n_select"] == 2
        assert cfg["temperature_initializer"]["class_name"] == "Constant"
        assert cfg["temperature_initializer"]["config"]["value"] == 0.01
        assert cfg["temperature_constraint"]["class_name"] == "NonNeg"
        assert cfg["temperature_regularizer"]["class_name"] == "L1"

        rank2 = psiz.keras.layers.SoftRank.from_config(cfg)
        assert rank2.name == "rs2"
        assert not rank2.trainable
        assert rank2.dtype == "float32"
        assert rank2.n_select == 2
        rank2.temperature_initializer.value == 0.01
        assert isinstance(rank2.temperature_constraint, keras.constraints.NonNeg)
        assert isinstance(rank2.temperature_regularizer, keras.regularizers.L1)

    def test_bad_input_v0(self):
        """Test outcome probabilty computation.

        Verify placeholder behavior.

        """
        softrank_3_3 = psiz.keras.layers.SoftRank(n_select=3)

        strengths = np.array(
            [
                [1, 2, 3],
                [0, 0, 0],
                [2, 1, 4],
                [4, 3, 1],
            ],
            dtype="float32",
        )

        with pytest.raises(Exception) as e_info:
            softrank_3_3(strengths)
        assert e_info.type == ValueError
        assert str(e_info.value) == (
            "The argument `n_select` must be less than the number of options "
            "implied by the inputs."
        )

    def test_bad_input_v1(self):
        """Test outcome probabilty computation.

        Verify placeholder behavior.

        """
        softrank_3_4 = psiz.keras.layers.SoftRank(n_select=4)

        strengths = np.array(
            [
                [1, 2, 3],
                [0, 0, 0],
                [2, 1, 4],
                [4, 3, 1],
            ],
            dtype="float32",
        )

        with pytest.raises(Exception) as e_info:
            softrank_3_4(strengths)
        assert e_info.type == ValueError
        assert str(e_info.value) == (
            "The argument `n_select` must be less than the number of options "
            "implied by the inputs."
        )
