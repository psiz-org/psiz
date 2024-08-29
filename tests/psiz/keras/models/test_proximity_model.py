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
"""Module for testing models."""


from pathlib import Path

import numpy as np
import keras
import pytest
import tensorflow as tf

import psiz


class ProximityModelA(keras.Model):
    """A proximity model.

    Gates:
        None

    """

    def __init__(self, proximity=None, **kwargs):
        """Initialize."""
        super(ProximityModelA, self).__init__(**kwargs)

        n_stimuli = 30
        n_dim = 10
        self.stimuli_axis = 1

        self.percept = keras.layers.Embedding(n_stimuli + 1, n_dim, mask_zero=True)
        if proximity is None:
            proximity = psiz.keras.layers.Minkowski(
                rho_initializer=keras.initializers.Constant(2.0),
                w_initializer=keras.initializers.Constant(1.0),
                trainable=False,
            )
        self.proximity = proximity

    def call(self, inputs):
        """Call."""
        z = self.percept(inputs["rate2_stimulus_set"])
        z_0, z_1 = keras.ops.split(z, [1], self.stimuli_axis)
        s = self.proximity([z_0, z_1])
        return s

    def get_config(self):
        config = super(ProximityModelA, self).get_config()
        return config


def build_proximity_subclass_a(proximity_layer):
    """Build subclassed `Model`."""
    model = ProximityModelA(proximity=proximity_layer)
    compile_kwargs = {
        "loss": keras.losses.MeanSquaredError(),
        "optimizer": keras.optimizers.Adam(learning_rate=0.001),
        "weighted_metrics": [keras.metrics.MeanSquaredError(name="mse")],
    }
    model.compile(**compile_kwargs)
    return model


def call_fit_evaluate_predict(model, tfds):
    """Simple test of call, fit, evaluate, and predict."""
    # Test isolated call.
    for data in tfds:
        x, y, sample_weight = keras.utils.unpack_x_y_sample_weight(data)
        y_pred = model(x, training=False)

    # Test fit.
    history = model.fit(tfds, epochs=3)
    assert not np.any(np.isnan(history.history["loss"]))

    # Test evaluate.
    eval0 = model.evaluate(tfds)
    assert not np.any(np.isnan(eval0))

    # Test predict.
    pred0 = model.predict(tfds)
    assert not np.any(np.isnan(pred0))


class TestProximity:
    """Test proximity model."""

    @pytest.mark.parametrize(
        "proximity_layer",
        [
            psiz.keras.layers.Minkowski(
                rho_initializer=keras.initializers.Constant(2.0),
                w_initializer=keras.initializers.Constant(1.0),
                trainable=False,
            ),
            psiz.keras.layers.CosineSimilarity(trainable=False),
            psiz.keras.layers.InnerProduct(trainable=False),
        ],
    )
    @pytest.mark.parametrize("is_eager", [True, False])
    def test_usage_subclass_a_v2(self, ds_rate2_v2, proximity_layer, is_eager):
        """Test subclass model, one group."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_rate2_v2
        model = build_proximity_subclass_a(proximity_layer)
        call_fit_evaluate_predict(model, tfds)
        keras.backend.clear_session()
