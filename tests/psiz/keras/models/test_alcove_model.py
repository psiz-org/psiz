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

import keras
import pytest
import tensorflow as tf

import psiz


class ALCOVEModelA(keras.Model):
    """An `ALCOVECell` model.

    Gates:
        None

    """

    def __init__(self, cell=None, **kwargs):
        """Initialize."""
        super(ALCOVEModelA, self).__init__(**kwargs)

        n_stimuli = 20
        n_dim = 4
        n_output = 3

        if cell is None:
            percept = keras.layers.Embedding(
                n_stimuli + 1,
                n_dim,
                mask_zero=True,
                trainable=False,
            )
            similarity = psiz.keras.layers.ExponentialSimilarity(
                beta_initializer=keras.initializers.Constant(3.0),
                tau_initializer=keras.initializers.Constant(1.0),
                gamma_initializer=keras.initializers.Constant(0.0),
                trainable=False,
            )
            cell = psiz.keras.layers.ALCOVECell(
                n_output,
                percept=percept,
                similarity=similarity,
                rho_initializer=keras.initializers.Constant(2.0),
                temperature_initializer=keras.initializers.Constant(1.0),
                lr_attention_initializer=keras.initializers.Constant(0.03),
                lr_association_initializer=keras.initializers.Constant(0.03),
                trainable=False,
            )
        self.cell = cell

        rnn = keras.layers.RNN(cell, return_sequences=True, stateful=False)
        self.behavior = rnn

    def call(self, inputs):
        """Call."""
        return self.behavior(inputs)

    def get_config(self):
        config = super(ALCOVEModelA, self).get_config()
        config.update({"cell": self.cell})
        return config

    @classmethod
    def from_config(cls, config):
        config["cell"] = keras.layers.deserialize(config["cell"])
        return cls(**config)


def call_fit_evaluate_predict(model, tfds):
    """Simple test of call, fit, evaluate, and predict."""
    # Test isolated call.
    for data in tfds:
        x, y, sample_weight = keras.utils.unpack_x_y_sample_weight(data)
        y_pred = model(x, training=False)

    # Test fit.
    model.fit(tfds, epochs=3)

    # Test evaluate.
    eval0 = model.evaluate(tfds)
    # assert not np.isnan(eval0)  TODO make work for returned nan or array of values

    # Test predict.
    pred0 = model.predict(tfds)
    # assert not np.isnan(eval0)  TODO make work for returned nan or array of values


def build_alcove_subclass_a():
    """Build subclassed `Model`."""
    model = ALCOVEModelA()
    compile_kwargs = {
        "loss": keras.losses.CategoricalCrossentropy(),
        "optimizer": keras.optimizers.Adam(learning_rate=0.001),
        "weighted_metrics": [keras.metrics.CategoricalAccuracy(name="accuracy")],
    }
    model.compile(**compile_kwargs)
    return model


def build_alcove_functional_v0():
    """Build model using functional API."""
    n_stimuli = 20
    n_dim = 4
    n_output = 3

    percept = keras.layers.Embedding(
        n_stimuli + 1,
        n_dim,
        mask_zero=True,
        trainable=False,
    )
    similarity = psiz.keras.layers.ExponentialSimilarity(
        beta_initializer=keras.initializers.Constant(3.0),
        tau_initializer=keras.initializers.Constant(1.0),
        gamma_initializer=keras.initializers.Constant(0.0),
        trainable=False,
    )
    cell = psiz.keras.layers.ALCOVECell(
        n_output,
        percept=percept,
        similarity=similarity,
        rho_initializer=keras.initializers.Constant(2.0),
        temperature_initializer=keras.initializers.Constant(1.0),
        lr_attention_initializer=keras.initializers.Constant(0.03),
        lr_association_initializer=keras.initializers.Constant(0.03),
        trainable=False,
    )
    rnn = keras.layers.RNN(cell, return_sequences=True, stateful=False)

    inp_stimulus_set = keras.Input(shape=(None, 1), name="categorize_stimulus_set")
    inp_objective_query_label = keras.Input(
        shape=(None, n_output), name="categorize_objective_query_label"
    )
    inputs = {
        "categorize_stimulus_set": inp_stimulus_set,
        "categorize_objective_query_label": inp_objective_query_label,
    }
    outputs = rnn(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs, name="functional_alcove")
    compile_kwargs = {
        "loss": keras.losses.CategoricalCrossentropy(),
        "optimizer": keras.optimizers.Adam(learning_rate=0.001),
        "weighted_metrics": [keras.metrics.CategoricalAccuracy(name="accuracy")],
    }
    model.compile(**compile_kwargs)
    return model


class TestALCOVECell:
    """Test using `ALCOVECell` layer."""

    @pytest.mark.parametrize("is_eager", [True, False])
    @pytest.mark.xfail(reason="Keras v3 RNN requires single input tensor.")
    def test_usage_subclass_a(self, ds_time_categorize_v0, is_eager):
        """Test subclassed model, one group."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_time_categorize_v0
        model = build_alcove_subclass_a()
        call_fit_evaluate_predict(model, tfds)
        keras.backend.clear_session()

    @pytest.mark.parametrize("is_eager", [True, False])
    @pytest.mark.xfail(reason="Keras v3 RNN requires single input tensor.")
    def test_save_load_subclass_a(self, ds_time_categorize_v0, is_eager, tmpdir):
        """Test serialization."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_time_categorize_v0
        model = build_alcove_subclass_a()
        model.fit(tfds, epochs=1)
        eval0 = model.evaluate(tfds)

        # Test storage.
        fp_model = Path(tmpdir) / "test_model.keras"
        model.save(fp_model)
        del model
        # Load the saved model.
        loaded = keras.models.load_model(
            fp_model,
            custom_objects={"ALCOVEModelA": ALCOVEModelA},
        )
        eval1 = loaded.evaluate(tfds)

        # Test for model equality.
        assert eval0[0] == eval1[0]
        assert eval0[1] == eval1[1]

    @pytest.mark.parametrize("is_eager", [True, False])
    @pytest.mark.xfail(reason="Keras v3 RNN requires single input tensor.")
    def test_usage_functional_v0(self, ds_time_categorize_v0, is_eager):
        """Test model using functional API."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_time_categorize_v0
        model = build_alcove_functional_v0()
        call_fit_evaluate_predict(model, tfds)
        keras.backend.clear_session()

    @pytest.mark.parametrize("is_eager", [True, False])
    @pytest.mark.xfail(reason="Keras v3 RNN requires single input tensor.")
    def test_save_load_functional_v0(self, ds_time_categorize_v0, is_eager, tmpdir):
        """Test serialization."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_time_categorize_v0
        model = build_alcove_functional_v0()
        model.fit(tfds, epochs=1)
        eval0 = model.evaluate(tfds)

        # Test storage.
        fp_model = Path(tmpdir) / "test_model.keras"
        model.save(fp_model)
        del model
        # Load the saved model.
        loaded = keras.models.load_model(
            fp_model,
        )
        eval1 = loaded.evaluate(tfds)

        # Test for model equality.
        assert eval0[0] == eval1[0]
        assert eval0[1] == eval1[1]
