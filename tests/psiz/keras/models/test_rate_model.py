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


class RateModelA(keras.Model):
    """A `Logistic` model.

    Gates:
        None

    """

    def __init__(self, **kwargs):
        """Initialize."""
        super(RateModelA, self).__init__(**kwargs)

        n_stimuli = 30
        n_dim = 10
        self.stimuli_axis = 1

        self.percept = keras.layers.Embedding(n_stimuli + 1, n_dim, mask_zero=True)
        self.proximity = psiz.keras.layers.Minkowski(
            rho_initializer=keras.initializers.Constant(2.0),
            w_initializer=keras.initializers.Constant(1.0),
            trainable=False,
            activation=psiz.keras.layers.ExponentialSimilarity(
                trainable=False,
                beta_initializer=keras.initializers.Constant(3.0),
                tau_initializer=keras.initializers.Constant(1.0),
                gamma_initializer=keras.initializers.Constant(0.0),
            ),
        )
        self.rate = psiz.keras.layers.Logistic()

    def call(self, inputs):
        """Call."""
        z = self.percept(inputs["rate2_stimulus_set"])
        z_0, z_1 = keras.ops.split(z, [1], self.stimuli_axis)
        s = self.proximity([z_0, z_1])
        return self.rate(s)

    def get_config(self):
        config = super(RateModelA, self).get_config()
        return config

    # TODO remove when done debugging
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value.
            # The loss function is configured in `compile()`.
            loss = self.compute_loss(
                y=y,
                y_pred=y_pred,
                sample_weight=sample_weight,
            )

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply(gradients, trainable_vars)

        # Update the metrics.
        # Metrics are configured in `compile()`.
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred, sample_weight=sample_weight)

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}


class RateModelB(keras.Model):
    """A `Logistic` model.

    Gates:
        Behavior layer (BraidGate:2) has two independent behavior
            layers each with their own percept and proximity (but shared
            similarity).

    """

    def __init__(self, **kwargs):
        """Initialize."""
        super(RateModelB, self).__init__(**kwargs)

        n_stimuli = 20
        n_dim = 2
        self.stimuli_axis = 1

        shared_activation = psiz.keras.layers.ExponentialSimilarity(
            trainable=False,
            beta_initializer=keras.initializers.Constant(10.0),
            tau_initializer=keras.initializers.Constant(1.0),
            gamma_initializer=keras.initializers.Constant(0.0),
        )

        # Group 0 layers.
        percept_0 = keras.layers.Embedding(n_stimuli + 1, n_dim, mask_zero=True)
        proximity_0 = psiz.keras.layers.Minkowski(
            rho_trainable=False,
            rho_initializer=keras.initializers.Constant(2.0),
            w_initializer=keras.initializers.Constant([1.2, 0.8]),
            w_constraint=psiz.keras.constraints.NonNegNorm(scale=n_dim, p=1.0),
            activation=shared_activation,
        )
        rate_0 = psiz.keras.layers.Logistic()

        # Group 1 layers.
        percept_1 = keras.layers.Embedding(n_stimuli + 1, n_dim, mask_zero=True)
        proximity_1 = psiz.keras.layers.Minkowski(
            rho_trainable=False,
            rho_initializer=keras.initializers.Constant(2.0),
            w_initializer=keras.initializers.Constant([0.7, 1.3]),
            w_constraint=psiz.keras.constraints.NonNegNorm(scale=n_dim, p=1.0),
            activation=shared_activation,
        )
        rate_1 = psiz.keras.layers.Logistic()

        # Create behavior-level branch.
        self.braid_percept = psiz.keras.layers.BraidGate(
            subnets=[percept_0, percept_1], gating_index=-1
        )
        self.braid_proximity = psiz.keras.layers.BraidGate(
            subnets=[proximity_0, proximity_1], gating_index=-1
        )
        self.braid_rate = psiz.keras.layers.BraidGate(
            subnets=[rate_0, rate_1], gating_index=-1
        )

    def call(self, inputs):
        """Call."""
        z = self.braid_percept(
            [inputs["rate2_stimulus_set"], inputs["behavior_gate_weights"]]
        )
        z_0, z_1 = keras.ops.split(z, [1], self.stimuli_axis)
        s = self.braid_proximity([z_0, z_1, inputs["behavior_gate_weights"]])
        return self.braid_rate([s, inputs["behavior_gate_weights"]])

    def get_config(self):
        config = super(RateModelB, self).get_config()
        return config


# TODO finish or move out
# class RateCellModelA(keras.Model):
#     """A `RateSimilarityCell` model.

#     Gates:
#         None

#     """

#     def __init__(self, **kwargs):
#         """Initialize."""
#         super(RateCellModelA, self).__init__(**kwargs)

#         n_stimuli = 30
#         n_dim = 10

#         percept = keras.layers.Embedding(n_stimuli + 1, n_dim, mask_zero=True)
#         proximity = psiz.keras.layers.Minkowski(
#             rho_initializer=keras.initializers.Constant(2.0),
#             w_initializer=keras.initializers.Constant(1.0),
#             activation=psiz.keras.layers.ExponentialSimilarity(
#                 trainable=False,
#                 beta_initializer=keras.initializers.Constant(3.0),
#                 tau_initializer=keras.initializers.Constant(1.0),
#                 gamma_initializer=keras.initializers.Constant(0.0),
#             ),
#             trainable=False,
#         )
#         rate_cell = psiz.keras.layers.RateSimilarityCell(percept=percept, kernel=proximity)
#         rnn = keras.layers.RNN(rate_cell, return_sequences=True)
#         self.behavior = rnn

#     def call(self, inputs):
#         """Call."""
#         return self.behavior(inputs)

#     def get_config(self):
#         config = super(RateCellModelA, self).get_config()
#         return config


def build_ratesim_subclass_a():
    """Build subclassed `Model`."""
    model = RateModelA()
    compile_kwargs = {
        "loss": keras.losses.MeanSquaredError(),
        "optimizer": keras.optimizers.Adam(learning_rate=0.001),
        "weighted_metrics": [keras.metrics.MeanSquaredError(name="mse")],
    }
    model.compile(**compile_kwargs)
    return model


def build_ratesim_functional_v0():
    """Build model using functional API."""
    n_stimuli = 30
    n_dim = 10
    stimuli_axis = 1

    percept = keras.layers.Embedding(n_stimuli + 1, n_dim, mask_zero=True)
    proximity = psiz.keras.layers.Minkowski(
        rho_initializer=keras.initializers.Constant(2.0),
        w_initializer=keras.initializers.Constant(1.0),
        trainable=False,
        activation=psiz.keras.layers.ExponentialSimilarity(
            trainable=False,
            beta_initializer=keras.initializers.Constant(3.0),
            tau_initializer=keras.initializers.Constant(1.0),
            gamma_initializer=keras.initializers.Constant(0.0),
        ),
    )
    rate = psiz.keras.layers.Logistic()

    inp_stimulus_set = keras.Input(shape=(2,), name="rate2_stimulus_set")
    inputs = {
        "rate2_stimulus_set": inp_stimulus_set,
    }
    z = percept(inputs["rate2_stimulus_set"])
    z_0, z_1 = keras.ops.split(z, [1], stimuli_axis)
    s = proximity([z_0, z_1])
    outputs = rate(s)
    model = keras.Model(inputs=inputs, outputs=outputs, name="functional_rate")
    compile_kwargs = {
        "loss": keras.losses.MeanSquaredError(),
        "optimizer": keras.optimizers.Adam(learning_rate=0.001),
        "weighted_metrics": [keras.metrics.MeanSquaredError(name="mse")],
    }
    model.compile(**compile_kwargs)
    return model


def build_ratesim_subclass_b():
    """Build subclassed `Model`."""
    model = RateModelB()
    compile_kwargs = {
        "loss": keras.losses.MeanSquaredError(),
        "optimizer": keras.optimizers.Adam(learning_rate=0.001),
        "weighted_metrics": [keras.metrics.MeanSquaredError(name="mse")],
    }
    model.compile(**compile_kwargs)
    return model


# TODO finish or move out
# def build_ratesimcell_subclass_a():
#     """Build subclassed `Model`."""
#     model = RateCellModelA()
#     compile_kwargs = {
#         "loss": keras.losses.MeanSquaredError(),
#         "optimizer": keras.optimizers.Adam(learning_rate=0.001),
#         "weighted_metrics": [keras.metrics.MeanSquaredError(name="mse")],
#     }
#     model.compile(**compile_kwargs)
#     return model


# TODO finish or move out
# def build_ratesimcell_functional_v0():
#     """Build model useing functional API."""
#     n_stimuli = 30
#     n_dim = 10

#     percept = keras.layers.Embedding(n_stimuli + 1, n_dim, mask_zero=True)
#     proximity = psiz.keras.layers.Minkowski(
#         rho_initializer=keras.initializers.Constant(2.0),
#         w_initializer=keras.initializers.Constant(1.0),
#         activation=psiz.keras.layers.ExponentialSimilarity(
#             trainable=False,
#             beta_initializer=keras.initializers.Constant(3.0),
#             tau_initializer=keras.initializers.Constant(1.0),
#             gamma_initializer=keras.initializers.Constant(0.0),
#         ),
#         trainable=False,
#     )
#     rate_cell = psiz.keras.layers.RateSimilarityCell(percept=percept, kernel=proximity)
#     rnn = keras.layers.RNN(rate_cell, return_sequences=True)

#     inp_stimulus_set = keras.Input(
#         shape=(
#             None,
#             2,
#         ),
#         name="rate2_stimulus_set",
#     )
#     inputs = {
#         "rate2_stimulus_set": inp_stimulus_set,
#     }
#     outputs = rnn(inputs)
#     model = keras.Model(inputs=inputs, outputs=outputs, name="functional_rate")
#     compile_kwargs = {
#         "loss": keras.losses.MeanSquaredError(),
#         "optimizer": keras.optimizers.Adam(learning_rate=0.001),
#         "weighted_metrics": [keras.metrics.MeanSquaredError(name="mse")],
#     }
#     model.compile(**compile_kwargs)
#     return model


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


class TestLogistic:
    """Test using `Logistic` layer."""

    @pytest.mark.parametrize("is_eager", [True, False])
    def test_usage_subclass_a(self, ds_rate2_v0, is_eager):
        """Test subclass model, one group."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_rate2_v0
        model = build_ratesim_subclass_a()
        call_fit_evaluate_predict(model, tfds)
        keras.backend.clear_session()

    @pytest.mark.parametrize("is_eager", [True, False])
    def test_save_load_subclass_a(self, ds_rate2_v0, is_eager, tmpdir):
        """Test serialization."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_rate2_v0
        model = build_ratesim_subclass_a()
        model.fit(tfds, epochs=1)
        eval0 = model.evaluate(tfds)

        # Test storage.
        fp_model = Path(tmpdir) / "test_model.keras"
        model.save(fp_model)
        del model
        # Load the saved model.
        loaded = keras.models.load_model(
            fp_model,
            custom_objects={"RateModelA": RateModelA},
        )
        eval1 = loaded.evaluate(tfds)

        # Test for model equality.
        assert eval0[0] == eval1[0]
        assert eval0[1] == eval1[1]

    @pytest.mark.parametrize("is_eager", [True, False])
    def test_usage_functional_v0(self, ds_rate2_v0, is_eager):
        """Test model using functional API."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_rate2_v0
        model = build_ratesim_functional_v0()
        call_fit_evaluate_predict(model, tfds)
        keras.backend.clear_session()

    @pytest.mark.parametrize("is_eager", [True, False])
    def test_save_load_functional_v0(self, ds_rate2_v0, is_eager, tmpdir):
        """Test serialization."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_rate2_v0
        model = build_ratesim_functional_v0()
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

    @pytest.mark.parametrize("is_eager", [True, False])
    def test_usage_subclass_b(self, ds_rate2_v1, is_eager):
        """Test subclass model, one group."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_rate2_v1
        model = build_ratesim_subclass_b()
        call_fit_evaluate_predict(model, tfds)
        keras.backend.clear_session()

    @pytest.mark.parametrize("is_eager", [True, False])
    def test_usage_subclass_a_v2(self, ds_rate2_v2, is_eager):
        """Test subclass model, one group."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_rate2_v2
        model = build_ratesim_subclass_a()
        call_fit_evaluate_predict(model, tfds)
        keras.backend.clear_session()


# TODO finish or move out
# class TestRateSimilarityCell:
#     """Test using `RateSimilarityCell` layer."""

#     @pytest.mark.parametrize("is_eager", [True, False])
#     @pytest.mark.xfail(reason="Keras v3 RNN requires single input tensor.")
#     def test_usage_subclass_a(self, ds_time_rate2_v0, is_eager):
#         """Test subclass model, one group."""
#         tf.config.run_functions_eagerly(is_eager)

#         tfds = ds_time_rate2_v0
#         model = build_ratesimcell_subclass_a()
#         call_fit_evaluate_predict(model, tfds)
#         keras.backend.clear_session()

#     @pytest.mark.parametrize("is_eager", [True, False])
#     @pytest.mark.xfail(reason="Keras v3 RNN requires single input tensor.")
#     def test_save_load_subclass_a(
#         self, ds_time_rate2_v0, is_eager, tmpdir
#     ):
#         """Test serialization."""
#         tf.config.run_functions_eagerly(is_eager)

#         tfds = ds_time_rate2_v0
#         model = build_ratesimcell_subclass_a()
#         model.fit(tfds, epochs=1)
#         eval0 = model.evaluate(tfds)

#         # Test storage.
#         fp_model = Path(tmpdir) / "test_model.keras"
#         model.save(fp_model)
#         del model
#         # Load the saved model.
#         loaded = keras.models.load_model(
#             fp_model, custom_objects={"RateCellModelA": RateCellModelA},
#         )
#         eval1 = loaded.evaluate(tfds)

#         # Test for model equality.
#         assert eval0[0] == eval1[0]
#         assert eval0[1] == eval1[1]

#     @pytest.mark.parametrize("is_eager", [True, False])
#     @pytest.mark.xfail(reason="Keras v3 RNN requires single input tensor.")
#     def test_usage_functional_v0(self, ds_time_rate2_v0, is_eager):
#         """Test model using functional API."""
#         tf.config.run_functions_eagerly(is_eager)

#         tfds = ds_time_rate2_v0
#         model = build_ratesimcell_functional_v0()
#         call_fit_evaluate_predict(model, tfds)
#         keras.backend.clear_session()

#     @pytest.mark.parametrize("is_eager", [True, False])
#     @pytest.mark.xfail(reason="Keras v3 RNN requires single input tensor.")
#     def test_save_load_functional_v0(self, ds_time_rate2_v0, is_eager, tmpdir):
#         """Test serialization."""
#         tf.config.run_functions_eagerly(is_eager)

#         tfds = ds_time_rate2_v0
#         model = build_ratesimcell_functional_v0()
#         model.fit(tfds, epochs=1)
#         eval0 = model.evaluate(tfds)

#         # Test storage.
#         fp_model = Path(tmpdir) / "test_model.keras"
#         model.save(fp_model)
#         del model
#         # Load the saved model.
#         loaded = keras.models.load_model(fp_model, )
#         eval1 = loaded.evaluate(tfds)

#         # Test for model equality.
#         assert eval0[0] == eval1[0]
#         assert eval0[1] == eval1[1]
