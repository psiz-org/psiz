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


class RankRateModelA(keras.Model):
    """A joint `SoftRank` and `Logistic` model."""

    def __init__(
        self, percept=None, proximity=None, soft_4rank2=None, rate=None, **kwargs
    ):
        """Initialize."""
        super(RankRateModelA, self).__init__(**kwargs)

        n_stimuli = 20
        n_dim = 3
        self.stimuli_axis = 1

        # Define a percept layer that will be shared across behaviors.
        if percept is None:
            percept = keras.layers.Embedding(n_stimuli + 1, n_dim, mask_zero=True)
        self.percept = percept

        # Define a proximity layer that will be shared across behaviors.
        if proximity is None:
            proximity = psiz.keras.layers.Minkowski(
                rho_initializer=keras.initializers.Constant(2.0),
                w_initializer=keras.initializers.Constant(1.0),
                trainable=False,
                activation=psiz.keras.layers.ExponentialSimilarity(
                    beta_initializer=keras.initializers.Constant(10.0),
                    tau_initializer=keras.initializers.Constant(1.0),
                    gamma_initializer=keras.initializers.Constant(0.001),
                    trainable=False,
                ),
            )
        self.proximity = proximity

        # Define a multi-output branches.
        if soft_4rank2 is None:
            soft_4rank2 = psiz.keras.layers.SoftRank(n_select=2)
        self.soft_4rank2 = soft_4rank2
        if rate is None:
            rate = psiz.keras.layers.Logistic()
        self.rate = rate

    def call(self, inputs):
        """Call."""
        # Rank branch.
        z = self.percept(inputs["given4rank2_stimulus_set"])
        z_q, z_r = keras.ops.split(z, [1], self.stimuli_axis)
        s = self.proximity([z_q, z_r])
        output_rank = self.soft_4rank2(s)

        # Rate branch.
        z = self.percept(inputs["rate2_stimulus_set"])
        z_0, z_1 = keras.ops.split(z, [1], self.stimuli_axis)
        s = self.proximity([z_0, z_1])
        output_rate = self.rate(s)

        return {"rank_branch": output_rank, "rate_branch": output_rate}

    def get_config(self):
        config = super(RankRateModelA, self).get_config()
        config.update(
            {
                "percept": self.percept,
                "proximity": self.proximity,
                "soft_4rank2": self.soft_4rank2,
                "rate": self.rate,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config["percept"] = keras.layers.deserialize(config["percept"])
        config["proximity"] = keras.layers.deserialize(config["proximity"])
        config["soft_4rank2"] = keras.layers.deserialize(config["soft_4rank2"])
        config["rate"] = keras.layers.deserialize(config["rate"])
        return cls(**config)


class RankRTModelA(keras.Model):
    """A `SoftRank` with response times model.

    Gates:
        None

    """

    def __init__(self, **kwargs):
        """Initialize."""
        super(RankRTModelA, self).__init__(**kwargs)

        n_stimuli = 20
        n_dim = 3
        self.stimuli_axis = 1

        # Define a percept layer that will be shared across behaviors.
        self.percept = keras.layers.Embedding(n_stimuli + 1, n_dim, mask_zero=True)

        # Define a proximity layer that will be shared across behaviors.
        self.proximity = psiz.keras.layers.Minkowski(
            rho_initializer=keras.initializers.Constant(2.0),
            w_initializer=keras.initializers.Constant(1.0),
            trainable=False,
            activation=psiz.keras.layers.ExponentialSimilarity(
                beta_initializer=keras.initializers.Constant(10.0),
                tau_initializer=keras.initializers.Constant(1.0),
                gamma_initializer=keras.initializers.Constant(0.001),
                trainable=False,
            ),
        )

        # Define a multi-output branches.
        self.soft_4rank1 = psiz.keras.layers.SoftRank(n_select=1)
        self.response_time = psiz.keras.layers.Logistic()

    def call(self, inputs):
        """Call."""
        # Rank branch.
        z = self.percept(inputs["given4rank1_stimulus_set"])
        z_q, z_r = keras.ops.split(z, [1], self.stimuli_axis)
        s = self.proximity([z_q, z_r])
        output_rank = self.soft_4rank1(s)

        # Response time branch.
        # Estimate response time as a function of soft rank outcome entropy.
        entropy = -keras.ops.sum(
            keras.ops.multiply(output_rank, keras.ops.log(output_rank)),
            axis=1,
            keepdims=True,
        )
        output_rt = self.response_time(entropy)

        outputs = {
            "rank_choice_branch": output_rank,
            "rank_rt_branch": output_rt,
        }
        return outputs

    def get_config(self):
        config = super(RankRTModelA, self).get_config()
        return config


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

    # Test predict.
    pred0 = model.predict(tfds)


def build_ranksim_ratesim_subclass_a():
    """Build subclassed `Model`."""
    model = RankRateModelA()
    compile_kwargs = {
        "optimizer": keras.optimizers.Adam(learning_rate=0.001),
        "loss": {
            "rank_branch": keras.losses.CategoricalCrossentropy(name="rank_loss"),
            "rate_branch": keras.losses.MeanSquaredError(name="rate_loss"),
        },
        "loss_weights": {"rank_branch": 1.0, "rate_branch": 1.0},
        "weighted_metrics": {
            "rank_branch": keras.metrics.CategoricalCrossentropy(name="cce"),
            "rate_branch": keras.metrics.MeanSquaredError(name="mse"),
        },
    }
    model.compile(**compile_kwargs)
    return model


def build_ranksim_ratesim_functional_v0():
    """Build model using functional API."""
    n_stimuli = 20
    n_dim = 3
    stimuli_axis = 1

    # Define a percept layer that will be shared across behaviors.
    percept = keras.layers.Embedding(n_stimuli + 1, n_dim, mask_zero=True)
    percept_2 = keras.layers.Embedding(
        n_stimuli + 1, n_dim, mask_zero=True
    )  # TODO decide on strategy: one shared layer or two independent

    # Define a proximity layer that will be shared across behaviors.
    proximity = psiz.keras.layers.Minkowski(
        rho_initializer=keras.initializers.Constant(2.0),
        w_initializer=keras.initializers.Constant(1.0),
        trainable=False,
        activation=psiz.keras.layers.ExponentialSimilarity(
            beta_initializer=keras.initializers.Constant(10.0),
            tau_initializer=keras.initializers.Constant(1.0),
            gamma_initializer=keras.initializers.Constant(0.001),
            trainable=False,
        ),
    )
    proximity_2 = psiz.keras.layers.Minkowski(  # TODO decide on strategy: one shared layer or two independent
        rho_initializer=keras.initializers.Constant(2.0),
        w_initializer=keras.initializers.Constant(1.0),
        trainable=False,
        activation=psiz.keras.layers.ExponentialSimilarity(
            beta_initializer=keras.initializers.Constant(10.0),
            tau_initializer=keras.initializers.Constant(1.0),
            gamma_initializer=keras.initializers.Constant(0.001),
            trainable=False,
        ),
    )

    # Define different behavioral branches.
    soft_4rank2 = psiz.keras.layers.SoftRank(n_select=2, name="rank_branch")
    rate = psiz.keras.layers.Logistic(name="rate_branch")

    inp_rank_stimulus_set = keras.Input(shape=(5,), name="given4rank2_stimulus_set")
    inp_rate_stimulus_set = keras.Input(shape=(2,), name="rate2_stimulus_set")
    inputs = {
        "given4rank2_stimulus_set": inp_rank_stimulus_set,
        "rate2_stimulus_set": inp_rate_stimulus_set,
    }

    # Rank branch.
    z_rank = percept(inputs["given4rank2_stimulus_set"])
    z_rank_q, z_rank_r = keras.ops.split(z_rank, [1], stimuli_axis)
    s_rank = proximity([z_rank_q, z_rank_r])
    output_rank = soft_4rank2(s_rank)

    # Rate branch.
    z_rate = percept_2(inputs["rate2_stimulus_set"])
    z_rate_0, z_rate_1 = keras.ops.split(z_rate, [1], stimuli_axis)
    s_rate = proximity_2([z_rate_0, z_rate_1])
    output_rate = rate(s_rate)

    outputs = {"rank_branch": output_rank, "rate_branch": output_rate}
    model = keras.Model(inputs=inputs, outputs=outputs, name="functional_rank_rate")
    compile_kwargs = {
        "optimizer": keras.optimizers.Adam(learning_rate=0.001),
        "loss": {
            "rank_branch": keras.losses.CategoricalCrossentropy(name="rank_loss"),
            "rate_branch": keras.losses.MeanSquaredError(name="rate_loss"),
        },
        "loss_weights": {"rank_branch": 1.0, "rate_branch": 1.0},
        "weighted_metrics": {
            "rank_branch": keras.metrics.CategoricalCrossentropy(name="rank_cce"),
            "rate_branch": keras.metrics.MeanSquaredError(name="rate_mse"),
        },
    }
    model.compile(**compile_kwargs)
    return model


def buld_ranksim_rt_subclass_a():
    """Build subclassed `Model`."""
    model = RankRTModelA()
    compile_kwargs = {
        "optimizer": keras.optimizers.Adam(learning_rate=0.001),
        "loss": {
            "rank_choice_branch": keras.losses.CategoricalCrossentropy(
                name="choice_loss"
            ),
            "rank_rt_branch": keras.losses.MeanSquaredError(name="rt_loss"),
        },
        "loss_weights": {"rank_choice_branch": 1.0, "rank_rt_branch": 1.0},
    }
    model.compile(**compile_kwargs)
    return model


class TestJointSoftRankRate:
    """Test using joint `SoftRank` and `Logistic` layers."""

    @pytest.mark.parametrize("is_eager", [True, False])
    def test_usage_subclass_a(self, ds_4rank2_rate2_v0, is_eager):
        """Test model using subclass API."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_4rank2_rate2_v0
        model = build_ranksim_ratesim_subclass_a()
        call_fit_evaluate_predict(model, tfds)
        keras.backend.clear_session()

    @pytest.mark.parametrize("is_eager", [True, False])
    def test_save_load_subclass_a(self, ds_4rank2_rate2_v0, is_eager, tmpdir):
        """Test serialization."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_4rank2_rate2_v0
        model = build_ranksim_ratesim_subclass_a()
        model.fit(tfds, epochs=1)
        eval0 = model.evaluate(tfds, return_dict=True)

        # Test storage.
        fp_model = Path(tmpdir) / "test_model.keras"
        model.save(fp_model)
        del model
        # Load the saved model.
        loaded = keras.models.load_model(
            fp_model,
            custom_objects={"RankRateModelA": RankRateModelA},
        )
        eval1 = loaded.evaluate(tfds, return_dict=True)

        # Test for model equality.
        assert eval0["rank_branch_cce"] == eval1["rank_branch_cce"]
        assert eval0["rate_branch_mse"] == eval1["rate_branch_mse"]

    @pytest.mark.parametrize("is_eager", [True, False])
    @pytest.mark.xfail(
        reason="Appears to be a Keras bug. Inputs are getting mixed up from Minkowski layer to softrank/logistic layers."
    )
    def test_usage_functional_v0(self, ds_4rank2_rate2_v0, is_eager):
        """Test model using functional API."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_4rank2_rate2_v0
        model = build_ranksim_ratesim_functional_v0()
        call_fit_evaluate_predict(model, tfds)
        keras.backend.clear_session()

    @pytest.mark.parametrize("is_eager", [True, False])
    @pytest.mark.xfail(reason="Not sure why failing.")
    def test_save_load_functional_v0(self, ds_4rank2_rate2_v0, is_eager, tmpdir):
        """Test serialization."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_4rank2_rate2_v0
        model = build_ranksim_ratesim_functional_v0()
        model.fit(tfds, epochs=1)
        eval0 = model.evaluate(tfds, return_dict=True)

        # Test storage.
        fp_model = Path(tmpdir) / "test_model.keras"
        model.save(fp_model)
        del model
        # Load the saved model.
        loaded = keras.models.load_model(
            fp_model,
        )
        eval1 = loaded.evaluate(tfds, return_dict=True)

        # Test for model equality.
        assert eval0["rank_branch_cce"] == eval1["rank_branch_cce"]
        assert eval0["rate_branch_mse"] == eval1["rate_branch_mse"]


class TestRankRT:
    """Test using multi-output rank similarity and response times."""

    @pytest.mark.parametrize("is_eager", [True, False])
    def test_usage_subclass_a(self, ds_4rank1_rt_v0, is_eager):
        """Test model using subclass API."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_4rank1_rt_v0
        model = buld_ranksim_rt_subclass_a()
        call_fit_evaluate_predict(model, tfds)
        keras.backend.clear_session()
