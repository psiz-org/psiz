# -*- coding: utf-8 -*-
# Copyright 2022 The PsiZ Authors. All Rights Reserved.
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
import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

import psiz


class RankModelA(keras.Model):
    """A `RankSimilarity` model.

    Gates:
        None

    """

    def __init__(self, percept=None, proximity=None, soft_4rank1=None, **kwargs):
        """Initialize."""
        super(RankModelA, self).__init__(**kwargs)

        n_stimuli = 20
        n_dim = 3
        self.stimuli_axis = 1

        if percept is None:
            percept = keras.layers.Embedding(
                n_stimuli + 1,
                n_dim,
                mask_zero=True,
            )
        self.percept = percept

        if proximity is None:
            proximity = psiz.keras.layers.Minkowski(
                rho_initializer=keras.initializers.Constant(2.0),
                w_initializer=keras.initializers.Constant(1.0),
                trainable=False,
                activation=psiz.keras.layers.ExponentialSimilarity(
                    beta_initializer=keras.initializers.Constant(10.0),
                    tau_initializer=keras.initializers.Constant(1.0),
                    gamma_initializer=keras.initializers.Constant(0.0),
                    trainable=False,
                ),
            )
        self.proximity = proximity

        if soft_4rank1 is None:
            soft_4rank1 = psiz.keras.layers.SoftRank(n_select=1)
        self.soft_4rank1 = soft_4rank1

    def call(self, inputs):
        """Call."""
        z = self.percept(inputs["given4rank1_stimulus_set"])
        z_q, z_r = keras.ops.split(z, [1], self.stimuli_axis)
        s = self.proximity([z_q, z_r])
        return self.soft_4rank1(s)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "percept": self.percept,
                "proximity": self.proximity,
                "soft_4rank1": self.soft_4rank1,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config["percept"] = keras.layers.deserialize(config["percept"])
        config["proximity"] = keras.layers.deserialize(config["proximity"])
        config["soft_4rank1"] = keras.layers.deserialize(config["soft_4rank1"])
        return cls(**config)


class RankModelB(keras.Model):
    """A `RankSimilarity` model.

    Gates:
        Kernel layer (BraidGate:2) with shared similarity layer.

    """

    def __init__(self, percept=None, braid_proximity=None, soft_4rank1=None, **kwargs):
        """Initialize."""
        super(RankModelB, self).__init__(**kwargs)

        n_stimuli = 30
        n_dim = 10
        self.stimuli_axis = 1

        if percept is None:
            percept = keras.layers.Embedding(n_stimuli + 1, n_dim, mask_zero=True)
        self.percept = percept

        if braid_proximity is None:
            shared_activation = psiz.keras.layers.ExponentialSimilarity(
                beta_initializer=keras.initializers.Constant(10.0),
                tau_initializer=keras.initializers.Constant(1.0),
                gamma_initializer=keras.initializers.Constant(0.0),
                trainable=False,
            )
            # Define group-specific kernels.
            proximity_0 = psiz.keras.layers.Minkowski(
                rho_initializer=keras.initializers.Constant(2.0),
                w_initializer=keras.initializers.Constant(
                    [1.2, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
                ),
                trainable=False,
                activation=shared_activation,
            )
            proximity_1 = psiz.keras.layers.Minkowski(
                rho_initializer=keras.initializers.Constant(2.0),
                w_initializer=keras.initializers.Constant(
                    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 1.2]
                ),
                trainable=False,
                activation=shared_activation,
            )
            braid_proximity = psiz.keras.layers.BraidGate(
                subnets=[proximity_0, proximity_1], gating_index=-1
            )
        self.braid_proximity = braid_proximity

        if soft_4rank1 is None:
            soft_4rank1 = psiz.keras.layers.SoftRank(n_select=1)
        self.soft_4rank1 = soft_4rank1

    def call(self, inputs):
        """Call."""
        z = self.percept(inputs["given4rank1_stimulus_set"])
        z_q, z_r = keras.ops.split(z, [1], self.stimuli_axis)
        s = self.braid_proximity([z_q, z_r, inputs["kernel_gate_weights"]])
        return self.soft_4rank1(s)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "percept": self.percept,
                "braid_proximity": self.braid_proximity,
                "soft_4rank1": self.soft_4rank1,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config["percept"] = keras.layers.deserialize(config["percept"])
        config["braid_proximity"] = keras.layers.deserialize(config["braid_proximity"])
        config["soft_4rank1"] = keras.layers.deserialize(config["soft_4rank1"])
        return cls(**config)


class RankModelC(keras.Model):
    """A `RankSimilarity` model.

    Gates:
        Percept layer (BraidGate:2)
        Kernel layer (BraidGate:2) with shared similarity layer.

    """

    def __init__(self, **kwargs):
        """Initialize."""
        super(RankModelC, self).__init__(**kwargs)

        n_stimuli = 30
        n_dim = 2
        self.stimuli_axis = 1

        # Define group-specific percept layers.
        percept_0 = keras.layers.Embedding(n_stimuli + 1, n_dim, mask_zero=True)
        percept_1 = keras.layers.Embedding(n_stimuli + 1, n_dim, mask_zero=True)
        self.braid_percept = psiz.keras.layers.BraidGate(
            subnets=[percept_0, percept_1], gating_index=-1
        )

        # Define group-specific proximity layers.
        shared_activation = psiz.keras.layers.ExponentialSimilarity(
            trainable=False,
            beta_initializer=keras.initializers.Constant(10.0),
            tau_initializer=keras.initializers.Constant(1.0),
            gamma_initializer=keras.initializers.Constant(0.0),
        )
        proximity_0 = psiz.keras.layers.Minkowski(
            rho_trainable=False,
            rho_initializer=keras.initializers.Constant(2.0),
            w_initializer=keras.initializers.Constant([1.2, 0.8]),
            w_constraint=psiz.keras.constraints.NonNegNorm(scale=n_dim, p=1.0),
            activation=shared_activation,
        )
        proximity_1 = psiz.keras.layers.Minkowski(
            rho_trainable=False,
            rho_initializer=keras.initializers.Constant(2.0),
            w_initializer=keras.initializers.Constant([0.7, 1.3]),
            w_constraint=psiz.keras.constraints.NonNegNorm(scale=n_dim, p=1.0),
            activation=shared_activation,
        )
        self.braid_proximity = psiz.keras.layers.BraidGate(
            subnets=[proximity_0, proximity_1], gating_index=-1
        )

        self.soft_4rank1 = psiz.keras.layers.SoftRank(n_select=1)

    def call(self, inputs):
        """Call."""
        z = self.braid_percept(
            [inputs["given4rank1_stimulus_set"], inputs["percept_gate_weights"]]
        )
        z_q, z_r = keras.ops.split(z, [1], self.stimuli_axis)
        s = self.braid_proximity([z_q, z_r, inputs["kernel_gate_weights"]])
        return self.soft_4rank1(s)

    def get_config(self):
        config = super(RankModelC, self).get_config()
        return config


class RankModelD(keras.Model):
    """A `RankSimilarity` model.

    Gates:
        Percept layer (BraidGate:2, BraidGate:2)

    """

    def __init__(self, **kwargs):
        """Initialize."""
        super(RankModelD, self).__init__(**kwargs)

        n_stimuli = 30
        n_dim = 2
        self.stimuli_axis = 1

        # Define heirarchical percept layers.
        percept_0 = keras.layers.Embedding(n_stimuli + 1, n_dim, mask_zero=True)
        percept_1 = keras.layers.Embedding(n_stimuli + 1, n_dim, mask_zero=True)
        percept_01 = psiz.keras.layers.BraidGate(
            subnets=[percept_0, percept_1], gating_index=-1, name="percept_01"
        )

        percept_2 = keras.layers.Embedding(n_stimuli + 1, n_dim, mask_zero=True)
        percept_3 = keras.layers.Embedding(n_stimuli + 1, n_dim, mask_zero=True)
        percept_23 = psiz.keras.layers.BraidGate(
            subnets=[percept_2, percept_3], gating_index=-1, name="percept_23"
        )

        self.braid_percept = psiz.keras.layers.BraidGate(
            subnets=[percept_01, percept_23], gating_index=-1, name="percept"
        )

        # Define proximity.
        self.proximity = psiz.keras.layers.Minkowski(
            rho_trainable=False,
            rho_initializer=keras.initializers.Constant(2.0),
            w_initializer=keras.initializers.Constant([1.2, 0.8]),
            w_constraint=psiz.keras.constraints.NonNegNorm(scale=n_dim, p=1.0),
            activation=psiz.keras.layers.ExponentialSimilarity(
                trainable=False,
                beta_initializer=keras.initializers.Constant(10.0),
                tau_initializer=keras.initializers.Constant(1.0),
                gamma_initializer=keras.initializers.Constant(0.0),
            ),
        )
        self.soft_4rank1 = psiz.keras.layers.SoftRank(n_select=1)

    def call(self, inputs):
        """Call."""
        # NOTE: Because the BraidGates were initialized with the argument
        # `gating_index=-1`, the corresponding gate weights are listed in the
        # reverse order that they are consumed. In this case, the last item
        # `percept_gate_weights_0`, is consumed first and used in routing at
        # the outermost braid gate.
        z = self.braid_percept(
            [
                inputs["given4rank1_stimulus_set"],
                inputs["percept_gate_weights_1"],
                inputs["percept_gate_weights_0"],
            ]
        )
        z_q, z_r = keras.ops.split(z, [1], self.stimuli_axis)
        s = self.proximity([z_q, z_r])
        return self.soft_4rank1(s)

    def get_config(self):
        config = super(RankModelD, self).get_config()
        return config


class RankModelE(keras.Model):
    """A `RankSimilarity` model.

    Gates:
        None

    """

    def __init__(self, **kwargs):
        """Initialize."""
        super(RankModelE, self).__init__(**kwargs)

        n_stimuli = 20
        n_dim = 3
        self.stimuli_axis = 1

        self.percept = keras.layers.Embedding(n_stimuli + 1, n_dim, mask_zero=True)
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
        self.soft_2rank1 = psiz.keras.layers.SoftRank(n_select=1)

    def call(self, inputs):
        """Call."""
        z = self.percept(inputs["given2rank1_stimulus_set"])
        z_q, z_r = keras.ops.split(z, [1], self.stimuli_axis)
        s = self.proximity([z_q, z_r])
        return self.soft_2rank1(s)

    def get_config(self):
        config = super(RankModelE, self).get_config()
        return config


class RankModelF(keras.Model):
    """A `RankSimilarity` model.

    Gates:
        None

    """

    def __init__(self, **kwargs):
        """Initialize."""
        super(RankModelF, self).__init__(**kwargs)

        n_stimuli = 20
        n_dim = 3
        self.stimuli_axis = 1

        self.percept = keras.layers.Embedding(n_stimuli + 1, n_dim, mask_zero=True)

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

        self.soft_8rank2 = psiz.keras.layers.SoftRank(n_select=2)

    def call(self, inputs):
        """Call."""
        z = self.percept(inputs["given8rank2_stimulus_set"])
        z_q, z_r = keras.ops.split(z, [1], self.stimuli_axis)
        s = self.proximity([z_q, z_r])
        return self.soft_8rank2(s)

    def get_config(self):
        config = super(RankModelF, self).get_config()
        return config


class MultiRankModelA(keras.Model):
    """A `RankSimilarity` model.

    Gates:
        BranchGate

    """

    def __init__(
        self, percept=None, proximity=None, soft_2rank1=None, soft_8rank2=None, **kwargs
    ):
        """Initialize."""
        super(MultiRankModelA, self).__init__(**kwargs)

        n_stimuli = 20
        n_dim = 3
        self.stimuli_axis = 1

        if percept is None:
            percept = keras.layers.Embedding(n_stimuli + 1, n_dim, mask_zero=True)
        self.percept = percept

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
        if soft_2rank1 is None:
            soft_2rank1 = psiz.keras.layers.SoftRank(n_select=1, trainable=False)
        self.soft_2rank1 = soft_2rank1
        if soft_8rank2 is None:
            soft_8rank2 = psiz.keras.layers.SoftRank(n_select=2, trainable=False)
        self.soft_8rank2 = soft_8rank2

    def call(self, inputs):
        """Call."""
        # The 2-rank-1 branch.
        z = self.percept(inputs["given2rank1_stimulus_set"])
        z_q, z_r = keras.ops.split(z, [1], self.stimuli_axis)
        s_2rank1 = self.proximity([z_q, z_r])
        prob_2rank1 = self.soft_2rank1(s_2rank1)

        # The 8-rank-2 branch.
        z = self.percept(inputs["given8rank2_stimulus_set"])
        z_q, z_r = keras.ops.split(z, [1], self.stimuli_axis)
        s_8rank2 = self.proximity([z_q, z_r])
        prob_8rank2 = self.soft_8rank2(s_8rank2)

        return {"given2rank1": prob_2rank1, "given8rank2": prob_8rank2}

    def get_config(self):
        config = super(MultiRankModelA, self).get_config()
        config.update(
            {
                "percept": self.percept,
                "proximity": self.proximity,
                "soft_2rank1": self.soft_2rank1,
                "soft_8rank2": self.soft_8rank2,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config["percept"] = keras.layers.deserialize(config["percept"])
        config["proximity"] = keras.layers.deserialize(config["proximity"])
        config["soft_2rank1"] = keras.layers.deserialize(config["soft_2rank1"])
        config["soft_8rank2"] = keras.layers.deserialize(config["soft_8rank2"])
        return cls(**config)

    # TODO remove when done debugging.
    # def train_step(self, data):
    #     # Unpack the data. Its structure depends on your model and
    #     # on what you pass to `fit()`.
    #     if len(data) == 3:
    #         x, y, sample_weight = data
    #     else:
    #         sample_weight = None
    #         x, y = data

    #     with tf.GradientTape() as tape:
    #         y_pred = self(x, training=True)  # Forward pass
    #         # Compute the loss value
    #         # (the loss function is configured in `compile()`)
    #         loss = self.compute_loss(y=y, y_pred=y_pred, sample_weight=sample_weight)

    #     # Compute gradients
    #     trainable_vars = self.trainable_variables
    #     gradients = tape.gradient(loss, trainable_vars)

    #     # Update weights
    #     self.optimizer.apply(gradients, trainable_vars)

    #     # Update metrics (includes the metric that tracks the loss)
    #     for metric in self.metrics:
    #         if metric.name == "loss":
    #             metric.update_state(loss)
    #         else:
    #             metric.update_state(y, y_pred)

    #     # Return a dict mapping metric names to current value
    #     return {m.name: m.result() for m in self.metrics}


# TODO finish or move out
# class RankCellModelA(keras.Model):
#     """A `RankSimilarityCell` model.

#     Gates:
#         None

#     """

#     def __init__(self, **kwargs):
#         """Initialize."""
#         super(RankCellModelA, self).__init__(**kwargs)

#         n_stimuli = 20
#         n_dim = 3
#         self.stimuli_axis = 2  # NOTE: Position different because of timestep axis.

#         self.percept = keras.layers.Embedding(n_stimuli + 1, n_dim, mask_zero=True)
#         self.proximity = psiz.keras.layers.Minkowski(
#             rho_initializer=keras.initializers.Constant(2.0),
#             w_initializer=keras.initializers.Constant(1.0),
#             trainable=False,
#             activation=psiz.keras.layers.ExponentialSimilarity(
#                 beta_initializer=keras.initializers.Constant(10.0),
#                 tau_initializer=keras.initializers.Constant(1.0),
#                 gamma_initializer=keras.initializers.Constant(0.001),
#                 trainable=False,
#             ),
#         )
#         rankcell_8_2 = psiz.keras.layers.SoftRankCell(n_select=2)
#         self.rnn = keras.layers.RNN(rankcell_8_2, return_sequences=True)

#     def call(self, inputs):
#         """Call."""
#         z = self.percept(inputs["given8rank2_stimulus_set"])
#         z_q, z_r = keras.ops.split(z, [1], self.stimuli_axis)
#         s = self.proximity([z_q, z_r])
#         output = self.rnn(s)
#         return output

#     def get_config(self):
#         config = super(RankCellModelA, self).get_config()
#         return config


class RateModelA(keras.Model):
    """A `RateSimilarity` model.

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


class RateModelB(keras.Model):
    """A `RateSimilarity` model.

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


class RankRateModelA(keras.Model):
    """A joint `RankSimilarity` and `RateSimilarity` model.

    Gates:
        Behavior layer (BranchGate:2)

    """

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

    # TODO remove when done debugging.
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
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compute_loss(y=y, y_pred=y_pred, sample_weight=sample_weight)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply(gradients, trainable_vars)

        # Update metrics (includes the metric that tracks the loss)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


class RankRTModelA(keras.Model):
    """A `RankSimilarity` with response times model.

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


def build_ranksim_subclass_a():
    """Build subclassed `Model`.

    RankSimilarity, one group, MLE.

    """
    model = RankModelA()
    compile_kwargs = {
        "loss": keras.losses.CategoricalCrossentropy(),
        "optimizer": keras.optimizers.Adam(learning_rate=0.001),
        "weighted_metrics": [keras.metrics.CategoricalCrossentropy(name="cce")],
    }
    model.compile(**compile_kwargs)
    return model


def build_ranksim_functional_v0():
    """Build model useing functional API.

    RankSimilarity, one group, MLE.

    """
    n_stimuli = 20
    n_dim = 3
    stimuli_axis = 1

    percept = keras.layers.Embedding(n_stimuli + 1, n_dim, mask_zero=True)
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
    soft_4rank1 = psiz.keras.layers.SoftRank(n_select=1)

    inputs = {
        "given4rank1_stimulus_set": keras.Input(
            shape=(5,), name="given4rank1_stimulus_set"
        ),
    }
    z = percept(inputs["given4rank1_stimulus_set"])
    z_q, z_r = keras.ops.split(z, [1], axis=stimuli_axis)
    s = proximity([z_q, z_r])
    outputs = soft_4rank1(s)

    model = keras.Model(inputs=inputs, outputs=outputs, name="functional_rank")
    compile_kwargs = {
        "loss": keras.losses.CategoricalCrossentropy(),
        "optimizer": keras.optimizers.Adam(learning_rate=0.001),
        "weighted_metrics": [keras.metrics.CategoricalCrossentropy(name="cce")],
    }
    model.compile(**compile_kwargs)
    return model


def build_ranksim_subclass_b():
    """Build subclassed `Model`.

    RankSimilarity, two kernels, MLE.

    """
    model = RankModelB()
    compile_kwargs = {
        "loss": keras.losses.CategoricalCrossentropy(),
        "optimizer": keras.optimizers.Adam(learning_rate=0.001),
        "weighted_metrics": [keras.metrics.CategoricalCrossentropy(name="cce")],
    }
    model.compile(**compile_kwargs)
    return model


def build_ranksim_subclass_c():
    """Build subclassed `Model`.

    RankSimilarity, two kernels, MLE.

    """
    model = RankModelC()
    compile_kwargs = {
        "loss": keras.losses.CategoricalCrossentropy(),
        "optimizer": keras.optimizers.Adam(learning_rate=0.001),
        "weighted_metrics": [keras.metrics.CategoricalCrossentropy(name="cce")],
    }
    model.compile(**compile_kwargs)
    return model


def build_ranksim_subclass_d():
    """Build subclassed `Model`.

    RankSimilarity, two kernels, MLE.

    """
    model = RankModelD()
    compile_kwargs = {
        "loss": keras.losses.CategoricalCrossentropy(),
        "optimizer": keras.optimizers.Adam(learning_rate=0.001),
        "weighted_metrics": [keras.metrics.CategoricalCrossentropy(name="cce")],
    }
    model.compile(**compile_kwargs)
    return model


def build_ranksim_subclass_e():
    """Build subclassed `Model`.

    RankSimilarity, one group, MLE.

    """
    model = RankModelE()
    compile_kwargs = {
        "loss": keras.losses.CategoricalCrossentropy(),
        "optimizer": keras.optimizers.Adam(learning_rate=0.001),
        "weighted_metrics": [keras.metrics.CategoricalCrossentropy(name="cce")],
    }
    model.compile(**compile_kwargs)
    return model


def build_ranksim_subclass_f():
    """Build subclassed `Model`.

    RankSimilarity, one group, MLE.

    """
    model = RankModelF()
    compile_kwargs = {
        "loss": keras.losses.CategoricalCrossentropy(),
        "optimizer": keras.optimizers.Adam(learning_rate=0.001),
        "weighted_metrics": [keras.metrics.CategoricalCrossentropy(name="cce")],
    }
    model.compile(**compile_kwargs)
    return model


def build_multirank_subclass_a():
    model = MultiRankModelA()
    compile_kwargs = {
        "optimizer": keras.optimizers.Adam(learning_rate=0.00001),
        "loss": {
            "given2rank1": keras.losses.CategoricalCrossentropy(
                name="given2rank1_loss"
            ),
            "given8rank2": keras.losses.CategoricalCrossentropy(
                name="given8rank2_loss"
            ),
        },
        "loss_weights": {"given2rank1": 1.0, "given8rank2": 1.0},
    }
    model.compile(**compile_kwargs)
    return model


# TODO finish or move out
# def build_ranksimcell_subclass_a():
#     """Build subclassed `Model`.

#     RankSimilarityCell, one group, MLE.

#     """
#     model = RankCellModelA()
#     compile_kwargs = {
#         "loss": keras.losses.CategoricalCrossentropy(),
#         "optimizer": keras.optimizers.Adam(learning_rate=0.001),
#         "weighted_metrics": [keras.metrics.CategoricalCrossentropy(name="cce")],
#     }
#     model.compile(**compile_kwargs)
#     return model


# TODO finish or move out
# def build_ranksimcell_functional_v0():
#     """Build model useing functional API."""
#     n_stimuli = 20
#     n_dim = 3

#     percept = keras.layers.Embedding(n_stimuli + 1, n_dim, mask_zero=True)
#     proximity = psiz.keras.layers.Minkowski(
#         rho_initializer=keras.initializers.Constant(2.0),
#         w_initializer=keras.initializers.Constant(1.0),
#         activation=psiz.keras.layers.ExponentialSimilarity(
#             beta_initializer=keras.initializers.Constant(10.0),
#             tau_initializer=keras.initializers.Constant(1.0),
#             gamma_initializer=keras.initializers.Constant(0.001),
#             trainable=False,
#         ),
#         trainable=False,
#     )
#     cell = psiz.keras.layers.RankSimilarityCell(
#         n_reference=8, n_select=2, percept=percept, kernel=proximity
#     )
#     rnn = keras.layers.RNN(cell, return_sequences=True)

#     inp_stimulus_set = keras.Input(shape=(None, 9), name="given8rank2_stimulus_set")
#     inputs = {
#         "given8rank2_stimulus_set": inp_stimulus_set,
#     }
#     outputs = rnn(inputs)
#     model = keras.Model(inputs=inputs, outputs=outputs, name="functional_rank")
#     compile_kwargs = {
#         "loss": keras.losses.CategoricalCrossentropy(),
#         "optimizer": keras.optimizers.Adam(learning_rate=0.001),
#         "weighted_metrics": [keras.metrics.CategoricalCrossentropy(name="cce")],
#     }
#     model.compile(**compile_kwargs)
#     return model


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

    # Define different behavioral branches.
    soft_4rank2 = psiz.keras.layers.SoftRank(n_select=2)
    rate = psiz.keras.layers.Logistic()

    inp_rank_stimulus_set = keras.Input(shape=(5,), name="given4rank1_stimulus_set")
    inp_rate_stimulus_set = keras.Input(shape=(2,), name="rate2_stimulus_set")
    # inp_gate_weights = keras.Input(shape=(1,), name="gate_weights_behavior")
    inputs = {
        "given4rank2_stimulus_set": inp_rank_stimulus_set,
        "rate2_stimulus_set": inp_rate_stimulus_set,
        # "gate_weights_behavior": inp_gate_weights,
    }

    # Rank branch.
    z = percept(inputs["given4rank2_stimulus_set"])
    z_q, z_r = keras.ops.split(z, [1], stimuli_axis)
    s = proximity([z_q, z_r])
    output_rank = soft_4rank2(s)

    # Rate branch.
    z = percept(inputs["rate2_stimulus_set"])
    z_0, z_1 = keras.ops.split(z, [1], stimuli_axis)
    s = proximity([z_0, z_1])
    output_rate = rate(s)

    outputs = {"rank_branch": output_rank, "rate_branch": output_rate}
    model = keras.Model(inputs=inputs, outputs=outputs, name="functional_rank_rate")
    compile_kwargs = {
        "optimizer": keras.optimizers.Adam(learning_rate=0.001),
        "loss": {
            "rank_branch": keras.losses.CategoricalCrossentropy(name="rank_loss"),
            "rate_branch": keras.losses.MeanSquaredError(name="rate_loss"),
        },
        "loss_weights": {"rank_branch": 1.0, "rate_branch": 1.0},
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


class TestRankSimilarity:
    """Test using `RankSimilarity` layer."""

    @pytest.mark.parametrize("is_eager", [True, False])
    def test_usage_subclass_a(self, ds_4rank1_v0, is_eager):
        """Test model using subclass API."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_4rank1_v0
        model = build_ranksim_subclass_a()
        call_fit_evaluate_predict(model, tfds)
        keras.backend.clear_session()

    @pytest.mark.parametrize("is_eager", [True, False])
    def test_save_load_subclass_a(self, ds_4rank1_v0, is_eager, tmpdir):
        """Test serialization."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_4rank1_v0
        model = build_ranksim_subclass_a()
        model.fit(tfds, epochs=1)
        eval0 = model.evaluate(tfds)

        # Test storage.
        fp_model = Path(tmpdir) / "test_model.keras"
        model.save(fp_model)
        del model
        # Load the saved model.
        loaded = keras.models.load_model(
            fp_model,
            custom_objects={"RankModelA": RankModelA},
        )
        eval1 = loaded.evaluate(tfds)

        # Test for model equality.
        assert eval0[0] == eval1[0]
        assert eval0[1] == eval1[1]

    @pytest.mark.parametrize("is_eager", [True, False])
    def test_usage_functional_v0(self, ds_4rank1_v0, is_eager):
        """Test model using functional API."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_4rank1_v0
        model = build_ranksim_functional_v0()
        call_fit_evaluate_predict(model, tfds)
        keras.backend.clear_session()

    @pytest.mark.parametrize("is_eager", [True, False])
    def test_save_load_functional_v0(self, ds_4rank1_v0, is_eager, tmpdir):
        """Test serialization."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_4rank1_v0
        model = build_ranksim_functional_v0()
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
    def test_usage_subclass_b(self, ds_4rank1_v1, is_eager):
        """Test model using subclass API."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_4rank1_v1
        model = build_ranksim_subclass_b()
        call_fit_evaluate_predict(model, tfds)
        keras.backend.clear_session()

    @pytest.mark.parametrize("is_eager", [True, False])
    def test_usage_subclass_c(self, ds_4rank1_v2, is_eager):
        """Test model using subclass API."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_4rank1_v2
        model = build_ranksim_subclass_c()
        call_fit_evaluate_predict(model, tfds)
        keras.backend.clear_session()

    @pytest.mark.parametrize("is_eager", [True, False])
    def test_usage_subclass_d(self, ds_4rank1_v3, is_eager):
        """Test model using subclass API."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_4rank1_v3
        model = build_ranksim_subclass_d()
        call_fit_evaluate_predict(model, tfds)
        keras.backend.clear_session()

    @pytest.mark.parametrize("is_eager", [True, False])
    def test_usage_subclass_e(self, ds_2rank1_v0, is_eager):
        """Test model using subclass API."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_2rank1_v0
        model = build_ranksim_subclass_e()
        call_fit_evaluate_predict(model, tfds)
        keras.backend.clear_session()

    @pytest.mark.parametrize("is_eager", [True, False])
    def test_usage_subclass_f(self, ds_8rank2_v0, is_eager):
        """Test model using subclass API."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_8rank2_v0
        model = build_ranksim_subclass_f()
        call_fit_evaluate_predict(model, tfds)
        keras.backend.clear_session()

    @pytest.mark.parametrize("is_eager", [True, False])
    def test_agent_subclass_a(self, ds_4rank1_v0, is_eager):
        """Test usage in 'agent mode'."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_4rank1_v0
        model = build_ranksim_subclass_a()

        def simulate_agent(x):
            n_class = 4
            outcome_probs = model(x)
            outcome_distribution = tfp.distributions.Categorical(probs=outcome_probs)
            outcome_idx = outcome_distribution.sample()
            outcome_one_hot = keras.ops.one_hot(
                outcome_idx, n_class
            )  # TODO verify this is correct
            return outcome_one_hot

        _ = tfds.map(lambda x, y, w: (x, simulate_agent(x), w))

        keras.backend.clear_session()


class TestMultiRankSimilarity:
    """Test using `RankSimilarity` layer."""

    @pytest.mark.parametrize("is_eager", [True, False])
    def test_usage_subclass_a(self, ds_2rank1_8rank2_v0, is_eager):
        """Test model using subclass API."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_2rank1_8rank2_v0
        model = build_multirank_subclass_a()
        call_fit_evaluate_predict(model, tfds)
        keras.backend.clear_session()

    # TODO here
    @pytest.mark.parametrize("is_eager", [True, False])
    def test_save_load_subclass_a(self, ds_2rank1_8rank2_v0, is_eager, tmpdir):
        """Test serialization."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_2rank1_8rank2_v0
        model = build_multirank_subclass_a()
        model.fit(tfds, epochs=1)
        eval0 = model.evaluate(tfds)
        predict0 = model.predict(tfds)

        # Test storage.
        fp_model = Path(tmpdir) / "test_model.keras"
        model.save(fp_model)
        del model
        # Load the saved model.
        loaded = keras.models.load_model(
            fp_model,
            custom_objects={"MultiRankModelA": MultiRankModelA},
        )
        eval1 = loaded.evaluate(tfds)
        predict1 = loaded.predict(tfds)

        # Test for model equality.
        assert eval0 == eval1
        tf.test.TestCase().assertAllClose(predict0, predict1, atol=1e-6)


# TODO finish or move out
# class TestRankSimilarityCell:
#     """Test using `RankSimilaritycell` layer."""

#     @pytest.mark.parametrize("is_eager", [True, False])
#     @pytest.mark.xfail(reason="Keras v3 RNN requires single input tensor.")
#     def test_usage_subclass_a(self, ds_time_8rank2_v0, is_eager):
#         """Test model using subclass API."""
#         tf.config.run_functions_eagerly(is_eager)

#         tfds = ds_time_8rank2_v0
#         model = build_ranksimcell_subclass_a()
#         call_fit_evaluate_predict(model, tfds)
#         keras.backend.clear_session()

#     @pytest.mark.parametrize("is_eager", [True, False])
#     @pytest.mark.xfail(reason="Keras v3 RNN requires single input tensor.")
#     def test_save_load_subclass_a(
#         self, ds_time_8rank2_v0, is_eager, tmpdir
#     ):
#         """Test serialization."""
#         tf.config.run_functions_eagerly(is_eager)

#         tfds = ds_time_8rank2_v0
#         model = build_ranksimcell_subclass_a()
#         model.fit(tfds, epochs=1)
#         eval0 = model.evaluate(tfds)

#         # Test storage.
#         fp_model = Path(tmpdir) / "test_model.keras"
#         model.save(fp_model)
#         del model
#         # Load the saved model.
#         loaded = keras.models.load_model(
#             fp_model, custom_objects={"RankCellModelA": RankCellModelA},
#         )
#         eval1 = loaded.evaluate(tfds)

#         # Test for model equality.
#         assert eval0[0] == eval1[0]
#         assert eval0[1] == eval1[1]

#     @pytest.mark.parametrize("is_eager", [True, False])
#     @pytest.mark.xfail(reason="Keras v3 RNN requires single input tensor.")
#     def test_usage_functional_v0(self, ds_time_8rank2_v0, is_eager):
#         """Test model using functional API."""
#         tf.config.run_functions_eagerly(is_eager)

#         tfds = ds_time_8rank2_v0
#         model = build_ranksimcell_functional_v0()
#         call_fit_evaluate_predict(model, tfds)
#         keras.backend.clear_session()

#     @pytest.mark.parametrize("is_eager", [True, False])
#     @pytest.mark.xfail(reason="Keras v3 RNN requires single input tensor.")
#     def test_save_load_functional_v0(self, ds_time_8rank2_v0, is_eager, tmpdir):
#         """Test serialization."""
#         tf.config.run_functions_eagerly(is_eager)

#         tfds = ds_time_8rank2_v0
#         model = build_ranksimcell_functional_v0()
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


class TestRateSimilarity:
    """Test using `RateSimilarity` layer."""

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


class TestJointRankRate:
    """Test using joint `RankSimilarity` and `RateSimilarity` layers."""

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
