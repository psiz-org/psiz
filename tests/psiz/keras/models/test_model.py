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


import pytest
import tensorflow as tf
import tensorflow_probability as tfp

import psiz


class RankModelA(tf.keras.Model):
    """A `RankSimilarity` model.

    Gates:
        None

    """

    def __init__(self, **kwargs):
        """Initialize."""
        super(RankModelA, self).__init__(**kwargs)

        n_stimuli = 20
        n_dim = 3
        self.stimuli_axis = 1

        self.percept = tf.keras.layers.Embedding(
            n_stimuli + 1,
            n_dim,
            mask_zero=True,
        )
        self.proximity = psiz.keras.layers.Minkowski(
            rho_initializer=tf.keras.initializers.Constant(2.0),
            w_initializer=tf.keras.initializers.Constant(1.0),
            trainable=False,
            activation=psiz.keras.layers.ExponentialSimilarity(
                beta_initializer=tf.keras.initializers.Constant(10.0),
                tau_initializer=tf.keras.initializers.Constant(1.0),
                gamma_initializer=tf.keras.initializers.Constant(0.0),
                trainable=False,
            ),
        )
        self.soft_4rank1 = psiz.keras.layers.SoftRank(n_select=1)

    def call(self, inputs):
        """Call."""
        z = self.percept(inputs["given4rank1_stimulus_set"])
        z_q, z_r = tf.split(z, [1, 4], self.stimuli_axis)
        s = self.proximity([z_q, z_r])
        return self.soft_4rank1(s)

    def get_config(self):
        config = super(RankModelA, self).get_config()
        return config


class RankModelB(tf.keras.Model):
    """A `RankSimilarity` model.

    Gates:
        Kernel layer (BraidGate:2) with shared similarity layer.

    """

    def __init__(self, **kwargs):
        """Initialize."""
        super(RankModelB, self).__init__(**kwargs)

        n_stimuli = 30
        n_dim = 10
        self.stimuli_axis = 1

        self.percept = tf.keras.layers.Embedding(n_stimuli + 1, n_dim, mask_zero=True)
        # Define group-specific kernels.
        shared_activation = psiz.keras.layers.ExponentialSimilarity(
            beta_initializer=tf.keras.initializers.Constant(10.0),
            tau_initializer=tf.keras.initializers.Constant(1.0),
            gamma_initializer=tf.keras.initializers.Constant(0.0),
            trainable=False,
        )
        proximity_0 = psiz.keras.layers.Minkowski(
            rho_initializer=tf.keras.initializers.Constant(2.0),
            w_initializer=tf.keras.initializers.Constant(
                [1.2, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
            ),
            trainable=False,
            activation=shared_activation,
        )
        proximity_1 = psiz.keras.layers.Minkowski(
            rho_initializer=tf.keras.initializers.Constant(2.0),
            w_initializer=tf.keras.initializers.Constant(
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 1.2]
            ),
            trainable=False,
            activation=shared_activation,
        )
        self.braid_proximity = psiz.keras.layers.BraidGate(
            subnets=[proximity_0, proximity_1], gating_index=-1
        )

        self.soft_4rank1 = psiz.keras.layers.SoftRank(n_select=1)

    def call(self, inputs):
        """Call."""
        z = self.percept(inputs["given4rank1_stimulus_set"])
        z_q, z_r = tf.split(z, [1, 4], self.stimuli_axis)
        s = self.braid_proximity([z_q, z_r, inputs["kernel_gate_weights"]])
        return self.soft_4rank1(s)

    def get_config(self):
        config = super(RankModelB, self).get_config()
        return config


class RankModelC(tf.keras.Model):
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
        percept_0 = tf.keras.layers.Embedding(n_stimuli + 1, n_dim, mask_zero=True)
        percept_1 = tf.keras.layers.Embedding(n_stimuli + 1, n_dim, mask_zero=True)
        self.braid_percept = psiz.keras.layers.BraidGate(
            subnets=[percept_0, percept_1], gating_index=-1
        )

        # Define group-specific proximity layers.
        shared_activation = psiz.keras.layers.ExponentialSimilarity(
            trainable=False,
            beta_initializer=tf.keras.initializers.Constant(10.0),
            tau_initializer=tf.keras.initializers.Constant(1.0),
            gamma_initializer=tf.keras.initializers.Constant(0.0),
        )
        proximity_0 = psiz.keras.layers.Minkowski(
            rho_trainable=False,
            rho_initializer=tf.keras.initializers.Constant(2.0),
            w_initializer=tf.keras.initializers.Constant([1.2, 0.8]),
            w_constraint=psiz.keras.constraints.NonNegNorm(scale=n_dim, p=1.0),
            activation=shared_activation,
        )
        proximity_1 = psiz.keras.layers.Minkowski(
            rho_trainable=False,
            rho_initializer=tf.keras.initializers.Constant(2.0),
            w_initializer=tf.keras.initializers.Constant([0.7, 1.3]),
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
        z_q, z_r = tf.split(z, [1, 4], self.stimuli_axis)
        s = self.braid_proximity([z_q, z_r, inputs["kernel_gate_weights"]])
        return self.soft_4rank1(s)

    def get_config(self):
        config = super(RankModelC, self).get_config()
        return config


class RankModelD(tf.keras.Model):
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
        percept_0 = tf.keras.layers.Embedding(n_stimuli + 1, n_dim, mask_zero=True)
        percept_1 = tf.keras.layers.Embedding(n_stimuli + 1, n_dim, mask_zero=True)
        percept_01 = psiz.keras.layers.BraidGate(
            subnets=[percept_0, percept_1], gating_index=-1, name="percept_01"
        )

        percept_2 = tf.keras.layers.Embedding(n_stimuli + 1, n_dim, mask_zero=True)
        percept_3 = tf.keras.layers.Embedding(n_stimuli + 1, n_dim, mask_zero=True)
        percept_23 = psiz.keras.layers.BraidGate(
            subnets=[percept_2, percept_3], gating_index=-1, name="percept_23"
        )

        self.braid_percept = psiz.keras.layers.BraidGate(
            subnets=[percept_01, percept_23], gating_index=-1, name="percept"
        )

        # Define proximity.
        self.proximity = psiz.keras.layers.Minkowski(
            rho_trainable=False,
            rho_initializer=tf.keras.initializers.Constant(2.0),
            w_initializer=tf.keras.initializers.Constant([1.2, 0.8]),
            w_constraint=psiz.keras.constraints.NonNegNorm(scale=n_dim, p=1.0),
            activation=psiz.keras.layers.ExponentialSimilarity(
                trainable=False,
                beta_initializer=tf.keras.initializers.Constant(10.0),
                tau_initializer=tf.keras.initializers.Constant(1.0),
                gamma_initializer=tf.keras.initializers.Constant(0.0),
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
        z_q, z_r = tf.split(z, [1, 4], self.stimuli_axis)
        s = self.proximity([z_q, z_r])
        return self.soft_4rank1(s)

    def get_config(self):
        config = super(RankModelD, self).get_config()
        return config


class RankModelE(tf.keras.Model):
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

        self.percept = tf.keras.layers.Embedding(n_stimuli + 1, n_dim, mask_zero=True)
        self.proximity = psiz.keras.layers.Minkowski(
            rho_initializer=tf.keras.initializers.Constant(2.0),
            w_initializer=tf.keras.initializers.Constant(1.0),
            trainable=False,
            activation=psiz.keras.layers.ExponentialSimilarity(
                beta_initializer=tf.keras.initializers.Constant(10.0),
                tau_initializer=tf.keras.initializers.Constant(1.0),
                gamma_initializer=tf.keras.initializers.Constant(0.001),
                trainable=False,
            ),
        )
        self.soft_2rank1 = psiz.keras.layers.SoftRank(n_select=1)

    def call(self, inputs):
        """Call."""
        z = self.percept(inputs["given2rank1_stimulus_set"])
        z_q, z_r = tf.split(z, [1, 2], self.stimuli_axis)
        s = self.proximity([z_q, z_r])
        return self.soft_2rank1(s)

    def get_config(self):
        config = super(RankModelE, self).get_config()
        return config


class RankModelF(tf.keras.Model):
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

        self.percept = tf.keras.layers.Embedding(n_stimuli + 1, n_dim, mask_zero=True)

        self.proximity = psiz.keras.layers.Minkowski(
            rho_initializer=tf.keras.initializers.Constant(2.0),
            w_initializer=tf.keras.initializers.Constant(1.0),
            trainable=False,
            activation=psiz.keras.layers.ExponentialSimilarity(
                beta_initializer=tf.keras.initializers.Constant(10.0),
                tau_initializer=tf.keras.initializers.Constant(1.0),
                gamma_initializer=tf.keras.initializers.Constant(0.001),
                trainable=False,
            ),
        )

        self.soft_8rank2 = psiz.keras.layers.SoftRank(n_select=2)

    def call(self, inputs):
        """Call."""
        z = self.percept(inputs["given8rank2_stimulus_set"])
        z_q, z_r = tf.split(z, [1, 8], self.stimuli_axis)
        s = self.proximity([z_q, z_r])
        return self.soft_8rank2(s)

    def get_config(self):
        config = super(RankModelF, self).get_config()
        return config


class MultiRankModelA(tf.keras.Model):
    """A `RankSimilarity` model.

    Gates:
        BranchGate

    """

    def __init__(self, **kwargs):
        """Initialize."""
        super(MultiRankModelA, self).__init__(**kwargs)

        n_stimuli = 20
        n_dim = 3
        self.stimuli_axis = 1

        self.percept = tf.keras.layers.Embedding(n_stimuli + 1, n_dim, mask_zero=True)

        self.proximity = psiz.keras.layers.Minkowski(
            rho_initializer=tf.keras.initializers.Constant(2.0),
            w_initializer=tf.keras.initializers.Constant(1.0),
            trainable=False,
            activation=psiz.keras.layers.ExponentialSimilarity(
                beta_initializer=tf.keras.initializers.Constant(10.0),
                tau_initializer=tf.keras.initializers.Constant(1.0),
                gamma_initializer=tf.keras.initializers.Constant(0.001),
                trainable=False,
            ),
        )
        self.soft_2rank1 = psiz.keras.layers.SoftRank(n_select=1)
        self.soft_8rank2 = psiz.keras.layers.SoftRank(n_select=2)

    def call(self, inputs):
        """Call."""
        # The 2-rank-1 branch.
        z = self.percept(inputs["given2rank1_stimulus_set"])
        z_q, z_r = tf.split(z, [1, 2], self.stimuli_axis)
        s = self.proximity([z_q, z_r])
        prob_2rank1 = self.soft_2rank1(s)

        # The 8-rank-2 branch.
        z = self.percept(inputs["given8rank2_stimulus_set"])
        z_q, z_r = tf.split(z, [1, 8], self.stimuli_axis)
        s = self.proximity([z_q, z_r])
        prob_8rank2 = self.soft_8rank2(s)

        return {"given2rank1": prob_2rank1, "given8rank2": prob_8rank2}

    def get_config(self):
        config = super(MultiRankModelA, self).get_config()
        return config


# TODO finish or move out
# class RankCellModelA(tf.keras.Model):
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

#         self.percept = tf.keras.layers.Embedding(n_stimuli + 1, n_dim, mask_zero=True)
#         self.proximity = psiz.keras.layers.Minkowski(
#             rho_initializer=tf.keras.initializers.Constant(2.0),
#             w_initializer=tf.keras.initializers.Constant(1.0),
#             trainable=False,
#             activation=psiz.keras.layers.ExponentialSimilarity(
#                 beta_initializer=tf.keras.initializers.Constant(10.0),
#                 tau_initializer=tf.keras.initializers.Constant(1.0),
#                 gamma_initializer=tf.keras.initializers.Constant(0.001),
#                 trainable=False,
#             ),
#         )
#         rankcell_8_2 = psiz.keras.layers.SoftRankCell(n_select=2)
#         self.rnn = tf.keras.layers.RNN(rankcell_8_2, return_sequences=True)

#     def call(self, inputs):
#         """Call."""
#         z = self.percept(inputs["given8rank2_stimulus_set"])
#         z_q, z_r = tf.split(z, [1, 8], self.stimuli_axis)
#         s = self.proximity([z_q, z_r])
#         output = self.rnn(s)
#         return output

#     def get_config(self):
#         config = super(RankCellModelA, self).get_config()
#         return config


class RateModelA(tf.keras.Model):
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

        self.percept = tf.keras.layers.Embedding(n_stimuli + 1, n_dim, mask_zero=True)
        self.proximity = psiz.keras.layers.Minkowski(
            rho_initializer=tf.keras.initializers.Constant(2.0),
            w_initializer=tf.keras.initializers.Constant(1.0),
            trainable=False,
            activation=psiz.keras.layers.ExponentialSimilarity(
                trainable=False,
                beta_initializer=tf.keras.initializers.Constant(3.0),
                tau_initializer=tf.keras.initializers.Constant(1.0),
                gamma_initializer=tf.keras.initializers.Constant(0.0),
            ),
        )
        self.rate = psiz.keras.layers.Logistic()

    def call(self, inputs):
        """Call."""
        z = self.percept(inputs["rate2_stimulus_set"])
        z_0, z_1 = tf.split(z, [1, 1], self.stimuli_axis)
        s = self.proximity([z_0, z_1])
        return self.rate(s)

    def get_config(self):
        config = super(RateModelA, self).get_config()
        return config


class RateModelB(tf.keras.Model):
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
            beta_initializer=tf.keras.initializers.Constant(10.0),
            tau_initializer=tf.keras.initializers.Constant(1.0),
            gamma_initializer=tf.keras.initializers.Constant(0.0),
        )

        # Group 0 layers.
        percept_0 = tf.keras.layers.Embedding(n_stimuli + 1, n_dim, mask_zero=True)
        proximity_0 = psiz.keras.layers.Minkowski(
            rho_trainable=False,
            rho_initializer=tf.keras.initializers.Constant(2.0),
            w_initializer=tf.keras.initializers.Constant([1.2, 0.8]),
            w_constraint=psiz.keras.constraints.NonNegNorm(scale=n_dim, p=1.0),
            activation=shared_activation,
        )
        rate_0 = psiz.keras.layers.Logistic()

        # Group 1 layers.
        percept_1 = tf.keras.layers.Embedding(n_stimuli + 1, n_dim, mask_zero=True)
        proximity_1 = psiz.keras.layers.Minkowski(
            rho_trainable=False,
            rho_initializer=tf.keras.initializers.Constant(2.0),
            w_initializer=tf.keras.initializers.Constant([0.7, 1.3]),
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
        z_0, z_1 = tf.split(z, [1, 1], self.stimuli_axis)
        s = self.braid_proximity([z_0, z_1, inputs["behavior_gate_weights"]])
        return self.braid_rate([s, inputs["behavior_gate_weights"]])

    def get_config(self):
        config = super(RateModelB, self).get_config()
        return config


# TODO finish or move out
# class RateCellModelA(tf.keras.Model):
#     """A `RateSimilarityCell` model.

#     Gates:
#         None

#     """

#     def __init__(self, **kwargs):
#         """Initialize."""
#         super(RateCellModelA, self).__init__(**kwargs)

#         n_stimuli = 30
#         n_dim = 10

#         percept = tf.keras.layers.Embedding(n_stimuli + 1, n_dim, mask_zero=True)
#         proximity = psiz.keras.layers.Minkowski(
#             rho_initializer=tf.keras.initializers.Constant(2.0),
#             w_initializer=tf.keras.initializers.Constant(1.0),
#             activation=psiz.keras.layers.ExponentialSimilarity(
#                 trainable=False,
#                 beta_initializer=tf.keras.initializers.Constant(3.0),
#                 tau_initializer=tf.keras.initializers.Constant(1.0),
#                 gamma_initializer=tf.keras.initializers.Constant(0.0),
#             ),
#             trainable=False,
#         )
#         rate_cell = psiz.keras.layers.RateSimilarityCell(percept=percept, kernel=proximity)
#         rnn = tf.keras.layers.RNN(rate_cell, return_sequences=True)
#         self.behavior = rnn

#     def call(self, inputs):
#         """Call."""
#         return self.behavior(inputs)

#     def get_config(self):
#         config = super(RateCellModelA, self).get_config()
#         return config


class ALCOVEModelA(tf.keras.Model):
    """An `ALCOVECell` model.

    Gates:
        None

    """

    def __init__(self, **kwargs):
        """Initialize."""
        super(ALCOVEModelA, self).__init__(**kwargs)

        n_stimuli = 20
        n_dim = 4
        n_output = 3

        percept = tf.keras.layers.Embedding(
            n_stimuli + 1,
            n_dim,
            mask_zero=True,
            trainable=False,
        )
        similarity = psiz.keras.layers.ExponentialSimilarity(
            beta_initializer=tf.keras.initializers.Constant(3.0),
            tau_initializer=tf.keras.initializers.Constant(1.0),
            gamma_initializer=tf.keras.initializers.Constant(0.0),
            trainable=False,
        )
        cell = psiz.keras.layers.ALCOVECell(
            n_output,
            percept=percept,
            similarity=similarity,
            rho_initializer=tf.keras.initializers.Constant(2.0),
            temperature_initializer=tf.keras.initializers.Constant(1.0),
            lr_attention_initializer=tf.keras.initializers.Constant(0.03),
            lr_association_initializer=tf.keras.initializers.Constant(0.03),
            trainable=False,
        )
        rnn = tf.keras.layers.RNN(cell, return_sequences=True, stateful=False)
        self.behavior = rnn

    def call(self, inputs):
        """Call."""
        return self.behavior(inputs)

    def get_config(self):
        config = super(ALCOVEModelA, self).get_config()
        return config


class RankRateModelA(tf.keras.Model):
    """A joint `RankSimilarity` and `RateSimilarity` model.

    Gates:
        Behavior layer (BranchGate:2)

    """

    def __init__(self, **kwargs):
        """Initialize."""
        super(RankRateModelA, self).__init__(**kwargs)

        n_stimuli = 20
        n_dim = 3
        self.stimuli_axis = 1

        # Define a percept layer that will be shared across behaviors.
        self.percept = tf.keras.layers.Embedding(n_stimuli + 1, n_dim, mask_zero=True)

        # Define a proximity layer that will be shared across behaviors.
        self.proximity = psiz.keras.layers.Minkowski(
            rho_initializer=tf.keras.initializers.Constant(2.0),
            w_initializer=tf.keras.initializers.Constant(1.0),
            trainable=False,
            activation=psiz.keras.layers.ExponentialSimilarity(
                beta_initializer=tf.keras.initializers.Constant(10.0),
                tau_initializer=tf.keras.initializers.Constant(1.0),
                gamma_initializer=tf.keras.initializers.Constant(0.001),
                trainable=False,
            ),
        )

        # Define a multi-output branches.
        self.soft_4rank2 = psiz.keras.layers.SoftRank(n_select=2)
        self.rate = psiz.keras.layers.Logistic()

    def call(self, inputs):
        """Call."""
        # Rank branch.
        z = self.percept(inputs["given4rank2_stimulus_set"])
        z_q, z_r = tf.split(z, [1, 4], self.stimuli_axis)
        s = self.proximity([z_q, z_r])
        output_rank = self.soft_4rank2(s)

        # Rate branch.
        z = self.percept(inputs["rate2_stimulus_set"])
        z_0, z_1 = tf.split(z, [1, 1], self.stimuli_axis)
        s = self.proximity([z_0, z_1])
        output_rate = self.rate(s)

        return {"rank_branch": output_rank, "rate_branch": output_rate}

    def get_config(self):
        config = super(RankRateModelA, self).get_config()
        return config


class RankRTModelA(tf.keras.Model):
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
        self.percept = tf.keras.layers.Embedding(n_stimuli + 1, n_dim, mask_zero=True)

        # Define a proximity layer that will be shared across behaviors.
        self.proximity = psiz.keras.layers.Minkowski(
            rho_initializer=tf.keras.initializers.Constant(2.0),
            w_initializer=tf.keras.initializers.Constant(1.0),
            trainable=False,
            activation=psiz.keras.layers.ExponentialSimilarity(
                beta_initializer=tf.keras.initializers.Constant(10.0),
                tau_initializer=tf.keras.initializers.Constant(1.0),
                gamma_initializer=tf.keras.initializers.Constant(0.001),
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
        z_q, z_r = tf.split(z, [1, 4], self.stimuli_axis)
        s = self.proximity([z_q, z_r])
        output_rank = self.soft_4rank1(s)

        # Response time branch.
        # Estimate response time as a function of soft rank outcome entropy.
        entropy = -tf.reduce_sum(
            tf.math.multiply(output_rank, tf.math.log(output_rank)),
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
        "loss": tf.keras.losses.CategoricalCrossentropy(),
        "optimizer": tf.keras.optimizers.Adam(learning_rate=0.001),
        "weighted_metrics": [tf.keras.metrics.CategoricalCrossentropy(name="cce")],
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

    percept = tf.keras.layers.Embedding(n_stimuli + 1, n_dim, mask_zero=True)
    proximity = psiz.keras.layers.Minkowski(
        rho_initializer=tf.keras.initializers.Constant(2.0),
        w_initializer=tf.keras.initializers.Constant(1.0),
        trainable=False,
        activation=psiz.keras.layers.ExponentialSimilarity(
            beta_initializer=tf.keras.initializers.Constant(10.0),
            tau_initializer=tf.keras.initializers.Constant(1.0),
            gamma_initializer=tf.keras.initializers.Constant(0.001),
            trainable=False,
        ),
    )
    soft_4rank1 = psiz.keras.layers.SoftRank(n_select=1)

    inputs = {
        "given4rank1_stimulus_set": tf.keras.Input(
            shape=(5,), name="given4rank1_stimulus_set"
        ),
    }
    z = percept(inputs["given4rank1_stimulus_set"])
    z_q, z_r = tf.split(z, [1, 4], stimuli_axis)
    s = proximity([z_q, z_r])
    outputs = soft_4rank1(s)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="functional_rank")
    compile_kwargs = {
        "loss": tf.keras.losses.CategoricalCrossentropy(),
        "optimizer": tf.keras.optimizers.Adam(learning_rate=0.001),
        "weighted_metrics": [tf.keras.metrics.CategoricalCrossentropy(name="cce")],
    }
    model.compile(**compile_kwargs)
    return model


def build_ranksim_subclass_b():
    """Build subclassed `Model`.

    RankSimilarity, two kernels, MLE.

    """
    model = RankModelB()
    compile_kwargs = {
        "loss": tf.keras.losses.CategoricalCrossentropy(),
        "optimizer": tf.keras.optimizers.Adam(learning_rate=0.001),
        "weighted_metrics": [tf.keras.metrics.CategoricalCrossentropy(name="cce")],
    }
    model.compile(**compile_kwargs)
    return model


def build_ranksim_subclass_c():
    """Build subclassed `Model`.

    RankSimilarity, two kernels, MLE.

    """
    model = RankModelC()
    compile_kwargs = {
        "loss": tf.keras.losses.CategoricalCrossentropy(),
        "optimizer": tf.keras.optimizers.Adam(learning_rate=0.001),
        "weighted_metrics": [tf.keras.metrics.CategoricalCrossentropy(name="cce")],
    }
    model.compile(**compile_kwargs)
    return model


def build_ranksim_subclass_d():
    """Build subclassed `Model`.

    RankSimilarity, two kernels, MLE.

    """
    model = RankModelD()
    compile_kwargs = {
        "loss": tf.keras.losses.CategoricalCrossentropy(),
        "optimizer": tf.keras.optimizers.Adam(learning_rate=0.001),
        "weighted_metrics": [tf.keras.metrics.CategoricalCrossentropy(name="cce")],
    }
    model.compile(**compile_kwargs)
    return model


def build_ranksim_subclass_e():
    """Build subclassed `Model`.

    RankSimilarity, one group, MLE.

    """
    model = RankModelE()
    compile_kwargs = {
        "loss": tf.keras.losses.CategoricalCrossentropy(),
        "optimizer": tf.keras.optimizers.Adam(learning_rate=0.001),
        "weighted_metrics": [tf.keras.metrics.CategoricalCrossentropy(name="cce")],
    }
    model.compile(**compile_kwargs)
    return model


def build_ranksim_subclass_f():
    """Build subclassed `Model`.

    RankSimilarity, one group, MLE.

    """
    model = RankModelF()
    compile_kwargs = {
        "loss": tf.keras.losses.CategoricalCrossentropy(),
        "optimizer": tf.keras.optimizers.Adam(learning_rate=0.001),
        "weighted_metrics": [tf.keras.metrics.CategoricalCrossentropy(name="cce")],
    }
    model.compile(**compile_kwargs)
    return model


def build_multirank_subclass_a():
    model = MultiRankModelA()
    compile_kwargs = {
        "optimizer": tf.keras.optimizers.Adam(learning_rate=0.001),
        "loss": {
            "given2rank1": tf.keras.losses.CategoricalCrossentropy(
                name="given2rank1_loss"
            ),
            "given8rank2": tf.keras.losses.CategoricalCrossentropy(
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
#         "loss": tf.keras.losses.CategoricalCrossentropy(),
#         "optimizer": tf.keras.optimizers.Adam(learning_rate=0.001),
#         "weighted_metrics": [tf.keras.metrics.CategoricalCrossentropy(name="cce")],
#     }
#     model.compile(**compile_kwargs)
#     return model


# TODO finish or move out
# def build_ranksimcell_functional_v0():
#     """Build model useing functional API."""
#     n_stimuli = 20
#     n_dim = 3

#     percept = tf.keras.layers.Embedding(n_stimuli + 1, n_dim, mask_zero=True)
#     proximity = psiz.keras.layers.Minkowski(
#         rho_initializer=tf.keras.initializers.Constant(2.0),
#         w_initializer=tf.keras.initializers.Constant(1.0),
#         activation=psiz.keras.layers.ExponentialSimilarity(
#             beta_initializer=tf.keras.initializers.Constant(10.0),
#             tau_initializer=tf.keras.initializers.Constant(1.0),
#             gamma_initializer=tf.keras.initializers.Constant(0.001),
#             trainable=False,
#         ),
#         trainable=False,
#     )
#     cell = psiz.keras.layers.RankSimilarityCell(
#         n_reference=8, n_select=2, percept=percept, kernel=proximity
#     )
#     rnn = tf.keras.layers.RNN(cell, return_sequences=True)

#     inp_stimulus_set = tf.keras.Input(shape=(None, 9), name="given8rank2_stimulus_set")
#     inputs = {
#         "given8rank2_stimulus_set": inp_stimulus_set,
#     }
#     outputs = rnn(inputs)
#     model = tf.keras.Model(inputs=inputs, outputs=outputs, name="functional_rank")
#     compile_kwargs = {
#         "loss": tf.keras.losses.CategoricalCrossentropy(),
#         "optimizer": tf.keras.optimizers.Adam(learning_rate=0.001),
#         "weighted_metrics": [tf.keras.metrics.CategoricalCrossentropy(name="cce")],
#     }
#     model.compile(**compile_kwargs)
#     return model


def build_ratesim_subclass_a():
    """Build subclassed `Model`."""
    model = RateModelA()
    compile_kwargs = {
        "loss": tf.keras.losses.MeanSquaredError(),
        "optimizer": tf.keras.optimizers.Adam(learning_rate=0.001),
        "weighted_metrics": [tf.keras.metrics.MeanSquaredError(name="mse")],
    }
    model.compile(**compile_kwargs)
    return model


def build_ratesim_functional_v0():
    """Build model using functional API."""
    n_stimuli = 30
    n_dim = 10
    stimuli_axis = 1

    percept = tf.keras.layers.Embedding(n_stimuli + 1, n_dim, mask_zero=True)
    proximity = psiz.keras.layers.Minkowski(
        rho_initializer=tf.keras.initializers.Constant(2.0),
        w_initializer=tf.keras.initializers.Constant(1.0),
        trainable=False,
        activation=psiz.keras.layers.ExponentialSimilarity(
            trainable=False,
            beta_initializer=tf.keras.initializers.Constant(3.0),
            tau_initializer=tf.keras.initializers.Constant(1.0),
            gamma_initializer=tf.keras.initializers.Constant(0.0),
        ),
    )
    rate = psiz.keras.layers.Logistic()

    inp_stimulus_set = tf.keras.Input(shape=(2,), name="rate2_stimulus_set")
    inputs = {
        "rate2_stimulus_set": inp_stimulus_set,
    }
    z = percept(inputs["rate2_stimulus_set"])
    z_0, z_1 = tf.split(z, [1, 1], stimuli_axis)
    s = proximity([z_0, z_1])
    outputs = rate(s)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="functional_rate")
    compile_kwargs = {
        "loss": tf.keras.losses.MeanSquaredError(),
        "optimizer": tf.keras.optimizers.Adam(learning_rate=0.001),
        "weighted_metrics": [tf.keras.metrics.MeanSquaredError(name="mse")],
    }
    model.compile(**compile_kwargs)
    return model


def build_ratesim_subclass_b():
    """Build subclassed `Model`."""
    model = RateModelB()
    compile_kwargs = {
        "loss": tf.keras.losses.MeanSquaredError(),
        "optimizer": tf.keras.optimizers.Adam(learning_rate=0.001),
        "weighted_metrics": [tf.keras.metrics.MeanSquaredError(name="mse")],
    }
    model.compile(**compile_kwargs)
    return model


# TODO finish or move out
# def build_ratesimcell_subclass_a():
#     """Build subclassed `Model`."""
#     model = RateCellModelA()
#     compile_kwargs = {
#         "loss": tf.keras.losses.MeanSquaredError(),
#         "optimizer": tf.keras.optimizers.Adam(learning_rate=0.001),
#         "weighted_metrics": [tf.keras.metrics.MeanSquaredError(name="mse")],
#     }
#     model.compile(**compile_kwargs)
#     return model


# TODO finish or move out
# def build_ratesimcell_functional_v0():
#     """Build model useing functional API."""
#     n_stimuli = 30
#     n_dim = 10

#     percept = tf.keras.layers.Embedding(n_stimuli + 1, n_dim, mask_zero=True)
#     proximity = psiz.keras.layers.Minkowski(
#         rho_initializer=tf.keras.initializers.Constant(2.0),
#         w_initializer=tf.keras.initializers.Constant(1.0),
#         activation=psiz.keras.layers.ExponentialSimilarity(
#             trainable=False,
#             beta_initializer=tf.keras.initializers.Constant(3.0),
#             tau_initializer=tf.keras.initializers.Constant(1.0),
#             gamma_initializer=tf.keras.initializers.Constant(0.0),
#         ),
#         trainable=False,
#     )
#     rate_cell = psiz.keras.layers.RateSimilarityCell(percept=percept, kernel=proximity)
#     rnn = tf.keras.layers.RNN(rate_cell, return_sequences=True)

#     inp_stimulus_set = tf.keras.Input(
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
#     model = tf.keras.Model(inputs=inputs, outputs=outputs, name="functional_rate")
#     compile_kwargs = {
#         "loss": tf.keras.losses.MeanSquaredError(),
#         "optimizer": tf.keras.optimizers.Adam(learning_rate=0.001),
#         "weighted_metrics": [tf.keras.metrics.MeanSquaredError(name="mse")],
#     }
#     model.compile(**compile_kwargs)
#     return model


def build_alcove_subclass_a():
    """Build subclassed `Model`."""
    model = ALCOVEModelA()
    compile_kwargs = {
        "loss": tf.keras.losses.CategoricalCrossentropy(),
        "optimizer": tf.keras.optimizers.Adam(learning_rate=0.001),
        "weighted_metrics": [tf.keras.metrics.CategoricalAccuracy(name="accuracy")],
    }
    model.compile(**compile_kwargs)
    return model


def build_alcove_functional_v0():
    """Build model using functional API."""
    n_stimuli = 20
    n_dim = 4
    n_output = 3

    percept = tf.keras.layers.Embedding(
        n_stimuli + 1,
        n_dim,
        mask_zero=True,
        trainable=False,
    )
    similarity = psiz.keras.layers.ExponentialSimilarity(
        beta_initializer=tf.keras.initializers.Constant(3.0),
        tau_initializer=tf.keras.initializers.Constant(1.0),
        gamma_initializer=tf.keras.initializers.Constant(0.0),
        trainable=False,
    )
    cell = psiz.keras.layers.ALCOVECell(
        n_output,
        percept=percept,
        similarity=similarity,
        rho_initializer=tf.keras.initializers.Constant(2.0),
        temperature_initializer=tf.keras.initializers.Constant(1.0),
        lr_attention_initializer=tf.keras.initializers.Constant(0.03),
        lr_association_initializer=tf.keras.initializers.Constant(0.03),
        trainable=False,
    )
    rnn = tf.keras.layers.RNN(cell, return_sequences=True, stateful=False)

    inp_stimulus_set = tf.keras.Input(shape=(None, 1), name="categorize_stimulus_set")
    inp_objective_query_label = tf.keras.Input(
        shape=(None, n_output), name="categorize_objective_query_label"
    )
    inputs = {
        "categorize_stimulus_set": inp_stimulus_set,
        "categorize_objective_query_label": inp_objective_query_label,
    }
    outputs = rnn(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="functional_alcove")
    compile_kwargs = {
        "loss": tf.keras.losses.CategoricalCrossentropy(),
        "optimizer": tf.keras.optimizers.Adam(learning_rate=0.001),
        "weighted_metrics": [tf.keras.metrics.CategoricalAccuracy(name="accuracy")],
    }
    model.compile(**compile_kwargs)
    return model


def build_ranksim_ratesim_subclass_a():
    """Build subclassed `Model`."""
    model = RankRateModelA()
    compile_kwargs = {
        "optimizer": tf.keras.optimizers.Adam(learning_rate=0.001),
        "loss": {
            "rank_branch": tf.keras.losses.CategoricalCrossentropy(name="rank_loss"),
            "rate_branch": tf.keras.losses.MeanSquaredError(name="rate_loss"),
        },
        "loss_weights": {"rank_branch": 1.0, "rate_branch": 1.0},
    }
    model.compile(**compile_kwargs)
    return model


def build_ranksim_ratesim_functional_v0():
    """Build model using functional API."""
    n_stimuli = 20
    n_dim = 3
    stimuli_axis = 1

    # Define a percept layer that will be shared across behaviors.
    percept = tf.keras.layers.Embedding(n_stimuli + 1, n_dim, mask_zero=True)

    # Define a proximity layer that will be shared across behaviors.
    proximity = psiz.keras.layers.Minkowski(
        rho_initializer=tf.keras.initializers.Constant(2.0),
        w_initializer=tf.keras.initializers.Constant(1.0),
        trainable=False,
        activation=psiz.keras.layers.ExponentialSimilarity(
            beta_initializer=tf.keras.initializers.Constant(10.0),
            tau_initializer=tf.keras.initializers.Constant(1.0),
            gamma_initializer=tf.keras.initializers.Constant(0.001),
            trainable=False,
        ),
    )

    # Define different behavioral branches.
    soft_4rank2 = psiz.keras.layers.SoftRank(n_select=2)
    rate = psiz.keras.layers.Logistic()

    inp_rank_stimulus_set = tf.keras.Input(shape=(5,), name="given4rank1_stimulus_set")
    inp_rate_stimulus_set = tf.keras.Input(shape=(2,), name="rate2_stimulus_set")
    inp_gate_weights = tf.keras.Input(shape=(1,), name="gate_weights_behavior")
    inputs = {
        "given4rank2_stimulus_set": inp_rank_stimulus_set,
        "rate2_stimulus_set": inp_rate_stimulus_set,
        "gate_weights_behavior": inp_gate_weights,
    }

    # Rank branch.
    z = percept(inputs["given4rank2_stimulus_set"])
    z_q, z_r = tf.split(z, [1, 4], stimuli_axis)
    s = proximity([z_q, z_r])
    output_rank = soft_4rank2(s)

    # Rate branch.
    z = percept(inputs["rate2_stimulus_set"])
    z_0, z_1 = tf.split(z, [1, 1], stimuli_axis)
    s = proximity([z_0, z_1])
    output_rate = rate(s)

    outputs = {"rank_branch": output_rank, "rate_branch": output_rate}
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="functional_rank_rate")
    compile_kwargs = {
        "optimizer": tf.keras.optimizers.Adam(learning_rate=0.001),
        "loss": {
            "rank_branch": tf.keras.losses.CategoricalCrossentropy(name="rank_loss"),
            "rate_branch": tf.keras.losses.MeanSquaredError(name="rate_loss"),
        },
        "loss_weights": {"rank_branch": 1.0, "rate_branch": 1.0},
    }
    model.compile(**compile_kwargs)
    return model


def buld_ranksim_rt_subclass_a():
    """Build subclassed `Model`."""
    model = RankRTModelA()
    compile_kwargs = {
        "optimizer": tf.keras.optimizers.Adam(learning_rate=0.001),
        "loss": {
            "rank_choice_branch": tf.keras.losses.CategoricalCrossentropy(
                name="choice_loss"
            ),
            "rank_rt_branch": tf.keras.losses.MeanSquaredError(name="rt_loss"),
        },
        "loss_weights": {"rank_choice_branch": 1.0, "rank_rt_branch": 1.0},
    }
    model.compile(**compile_kwargs)
    return model


def call_fit_evaluate_predict(model, tfds):
    """Simple test of call, fit, evaluate, and predict."""
    # Test isolated call.
    for data in tfds:
        x, _, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        _ = model(x, training=False)

    # Test fit.
    model.fit(tfds, epochs=3)

    # Test evaluate.
    _ = model.evaluate(tfds)

    # Test predict.
    _ = model.predict(tfds)


class TestRankSimilarity:
    """Test using `RankSimilarity` layer."""

    @pytest.mark.parametrize("is_eager", [True, False])
    def test_usage_subclass_a(self, ds_4rank1_v0, is_eager):
        """Test model using subclass API."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_4rank1_v0
        model = build_ranksim_subclass_a()
        call_fit_evaluate_predict(model, tfds)
        tf.keras.backend.clear_session()

    @pytest.mark.parametrize("is_eager", [True, False])
    @pytest.mark.parametrize("save_traces", [True, False])
    def test_save_load_subclass_a(self, ds_4rank1_v0, is_eager, save_traces, tmpdir):
        """Test serialization."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_4rank1_v0
        model = build_ranksim_subclass_a()
        model.fit(tfds, epochs=1)
        result0 = model.evaluate(tfds)

        # Test storage.
        fp_model = tmpdir.join("test_model")
        model.save(fp_model, save_traces=save_traces)
        del model
        # Load the saved model.
        loaded = tf.keras.models.load_model(
            fp_model, custom_objects={"RankModelA": RankModelA}
        )
        result1 = loaded.evaluate(tfds)

        # Test for model equality.
        assert result0[0] == result1[0]
        assert result0[1] == result1[1]

    @pytest.mark.parametrize("is_eager", [True, False])
    def test_usage_functional_v0(self, ds_4rank1_v0, is_eager):
        """Test model using functional API."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_4rank1_v0
        model = build_ranksim_functional_v0()
        call_fit_evaluate_predict(model, tfds)
        tf.keras.backend.clear_session()

    @pytest.mark.parametrize("is_eager", [True, False])
    def test_save_load_functional_v0(self, ds_4rank1_v0, is_eager, tmpdir):
        """Test serialization."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_4rank1_v0
        model = build_ranksim_functional_v0()
        model.fit(tfds, epochs=1)
        result0 = model.evaluate(tfds)

        # Test storage.
        fp_model = tmpdir.join("test_model")
        model.save(fp_model)
        del model
        # Load the saved model.
        loaded = tf.keras.models.load_model(fp_model)
        result1 = loaded.evaluate(tfds)

        # Test for model equality.
        assert result0[0] == result1[0]
        assert result0[1] == result1[1]

    @pytest.mark.parametrize("is_eager", [True, False])
    def test_usage_subclass_b(self, ds_4rank1_v1, is_eager):
        """Test model using subclass API."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_4rank1_v1
        model = build_ranksim_subclass_b()
        call_fit_evaluate_predict(model, tfds)
        tf.keras.backend.clear_session()

    @pytest.mark.parametrize("is_eager", [True, False])
    def test_usage_subclass_c(self, ds_4rank1_v2, is_eager):
        """Test model using subclass API."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_4rank1_v2
        model = build_ranksim_subclass_c()
        call_fit_evaluate_predict(model, tfds)
        tf.keras.backend.clear_session()

    @pytest.mark.parametrize("is_eager", [True, False])
    def test_usage_subclass_d(self, ds_4rank1_v3, is_eager):
        """Test model using subclass API."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_4rank1_v3
        model = build_ranksim_subclass_d()
        call_fit_evaluate_predict(model, tfds)
        tf.keras.backend.clear_session()

    @pytest.mark.parametrize("is_eager", [True, False])
    def test_usage_subclass_e(self, ds_2rank1_v0, is_eager):
        """Test model using subclass API."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_2rank1_v0
        model = build_ranksim_subclass_e()
        call_fit_evaluate_predict(model, tfds)
        tf.keras.backend.clear_session()

    @pytest.mark.parametrize("is_eager", [True, False])
    def test_usage_subclass_f(self, ds_8rank2_v0, is_eager):
        """Test model using subclass API."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_8rank2_v0
        model = build_ranksim_subclass_f()
        call_fit_evaluate_predict(model, tfds)
        tf.keras.backend.clear_session()

    @pytest.mark.parametrize("is_eager", [True, False])
    def test_agent_subclass_a(self, ds_4rank1_v0, is_eager):
        """Test usage in 'agent mode'."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_4rank1_v0
        model = build_ranksim_subclass_a()

        def simulate_agent(x):
            depth = 4
            outcome_probs = model(x)
            outcome_distribution = tfp.distributions.Categorical(probs=outcome_probs)
            outcome_idx = outcome_distribution.sample()
            outcome_one_hot = tf.one_hot(outcome_idx, depth)
            return outcome_one_hot

        _ = tfds.map(lambda x, y, w: (x, simulate_agent(x), w))

        tf.keras.backend.clear_session()


class TestMultiRankSimilarity:
    """Test using `RankSimilarity` layer."""

    @pytest.mark.parametrize("is_eager", [True, False])
    def test_usage_subclass_a(self, ds_2rank1_8rank2_v0, is_eager):
        """Test model using subclass API."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_2rank1_8rank2_v0
        model = build_multirank_subclass_a()
        call_fit_evaluate_predict(model, tfds)
        tf.keras.backend.clear_session()

    @pytest.mark.parametrize("is_eager", [True, False])
    @pytest.mark.parametrize("save_traces", [True, False])
    def test_save_load_subclass_a(
        self, ds_2rank1_8rank2_v0, is_eager, save_traces, tmpdir
    ):
        """Test serialization."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_2rank1_8rank2_v0
        model = build_multirank_subclass_a()
        model.fit(tfds, epochs=1)
        result0 = model.evaluate(tfds)

        # Test storage.
        fp_model = tmpdir.join("test_model")
        model.save(fp_model, save_traces=save_traces)
        del model
        # Load the saved model.
        loaded = tf.keras.models.load_model(
            fp_model, custom_objects={"MultiRankModelA": MultiRankModelA}
        )
        result1 = loaded.evaluate(tfds)

        # Test for model equality.
        assert result0[0] == result1[0]
        assert result0[1] == result1[1]


# TODO finish or move out
# class TestRankSimilarityCell:
#     """Test using `RankSimilaritycell` layer."""

#     @pytest.mark.parametrize("is_eager", [True, False])
#
#     def test_usage_subclass_a(self, ds_time_8rank2_v0, is_eager):
#         """Test model using subclass API."""
#         tf.config.run_functions_eagerly(is_eager)

#         tfds = ds_time_8rank2_v0
#         model = build_ranksimcell_subclass_a()
#         call_fit_evaluate_predict(model, tfds)
#         tf.keras.backend.clear_session()

#     @pytest.mark.parametrize("is_eager", [True, False])
#     @pytest.mark.parametrize("save_traces", [True, False])
#     def test_save_load_subclass_a(
#         self, ds_time_8rank2_v0, is_eager, save_traces, tmpdir
#     ):
#         """Test serialization."""
#         tf.config.run_functions_eagerly(is_eager)

#         tfds = ds_time_8rank2_v0
#         model = build_ranksimcell_subclass_a()
#         model.fit(tfds, epochs=1)
#         result0 = model.evaluate(tfds)

#         # Test storage.
#         fp_model = tmpdir.join("test_model")
#         model.save(fp_model, save_traces=save_traces)
#         del model
#         # Load the saved model.
#         loaded = tf.keras.models.load_model(
#             fp_model, custom_objects={"RankCellModelA": RankCellModelA}
#         )
#         result1 = loaded.evaluate(tfds)

#         # Test for model equality.
#         assert result0[0] == result1[0]
#         assert result0[1] == result1[1]

#     @pytest.mark.parametrize("is_eager", [True, False])
#     def test_usage_functional_v0(self, ds_time_8rank2_v0, is_eager):
#         """Test model using functional API."""
#         tf.config.run_functions_eagerly(is_eager)

#         tfds = ds_time_8rank2_v0
#         model = build_ranksimcell_functional_v0()
#         call_fit_evaluate_predict(model, tfds)
#         tf.keras.backend.clear_session()

#     @pytest.mark.parametrize("is_eager", [True, False])
#     def test_save_load_functional_v0(self, ds_time_8rank2_v0, is_eager, tmpdir):
#         """Test serialization."""
#         tf.config.run_functions_eagerly(is_eager)

#         tfds = ds_time_8rank2_v0
#         model = build_ranksimcell_functional_v0()
#         model.fit(tfds, epochs=1)
#         result0 = model.evaluate(tfds)

#         # Test storage.
#         fp_model = tmpdir.join("test_model")
#         model.save(fp_model)
#         del model
#         # Load the saved model.
#         loaded = tf.keras.models.load_model(fp_model)
#         result1 = loaded.evaluate(tfds)

#         # Test for model equality.
#         assert result0[0] == result1[0]
#         assert result0[1] == result1[1]


class TestRateSimilarity:
    """Test using `RateSimilarity` layer."""

    @pytest.mark.parametrize("is_eager", [True, False])
    def test_usage_subclass_a(self, ds_rate2_v0, is_eager):
        """Test subclass model, one group."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_rate2_v0
        model = build_ratesim_subclass_a()
        call_fit_evaluate_predict(model, tfds)
        tf.keras.backend.clear_session()

    @pytest.mark.parametrize("is_eager", [True, False])
    @pytest.mark.parametrize("save_traces", [True, False])
    def test_save_load_subclass_a(self, ds_rate2_v0, is_eager, save_traces, tmpdir):
        """Test serialization."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_rate2_v0
        model = build_ratesim_subclass_a()
        model.fit(tfds, epochs=1)
        result0 = model.evaluate(tfds)

        # Test storage.
        fp_model = tmpdir.join("test_model")
        model.save(fp_model, save_traces=save_traces)
        del model
        # Load the saved model.
        loaded = tf.keras.models.load_model(
            fp_model, custom_objects={"RateModelA": RateModelA}
        )
        result1 = loaded.evaluate(tfds)

        # Test for model equality.
        assert result0[0] == result1[0]
        assert result0[1] == result1[1]

    @pytest.mark.parametrize("is_eager", [True, False])
    def test_usage_functional_v0(self, ds_rate2_v0, is_eager):
        """Test model using functional API."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_rate2_v0
        model = build_ratesim_functional_v0()
        call_fit_evaluate_predict(model, tfds)
        tf.keras.backend.clear_session()

    @pytest.mark.parametrize("is_eager", [True, False])
    def test_save_load_functional_v0(self, ds_rate2_v0, is_eager, tmpdir):
        """Test serialization."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_rate2_v0
        model = build_ratesim_functional_v0()
        model.fit(tfds, epochs=1)
        result0 = model.evaluate(tfds)

        # Test storage.
        fp_model = tmpdir.join("test_model")
        model.save(fp_model)
        del model
        # Load the saved model.
        loaded = tf.keras.models.load_model(fp_model)
        result1 = loaded.evaluate(tfds)

        # Test for model equality.
        assert result0[0] == result1[0]
        assert result0[1] == result1[1]

    @pytest.mark.parametrize("is_eager", [True, False])
    def test_usage_subclass_b(self, ds_rate2_v1, is_eager):
        """Test subclass model, one group."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_rate2_v1
        model = build_ratesim_subclass_b()
        call_fit_evaluate_predict(model, tfds)
        tf.keras.backend.clear_session()


# TODO finish or move out
# class TestRateSimilarityCell:
#     """Test using `RateSimilarityCell` layer."""

#     @pytest.mark.parametrize("is_eager", [True, False])
#     def test_usage_subclass_a(self, ds_time_rate2_v0, is_eager):
#         """Test subclass model, one group."""
#         tf.config.run_functions_eagerly(is_eager)

#         tfds = ds_time_rate2_v0
#         model = build_ratesimcell_subclass_a()
#         call_fit_evaluate_predict(model, tfds)
#         tf.keras.backend.clear_session()

#     @pytest.mark.parametrize("is_eager", [True, False])
#     @pytest.mark.parametrize("save_traces", [True, False])
#     def test_save_load_subclass_a(
#         self, ds_time_rate2_v0, is_eager, save_traces, tmpdir
#     ):
#         """Test serialization."""
#         tf.config.run_functions_eagerly(is_eager)

#         tfds = ds_time_rate2_v0
#         model = build_ratesimcell_subclass_a()
#         model.fit(tfds, epochs=1)
#         result0 = model.evaluate(tfds)

#         # Test storage.
#         fp_model = tmpdir.join("test_model")
#         model.save(fp_model, save_traces=save_traces)
#         del model
#         # Load the saved model.
#         loaded = tf.keras.models.load_model(
#             fp_model, custom_objects={"RateCellModelA": RateCellModelA}
#         )
#         result1 = loaded.evaluate(tfds)

#         # Test for model equality.
#         assert result0[0] == result1[0]
#         assert result0[1] == result1[1]

#     @pytest.mark.parametrize("is_eager", [True, False])
#     def test_usage_functional_v0(self, ds_time_rate2_v0, is_eager):
#         """Test model using functional API."""
#         tf.config.run_functions_eagerly(is_eager)

#         tfds = ds_time_rate2_v0
#         model = build_ratesimcell_functional_v0()
#         call_fit_evaluate_predict(model, tfds)
#         tf.keras.backend.clear_session()

#     @pytest.mark.parametrize("is_eager", [True, False])
#     def test_save_load_functional_v0(self, ds_time_rate2_v0, is_eager, tmpdir):
#         """Test serialization."""
#         tf.config.run_functions_eagerly(is_eager)

#         tfds = ds_time_rate2_v0
#         model = build_ratesimcell_functional_v0()
#         model.fit(tfds, epochs=1)
#         result0 = model.evaluate(tfds)

#         # Test storage.
#         fp_model = tmpdir.join("test_model")
#         model.save(fp_model)
#         del model
#         # Load the saved model.
#         loaded = tf.keras.models.load_model(fp_model)
#         result1 = loaded.evaluate(tfds)

#         # Test for model equality.
#         assert result0[0] == result1[0]
#         assert result0[1] == result1[1]


class TestALCOVECell:
    """Test using `ALCOVECell` layer."""

    @pytest.mark.parametrize("is_eager", [True, False])
    def test_usage_subclass_a(self, ds_time_categorize_v0, is_eager):
        """Test subclassed model, one group."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_time_categorize_v0
        model = build_alcove_subclass_a()
        call_fit_evaluate_predict(model, tfds)
        tf.keras.backend.clear_session()

    @pytest.mark.parametrize("is_eager", [True, False])
    @pytest.mark.parametrize("save_traces", [True, False])
    def test_save_load_subclass_a(
        self, ds_time_categorize_v0, is_eager, save_traces, tmpdir
    ):
        """Test serialization."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_time_categorize_v0
        model = build_alcove_subclass_a()
        model.fit(tfds, epochs=1)
        result0 = model.evaluate(tfds)

        # Test storage.
        fp_model = tmpdir.join("test_model")
        model.save(fp_model, save_traces=save_traces)
        del model
        # Load the saved model.
        loaded = tf.keras.models.load_model(
            fp_model, custom_objects={"ALCOVEModelA": ALCOVEModelA}
        )
        result1 = loaded.evaluate(tfds)

        # Test for model equality.
        assert result0[0] == result1[0]
        assert result0[1] == result1[1]

    @pytest.mark.parametrize("is_eager", [True, False])
    def test_usage_functional_v0(self, ds_time_categorize_v0, is_eager):
        """Test model using functional API."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_time_categorize_v0
        model = build_alcove_functional_v0()
        call_fit_evaluate_predict(model, tfds)
        tf.keras.backend.clear_session()

    @pytest.mark.parametrize("is_eager", [True, False])
    def test_save_load_functional_v0(self, ds_time_categorize_v0, is_eager, tmpdir):
        """Test serialization."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_time_categorize_v0
        model = build_alcove_functional_v0()
        model.fit(tfds, epochs=1)
        result0 = model.evaluate(tfds)

        # Test storage.
        fp_model = tmpdir.join("test_model")
        model.save(fp_model)
        del model
        # Load the saved model.
        loaded = tf.keras.models.load_model(fp_model)
        result1 = loaded.evaluate(tfds)

        # Test for model equality.
        assert result0[0] == result1[0]
        assert result0[1] == result1[1]


class TestJointRankRate:
    """Test using joint `RankSimilarity` and `RateSimilarity` layers."""

    @pytest.mark.parametrize("is_eager", [True, False])
    def test_usage_subclass_a(self, ds_4rank2_rate2_v0, is_eager):
        """Test model using subclass API."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_4rank2_rate2_v0
        model = build_ranksim_ratesim_subclass_a()
        call_fit_evaluate_predict(model, tfds)
        tf.keras.backend.clear_session()

    @pytest.mark.parametrize("save_traces", [True, False])
    @pytest.mark.parametrize("is_eager", [True, False])
    def test_save_load_subclass_a(
        self, ds_4rank2_rate2_v0, is_eager, save_traces, tmpdir
    ):
        """Test serialization."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_4rank2_rate2_v0
        model = build_ranksim_ratesim_subclass_a()
        model.fit(tfds, epochs=1)
        result0 = model.evaluate(tfds, return_dict=True)

        # Test storage.
        fp_model = tmpdir.join("test_model")
        model.save(fp_model, save_traces=save_traces)
        del model
        # Load the saved model.
        loaded = tf.keras.models.load_model(
            fp_model, custom_objects={"RankRateModelA": RankRateModelA}
        )
        result1 = loaded.evaluate(tfds, return_dict=True)

        # Test for model equality.
        # TODO placeholder trials generating nan's when rank similarity loss
        # is computed.
        assert result0["rank_branch_loss"] == result1["rank_branch_loss"]
        assert result0["rate_branch_loss"] == result1["rate_branch_loss"]

    @pytest.mark.parametrize("is_eager", [True, False])
    def test_usage_functional_v0(self, ds_4rank2_rate2_v0, is_eager):
        """Test model using functional API."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_4rank2_rate2_v0
        model = build_ranksim_ratesim_functional_v0()
        call_fit_evaluate_predict(model, tfds)
        tf.keras.backend.clear_session()

    @pytest.mark.parametrize("is_eager", [True, False])
    @pytest.mark.xfail(reason="Not sure why failing.")
    def test_save_load_functional_v0(self, ds_4rank2_rate2_v0, is_eager, tmpdir):
        """Test serialization."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_4rank2_rate2_v0
        model = build_ranksim_ratesim_functional_v0()
        model.fit(tfds, epochs=1)
        result0 = model.evaluate(tfds, return_dict=True)

        # Test storage.
        fp_model = tmpdir.join("test_model")
        model.save(fp_model)
        del model
        # Load the saved model.
        loaded = tf.keras.models.load_model(fp_model)
        result1 = loaded.evaluate(tfds, return_dict=True)

        # Test for model equality.
        # TODO placeholder trials generating nan's when rank similarity loss
        # is computed.
        assert result0["rank_branch_loss"] == result1["rank_branch_loss"]
        assert result0["rate_branch_loss"] == result1["rate_branch_loss"]


class TestRankRT:
    """Test using multi-output rank similarity and response times."""

    @pytest.mark.parametrize("is_eager", [True, False])
    def test_usage_subclass_a(self, ds_4rank1_rt_v0, is_eager):
        """Test model using subclass API."""
        tf.config.run_functions_eagerly(is_eager)

        tfds = ds_4rank1_rt_v0
        model = buld_ranksim_rt_subclass_a()
        call_fit_evaluate_predict(model, tfds)
        tf.keras.backend.clear_session()
