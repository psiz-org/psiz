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

import numpy as np
import pytest
import tensorflow as tf

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

        percept = tf.keras.layers.Embedding(
            n_stimuli + 1, n_dim, mask_zero=True
        )
        kernel = psiz.keras.layers.DistanceBased(
            distance=psiz.keras.layers.Minkowski(
                rho_initializer=tf.keras.initializers.Constant(2.),
                w_initializer=tf.keras.initializers.Constant(1.),
                trainable=False,
            ),
            similarity=psiz.keras.layers.ExponentialSimilarity(
                beta_initializer=tf.keras.initializers.Constant(10.),
                tau_initializer=tf.keras.initializers.Constant(1.),
                gamma_initializer=tf.keras.initializers.Constant(0.001),
                trainable=False,
            )
        )
        behavior = psiz.keras.layers.RankSimilarity(
            percept=percept, kernel=kernel
        )

        self.behavior = behavior

    def call(self, inputs, training=None):
        """Call."""
        return self.behavior(inputs)

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

        percept = tf.keras.layers.Embedding(
            n_stimuli + 1, n_dim, mask_zero=True
        )
        # Define group-specific kernels.
        shared_similarity = psiz.keras.layers.ExponentialSimilarity(
            beta_initializer=tf.keras.initializers.Constant(1.),
            tau_initializer=tf.keras.initializers.Constant(1.),
            gamma_initializer=tf.keras.initializers.Constant(0.),
            trainable=False
        )
        kernel_0 = psiz.keras.layers.DistanceBased(
            distance=psiz.keras.layers.Minkowski(
                rho_trainable=False,
                rho_initializer=tf.keras.initializers.Constant(2.),
                w_initializer=tf.keras.initializers.Constant(
                    [1.2, 1., .1, .1, .1, .1, .1, .1, .1, .1]
                ),
            ),
            similarity=shared_similarity
        )
        kernel_1 = psiz.keras.layers.DistanceBased(
            distance=psiz.keras.layers.Minkowski(
                rho_trainable=False,
                rho_initializer=tf.keras.initializers.Constant(2.),
                w_initializer=tf.keras.initializers.Constant(
                    [.1, .1, .1, .1, .1, .1, .1, .1, 1., 1.2]
                ),
            ),
            similarity=shared_similarity
        )
        kernel = psiz.keras.layers.BraidGate(
            subnets=[kernel_0, kernel_1], gating_index=-1
        )

        kernel_adapter = psiz.keras.layers.GateAdapter(
            gating_keys='kernel_gate_weights',
            format_inputs_as_tuple=True
        )
        behavior = psiz.keras.layers.RankSimilarity(
            percept=percept,
            kernel=kernel,
            kernel_adapter=kernel_adapter
        )
        self.behavior = behavior

    def call(self, inputs, training=None):
        """Call."""
        return self.behavior(inputs)

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

        n_stimuli = 3
        n_dim = 2

        # Define group-specific percept layers.
        percept_0 = tf.keras.layers.Embedding(
            n_stimuli + 1, n_dim, mask_zero=True,
            embeddings_initializer=tf.keras.initializers.Constant(
                np.array(
                    [
                        [0.0, 0.0], [.1, .1], [.15, .2], [.4, .5]
                    ], dtype=np.float32
                )
            )
        )
        percept_1 = tf.keras.layers.Embedding(
            n_stimuli + 1, n_dim, mask_zero=True,
            embeddings_initializer=tf.keras.initializers.Constant(
                np.array(
                    [
                        [0.0, 0.0], [.15, .2], [.4, .5], [.1, .1]
                    ], dtype=np.float32
                )
            )
        )
        percept = psiz.keras.layers.BraidGate(
            subnets=[percept_0, percept_1], gating_index=-1
        )

        # Define group-specific kernel layers.
        shared_similarity = psiz.keras.layers.ExponentialSimilarity(
            trainable=False,
            beta_initializer=tf.keras.initializers.Constant(10.),
            tau_initializer=tf.keras.initializers.Constant(1.),
            gamma_initializer=tf.keras.initializers.Constant(0.)
        )
        kernel_0 = psiz.keras.layers.DistanceBased(
            distance=psiz.keras.layers.Minkowski(
                rho_trainable=False,
                rho_initializer=tf.keras.initializers.Constant(2.),
                w_initializer=tf.keras.initializers.Constant(
                    [1.2, .8]
                ),
                w_constraint=psiz.keras.constraints.NonNegNorm(
                    scale=n_dim, p=1.
                ),
            ),
            similarity=shared_similarity
        )
        kernel_1 = psiz.keras.layers.DistanceBased(
            distance=psiz.keras.layers.Minkowski(
                rho_trainable=False,
                rho_initializer=tf.keras.initializers.Constant(2.),
                w_initializer=tf.keras.initializers.Constant(
                    [.7, 1.3]
                ),
                w_constraint=psiz.keras.constraints.NonNegNorm(
                    scale=n_dim, p=1.
                ),
            ),
            similarity=shared_similarity
        )
        kernel = psiz.keras.layers.BraidGate(
            subnets=[kernel_0, kernel_1], gating_index=-1
        )

        percept_adapter = psiz.keras.layers.GateAdapter(
            gating_keys=['percept_gate_weights'],
            format_inputs_as_tuple=True
        )
        kernel_adapter = psiz.keras.layers.GateAdapter(
            gating_keys=['kernel_gate_weights'],
            format_inputs_as_tuple=True
        )
        behavior = psiz.keras.layers.RankSimilarity(
            percept=percept,
            kernel=kernel,
            percept_adapter=percept_adapter,
            kernel_adapter=kernel_adapter
        )
        self.behavior = behavior

    def call(self, inputs, training=None):
        """Call."""
        return self.behavior(inputs)

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

        n_stimuli = 3
        n_dim = 2

        # Define heirarchical percept layers.
        percept_0 = tf.keras.layers.Embedding(
            n_stimuli + 1, n_dim, mask_zero=True,
            embeddings_initializer=tf.keras.initializers.Constant(
                np.array(
                    [
                        [0.0, 0.0], [.1, .1], [.15, .2], [.4, .5]
                    ], dtype=np.float32
                )
            )
        )
        percept_1 = tf.keras.layers.Embedding(
            n_stimuli + 1, n_dim, mask_zero=True,
            embeddings_initializer=tf.keras.initializers.Constant(
                np.array(
                    [
                        [0.0, 0.0], [.15, .2], [.4, .5], [.1, .1]
                    ], dtype=np.float32
                )
            )
        )
        percept_01 = psiz.keras.layers.BraidGate(
            subnets=[percept_0, percept_1], gating_index=-1, name='percept_01'
        )

        percept_2 = tf.keras.layers.Embedding(
            n_stimuli + 1, n_dim, mask_zero=True,
            embeddings_initializer=tf.keras.initializers.Constant(
                np.array(
                    [
                        [0.0, 0.0], [.1, .1], [.15, .2], [.4, .5]
                    ], dtype=np.float32
                )
            )
        )
        percept_3 = tf.keras.layers.Embedding(
            n_stimuli + 1, n_dim, mask_zero=True,
            embeddings_initializer=tf.keras.initializers.Constant(
                np.array(
                    [
                        [0.0, 0.0], [.15, .2], [.4, .5], [.1, .1]
                    ], dtype=np.float32
                )
            )
        )
        percept_23 = psiz.keras.layers.BraidGate(
            subnets=[percept_2, percept_3], gating_index=-1, name='percept_23'
        )

        percept = psiz.keras.layers.BraidGate(
            subnets=[percept_01, percept_23], gating_index=-1, name='percept'
        )
        percept_adapter = psiz.keras.layers.GateAdapter(
            gating_keys=[
                'percept_gate_weights_1', 'percept_gate_weights_0'
            ],
            format_inputs_as_tuple=True
        )

        # Define kernel.
        kernel = psiz.keras.layers.DistanceBased(
            distance=psiz.keras.layers.Minkowski(
                rho_trainable=False,
                rho_initializer=tf.keras.initializers.Constant(2.),
                w_initializer=tf.keras.initializers.Constant(
                    [1.2, .8]
                ),
                w_constraint=psiz.keras.constraints.NonNegNorm(
                    scale=n_dim, p=1.
                ),
            ),
            similarity=psiz.keras.layers.ExponentialSimilarity(
                trainable=False,
                beta_initializer=tf.keras.initializers.Constant(10.),
                tau_initializer=tf.keras.initializers.Constant(1.),
                gamma_initializer=tf.keras.initializers.Constant(0.)
            )
        )

        behavior = psiz.keras.layers.RankSimilarity(
            percept=percept, kernel=kernel, percept_adapter=percept_adapter
        )
        self.behavior = behavior

    def call(self, inputs, training=None):
        """Call."""
        return self.behavior(inputs)

    def get_config(self):
        config = super(RankModelD, self).get_config()
        return config


class RankCellModelA(tf.keras.Model):
    """A `RankSimilarityCell` model.

    Gates:
        None

    """

    def __init__(self, **kwargs):
        """Initialize."""
        super(RankCellModelA, self).__init__(**kwargs)

        n_stimuli = 20
        n_dim = 3

        percept = tf.keras.layers.Embedding(
            n_stimuli + 1, n_dim, mask_zero=True
        )
        kernel = psiz.keras.layers.DistanceBased(
            distance=psiz.keras.layers.Minkowski(
                rho_initializer=tf.keras.initializers.Constant(2.),
                w_initializer=tf.keras.initializers.Constant(1.),
                trainable=False,
            ),
            similarity=psiz.keras.layers.ExponentialSimilarity(
                beta_initializer=tf.keras.initializers.Constant(10.),
                tau_initializer=tf.keras.initializers.Constant(1.),
                gamma_initializer=tf.keras.initializers.Constant(0.001),
                trainable=False,
            )
        )
        cell = psiz.keras.layers.RankSimilarityCell(
            percept=percept, kernel=kernel
        )
        rnn = tf.keras.layers.RNN(cell, return_sequences=True)
        self.behavior = rnn

    def call(self, inputs, training=None):
        """Call."""
        return self.behavior(inputs)

    def get_config(self):
        config = super(RankCellModelA, self).get_config()
        return config


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

        percept = tf.keras.layers.Embedding(
            n_stimuli + 1, n_dim, mask_zero=True
        )
        kernel = psiz.keras.layers.DistanceBased(
            distance=psiz.keras.layers.Minkowski(
                rho_initializer=tf.keras.initializers.Constant(2.),
                w_initializer=tf.keras.initializers.Constant(1.),
                trainable=False
            ),
            similarity=psiz.keras.layers.ExponentialSimilarity(
                trainable=False,
                beta_initializer=tf.keras.initializers.Constant(3.),
                tau_initializer=tf.keras.initializers.Constant(1.),
                gamma_initializer=tf.keras.initializers.Constant(0.),
            )
        )
        behavior = psiz.keras.layers.RateSimilarity(
            percept=percept, kernel=kernel
        )
        self.behavior = behavior

    def call(self, inputs, training=None):
        """Call."""
        return self.behavior(inputs)

    def get_config(self):
        config = super(RateModelA, self).get_config()
        return config


class RateModelB(tf.keras.Model):
    """A `RateSimilarity` model.

    Gates:
        Behavior layer (BraidGate:2) has two independent behavior
            layers each with their own percept and kernel (but shared
            similarity).

    """

    def __init__(self, **kwargs):
        """Initialize."""
        super(RateModelB, self).__init__(**kwargs)

        n_stimuli = 20
        n_dim = 2

        shared_similarity = psiz.keras.layers.ExponentialSimilarity(
            trainable=False,
            beta_initializer=tf.keras.initializers.Constant(10.),
            tau_initializer=tf.keras.initializers.Constant(1.),
            gamma_initializer=tf.keras.initializers.Constant(0.)
        )

        # Group 0 layers.
        percept_0 = tf.keras.layers.Embedding(
            n_stimuli + 1, n_dim, mask_zero=True
        )
        kernel_0 = psiz.keras.layers.DistanceBased(
            distance=psiz.keras.layers.Minkowski(
                rho_trainable=False,
                rho_initializer=tf.keras.initializers.Constant(2.),
                w_initializer=tf.keras.initializers.Constant(
                    [1.2, .8]
                ),
                w_constraint=psiz.keras.constraints.NonNegNorm(
                    scale=n_dim, p=1.
                ),
            ),
            similarity=shared_similarity
        )
        behavior_0 = psiz.keras.layers.RateSimilarity(
            percept=percept_0, kernel=kernel_0,
        )

        # Group 1 layers.
        percept_1 = tf.keras.layers.Embedding(
            n_stimuli + 1, n_dim, mask_zero=True
        )
        kernel_1 = psiz.keras.layers.DistanceBased(
            distance=psiz.keras.layers.Minkowski(
                rho_trainable=False,
                rho_initializer=tf.keras.initializers.Constant(2.),
                w_initializer=tf.keras.initializers.Constant(
                    [.7, 1.3]
                ),
                w_constraint=psiz.keras.constraints.NonNegNorm(
                    scale=n_dim, p=1.
                ),
            ),
            similarity=shared_similarity
        )
        behavior_1 = psiz.keras.layers.RateSimilarity(
            percept=percept_1, kernel=kernel_1,
        )

        # Create behavior-level branch.
        behavior = psiz.keras.layers.BraidGate(
            subnets=[behavior_0, behavior_1],
            gating_key='behavior_gate_weights'
        )
        self.behavior = behavior

    def call(self, inputs, training=None):
        """Call."""
        return self.behavior(inputs)

    def get_config(self):
        config = super(RateModelB, self).get_config()
        return config


class RateCellModelA(tf.keras.Model):
    """A `RateSimilarityCell` model.

    Gates:
        None

    """

    def __init__(self, **kwargs):
        """Initialize."""
        super(RateCellModelA, self).__init__(**kwargs)

        n_stimuli = 30
        n_dim = 10

        percept = tf.keras.layers.Embedding(
            n_stimuli + 1, n_dim, mask_zero=True
        )
        kernel = psiz.keras.layers.DistanceBased(
            distance=psiz.keras.layers.Minkowski(
                rho_initializer=tf.keras.initializers.Constant(2.),
                w_initializer=tf.keras.initializers.Constant(1.),
                trainable=False
            ),
            similarity=psiz.keras.layers.ExponentialSimilarity(
                trainable=False,
                beta_initializer=tf.keras.initializers.Constant(3.),
                tau_initializer=tf.keras.initializers.Constant(1.),
                gamma_initializer=tf.keras.initializers.Constant(0.),
            )
        )
        rate_cell = psiz.keras.layers.RateSimilarityCell(
            percept=percept, kernel=kernel
        )
        rnn = tf.keras.layers.RNN(rate_cell, return_sequences=True)
        self.behavior = rnn

    def call(self, inputs, training=None):
        """Call."""
        return self.behavior(inputs)

    def get_config(self):
        config = super(RateCellModelA, self).get_config()
        return config


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
            n_stimuli + 1, n_dim, mask_zero=True,
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
            lr_attention_initializer=tf.keras.initializers.Constant(.03),
            lr_association_initializer=tf.keras.initializers.Constant(.03),
            trainable=False
        )
        rnn = tf.keras.layers.RNN(cell, return_sequences=True, stateful=False)
        self.behavior = rnn

    def call(self, inputs, training=None):
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

        # Define a percept layer that will be shared across behaviors.
        percept = tf.keras.layers.Embedding(
            n_stimuli + 1, n_dim, mask_zero=True
        )
        # Define a kernel layer that will be shared across behaviors.
        kernel = psiz.keras.layers.DistanceBased(
            distance=psiz.keras.layers.Minkowski(
                rho_initializer=tf.keras.initializers.Constant(2.),
                w_initializer=tf.keras.initializers.Constant(1.),
                trainable=False,
            ),
            similarity=psiz.keras.layers.ExponentialSimilarity(
                beta_initializer=tf.keras.initializers.Constant(10.),
                tau_initializer=tf.keras.initializers.Constant(1.),
                gamma_initializer=tf.keras.initializers.Constant(0.001),
                trainable=False,
            )
        )

        # Define a multi-behavior module
        rank = psiz.keras.layers.RankSimilarity(
            percept=percept, kernel=kernel
        )
        rate = psiz.keras.layers.RateSimilarity(
            percept=percept, kernel=kernel
        )
        behavior_branch = psiz.keras.layers.BranchGate(
            subnets=[rank, rate],
            gating_key='gate_weights_behavior',
            output_names=['rank_branch', 'rate_branch'],
            name="behav_branch",
        )
        self.behavior = behavior_branch

    def call(self, inputs, training=None):
        """Call."""
        return self.behavior(inputs)

    def get_config(self):
        config = super(RankRateModelA, self).get_config()
        return config


def build_ranksim_subclass_a():
    """Build subclassed `Model`.

    RankSimilarity, one group, MLE.

    """
    model = RankModelA()
    compile_kwargs = {
        'loss': tf.keras.losses.CategoricalCrossentropy(),
        'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
        'weighted_metrics': [
            tf.keras.metrics.CategoricalCrossentropy(name='cce')
        ]
    }
    model.compile(**compile_kwargs)
    return model


def build_ranksim_functional_v0():
    """Build model useing functional API.

    RankSimilarity, one group, MLE.

    """
    n_stimuli = 20
    n_dim = 3

    percept = tf.keras.layers.Embedding(
        n_stimuli + 1, n_dim, mask_zero=True
    )
    kernel = psiz.keras.layers.DistanceBased(
        distance=psiz.keras.layers.Minkowski(
            rho_initializer=tf.keras.initializers.Constant(2.),
            w_initializer=tf.keras.initializers.Constant(1.),
            trainable=False,
        ),
        similarity=psiz.keras.layers.ExponentialSimilarity(
            beta_initializer=tf.keras.initializers.Constant(10.),
            tau_initializer=tf.keras.initializers.Constant(1.),
            gamma_initializer=tf.keras.initializers.Constant(0.001),
            trainable=False,
        )
    )
    behavior = psiz.keras.layers.RankSimilarity(
        percept=percept, kernel=kernel
    )

    inp_stimulus_set = tf.keras.Input(
        shape=(9, 56), name='rank_similarity_stimulus_set'
    )
    inp_is_select = tf.keras.Input(
        shape=(9, 1), name='rank_similarity_is_select'
    )
    inputs = {
        'rank_similarity_stimulus_set': inp_stimulus_set,
        'rank_similarity_is_select': inp_is_select
    }
    outputs = behavior(inputs)
    model = tf.keras.Model(
        inputs=inputs, outputs=outputs, name="functional_rank"
    )
    compile_kwargs = {
        'loss': tf.keras.losses.CategoricalCrossentropy(),
        'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
        'weighted_metrics': [
            tf.keras.metrics.CategoricalCrossentropy(name='cce')
        ]
    }
    model.compile(**compile_kwargs)
    return model


def build_ranksim_subclass_b():
    """Build subclassed `Model`.

    RankSimilarity, two kernels, MLE.

    """
    model = RankModelB()
    compile_kwargs = {
        'loss': tf.keras.losses.CategoricalCrossentropy(),
        'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
        'weighted_metrics': [
            tf.keras.metrics.CategoricalCrossentropy(name='cce')
        ]
    }
    model.compile(**compile_kwargs)
    return model


def build_ranksim_subclass_c():
    """Build subclassed `Model`.

    RankSimilarity, two kernels, MLE.

    """
    model = RankModelC()
    compile_kwargs = {
        'loss': tf.keras.losses.CategoricalCrossentropy(),
        'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
        'weighted_metrics': [
            tf.keras.metrics.CategoricalCrossentropy(name='cce')
        ]
    }
    model.compile(**compile_kwargs)
    return model


def build_ranksim_subclass_d():
    """Build subclassed `Model`.

    RankSimilarity, two kernels, MLE.

    """
    model = RankModelD()
    compile_kwargs = {
        'loss': tf.keras.losses.CategoricalCrossentropy(),
        'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
        'weighted_metrics': [
            tf.keras.metrics.CategoricalCrossentropy(name='cce')
        ]
    }
    model.compile(**compile_kwargs)
    return model


def build_ranksimcell_subclass_a():
    """Build subclassed `Model`.

    RankSimilarityCell, one group, MLE.

    """
    model = RankCellModelA()
    compile_kwargs = {
        'loss': tf.keras.losses.CategoricalCrossentropy(),
        'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
        'weighted_metrics': [
            tf.keras.metrics.CategoricalCrossentropy(name='cce')
        ]
    }
    model.compile(**compile_kwargs)
    return model


def build_ranksimcell_functional_v0():
    """Build model useing functional API."""
    n_stimuli = 20
    n_dim = 3

    percept = tf.keras.layers.Embedding(
        n_stimuli + 1, n_dim, mask_zero=True
    )
    kernel = psiz.keras.layers.DistanceBased(
        distance=psiz.keras.layers.Minkowski(
            rho_initializer=tf.keras.initializers.Constant(2.),
            w_initializer=tf.keras.initializers.Constant(1.),
            trainable=False,
        ),
        similarity=psiz.keras.layers.ExponentialSimilarity(
            beta_initializer=tf.keras.initializers.Constant(10.),
            tau_initializer=tf.keras.initializers.Constant(1.),
            gamma_initializer=tf.keras.initializers.Constant(0.001),
            trainable=False,
        )
    )
    cell = psiz.keras.layers.RankSimilarityCell(
        percept=percept, kernel=kernel
    )
    rnn = tf.keras.layers.RNN(cell, return_sequences=True)

    inp_stimulus_set = tf.keras.Input(
        shape=(None, 9, 56), name='rank_similarity_stimulus_set'
    )
    inp_is_select = tf.keras.Input(
        shape=(None, 9, 1), name='rank_similarity_is_select'
    )
    inputs = {
        'rank_similarity_stimulus_set': inp_stimulus_set,
        'rank_similarity_is_select': inp_is_select
    }
    outputs = rnn(inputs)
    model = tf.keras.Model(
        inputs=inputs, outputs=outputs, name="functional_rank"
    )
    compile_kwargs = {
        'loss': tf.keras.losses.CategoricalCrossentropy(),
        'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
        'weighted_metrics': [
            tf.keras.metrics.CategoricalCrossentropy(name='cce')
        ]
    }
    model.compile(**compile_kwargs)
    return model


def build_ratesim_subclass_a():
    """Build subclassed `Model`."""
    model = RateModelA()
    compile_kwargs = {
        'loss': tf.keras.losses.MeanSquaredError(),
        'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
        'weighted_metrics': [
            tf.keras.metrics.MeanSquaredError(name='mse')
        ]
    }
    model.compile(**compile_kwargs)
    return model


def build_ratesim_functional_v0():
    """Build model using functional API."""
    n_stimuli = 30
    n_dim = 10

    percept = tf.keras.layers.Embedding(
        n_stimuli + 1, n_dim, mask_zero=True
    )
    kernel = psiz.keras.layers.DistanceBased(
        distance=psiz.keras.layers.Minkowski(
            rho_initializer=tf.keras.initializers.Constant(2.),
            w_initializer=tf.keras.initializers.Constant(1.),
            trainable=False
        ),
        similarity=psiz.keras.layers.ExponentialSimilarity(
            trainable=False,
            beta_initializer=tf.keras.initializers.Constant(3.),
            tau_initializer=tf.keras.initializers.Constant(1.),
            gamma_initializer=tf.keras.initializers.Constant(0.),
        )
    )
    behavior = psiz.keras.layers.RateSimilarity(
        percept=percept, kernel=kernel
    )

    inp_stimulus_set = tf.keras.Input(
        shape=(2,), name='rate_similarity_stimulus_set'
    )
    inputs = {
        'rate_similarity_stimulus_set': inp_stimulus_set,
    }
    outputs = behavior(inputs)
    model = tf.keras.Model(
        inputs=inputs, outputs=outputs, name="functional_rate"
    )
    compile_kwargs = {
        'loss': tf.keras.losses.MeanSquaredError(),
        'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
        'weighted_metrics': [
            tf.keras.metrics.MeanSquaredError(name='mse')
        ]
    }
    model.compile(**compile_kwargs)
    return model


def build_ratesim_subclass_b():
    """Build subclassed `Model`."""
    model = RateModelB()
    compile_kwargs = {
        'loss': tf.keras.losses.MeanSquaredError(),
        'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
        'weighted_metrics': [
            tf.keras.metrics.MeanSquaredError(name='mse')
        ]
    }
    model.compile(**compile_kwargs)
    return model


def build_ratesimcell_subclass_a():
    """Build subclassed `Model`."""
    model = RateCellModelA()
    compile_kwargs = {
        'loss': tf.keras.losses.MeanSquaredError(),
        'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
        'weighted_metrics': [
            tf.keras.metrics.MeanSquaredError(name='mse')
        ]
    }
    model.compile(**compile_kwargs)
    return model


def build_ratesimcell_functional_v0():
    """Build model useing functional API."""
    n_stimuli = 30
    n_dim = 10

    percept = tf.keras.layers.Embedding(
        n_stimuli + 1, n_dim, mask_zero=True
    )
    kernel = psiz.keras.layers.DistanceBased(
        distance=psiz.keras.layers.Minkowski(
            rho_initializer=tf.keras.initializers.Constant(2.),
            w_initializer=tf.keras.initializers.Constant(1.),
            trainable=False
        ),
        similarity=psiz.keras.layers.ExponentialSimilarity(
            trainable=False,
            beta_initializer=tf.keras.initializers.Constant(3.),
            tau_initializer=tf.keras.initializers.Constant(1.),
            gamma_initializer=tf.keras.initializers.Constant(0.),
        )
    )
    rate_cell = psiz.keras.layers.RateSimilarityCell(
        percept=percept, kernel=kernel
    )
    rnn = tf.keras.layers.RNN(rate_cell, return_sequences=True)

    inp_stimulus_set = tf.keras.Input(
        shape=(None, 2,), name='rate_similarity_stimulus_set'
    )
    inputs = {
        'rate_similarity_stimulus_set': inp_stimulus_set,
    }
    outputs = rnn(inputs)
    model = tf.keras.Model(
        inputs=inputs, outputs=outputs, name="functional_rate"
    )
    compile_kwargs = {
        'loss': tf.keras.losses.MeanSquaredError(),
        'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
        'weighted_metrics': [
            tf.keras.metrics.MeanSquaredError(name='mse')
        ]
    }
    model.compile(**compile_kwargs)
    return model


def build_alcove_subclass_a():
    """Build subclassed `Model`."""
    model = ALCOVEModelA()
    compile_kwargs = {
        'loss': tf.keras.losses.CategoricalCrossentropy(),
        'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
        'weighted_metrics': [
            tf.keras.metrics.CategoricalAccuracy(name='accuracy')
        ]
    }
    model.compile(**compile_kwargs)
    return model


def build_alcove_functional_v0():
    """Build model using functional API."""
    n_stimuli = 20
    n_dim = 4
    n_output = 3

    percept = tf.keras.layers.Embedding(
        n_stimuli + 1, n_dim, mask_zero=True,
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
        lr_attention_initializer=tf.keras.initializers.Constant(.03),
        lr_association_initializer=tf.keras.initializers.Constant(.03),
        trainable=False
    )
    rnn = tf.keras.layers.RNN(cell, return_sequences=True, stateful=False)

    inp_stimulus_set = tf.keras.Input(
        shape=(None, 1), name='categorize_stimulus_set'
    )
    inp_correct_label = tf.keras.Input(
        shape=(None, 1,), name='categorize_correct_label'
    )
    inputs = {
        'categorize_stimulus_set': inp_stimulus_set,
        'categorize_correct_label': inp_correct_label,
    }
    outputs = rnn(inputs)
    model = tf.keras.Model(
        inputs=inputs, outputs=outputs, name="functional_alcove"
    )
    compile_kwargs = {
        'loss': tf.keras.losses.CategoricalCrossentropy(),
        'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
        'weighted_metrics': [
            tf.keras.metrics.CategoricalAccuracy(name='accuracy')
        ]
    }
    model.compile(**compile_kwargs)
    return model


def buld_ranksim_ratesim_subclass_a():
    """Build subclassed `Model`."""
    model = RankRateModelA()
    compile_kwargs = {
        'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
        'loss': {
            'rank_branch': tf.keras.losses.CategoricalCrossentropy(
                name='rank_loss'
            ),
            'rate_branch': tf.keras.losses.MeanSquaredError(
                name='rate_loss'
            ),
        },
        'loss_weights': {'rank_branch': 1.0, 'rate_branch': 1.0},
    }
    model.compile(**compile_kwargs)
    return model


def build_ranksim_ratesim_functional_v0():
    """Build model using functional API."""
    n_stimuli = 20
    n_dim = 3

    # Define a percept layer that will be shared across behaviors.
    percept = tf.keras.layers.Embedding(
        n_stimuli + 1, n_dim, mask_zero=True
    )
    # Define a kernel layer that will be shared across behaviors.
    kernel = psiz.keras.layers.DistanceBased(
        distance=psiz.keras.layers.Minkowski(
            rho_initializer=tf.keras.initializers.Constant(2.),
            w_initializer=tf.keras.initializers.Constant(1.),
            trainable=False,
        ),
        similarity=psiz.keras.layers.ExponentialSimilarity(
            beta_initializer=tf.keras.initializers.Constant(10.),
            tau_initializer=tf.keras.initializers.Constant(1.),
            gamma_initializer=tf.keras.initializers.Constant(0.001),
            trainable=False,
        )
    )

    # Define a multi-behavior module
    rank = psiz.keras.layers.RankSimilarity(
        percept=percept, kernel=kernel
    )
    rate = psiz.keras.layers.RateSimilarity(
        percept=percept, kernel=kernel
    )
    behavior_branch = psiz.keras.layers.BranchGate(
        subnets=[rank, rate],
        gating_key='gate_weights_behavior',
        output_names=['rank_branch', 'rate_branch'],
        name="behav_branch",
    )

    inp_rank_stimulus_set = tf.keras.Input(
        shape=(9, 56), name='rank_similarity_stimulus_set'
    )
    inp_rank_is_select = tf.keras.Input(
        shape=(9, 1), name='rank_similarity_is_select'
    )
    inp_rate_stimulus_set = tf.keras.Input(
        shape=(2,), name='rate_similarity_stimulus_set'
    )
    inp_gate_weights = tf.keras.Input(
        shape=(1,), name='gate_weights_behavior'
    )
    inputs = {
        'rank_similarity_stimulus_set': inp_rank_stimulus_set,
        'rank_similarity_is_select': inp_rank_is_select,
        'rate_similarity_stimulus_set': inp_rate_stimulus_set,
        'gate_weights_behavior': inp_gate_weights
    }
    outputs = behavior_branch(inputs)
    model = tf.keras.Model(
        inputs=inputs, outputs=outputs, name="functional_rank_rate"
    )
    compile_kwargs = {
        'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
        'loss': {
            'rank_branch': tf.keras.losses.CategoricalCrossentropy(
                name='rank_loss'
            ),
            'rate_branch': tf.keras.losses.MeanSquaredError(
                name='rate_loss'
            ),
        },
        'loss_weights': {'rank_branch': 1.0, 'rate_branch': 1.0},
    }
    model.compile(**compile_kwargs)
    return model


def call_fit_evaluate_predict(model, ds):
    """Simple test of call, fit, evaluate, and predict."""
    # Test isolated call.
    for data in ds:
        x, _, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        _ = model(x, training=False)

    # Test fit.
    model.fit(ds, epochs=3)

    # Test evaluate.
    _ = model.evaluate(ds)

    # Test predict.
    _ = model.predict(ds)


class TestRankSimilarity:
    """Test using `RankSimilarity` layer."""

    @pytest.mark.parametrize(
        "is_eager", [True, False]
    )
    def test_usage_subclass_a(self, ds_ranksim_v0, is_eager):
        """Test model using subclass API."""
        tf.config.run_functions_eagerly(is_eager)

        ds = ds_ranksim_v0
        model = build_ranksim_subclass_a()
        call_fit_evaluate_predict(model, ds)
        tf.keras.backend.clear_session()

    @pytest.mark.parametrize(
        "is_eager", [True, False]
    )
    @pytest.mark.parametrize(
        "save_traces", [True, False]
    )
    def test_save_load_subclass_a(
        self, ds_ranksim_v0, is_eager, save_traces, tmpdir
    ):
        """Test serialization."""
        tf.config.run_functions_eagerly(is_eager)

        ds = ds_ranksim_v0
        model = build_ranksim_subclass_a()
        model.fit(ds, epochs=1)
        result0 = model.evaluate(ds)

        # Test storage.
        fp_model = tmpdir.join('test_model')
        model.save(fp_model, save_traces=save_traces)
        del model
        # Load the saved model.
        loaded = tf.keras.models.load_model(
            fp_model, custom_objects={"RankModelA": RankModelA}
        )
        result1 = loaded.evaluate(ds)

        # Test for model equality.
        assert result0[0] == result1[0]
        assert result0[1] == result1[1]

    @pytest.mark.parametrize(
        "is_eager", [True, False]
    )
    def test_usage_functional_v0(self, ds_ranksim_v0, is_eager):
        """Test model using functional API."""
        tf.config.run_functions_eagerly(is_eager)

        ds = ds_ranksim_v0
        model = build_ranksim_functional_v0()
        call_fit_evaluate_predict(model, ds)
        tf.keras.backend.clear_session()

    @pytest.mark.parametrize(
        "is_eager", [True, False]
    )
    def test_save_load_functional_v0(self, ds_ranksim_v0, is_eager, tmpdir):
        """Test serialization."""
        tf.config.run_functions_eagerly(is_eager)

        ds = ds_ranksim_v0
        model = build_ranksim_functional_v0()
        model.fit(ds, epochs=1)
        result0 = model.evaluate(ds)

        # Test storage.
        fp_model = tmpdir.join('test_model')
        model.save(fp_model)
        del model
        # Load the saved model.
        loaded = tf.keras.models.load_model(fp_model)
        result1 = loaded.evaluate(ds)

        # Test for model equality.
        assert result0[0] == result1[0]
        assert result0[1] == result1[1]

    @pytest.mark.parametrize(
        "is_eager", [True, False]
    )
    def test_usage_subclass_b(self, ds_ranksim_v1, is_eager):
        """Test model using subclass API."""
        tf.config.run_functions_eagerly(is_eager)

        ds = ds_ranksim_v1
        model = build_ranksim_subclass_b()
        call_fit_evaluate_predict(model, ds)
        tf.keras.backend.clear_session()

    @pytest.mark.parametrize(
        "is_eager", [True, False]
    )
    def test_usage_subclass_c(self, ds_ranksim_v2, is_eager):
        """Test model using subclass API."""
        tf.config.run_functions_eagerly(is_eager)

        ds = ds_ranksim_v2
        model = build_ranksim_subclass_c()
        call_fit_evaluate_predict(model, ds)
        tf.keras.backend.clear_session()

    @pytest.mark.parametrize(
        "is_eager", [True, False]
    )
    def test_usage_subclass_d(self, ds_ranksim_v3, is_eager):
        """Test model using subclass API."""
        tf.config.run_functions_eagerly(is_eager)

        ds = ds_ranksim_v3
        model = build_ranksim_subclass_d()
        call_fit_evaluate_predict(model, ds)
        tf.keras.backend.clear_session()


class TestRankSimilarityCell:
    """Test using `RankSimilaritycell` layer."""

    @pytest.mark.parametrize(
        "is_eager", [True, False]
    )
    def test_usage_subclass_a(self, ds_ranksimcell_v0, is_eager):
        """Test model using subclass API."""
        tf.config.run_functions_eagerly(is_eager)

        ds = ds_ranksimcell_v0
        model = build_ranksimcell_subclass_a()
        call_fit_evaluate_predict(model, ds)
        tf.keras.backend.clear_session()

    @pytest.mark.parametrize(
        "is_eager", [True, False]
    )
    @pytest.mark.parametrize(
        "save_traces", [True, False]
    )
    def test_save_load_subclass_a(
        self, ds_ranksimcell_v0, is_eager, save_traces, tmpdir
    ):
        """Test serialization."""
        tf.config.run_functions_eagerly(is_eager)

        ds = ds_ranksimcell_v0
        model = build_ranksimcell_subclass_a()
        model.fit(ds, epochs=1)
        result0 = model.evaluate(ds)

        # Test storage.
        fp_model = tmpdir.join('test_model')
        model.save(fp_model, save_traces=save_traces)
        del model
        # Load the saved model.
        loaded = tf.keras.models.load_model(
            fp_model, custom_objects={"RankCellModelA": RankCellModelA}
        )
        result1 = loaded.evaluate(ds)

        # Test for model equality.
        assert result0[0] == result1[0]
        assert result0[1] == result1[1]

    @pytest.mark.parametrize(
        "is_eager", [True, False]
    )
    def test_usage_functional_v0(self, ds_ranksimcell_v0, is_eager):
        """Test model using functional API."""
        tf.config.run_functions_eagerly(is_eager)

        ds = ds_ranksimcell_v0
        model = build_ranksimcell_functional_v0()
        call_fit_evaluate_predict(model, ds)
        tf.keras.backend.clear_session()

    @pytest.mark.parametrize(
        "is_eager", [True, False]
    )
    def test_save_load_functional_v0(
        self, ds_ranksimcell_v0, is_eager, tmpdir
    ):
        """Test serialization."""
        tf.config.run_functions_eagerly(is_eager)

        ds = ds_ranksimcell_v0
        model = build_ranksimcell_functional_v0()
        model.fit(ds, epochs=1)
        result0 = model.evaluate(ds)

        # Test storage.
        fp_model = tmpdir.join('test_model')
        model.save(fp_model)
        del model
        # Load the saved model.
        loaded = tf.keras.models.load_model(fp_model)
        result1 = loaded.evaluate(ds)

        # Test for model equality.
        assert result0[0] == result1[0]
        assert result0[1] == result1[1]


class TestRateSimilarity:
    """Test using `RateSimilarity` layer."""

    @pytest.mark.parametrize(
        "is_eager", [True, False]
    )
    def test_usage_subclass_a(self, ds_ratesim_v0, is_eager):
        """Test subclass model, one group."""
        tf.config.run_functions_eagerly(is_eager)

        ds = ds_ratesim_v0
        model = build_ratesim_subclass_a()
        call_fit_evaluate_predict(model, ds)
        tf.keras.backend.clear_session()

    @pytest.mark.parametrize(
        "is_eager", [True, False]
    )
    @pytest.mark.parametrize(
        "save_traces", [True, False]
    )
    def test_save_load_subclass_a(
        self, ds_ratesim_v0, is_eager, save_traces, tmpdir
    ):
        """Test serialization."""
        tf.config.run_functions_eagerly(is_eager)

        ds = ds_ratesim_v0
        model = build_ratesim_subclass_a()
        model.fit(ds, epochs=1)
        result0 = model.evaluate(ds)

        # Test storage.
        fp_model = tmpdir.join('test_model')
        model.save(fp_model, save_traces=save_traces)
        del model
        # Load the saved model.
        loaded = tf.keras.models.load_model(
            fp_model, custom_objects={"RateModelA": RateModelA}
        )
        result1 = loaded.evaluate(ds)

        # Test for model equality.
        assert result0[0] == result1[0]
        assert result0[1] == result1[1]

    @pytest.mark.parametrize(
        "is_eager", [True, False]
    )
    def test_usage_functional_v0(self, ds_ratesim_v0, is_eager):
        """Test model using functional API."""
        tf.config.run_functions_eagerly(is_eager)

        ds = ds_ratesim_v0
        model = build_ratesim_functional_v0()
        call_fit_evaluate_predict(model, ds)
        tf.keras.backend.clear_session()

    @pytest.mark.parametrize(
        "is_eager", [True, False]
    )
    def test_save_load_functional_v0(self, ds_ratesim_v0, is_eager, tmpdir):
        """Test serialization."""
        tf.config.run_functions_eagerly(is_eager)

        ds = ds_ratesim_v0
        model = build_ratesim_functional_v0()
        model.fit(ds, epochs=1)
        result0 = model.evaluate(ds)

        # Test storage.
        fp_model = tmpdir.join('test_model')
        model.save(fp_model)
        del model
        # Load the saved model.
        loaded = tf.keras.models.load_model(fp_model)
        result1 = loaded.evaluate(ds)

        # Test for model equality.
        assert result0[0] == result1[0]
        assert result0[1] == result1[1]

    @pytest.mark.parametrize(
        "is_eager", [True, False]
    )
    def test_usage_subclass_b(self, ds_ratesim_v1, is_eager):
        """Test subclass model, one group."""
        tf.config.run_functions_eagerly(is_eager)

        ds = ds_ratesim_v1
        model = build_ratesim_subclass_b()
        call_fit_evaluate_predict(model, ds)
        tf.keras.backend.clear_session()


class TestRateSimilarityCell:
    """Test using `RateSimilarityCell` layer."""

    @pytest.mark.parametrize(
        "is_eager", [True, False]
    )
    def test_usage_subclass_a(self, ds_ratesimcell_v0, is_eager):
        """Test subclass model, one group."""
        tf.config.run_functions_eagerly(is_eager)

        ds = ds_ratesimcell_v0
        model = build_ratesimcell_subclass_a()
        call_fit_evaluate_predict(model, ds)
        tf.keras.backend.clear_session()

    @pytest.mark.parametrize(
        "is_eager", [True, False]
    )
    @pytest.mark.parametrize(
        "save_traces", [True, False]
    )
    def test_save_load_subclass_a(
        self, ds_ratesimcell_v0, is_eager, save_traces, tmpdir
    ):
        """Test serialization."""
        tf.config.run_functions_eagerly(is_eager)

        ds = ds_ratesimcell_v0
        model = build_ratesimcell_subclass_a()
        model.fit(ds, epochs=1)
        result0 = model.evaluate(ds)

        # Test storage.
        fp_model = tmpdir.join('test_model')
        model.save(fp_model, save_traces=save_traces)
        del model
        # Load the saved model.
        loaded = tf.keras.models.load_model(
            fp_model, custom_objects={"RateCellModelA": RateCellModelA}
        )
        result1 = loaded.evaluate(ds)

        # Test for model equality.
        assert result0[0] == result1[0]
        assert result0[1] == result1[1]

    @pytest.mark.parametrize(
        "is_eager", [True, False]
    )
    def test_usage_functional_v0(self, ds_ratesimcell_v0, is_eager):
        """Test model using functional API."""
        tf.config.run_functions_eagerly(is_eager)

        ds = ds_ratesimcell_v0
        model = build_ratesimcell_functional_v0()
        call_fit_evaluate_predict(model, ds)
        tf.keras.backend.clear_session()

    @pytest.mark.parametrize(
        "is_eager", [True, False]
    )
    def test_save_load_functional_v0(
        self, ds_ratesimcell_v0, is_eager, tmpdir
    ):
        """Test serialization."""
        tf.config.run_functions_eagerly(is_eager)

        ds = ds_ratesimcell_v0
        model = build_ratesimcell_functional_v0()
        model.fit(ds, epochs=1)
        result0 = model.evaluate(ds)

        # Test storage.
        fp_model = tmpdir.join('test_model')
        model.save(fp_model)
        del model
        # Load the saved model.
        loaded = tf.keras.models.load_model(fp_model)
        result1 = loaded.evaluate(ds)

        # Test for model equality.
        assert result0[0] == result1[0]
        assert result0[1] == result1[1]


class TestALCOVECell:
    """Test using `ALCOVECell` layer."""

    @pytest.mark.parametrize(
        "is_eager", [True, False]
    )
    def test_usage_subclass_a(self, ds_categorize_v0, is_eager):
        """Test subclassed model, one group."""
        tf.config.run_functions_eagerly(is_eager)

        ds = ds_categorize_v0
        model = build_alcove_subclass_a()
        call_fit_evaluate_predict(model, ds)
        tf.keras.backend.clear_session()

    @pytest.mark.parametrize(
        "is_eager", [True, False]
    )
    @pytest.mark.parametrize(
        "save_traces", [True, False]
    )
    def test_save_load_subclass_a(
        self, ds_categorize_v0, is_eager, save_traces, tmpdir
    ):
        """Test serialization."""
        tf.config.run_functions_eagerly(is_eager)

        ds = ds_categorize_v0
        model = build_alcove_subclass_a()
        model.fit(ds, epochs=1)
        result0 = model.evaluate(ds)

        # Test storage.
        fp_model = tmpdir.join('test_model')
        model.save(fp_model, save_traces=save_traces)
        del model
        # Load the saved model.
        loaded = tf.keras.models.load_model(
            fp_model, custom_objects={"ALCOVEModelA": ALCOVEModelA}
        )
        result1 = loaded.evaluate(ds)

        # Test for model equality.
        assert result0[0] == result1[0]
        assert result0[1] == result1[1]

    @pytest.mark.parametrize(
        "is_eager", [True, False]
    )
    def test_usage_functional_v0(self, ds_categorize_v0, is_eager):
        """Test model using functional API."""
        tf.config.run_functions_eagerly(is_eager)

        ds = ds_categorize_v0
        model = build_alcove_functional_v0()
        call_fit_evaluate_predict(model, ds)
        tf.keras.backend.clear_session()

    @pytest.mark.parametrize(
        "is_eager", [True, False]
    )
    def test_save_load_functional_v0(self, ds_categorize_v0, is_eager, tmpdir):
        """Test serialization."""
        tf.config.run_functions_eagerly(is_eager)

        ds = ds_categorize_v0
        model = build_alcove_functional_v0()
        model.fit(ds, epochs=1)
        result0 = model.evaluate(ds)

        # Test storage.
        fp_model = tmpdir.join('test_model')
        model.save(fp_model)
        del model
        # Load the saved model.
        loaded = tf.keras.models.load_model(fp_model)
        result1 = loaded.evaluate(ds)

        # Test for model equality.
        assert result0[0] == result1[0]
        assert result0[1] == result1[1]


class TestJointRankRate:
    """Test using joint `RankSimilarity` and `RateSimilarity` layers."""

    @pytest.mark.parametrize(
        "is_eager", [True, False]
    )
    def test_usage_subclass_a(self, ds_ranksim_ratesim_v0, is_eager):
        """Test model using subclass API."""
        tf.config.run_functions_eagerly(is_eager)

        ds = ds_ranksim_ratesim_v0
        model = buld_ranksim_ratesim_subclass_a()
        call_fit_evaluate_predict(model, ds)
        tf.keras.backend.clear_session()

    # TODO add `save_traces` parameterization once other cases working
    # @pytest.mark.parametrize(
    #     "save_traces", [True, False]
    # )
    @pytest.mark.parametrize(
        "is_eager", [True, False]
    )
    @pytest.mark.xfail(
        reason="nan's in loss of RankSimilarity branch"
    )
    def test_save_load_subclass_a(
        self, ds_ranksim_ratesim_v0, is_eager, tmpdir
    ):
        """Test serialization."""
        tf.config.run_functions_eagerly(is_eager)

        ds = ds_ranksim_ratesim_v0
        model = buld_ranksim_ratesim_subclass_a()
        model.fit(ds, epochs=1)
        result0 = model.evaluate(ds, return_dict=True)

        # Test storage.
        fp_model = tmpdir.join('test_model')
        model.save(fp_model)
        del model
        # Load the saved model.
        loaded = tf.keras.models.load_model(
            fp_model, custom_objects={"RankModelA": RankModelA}
        )
        result1 = loaded.evaluate(ds, return_dict=True)

        # Test for model equality.
        # TODO placeholder trials generating nan's when rank similarity loss
        # is computed.
        assert result0['rank_branch_loss'] == result1['rank_branch_loss']
        assert result0['rate_branch_loss'] == result1['rate_branch_loss']

    @pytest.mark.parametrize(
        "is_eager", [True, False]
    )
    def test_usage_functional_v0(self, ds_ranksim_ratesim_v0, is_eager):
        """Test model using functional API."""
        tf.config.run_functions_eagerly(is_eager)

        ds = ds_ranksim_ratesim_v0
        model = build_ranksim_ratesim_functional_v0()
        call_fit_evaluate_predict(model, ds)
        tf.keras.backend.clear_session()

    @pytest.mark.parametrize(
        "is_eager", [True, False]
    )
    @pytest.mark.xfail(
        reason="nan's in loss of RankSimilarity branch"
    )
    def test_save_load_functional_v0(
        self, ds_ranksim_ratesim_v0, is_eager, tmpdir
    ):
        """Test serialization."""
        tf.config.run_functions_eagerly(is_eager)

        ds = ds_ranksim_ratesim_v0
        model = build_ranksim_ratesim_functional_v0()
        model.fit(ds, epochs=1)
        result0 = model.evaluate(ds, return_dict=True)

        # Test storage.
        fp_model = tmpdir.join('test_model')
        model.save(fp_model)
        del model
        # Load the saved model.
        loaded = tf.keras.models.load_model(fp_model)
        result1 = loaded.evaluate(ds, return_dict=True)

        # Test for model equality.
        # TODO placeholder trials generating nan's when rank similarity loss
        # is computed.
        assert result0['rank_branch_loss'] == result1['rank_branch_loss']
        assert result0['rate_branch_loss'] == result1['rate_branch_loss']
