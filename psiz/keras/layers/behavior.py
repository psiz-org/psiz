# -*- coding: utf-8 -*-
# Copyright 2020 The PsiZ Authors. All Rights Reserved.
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
"""Module of TensorFlow behavior layers.

Classes:
    Behavior: An abstract behavior layer.
    RankBehavior: A rank behavior layer.
    RateBehavior: A rate behavior layer.
    SortBehavior: A sort behavior layer.

"""

import tensorflow as tf
from tensorflow.python.keras import backend as K

import psiz.keras.constraints as pk_constraints
from psiz.models.base import GroupLevel


class Behavior(GroupLevel):
    """An abstract behavior layer."""

    def __init__(self, **kwargs):
        """Initialize.

        Arguments:
            kwargs (optional): Additional keyword arguments.

        """
        super(Behavior, self).__init__(**kwargs)

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        return config

    def call(self, inputs):
        raise NotImplementedError


@tf.keras.utils.register_keras_serializable(
    package='psiz.keras.layers', name='RankBehavior'
)
class RankBehavior(Behavior):
    """A rank behavior layer.

    Embodies a `_tf_ranked_sequence_probability` call.

    """

    def __init__(self, **kwargs):
        """Initialize.

        Arguments:
            kwargs (optional): Additional keyword arguments.

        """
        super(RankBehavior, self).__init__(**kwargs)

    def call(self, inputs):
        """Return probability of a ranked selection sequence.

        See: _ranked_sequence_probability for NumPy implementation.

        Arguments:
            inputs:
                sim_qr: A tensor containing the precomputed
                    similarities between the query stimuli and
                    corresponding reference stimuli.
                    shape = (batch_size, n_max_reference, n_outcome)
                is_select: A Boolean tensor indicating if a reference
                    was selected.
                    shape = (batch_size, n_max_reference, n_outcome)

        """
        sim_qr = inputs[0]
        is_select = inputs[1]
        is_outcome = inputs[2]

        # Initialize sequence log-probability. Note that log(prob=1)=1.
        # sample_size = tf.shape(sim_qr)[0] 
        # batch_size = tf.shape(sim_qr)[1]
        # n_outcome = tf.shape(sim_qr)[3]
        # seq_log_prob = tf.zeros(
        #     [sample_size, batch_size, n_outcome], dtype=K.floatx()
        # )

        # Compute denominator based on formulation of Luce's choice rule.
        denom = tf.cumsum(sim_qr, axis=2, reverse=True)

        # Compute log-probability of each selection, assuming all selections
        # occurred. Add fuzz factor to avoid log(0)
        sim_qr = tf.maximum(sim_qr, tf.keras.backend.epsilon())
        denom = tf.maximum(denom, tf.keras.backend.epsilon())
        log_prob = tf.math.log(sim_qr) - tf.math.log(denom)

        # Mask non-existent selections.
        log_prob = is_select * log_prob

        # Compute sequence log-probability
        seq_log_prob = tf.reduce_sum(log_prob, axis=2)
        seq_prob = tf.math.exp(seq_log_prob)
        return is_outcome * seq_prob

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        return config


@tf.keras.utils.register_keras_serializable(
    package='psiz.keras.layers', name='RateBehavior'
)
class RateBehavior(Behavior):
    """A rate behavior layer.

    Similarities are converted to probabilities using a parameterized
    logistic function,

    p(x) = lower + ((upper - lower) / (1 + exp(-rate*(x - midpoint))))

    with the following variable meanings:
    `lower`: The lower asymptote of the function's range.
    `upper`: The upper asymptote of the function's range.
    `midpoint`: The midpoint of the function's domain and point of
        maximum growth.
    `rate`: The growth rate of the logistic function.

    """

    def __init__(
            self, lower_initializer=None, upper_initializer=None,
            midpoint_initializer=None, rate_initializer=None,
            lower_trainable=False, upper_trainable=False,
            midpoint_trainable=True, rate_trainable=True, **kwargs):
        """Initialize.

        Arguments:
            lower_initializer (optional): TensorFlow initializer.
            upper_initializer (optional): TensorFlow initializer.
            midpoint_initializer (optional): TensorFlow initializer.
            rate_initializer (optional): TensorFlow initializer.
            lower_trainable (optional): Boolean indicating if variable
                is trainable.
            upper_trainable (optional): Boolean indicating if variable
                is trainable.
            fid_midpoint (optional): Boolean indicating if variable is
                trainable.
            rate_trainable (optional): Boolean indicating if variable
                is trainable.
            kwargs (optional): Additional keyword arguments.

        """
        super(RateBehavior, self).__init__(**kwargs)

        self.lower_trainable = lower_trainable
        if lower_initializer is None:
            lower_initializer = tf.keras.initializers.Constant(0.)
        self.lower_initializer = tf.keras.initializers.get(lower_initializer)
        self.lower = self.add_weight(
            shape=[], initializer=self.lower_initializer,
            trainable=self.lower_trainable, name="lower", dtype=K.floatx(),
            constraint=pk_constraints.GreaterEqualThan(min_value=0.0)
        )

        self.upper_trainable = upper_trainable
        if upper_initializer is None:
            upper_initializer = tf.keras.initializers.Constant(1.)
        self.upper_initializer = tf.keras.initializers.get(upper_initializer)
        self.upper = self.add_weight(
            shape=[], initializer=self.upper_initializer,
            trainable=self.upper_trainable, name="upper", dtype=K.floatx(),
            constraint=pk_constraints.LessEqualThan(max_value=1.0)
        )

        self.midpoint_trainable = midpoint_trainable
        if midpoint_initializer is None:
            midpoint_initializer = tf.keras.initializers.Constant(.5)
        self.midpoint_initializer = tf.keras.initializers.get(
            midpoint_initializer
        )
        self.midpoint = self.add_weight(
            shape=[], initializer=self.midpoint_initializer,
            trainable=self.midpoint_trainable, name="midpoint",
            dtype=K.floatx(),
            constraint=pk_constraints.MinMax(0.0, 1.0)
        )

        self.rate_trainable = rate_trainable
        if rate_initializer is None:
            rate_initializer = tf.keras.initializers.Constant(5.)
        self.rate_initializer = tf.keras.initializers.get(rate_initializer)
        self.rate = self.add_weight(
            shape=[], initializer=self.rate_initializer,
            trainable=self.rate_trainable, name="rate", dtype=K.floatx(),
        )

    def call(self, inputs):
        """Return predicted rating of a trial.

        Arguments:
            inputs:
                sim_qr: A tensor containing the precomputed
                    similarities between the query stimuli and
                    corresponding reference stimuli (only 1 reference).
                    shape = (batch_size, 1, 1)

        Returns:
            probs: The probabilites as determined by a parameterized
                logistic function.

        """
        sim_qr = inputs[0]
        prob = self.lower + tf.math.divide(
            self.upper - self.lower,
            1 + tf.math.exp(-self.rate*(sim_qr - self.midpoint))
        )
        return prob

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        config.update({
            'lower_trainable': self.lower_trainable,
            'upper_trainable': self.upper_trainable,
            'midpoint_trainable': self.midpoint_trainable,
            'rate_trainable': self.rate_trainable,
            'lower_initializer': tf.keras.initializers.serialize(
                self.lower_initializer
            ),
            'upper_initializer': tf.keras.initializers.serialize(
                self.upper_initializer
            ),
            'midpoint_initializer': tf.keras.initializers.serialize(
                self.midpoint_initializer
            ),
            'rate_initializer': tf.keras.initializers.serialize(
                self.rate_initializer
            ),
        })
        return config


@tf.keras.utils.register_keras_serializable(
    package='psiz.keras.layers', name='SortBehavior'
)
class SortBehavior(Behavior):
    """A sort behavior layer.

    TODO

    """

    def __init__(
            self, lower_initializer=None, upper_initializer=None,
            midpoint_initializer=None, rate_initializer=None,
            lower_trainable=True, upper_trainable=True,
            midpoint_trainable=True, rate_trainable=True, **kwargs):
        """Initialize.

        Arguments:
            TODO
            kwargs (optional): Additional keyword arguments.

        """
        super(SortBehavior, self).__init__(**kwargs)
        raise NotImplementedError

    def call(self, inputs):
        """Return probability of outcome.

        Arguments:
            inputs:
                sim_qr: A tensor containing the precomputed
                    similarities between the query stimuli and
                    corresponding reference stimuli (only 1 reference).
                    shape = (batch_size, 1, 1)

        Returns:
            probs: The probabilites as determined by a parameterized
                logistic function.

        """
        raise NotImplementedError
        return None

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        # config.update({})  TODO
        return config
