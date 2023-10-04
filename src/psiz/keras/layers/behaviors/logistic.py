# -*- coding: utf-8 -*-
# Copyright 2023 The PsiZ Authors. All Rights Reserved.
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
    Logistic

"""

import tensorflow as tf
from tensorflow.keras import backend


@tf.keras.utils.register_keras_serializable(
    package="psiz.keras.layers", name="Logistic"
)
class Logistic(tf.keras.layers.Layer):
    """A layer for learning a parameterized logistic function.

    Inputs are converted via the parameterized
    logistic function,

    f(x) = upper / (1 + exp(-rate*(x - midpoint))))

    with the following variable meanings:
    `upper`: The upper asymptote of the function's range.
    `midpoint`: The midpoint of the function's domain and point of
    maximum growth.
    `rate`: The growth rate of the logistic function.

    """

    def __init__(
        self,
        upper_initializer=None,
        midpoint_initializer=None,
        rate_initializer=None,
        upper_constraint=None,
        midpoint_constraint=None,
        rate_constraint=None,
        **kwargs
    ):
        """Initialize.

        Default initialization is the standard logistic function (i.e.,
        sigmoid function).

        Args:
            upper_initializer (optional): Initializer for upper scalar.
            midpoint_initializer (optional): Initializer for midpoint
                scalar.
            rate_initializer (optional): Initializer for rate scalar.
            upper_constraint (optional): Constraint function applied to
                the upper scalar. Default is a nonnegative constraint.
            midpoint_constraint (optional): Constraint function applied
                to the midpoint scalar. No default constraint.
            rate_constraint (optional): Constraint function applied to
                the rate scalar. No default constraint.
            kwargs (optional): Additional keyword arguments.


        """
        super(Logistic, self).__init__(**kwargs)

        if upper_initializer is None:
            upper_initializer = tf.keras.initializers.Constant(1.0)
        self.upper_initializer = tf.keras.initializers.get(upper_initializer)
        if upper_constraint is None:
            upper_constraint = tf.keras.constraints.NonNeg()
        self.upper_constraint = tf.keras.constraints.get(upper_constraint)
        self.upper = self.add_weight(
            shape=[],
            initializer=self.upper_initializer,
            trainable=self.trainable,
            name="upper",
            dtype=backend.floatx(),
            constraint=upper_constraint,
        )

        if midpoint_initializer is None:
            midpoint_initializer = tf.keras.initializers.Constant(0.0)
        self.midpoint_initializer = tf.keras.initializers.get(midpoint_initializer)
        self.midpoint_constraint = tf.keras.constraints.get(midpoint_constraint)
        self.midpoint = self.add_weight(
            shape=[],
            initializer=self.midpoint_initializer,
            trainable=self.trainable,
            name="midpoint",
            dtype=backend.floatx(),
            constraint=midpoint_constraint,
        )

        if rate_initializer is None:
            rate_initializer = tf.keras.initializers.Constant(1.0)
        self.rate_initializer = tf.keras.initializers.get(rate_initializer)
        self.rate_constraint = tf.keras.constraints.get(rate_constraint)
        self.rate = self.add_weight(
            shape=[],
            initializer=self.rate_initializer,
            trainable=self.trainable,
            name="rate",
            dtype=backend.floatx(),
            constraint=rate_constraint,
        )

    def call(self, inputs, training=None):
        """Return logistic function output.

        Args:
            inputs: A tensor of inputs to the logistic function.
                shape=(batch_size, n, [m, ...])

        Returns:
            y: The output of the parameterized logistic function.
                shape=(batch_size, n, [m, ...])

        """
        y = tf.math.divide(
            self.upper,
            1 + tf.math.exp(-self.rate * (inputs - self.midpoint)),
        )

        return y

    def get_config(self):
        """Return layer configuration."""
        config = super(Logistic, self).get_config()
        config.update(
            {
                "upper_constraint": tf.keras.constraints.serialize(
                    self.upper_constraint
                ),
                "midpoint_constraint": tf.keras.constraints.serialize(
                    self.midpoint_constraint
                ),
                "rate_constraint": tf.keras.constraints.serialize(self.rate_constraint),
                "upper_initializer": tf.keras.initializers.serialize(
                    self.upper_initializer
                ),
                "midpoint_initializer": tf.keras.initializers.serialize(
                    self.midpoint_initializer
                ),
                "rate_initializer": tf.keras.initializers.serialize(
                    self.rate_initializer
                ),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return super().from_config(config)
