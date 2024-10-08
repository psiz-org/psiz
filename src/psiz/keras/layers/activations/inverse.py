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
"""Module of Keras activation layers.

Classes:
    InverseSimilarity: A parameterized inverse similarity layer.

"""


import keras

import psiz.keras.constraints as pk_constraints


@keras.saving.register_keras_serializable(
    package="psiz.keras.layers", name="InverseSimilarity"
)
class InverseSimilarity(keras.layers.Layer):
    """Inverse-distance similarity function.

    The inverse-distance similarity function is parameterized as:
    s(x,y) = 1 / (d(x,y)**tau + mu),
    where x and y are n-dimensional vectors.
    """

    def __init__(
        self,
        fit_tau=True,
        fit_mu=True,
        tau_initializer=None,
        mu_initializer=None,
        **kwargs
    ):
        """Initialize.

        Args:
            fit_tau (optional): Boolean indicating if variable is
                trainable.
            fit_mu (optional): Boolean indicating if variable is
                trainable.
            tau_initializer (optional): Initializer for tau.
            mu_initializer (optional): Initializer for mu.

        """
        super(InverseSimilarity, self).__init__(**kwargs)

        self.fit_tau = fit_tau
        if tau_initializer is None:
            tau_initializer = keras.initializers.RandomUniform(minval=1.0, maxval=2.0)
        self.tau_initializer = keras.initializers.get(tau_initializer)

        self.fit_mu = fit_mu
        if mu_initializer is None:
            mu_initializer = keras.initializers.RandomUniform(
                minval=0.0000000001, maxval=0.001
            )
        self.mu_initializer = keras.initializers.get(tau_initializer)

    def build(self, input_shape):
        """Build."""
        if self.built:
            return
        tau_trainable = self.trainable and self.fit_tau
        mu_trainable = self.trainable and self.fit_mu
        with keras.name_scope(self.name):
            self.tau = self.add_weight(
                shape=[],
                initializer=self.tau_initializer,
                trainable=tau_trainable,
                name="tau",
                dtype=keras.backend.floatx(),
                constraint=pk_constraints.GreaterEqualThan(min_value=1.0),
            )
            self.mu = self.add_weight(
                shape=[],
                initializer=self.tau_initializer,
                trainable=mu_trainable,
                name="mu",
                dtype=keras.backend.floatx(),
                constraint=pk_constraints.GreaterEqualThan(min_value=2.2204e-16),
            )

    def call(self, inputs):
        """Call.

        Args:
            inputs: A tensor of distances.

        Returns:
            A tensor of similarities.

        """
        return 1 / (keras.ops.power(inputs, self.tau) + self.mu)

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        config.update(
            {
                "fit_tau": self.fit_tau,
                "fit_mu": self.fit_mu,
                "tau_initializer": keras.initializers.serialize(self.tau_initializer),
                "mu_initializer": keras.initializers.serialize(self.mu_initializer),
            }
        )
        return config
