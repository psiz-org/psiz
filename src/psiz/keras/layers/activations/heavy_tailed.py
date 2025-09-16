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
    HeavyTailedSimilarity: A parameterized heavy-tailed similarity
        layer.

"""


import keras

import psiz.keras.constraints as pk_constraints


@keras.saving.register_keras_serializable(
    package="psiz.keras.layers", name="HeavyTailedSimilarity"
)
class HeavyTailedSimilarity(keras.layers.Layer):
    """Heavy-tailed family similarity function.

    The heavy-tailed similarity function is parameterized as:

    s(x,y) = (kappa + (d(x,y).^tau)).^(-alpha),

    where x and y are n-dimensional vectors. The heavy-tailed family is
    a generalization of the Student-t family.

    """

    def __init__(
        self,
        tau_initializer=None,
        kappa_initializer=None,
        alpha_initializer=None,
        tau_trainable=None,
        kappa_trainable=None,
        alpha_trainable=None,
        fit_tau=True,
        fit_kappa=True,
        fit_alpha=True,
        **kwargs
    ):
        """Initialize.

        Args:
            tau_initializer (optional): Initializer for tau.
            kappa_initializer (optional): Initializer for kappa.
            alpha_initializer (optional): Initializer for alpha.
            tau_trainable (optional): Boolean indicating if variable is
                trainable.
            kappa_trainable (optional): Boolean indicating if variable is
                trainable.
            alpha_trainable (optional): Boolean indicating if variable is
                trainable.
            fit_tau (deprecated, optional): alias for tau_trainable.
            fit_kappa (deprecated, optional): alias for kappa_trainable.
            fit_alpha (deprecated, optional): alias for alpha_trainable.

        """
        super(HeavyTailedSimilarity, self).__init__(**kwargs)

        if tau_trainable is not None:
            self.fit_tau = tau_trainable
            self.tau_trainable = tau_trainable
        else:
            self.fit_tau = fit_tau
            self.tau_trainable = fit_tau
        if tau_initializer is None:
            tau_initializer = keras.initializers.RandomUniform(minval=1.0, maxval=2.0)
        self.tau_initializer = keras.initializers.get(tau_initializer)

        if kappa_trainable is not None:
            self.fit_kappa = kappa_trainable
            self.kappa_trainable = kappa_trainable
        else:
            self.fit_kappa = fit_kappa
            self.kappa_trainable = fit_kappa
        if kappa_initializer is None:
            kappa_initializer = keras.initializers.RandomUniform(
                minval=1.0, maxval=11.0
            )
        self.kappa_initializer = keras.initializers.get(kappa_initializer)

        if alpha_trainable is not None:
            self.fit_alpha = alpha_trainable
            self.alpha_trainable = alpha_trainable
        else:
            self.fit_alpha = fit_alpha
            self.alpha_trainable = fit_alpha
        if alpha_initializer is None:
            alpha_initializer = keras.initializers.RandomUniform(
                minval=1.0, maxval=10.0
            )
        self.alpha_initializer = keras.initializers.get(alpha_initializer)

    def build(self, input_shape):
        """Build."""
        if self.built:
            return
        tau_trainable = self.trainable and self.tau_trainable
        kappa_trainable = self.trainable and self.kappa_trainable
        alpha_trainable = self.trainable and self.alpha_trainable
        with keras.name_scope(self.name):
            self.tau = self.add_weight(
                shape=[],
                initializer=self.tau_initializer,
                trainable=tau_trainable,
                name="tau",
                constraint=pk_constraints.GreaterEqualThan(min_value=1.0),
            )
            self.kappa = self.add_weight(
                shape=[],
                initializer=self.kappa_initializer,
                trainable=kappa_trainable,
                name="kappa",
                constraint=pk_constraints.GreaterEqualThan(min_value=0.0),
            )
            self.alpha = self.add_weight(
                shape=[],
                initializer=self.alpha_initializer,
                trainable=alpha_trainable,
                name="alpha",
                constraint=pk_constraints.GreaterEqualThan(min_value=0.0),
            )

    def call(self, inputs):
        """Call.

        Args:
            inputs: A tensor of distances.

        Returns:
            A tensor of similarities.

        """
        return keras.ops.power(
            self.kappa + keras.ops.power(inputs, self.tau),
            (keras.ops.negative(self.alpha)),
        )

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        config.update(
            {
                "tau_initializer": keras.initializers.serialize(self.tau_initializer),
                "kappa_initializer": keras.initializers.serialize(
                    self.kappa_initializer
                ),
                "alpha_initializer": keras.initializers.serialize(
                    self.alpha_initializer
                ),
                "tau_trainable": self.tau_trainable,
                "kappa_trainable": self.kappa_trainable,
                "alpha_trainable": self.alpha_trainable,
            }
        )
        return config
