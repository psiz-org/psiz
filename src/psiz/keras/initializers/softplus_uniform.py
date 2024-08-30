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
"""Module of custom TensorFlow initializers.

Classes:
    SoftplusUniform:

"""


import keras
import tensorflow_probability as tfp


@keras.saving.register_keras_serializable(package="psiz.keras.initializers")
class SoftplusUniform(keras.initializers.Initializer):
    """Initializer using an inverse-softplus-uniform distribution."""

    def __init__(self, minval=-0.05, maxval=0.05, hinge_softness=1.0, seed=None):
        """Initialize.

        Args:
            minval: Minimum value of a uniform random sampler.
            maxval: Maximum value of a uniform random sampler.
            hinge_softness (optional): A float controlling the shape of
                the softplus function.
            seed (optional): An integer seed.

        """
        self.minval = minval
        self.maxval = maxval
        self.hinge_softness = hinge_softness
        self._init_seed = seed
        self.seed = seed if seed is not None else keras.random.SeedGenerator(seed=252)
        super().__init__()

    def __call__(self, shape, dtype=None):
        """Call."""
        w = keras.random.uniform(
            shape=shape,
            minval=self.minval,
            maxval=self.maxval,
            seed=self.seed,
            dtype=dtype,
        )

        def generalized_softplus_inverse(x, c):
            return c * tfp.math.softplus_inverse(x / c)

        # TODO(roads) critical handle zeros
        return generalized_softplus_inverse(w, self.hinge_softness)

    def get_config(self):
        """Return configuration."""
        seed_config = keras.saving.serialize_keras_object(self._init_seed)
        config = {
            "minval": self.minval,
            "maxval": self.maxval,
            "hinge_softness": self.hinge_softness,
            "seed": seed_config,
        }
        return config
