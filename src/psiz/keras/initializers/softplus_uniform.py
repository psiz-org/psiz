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
"""Module of custom TensorFlow initializers.

Classes:
    SoftplusUniform:

"""

import tensorflow as tf
from tensorflow.keras import initializers
from tensorflow.keras import backend as K
import tensorflow_probability as tfp


@tf.keras.utils.register_keras_serializable(package="psiz.keras.initializers")
class SoftplusUniform(initializers.Initializer):
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
        self.seed = seed

    def __call__(self, shape, dtype=None, **kwargs):
        """Call."""
        # pylint: disable=unexpected-keyword-arg
        if dtype is None:
            dtype = K.floatx()

        w = tf.random.uniform(
            shape,
            minval=self.minval,
            maxval=self.maxval,
            dtype=dtype,
            seed=self.seed,
            name=None,
        )

        def generalized_softplus_inverse(x, c):
            return c * tfp.math.softplus_inverse(x / c)

        # TODO critical handle zeros
        return generalized_softplus_inverse(w, self.hinge_softness)

    def get_config(self):
        """Return configuration."""
        config = {
            "minval": self.minval,
            "maxval": self.maxval,
            "hinge_softness": self.hinge_softness,
            "seed": self.seed,
        }
        return config
