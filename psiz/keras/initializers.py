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
# ==============================================================================

"""Module of custom TensorFlow initializers.

Classes:
    RandomScaleMVN:
    RandomAttention:


"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import backend as K
from tensorflow.keras import initializers


@tf.keras.utils.register_keras_serializable(package='psiz.keras.initializers')
class RandomScaleMVN(initializers.Initializer):
    """Initializer that generates tensors with a normal distribution."""

    def __init__(
            self, mean=0.0, stddev=1.0, minval=-4.0, maxval=-1.0, seed=None):
        """Initialize.

        Arguments:
            mean: A python scalar or a scalar tensor. Mean of the
                random values to generate.
            stddev: A scalar indicating the initial standard deviation.
            minval: Minimum value of a uniform random sampler for each
                dimension.
            maxval: Maximum value of a uniform random sampler for each
                dimension.
            seed (optional): A Python integer. Used to create random
                seeds. See `tf.set_random_seed` for behavior.

        """
        self.mean = mean
        self.stddev = stddev
        self.minval = minval
        self.maxval = maxval
        self.seed = seed

    def __call__(self, shape, dtype=K.floatx()):
        """Call."""
        p = tf.random.uniform(
            [1],
            minval=self.minval,
            maxval=self.maxval,
            dtype=dtype,
            seed=self.seed,
            name=None
        )
        scale = tf.pow(
            tf.constant(10., dtype=dtype), p
        )

        stddev = scale * self.stddev
        return tf.random.normal(
            shape, mean=self.mean, stddev=stddev, dtype=dtype, seed=self.seed
        )

    def get_config(self):
        """Return configuration."""
        config = {
            "mean": self.mean,
            "stddev": self.stddev,
            "minval": self.minval,
            "maxval": self.maxval,
            "seed": self.seed
        }
        return config


@tf.keras.utils.register_keras_serializable(package='psiz.keras.initializers')
class RandomAttention(initializers.Initializer):
    """Initializer that generates tensors for attention weights."""

    def __init__(self, concentration, scale=1.0, seed=None):
        """Initialize.

        Arguments:
            concentration: An array-like set of values indicating the
                concentration parameters (i.e., alpha values) governing
                a Dirichlet distribution.
            scale (optional): Scalar indicating how the Dirichlet
                sample should be scaled.
            seed (optional): A seed for deterministic behavior.

        """
        self.concentration = np.asarray(concentration)
        self.scale = scale
        self.seed = seed

    def __call__(self, shape, dtype=K.floatx()):
        """Call."""
        dist = tfp.distributions.Dirichlet(
            tf.cast(self.concentration, dtype=dtype)
        )
        sample = tf.cast(self.scale, dtype=dtype) * dist.sample(
            [shape[0]], seed=self.seed
        )
        return sample

    def get_config(self):
        """Return configuration."""
        return {
            "concentration": self.concentration.tolist(),
            "scale": self.scale,
            "seed": self.seed
        }
