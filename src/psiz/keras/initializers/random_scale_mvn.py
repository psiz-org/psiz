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
    RandomScaleMVN:

"""

import tensorflow as tf
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

    def __call__(self, shape, dtype=None, **kwargs):
        """Call."""

        # pylint: disable=unexpected-keyword-arg
        p = tf.random.uniform(
            [1], minval=self.minval, maxval=self.maxval, dtype=dtype,
            seed=self.seed, name=None
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
