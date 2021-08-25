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
    RandomAttention:

"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import initializers
import tensorflow_probability as tfp


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

    def __call__(self, shape, dtype=None, **kwargs):
        """Call."""
        dist = tfp.distributions.Dirichlet(
            tf.cast(self.concentration, dtype)
        )
        sample = tf.cast(self.scale, dtype) * dist.sample(
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
