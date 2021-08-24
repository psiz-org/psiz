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
"""Module of TensorFlow embedding layers.

Classes:
    StochasticEmbedding: An abstract class for stochastic embeddings.

"""

import tensorflow as tf
from tensorflow.python.keras import backend as K


class StochasticEmbedding(tf.keras.layers.Layer):
    """Abstract base class for stochastic embeddings.

    Intended to be a drop-in stochastic replacement for
    `tf.keras.layers.Embedding`.

    """
    def __init__(
            self, input_dim, output_dim, mask_zero=False, input_length=1,
            sample_shape=(), **kwargs):
        """Initialize.

        Arguments:
            input_dim:
            output_dim:
            mask_zero (optional):
            input_length (optional):
            sample_shape (optional):
            kwargs: Additional key-word arguments.

        """
        if 'input_shape' not in kwargs:
            if input_length:
                kwargs['input_shape'] = (input_length,)
            else:
                kwargs['input_shape'] = (None,)
        dtype = kwargs.pop('dtype', K.floatx())
        # We set autocast to False, as we do not want to cast floating-
        # point inputs to self.dtype. In call(), we cast to int32, and
        # casting to self.dtype before casting to int32 might cause the
        # int32 values to be different due to a loss of precision.
        kwargs['autocast'] = False
        super().__init__(dtype=dtype, **kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mask_zero = mask_zero
        self.supports_masking = mask_zero
        self.input_length = input_length
        self._supports_ragged_inputs = True
        self.sample_shape = sample_shape

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        config.update({
            'input_dim': int(self.input_dim),
            'output_dim': int(self.output_dim),
            'mask_zero': self.mask_zero,
            'input_length': int(self.input_length),
            'sample_shape': self.sample_shape
        })
        return config
