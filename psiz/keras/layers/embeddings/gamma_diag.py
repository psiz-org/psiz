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
    EmbeddingGammaDiag: A Gamma distribution embedding layer.

"""

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
import tensorflow_probability as tfp

import psiz.distributions
import psiz.keras.constraints


@tf.keras.utils.register_keras_serializable(
    package='psiz.keras.layers', name='EmbeddingGammaDiag'
)
class EmbeddingGammaDiag(tf.keras.layers.Layer):
    """Gamma distribution embedding.

    Each embedding point is characterized by a Gamma distribution.

    """
    def __init__(
            self, input_dim, output_dim, mask_zero=False, input_length=1,
            concentration_initializer=None, rate_initializer=None,
            concentration_regularizer=None, rate_regularizer=None,
            concentration_constraint=None, rate_constraint=None,
            concentration_trainable=True, rate_trainable=True,
            sample_shape=(), **kwargs):
        """Initialize.

        Arguments:
            input_dim:
            output_dim:
            mask_zero (optional):
            input_length (optional):
            concentration_initializer (optional):
            rate_initializer (optional):
            concentration_regularizer (optional):
            rate_regularizer (optional):
            concentration_constraint (optional):
            rate_constraint (optional):
            concentration_trainable (optional):
            rate_trainable (optional):
            kwargs: Additional key-word arguments.

        Notes:
            The trinability of a particular variable is determined by a
                logical "and" between `self.trainable` (the
                layer-wise attribute) and `self.x_trainable` (the
                attribute that specifically controls the variable `x`).
            Uses a constraint-based approach instead of a parameter
                transformation approach in order to avoid low viscosity
                issues with small parameter values.

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
        super(EmbeddingGammaDiag, self).__init__(dtype=dtype, **kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mask_zero = mask_zero
        self.supports_masking = mask_zero
        self.input_length = input_length
        self._supports_ragged_inputs = True
        self.sample_shape = sample_shape

        # Handle initializer.
        if concentration_initializer is None:
            concentration_initializer = tf.keras.initializers.RandomUniform(
                1., 3.
            )
        self.concentration_initializer = tf.keras.initializers.get(
            concentration_initializer
        )
        if rate_initializer is None:
            rate_initializer = tf.keras.initializers.RandomUniform(0., 1.)
        self.rate_initializer = tf.keras.initializers.get(
            rate_initializer
        )

        # Handle regularizer.
        self.concentration_regularizer = tf.keras.regularizers.get(
            concentration_regularizer
        )
        self.rate_regularizer = tf.keras.regularizers.get(
            rate_regularizer
        )

        # Handle constraints.
        if concentration_constraint is None:
            concentration_constraint = psiz.keras.constraints.GreaterEqualThan(
                min_value=1.
            )
        self.concentration_constraint = tf.keras.constraints.get(
            concentration_constraint
        )
        if rate_constraint is None:
            rate_constraint = psiz.keras.constraints.GreaterThan(min_value=0.)
        self.rate_constraint = tf.keras.constraints.get(
            rate_constraint
        )

        self.concentration_trainable = (
            self.trainable and concentration_trainable
        )
        self.rate_trainable = self.trainable and rate_trainable

        # If self.dtype is None, build weights using the default dtype.
        dtype = tf.as_dtype(self.dtype or K.floatx())

        # Note: most sparse optimizers do not have GPU kernels defined.
        # When building graphs, the placement algorithm is able to
        # place variables on CPU since it knows all kernels using the
        # variable only exist on CPU. When eager execution is enabled,
        # the placement decision has to be made right now. Checking for
        # the presence of GPUs to avoid complicating the TPU codepaths
        # which can handle sparse optimizers.
        if context.executing_eagerly() and context.context().num_gpus():
            with ops.device('cpu:0'):
                self.embeddings = self._build_embeddings_distribution(dtype)
        else:
            self.embeddings = self._build_embeddings_distribution(dtype)

    def _build_embeddings_distribution(self, dtype):
        """Build embeddings distribution."""
        # Handle concentration variables.
        self.concentration = self.add_weight(
            name='concentration', shape=[self.input_dim, self.output_dim],
            dtype=dtype, initializer=self.concentration_initializer,
            regularizer=self.concentration_regularizer,
            trainable=self.concentration_trainable,
            constraint=self.concentration_constraint
        )

        # Handle rate variables.
        self.rate = self.add_weight(
            name='untransformed_rate', shape=[self.input_dim, self.output_dim],
            dtype=dtype, initializer=self.rate_initializer,
            regularizer=self.rate_regularizer,
            trainable=self.rate_trainable,
            constraint=self.rate_constraint
        )

        dist = psiz.distributions.Gamma(self.concentration, self.rate)
        batch_ndims = tf.size(dist.batch_shape_tensor())
        return tfp.distributions.Independent(
            dist, reinterpreted_batch_ndims=batch_ndims
        )

    def call(self, inputs):
        """Call."""
        dtype = K.dtype(inputs)
        if dtype != 'int32' and dtype != 'int64':
            inputs = math_ops.cast(inputs, 'int32')

        # Delay reification until end of subclass call in order to
        # generate independent samples for each instance in batch_size.
        inputs_concentration = embedding_ops.embedding_lookup(
            self.embeddings.distribution.concentration, inputs
        )
        inputs_rate = embedding_ops.embedding_lookup(
            self.embeddings.distribution.rate, inputs
        )
        # return [inputs_concentration, inputs_rate]

        # [inputs_concetration, inputs_rate] = super().call(inputs)
        # Use reparameterization trick.
        dist_batch = psiz.distributions.Gamma(
            inputs_concentration, inputs_rate
        )
        # Reify output using samples.
        return dist_batch.sample(self.sample_shape)

    def get_config(self):
        """Return layer configuration."""
        config = super(EmbeddingGammaDiag, self).get_config()
        config.update({
            'input_dim': int(self.input_dim),
            'output_dim': int(self.output_dim),
            'mask_zero': self.mask_zero,
            'input_length': int(self.input_length),
            'concentration_initializer':
                tf.keras.initializers.serialize(
                    self.concentration_initializer
                ),
            'rate_initializer':
                tf.keras.initializers.serialize(self.rate_initializer),
            'concentration_regularizer':
                tf.keras.regularizers.serialize(
                    self.concentration_regularizer
                ),
            'rate_regularizer':
                tf.keras.regularizers.serialize(self.rate_regularizer),
            'concentration_constraint':
                tf.keras.constraints.serialize(self.concentration_constraint),
            'rate_constraint':
                tf.keras.constraints.serialize(self.rate_constraint),
            'concentration_trainable': self.concentration_trainable,
            'rate_trainable': self.rate_trainable,
        })
        return config
