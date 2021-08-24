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
    _EmbeddingLocScale: An abstract embedding class for location, scale
        family of distributions.

"""

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.eager import context
# from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.framework import ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
import tensorflow_probability as tfp

from psiz.keras.layers.embeddings.stochastic_embedding import (
    StochasticEmbedding
)


class _EmbeddingLocScale(StochasticEmbedding):
    """A private base class for a location-scale embedding.

    Each embedding point is characterized by a location-scale
    distribution.

    """
    def __init__(
            self, input_dim, output_dim, mask_zero=False, input_length=1,
            loc_initializer=None, scale_initializer=None, loc_regularizer=None,
            scale_regularizer=None, loc_constraint=None, scale_constraint=None,
            loc_trainable=True, scale_trainable=True, sample_shape=(),
            **kwargs):
        """Initialize.

        Arguments:
            input_dim:
            output_dim:
            mask_zero (optional):
            input_length (optional):
            loc_initializer (optional):
            scale_initializer (optional):
            loc_regularizer (optional):
            scale_regularizer (optional):
            loc_constraint (optional):
            scale_constraint (optional):
            loc_trainable (optional):
            scale_trainable (optional):
            kwargs: Additional key-word arguments.

        Notes:
            The trinability of a particular variable is determined by a
            logical "and" between `self.trainable` (the
            layer-wise attribute) and `self.x_trainable` (the
            attribute that specifically controls the variable `x`).

        """
        super(_EmbeddingLocScale, self).__init__(
            input_dim, output_dim, mask_zero=mask_zero,
            input_length=input_length, sample_shape=sample_shape, **kwargs
        )

        # Handle initializers.
        if loc_initializer is None:
            loc_initializer = tf.keras.initializers.RandomUniform()
        self.loc_initializer = tf.keras.initializers.get(loc_initializer)
        if scale_initializer is None:
            scale_initializer = (
                tf.keras.initializers.RandomNormal(
                    mean=tfp.math.softplus_inverse(1.), stddev=.001
                )
            )
        self.scale_initializer = tf.keras.initializers.get(
            scale_initializer
        )

        # Handle regularizers.
        self.loc_regularizer = tf.keras.regularizers.get(
            loc_regularizer
        )
        self.scale_regularizer = tf.keras.regularizers.get(
            scale_regularizer
        )

        # Handle constraints.
        self.loc_constraint = tf.keras.constraints.get(
            loc_constraint
        )
        self.scale_constraint = tf.keras.constraints.get(
            scale_constraint
        )

        self.loc_trainable = self.trainable and loc_trainable
        self.scale_trainable = self.trainable and scale_trainable

        # If self.dtype is None, build weights using the default dtype.
        dtype = tf.as_dtype(self.dtype or K.floatx())

        # Note: most sparse optimizers do not have GPU kernels defined.
        # When building graphs, the placement algorithm is able to
        # place variables on CPU since it knows all kernels using the
        # variable only exist on CPU. When eager execution is enabled,
        # the placement decision has to be made right now. Checking for
        # the presence of GPUs to avoid complicating the TPU codepaths
        # which can handle sparse optimizers.
        with tf.name_scope(self.name):
            if context.executing_eagerly() and context.context().num_gpus():
                with ops.device('cpu:0'):
                    self.embeddings = self._build_embeddings_distribution(
                        dtype
                    )
            else:
                self.embeddings = self._build_embeddings_distribution(dtype)

    # @tf_utils.shape_type_conversion
    # def build(self, input_shape):
    #     """Build."""
    #     # If self.dtype is None, build weights using the default dtype.
    #     dtype = tf.as_dtype(self.dtype or K.floatx())

    #     # Note: most sparse optimizers do not have GPU kernels defined.
    #     # When building graphs, the placement algorithm is able to
    #     # place variables on CPU since it knows all kernels using the
    #     # variable only exist on CPU. When eager execution is enabled,
    #     # the placement decision has to be made right now. Checking for
    #     # the presence of GPUs to avoid complicating the TPU codepaths
    #     # which can handle sparse optimizers.
    #     if context.executing_eagerly() and context.context().num_gpus():
    #         with ops.device('cpu:0'):
    #             self.embeddings = self._build_embeddings_distribution(dtype)
    #     else:
    #         self.embeddings = self._build_embeddings_distribution(dtype)
    #     self.built = True

    def call(self, inputs):
        """Call."""
        dtype = K.dtype(inputs)
        if dtype != 'int32' and dtype != 'int64':
            inputs = math_ops.cast(inputs, 'int32')

        # Delay reification until end of subclass call in order to
        # generate independent samples for each instance in batch_size.
        inputs_loc = embedding_ops.embedding_lookup(
            self.embeddings.distribution.loc, inputs
        )
        inputs_scale = embedding_ops.embedding_lookup(
            self.embeddings.distribution.scale, inputs
        )
        return [inputs_loc, inputs_scale]

    def get_config(self):
        """Return layer configuration."""
        config = super(_EmbeddingLocScale, self).get_config()
        config.update({
            'loc_initializer':
                tf.keras.initializers.serialize(self.loc_initializer),
            'scale_initializer':
                tf.keras.initializers.serialize(self.scale_initializer),
            'loc_regularizer':
                tf.keras.regularizers.serialize(self.loc_regularizer),
            'scale_regularizer':
                tf.keras.regularizers.serialize(self.scale_regularizer),
            'loc_constraint':
                tf.keras.constraints.serialize(self.loc_constraint),
            'scale_constraint':
                tf.keras.constraints.serialize(self.scale_constraint),
            'loc_trainable': self.loc_trainable,
            'scale_trainable': self.scale_trainable,
        })
        return config
