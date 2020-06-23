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
    EmbeddingNormalDiag: A normal distribution embedding layer.
    EmbeddingLaplaceDiag: A Laplace distribution embedding layer.
    EmbeddingLogNormalDiag: A log-normal distribution embedding layer.
    EmbeddingLogitNormalDiag: A logit-normal distribution embedding
        layer.
    EmbeddingVariational: A variational embedding layer.

"""

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.eager import context
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
import tensorflow_probability as tfp

from psiz.keras.layers.variational import Variational


class _EmbeddingLocScale(tf.keras.layers.Layer):
    """A private base class for a location-scale embedding.

    Each embedding point is characterized by a location-scale
    distribution.

    """
    def __init__(
            self, input_dim, output_dim, mask_zero=False, input_length=None,
            loc_initializer=None, scale_initializer=None, loc_regularizer=None,
            scale_regularizer=None, loc_constraint=None, scale_constraint=None,
            loc_trainable=True, scale_trainable=True, **kwargs):
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
        super(_EmbeddingLocScale, self).__init__(dtype=dtype, **kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mask_zero = mask_zero
        self.supports_masking = mask_zero
        self.input_length = input_length
        self._supports_ragged_inputs = True

        # Handle initializer.
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

        # Handle regularizer.
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

        self.loc_trainable = loc_trainable
        self.scale_trainable = scale_trainable

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
            with tf.python.framework.ops.device('cpu:0'):
                self.embeddings = self._build_embeddings_distribution(dtype)
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
    #         with tf.python.framework.ops.device('cpu:0'):
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
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'mask_zero': self.mask_zero,
            'input_length': self.input_length,
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

    @property
    def embeddings_mode(self):
        """Getter method for mode of `embeddings`."""
        return self.embeddings.mode()


@tf.keras.utils.register_keras_serializable(
    package='psiz.keras.layers', name='EmbeddingNormalDiag'
)
class EmbeddingNormalDiag(_EmbeddingLocScale):
    """A distribution-based embedding.

    Each embedding point is characterized by a Normal distribution with
    a diagonal scale matrix.

    """
    def __init__(self, input_dim, output_dim, **kwargs):
        """Initialize."""
        super(EmbeddingNormalDiag, self).__init__(
            input_dim, output_dim, **kwargs
        )

    def _build_embeddings_distribution(self, dtype):
        """Build embeddings distribution."""
        # Handle location variables.
        loc_trainable = self.trainable and self.loc_trainable
        loc = self.add_weight(
            name='loc', shape=[self.input_dim, self.output_dim], dtype=dtype,
            initializer=self.loc_initializer, regularizer=self.loc_regularizer,
            trainable=loc_trainable, constraint=self.loc_constraint
        )

        # Handle scale variables.
        scale_trainable = self.trainable and self.scale_trainable
        untransformed_scale = self.add_weight(
            name='untransformed_scale',
            shape=[self.input_dim, self.output_dim], dtype=dtype,
            initializer=self.scale_initializer,
            regularizer=self.scale_regularizer, trainable=scale_trainable,
            constraint=self.scale_constraint
        )
        scale = tfp.util.DeferredTensor(
            untransformed_scale,
            lambda x: (K.epsilon() + tf.nn.softplus(x))
        )

        dist = tfp.distributions.Normal(loc=loc, scale=scale)
        batch_ndims = tf.size(dist.batch_shape_tensor())
        return tfp.distributions.Independent(
            dist, reinterpreted_batch_ndims=batch_ndims
        )

    def call(self, inputs):
        """Call."""
        [inputs_loc, inputs_scale] = super().call(inputs)
        # Use reparameterization trick.
        dist_batch = tfp.distributions.Normal(
            loc=inputs_loc, scale=inputs_scale
        )
        # Reify output using samples.
        return dist_batch.sample()


@tf.keras.utils.register_keras_serializable(
    package='psiz.keras.layers', name='EmbeddingLaplaceDiag'
)
class EmbeddingLaplaceDiag(_EmbeddingLocScale):
    """A distribution-based embedding.

    Each embedding point is characterized by a Laplace distribution with
    a diagonal scale matrix.

    """
    def __init__(self, input_dim, output_dim, **kwargs):
        """Initialize."""
        super(EmbeddingLaplaceDiag, self).__init__(
            input_dim, output_dim, **kwargs
        )

    def _build_embeddings_distribution(self, dtype):
        """Build embeddings distribution."""
        # Handle location variables.
        loc_trainable = self.trainable and self.loc_trainable
        loc = self.add_weight(
            name='loc', shape=[self.input_dim, self.output_dim], dtype=dtype,
            initializer=self.loc_initializer, regularizer=self.loc_regularizer,
            trainable=loc_trainable, constraint=self.loc_constraint
        )

        # Handle scale variables.
        scale_trainable = self.trainable and self.scale_trainable
        untransformed_scale = self.add_weight(
            name='untransformed_scale',
            shape=[self.input_dim, self.output_dim], dtype=dtype,
            initializer=self.scale_initializer,
            regularizer=self.scale_regularizer, trainable=scale_trainable,
            constraint=self.scale_constraint
        )
        scale = tfp.util.DeferredTensor(
            untransformed_scale,
            lambda x: (K.epsilon() + tf.nn.softplus(x))
        )

        dist = tfp.distributions.Laplace(loc=loc, scale=scale)
        batch_ndims = tf.size(dist.batch_shape_tensor())
        return tfp.distributions.Independent(
            dist, reinterpreted_batch_ndims=batch_ndims
        )

    def call(self, inputs):
        """Call."""
        [inputs_loc, inputs_scale] = super().call(inputs)
        # Use reparameterization trick.
        dist_batch = tfp.distributions.Laplace(
            loc=inputs_loc, scale=inputs_scale
        )
        # Reify output using samples.
        return dist_batch.sample()


@tf.keras.utils.register_keras_serializable(
    package='psiz.keras.layers', name='EmbeddingLogNormalDiag'
)
class EmbeddingLogNormalDiag(_EmbeddingLocScale):
    """A distribution-based embedding.

    Each embedding point is characterized by a Log-Normal distribution
    with a diagonal scale matrix.

    """
    def __init__(self, input_dim, output_dim, **kwargs):
        """Initialize."""
        # Overide default scale initializer.
        scale_initializer = kwargs.pop('scale_initializer', None)
        if scale_initializer is None:
            scale_initializer = tf.keras.initializers.RandomNormal(
                mean=tfp.math.softplus_inverse(2.), stddev=.01
            )

        super(EmbeddingLogNormalDiag, self).__init__(
            input_dim, output_dim, scale_initializer=scale_initializer,
            **kwargs
        )

    def _build_embeddings_distribution(self, dtype):
        """Build embeddings distribution."""
        # Handle location variables.
        loc_trainable = self.trainable and self.loc_trainable
        loc = self.add_weight(
            name='loc', shape=[self.input_dim, self.output_dim], dtype=dtype,
            initializer=self.loc_initializer, regularizer=self.loc_regularizer,
            trainable=loc_trainable, constraint=self.loc_constraint
        )

        # Handle scale variables.
        scale_trainable = self.trainable and self.scale_trainable
        untransformed_scale = self.add_weight(
            name='untransformed_scale',
            shape=[self.input_dim, self.output_dim], dtype=dtype,
            initializer=self.scale_initializer,
            regularizer=self.scale_regularizer, trainable=scale_trainable,
            constraint=self.scale_constraint
        )
        scale = tfp.util.DeferredTensor(
            untransformed_scale, lambda x: (K.epsilon() + tf.nn.softplus(x))
        )

        dist = tfp.distributions.LogNormal(loc=loc, scale=scale)
        batch_ndims = tf.size(dist.batch_shape_tensor())
        return tfp.distributions.Independent(
            dist, reinterpreted_batch_ndims=batch_ndims
        )

    def call(self, inputs):
        """Call."""
        [inputs_loc, inputs_scale] = super().call(inputs)
        # Use reparameterization trick.
        dist_batch = tfp.distributions.LogNormal(
            loc=inputs_loc, scale=inputs_scale
        )
        # Reify output using samples.
        return dist_batch.sample()


@tf.keras.utils.register_keras_serializable(
    package='psiz.keras.layers', name='EmbeddingLogitNormalDiag'
)
class EmbeddingLogitNormalDiag(_EmbeddingLocScale):
    """A distribution-based embedding.

    Each embedding point is characterized by LogitNormal distribution
    with a diagonal scale matrix.

    """
    def __init__(self, input_dim, output_dim, **kwargs):
        """Initialize."""
        # Overide default scale initializer.
        scale_initializer = kwargs.pop('scale_initializer', None)
        if scale_initializer is None:
            scale_initializer = tf.keras.initializers.RandomNormal(0.3, .01)

        super(EmbeddingLogitNormalDiag, self).__init__(
            input_dim, output_dim, scale_initializer=scale_initializer,
            **kwargs
        )

    def _build_embeddings_distribution(self, dtype):
        """Build embeddings distribution."""
        # Handle location variables.
        loc_trainable = self.trainable and self.loc_trainable
        untransformed_loc = self.add_weight(
            name='untransformed_loc', shape=[self.input_dim, self.output_dim],
            dtype=dtype, initializer=self.loc_initializer,
            regularizer=self.loc_regularizer, trainable=loc_trainable,
            constraint=self.loc_constraint
        )

        # Handle scale variables.
        scale_trainable = self.trainable and self.scale_trainable
        untransformed_scale = self.add_weight(
            name='untransformed_scale',
            shape=[self.input_dim, self.output_dim], dtype=dtype,
            initializer=self.scale_initializer,
            regularizer=self.scale_regularizer, trainable=scale_trainable,
            constraint=self.scale_constraint
        )
        scale = tfp.util.DeferredTensor(
            untransformed_scale, lambda x: (K.epsilon() + tf.math.exp(x))
        )

        dist = tfp.distributions.LogitNormal(
            loc=untransformed_loc, scale=scale
        )
        batch_ndims = tf.size(dist.batch_shape_tensor())
        return tfp.distributions.Independent(
            dist, reinterpreted_batch_ndims=batch_ndims
        )

    def call(self, inputs):
        """Call."""
        [inputs_loc, inputs_scale] = super().call(inputs)
        # Use reparameterization trick.
        dist_batch = tfp.distributions.LogitNormal(
            loc=inputs_loc, scale=inputs_scale
        )
        # Reify output using samples.
        return dist_batch.sample()

    @property
    def embeddings_mode(self):
        """Getter method for mode of `embeddings`."""
        # Use median as approximation of mode. For logit-normal distribution,
        # `median = logistic(loc)`. TODO
        return tf.math.sigmoid(self.embeddings.distribution.loc)


@tf.keras.utils.register_keras_serializable(
    package='psiz.keras.layers', name='EmbeddingVariational'
)
class EmbeddingVariational(Variational):
    """Variational analog of Embedding layer."""

    def __init__(self, **kwargs):
        """Initialize.

        Arguments:
            kwargs: Additional key-word arguments.

        """
        super(EmbeddingVariational, self).__init__(**kwargs)

    def call(self, inputs):
        """Call."""
        # Run forward pass through variational posterior layer.
        outputs = self.posterior(inputs)

        # Apply KL divergence between posterior and prior.
        self.add_kl_loss(self.posterior.embeddings, self.prior.embeddings)

        # TODO Remove debug metrics.
        # self.add_metric(
        #     self.kl_anneal, aggregation='mean', name='kl_anneal'
        # )

        m = self.posterior.embeddings.distribution.loc[1:]
        # m = self.posterior.embeddings.distribution.bijector(
        #     self.posterior.embeddings.distribution.distribution.loc[1:]
        # )
        s = self.posterior.embeddings.distribution.scale[1:]
        self.add_metric(
            tf.reduce_mean(m),
            aggregation='mean', name='po_loc_avg'
        )
        self.add_metric(
            tf.reduce_mean(s),
            aggregation='mean', name='po_scale_avg'
        )

        # m = self.posterior.embeddings.distribution.mode()[1:]
        # s = self.posterior.embeddings.distribution.stddev()[1:]
        # self.add_metric(
        #     tf.reduce_mean(m),
        #     aggregation='mean', name='po_mode_avg'
        # )
        # self.add_metric(
        #     tf.reduce_mean(s),
        #     aggregation='mean', name='po_stddev_avg'
        # )

        # m = self.prior.embeddings.distribution.loc[1:]
        s = self.prior.embeddings.distribution.scale[1:]
        # self.add_metric(
        #     tf.reduce_mean(m),
        #     aggregation='mean', name='pr_loc'
        # )
        self.add_metric(
            tf.reduce_mean(s),
            aggregation='mean', name='pr_scale'
        )

        return outputs

    @property
    def input_dim(self):
        """Getter method for embeddings `input_dim`."""
        return self.posterior.input_dim

    @property
    def output_dim(self):
        """Getter method for embeddings `output_dim`."""
        return self.posterior.output_dim

    @property
    def mask_zero(self):
        """Getter method for embeddings `mask_zero`."""
        return self.posterior.mask_zero

    @property
    def embeddings(self):
        """Getter method for embeddings posterior mode."""
        return self.posterior.embeddings_mode
