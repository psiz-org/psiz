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
"""Module of TensorFlow distance layers.

Classes:
    MinkowskiStochastic: A stochastic Minkowski distance layer.

"""

import tensorflow as tf
from tensorflow.python.keras import backend as K
import tensorflow_probability as tfp

from psiz.tfp.distributions.truncated_normal import TruncatedNormal
import psiz.keras.constraints as pk_constraints
from psiz.keras.layers.ops.core import wpnorm


@tf.keras.utils.register_keras_serializable(
    package='psiz.keras.layers', name='MinkowskiStochastic'
)
class MinkowskiStochastic(tf.keras.layers.Layer):
    """A stochastic Minkowski distance layer."""
    def __init__(
            self,
            rho_loc_trainable=True,
            rho_loc_initializer=None,
            rho_loc_constraint=None,
            rho_loc_regularizer=None,
            rho_scale_trainable=True,
            rho_scale_initializer=None,
            rho_scale_constraint=None,
            rho_scale_regularizer=None,
            w_loc_trainable=True,
            w_loc_initializer=None,
            w_loc_constraint=None,
            w_loc_regularizer=None,
            w_scale_trainable=True,
            w_scale_initializer=None,
            w_scale_constraint=None,
            w_scale_regularizer=None,
            **kwargs):
        """Initialize."""
        super(MinkowskiStochastic, self).__init__(**kwargs)

        # Additional defaults.
        self.rho_low = 1.0
        self.rho_high = 1000000.
        self.w_low = 0.0
        self.w_high = 1000000.

        if rho_loc_initializer is None:
            rho_loc_initializer = tf.keras.initializers.Constant(2.)
        self.rho_loc_initializer = tf.keras.initializers.get(
            rho_loc_initializer
        )
        if rho_loc_constraint is None:
            rho_loc_constraint = pk_constraints.GreaterEqualThan(1.)
        self.rho_loc_constraint = tf.keras.constraints.get(
            rho_loc_constraint
        )
        self.rho_loc_regularizer = tf.keras.regularizers.get(
            rho_loc_regularizer
        )

        if rho_scale_initializer is None:
            rho_scale_initializer = tf.keras.initializers.Constant(-13.)
        self.rho_scale_initializer = tf.keras.initializers.get(
            rho_scale_initializer
        )
        self.rho_scale_constraint = tf.keras.constraints.get(
            rho_scale_constraint
        )
        self.rho_scale_regularizer = tf.keras.regularizers.get(
            rho_scale_regularizer
        )

        if w_loc_initializer is None:
            w_loc_initializer = tf.keras.initializers.Constant(1.)
        self.w_loc_initializer = tf.keras.initializers.get(
            w_loc_initializer
        )
        if w_loc_constraint is None:
            w_loc_constraint = tf.keras.constraints.NonNeg()
        self.w_loc_constraint = tf.keras.constraints.get(
            w_loc_constraint
        )
        self.w_loc_regularizer = tf.keras.regularizers.get(
            w_loc_regularizer
        )
        if w_scale_initializer is None:
            w_scale_initializer = tf.keras.initializers.Constant(-13.)
        self.w_scale_initializer = tf.keras.initializers.get(
            w_scale_initializer
        )
        self.w_scale_constraint = tf.keras.constraints.get(
            w_scale_constraint
        )
        self.w_scale_regularizer = tf.keras.regularizers.get(
            w_scale_regularizer
        )

        self.rho_loc_trainable = self.trainable and rho_loc_trainable
        self.rho_scale_trainable = self.trainable and rho_scale_trainable

        self.w_loc_trainable = self.trainable and w_loc_trainable
        self.w_scale_trainable = self.trainable and w_scale_trainable

        self.rho = None
        self.rho_loc = None
        self.rho_untransformed_scale = None
        self.w = None
        self.w_loc = None
        self.w_untransformed_scale = None

    def build(self, input_shape):
        """Build."""
        dtype = tf.as_dtype(self.dtype or K.floatx())
        self.rho = self._build_rho(input_shape, dtype)
        self.w = self._build_w(input_shape, dtype)

    def _build_rho(self, input_shape, dtype):
        # pylint: disable=unused-argument
        with tf.name_scope(self.name):
            self.rho_loc = self.add_weight(
                name='rho_loc',
                shape=[], dtype=dtype,
                initializer=self.rho_loc_initializer,
                regularizer=self.rho_loc_regularizer,
                trainable=self.rho_loc_trainable,
                constraint=self.rho_loc_constraint
            )

            # Handle scale variables.
            self.rho_untransformed_scale = self.add_weight(
                name='rho_untransformed_scale',
                shape=[], dtype=dtype,
                initializer=self.rho_scale_initializer,
                regularizer=self.rho_scale_regularizer,
                trainable=self.rho_scale_trainable,
                constraint=self.rho_scale_constraint
            )
        rho_scale = tfp.util.DeferredTensor(
            self.rho_untransformed_scale,
            lambda x: (K.epsilon() + tf.nn.softplus(x))
        )

        rho_dist = TruncatedNormal(
            self.rho_loc, rho_scale, self.rho_low, self.rho_high
        )
        batch_ndims = tf.size(rho_dist.batch_shape_tensor())
        return tfp.distributions.Independent(
            rho_dist, reinterpreted_batch_ndims=batch_ndims
        )

    def _build_w(self, input_shape, dtype):
        with tf.name_scope(self.name):
            self.w_loc = self.add_weight(
                name='w_loc',
                shape=[input_shape[0][-1]], dtype=dtype,
                initializer=self.w_loc_initializer,
                regularizer=self.w_loc_regularizer,
                trainable=self.w_loc_trainable,
                constraint=self.w_loc_constraint
            )

            # Handle scale variables.
            self.w_untransformed_scale = self.add_weight(
                name='w_untransformed_scale',
                shape=[input_shape[0][-1]], dtype=dtype,
                initializer=self.w_scale_initializer,
                regularizer=self.w_scale_regularizer,
                trainable=self.w_scale_trainable,
                constraint=self.w_scale_constraint
            )
        w_scale = tfp.util.DeferredTensor(
            self.w_untransformed_scale,
            lambda x: (K.epsilon() + tf.nn.softplus(x))
        )

        w_dist = TruncatedNormal(
            self.w_loc, w_scale, self.w_low, self.w_high
        )
        batch_ndims = tf.size(w_dist.batch_shape_tensor())
        return tfp.distributions.Independent(
            w_dist, reinterpreted_batch_ndims=batch_ndims
        )

    def call(self, inputs):
        """Call.

        Arguments:
            inputs: A list of two tf.Tensor's denoting a the set of
                vectors to compute pairwise distance. Each tensor is
                assumed to have the same shape and be at least rank-2.
                Any additional tensors in the list are ignored.
                shape = (batch_size, [n, m, ...] n_dim)

        Returns:
            shape=(batch_size, [n, m, ...])

        """
        z_0 = inputs[0]
        z_1 = inputs[1]
        x = z_0 - z_1
        x_shape = tf.shape(x)

        # Sample free parameters based on input shape.
        # Note that `wpnorm` expects `rho` to have one less rank than
        # `x` and `w`, i.e., it does not have a trailing `n_dim`.
        # We wrap the sample call in a conditional to protect against
        # batch_size=0.
        batch_size = x_shape[0]
        rho = tf.cond(
            batch_size == 0,
            lambda: tf.zeros(x_shape[0:-1]),
            lambda: self.rho.sample(x_shape[0:-1])
        )
        w = tf.cond(
            batch_size == 0,
            lambda: tf.zeros(x_shape),
            lambda: self.w.sample(x_shape[0:-1])
        )

        # Weighted Minkowski distance.
        d_qr = wpnorm(x, w, rho)
        d_qr = tf.squeeze(d_qr, [-1])
        return d_qr

    def get_config(self):
        config = super().get_config()
        config.update({
            'rho_loc_initializer':
                tf.keras.initializers.serialize(self.rho_loc_initializer),
            'rho_scale_initializer':
                tf.keras.initializers.serialize(self.rho_scale_initializer),
            'w_loc_initializer':
                tf.keras.initializers.serialize(self.w_loc_initializer),
            'w_scale_initializer':
                tf.keras.initializers.serialize(self.w_scale_initializer),
            'rho_loc_regularizer':
                tf.keras.regularizers.serialize(self.rho_loc_regularizer),
            'rho_scale_regularizer':
                tf.keras.regularizers.serialize(self.rho_scale_regularizer),
            'w_loc_regularizer':
                tf.keras.regularizers.serialize(self.w_loc_regularizer),
            'w_scale_regularizer':
                tf.keras.regularizers.serialize(self.w_scale_regularizer),
            'rho_loc_constraint':
                tf.keras.constraints.serialize(self.rho_loc_constraint),
            'rho_scale_constraint':
                tf.keras.constraints.serialize(self.rho_scale_constraint),
            'w_loc_constraint':
                tf.keras.constraints.serialize(self.w_loc_constraint),
            'w_scale_constraint':
                tf.keras.constraints.serialize(self.w_scale_constraint),
            'rho_loc_trainable': self.rho_loc_trainable,
            'rho_scale_trainable': self.rho_scale_trainable,
            'w_loc_trainable': self.w_loc_trainable,
            'w_scale_trainable': self.w_scale_trainable,
        })
        return config
