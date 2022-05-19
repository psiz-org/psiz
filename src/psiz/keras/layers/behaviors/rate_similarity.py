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
"""Module of TensorFlow behavior layers.

Classes:
    RateSimilarity: A rate similarity layer.

"""

import tensorflow as tf
from tensorflow.python.keras import backend as K

import psiz.keras.constraints as pk_constraints
from psiz.keras.layers.behaviors.behavior import Behavior


@tf.keras.utils.register_keras_serializable(
    package='psiz.keras.layers', name='RateSimilarity'
)
class RateSimilarity(Behavior):
    """A rate behavior layer.

    Similarities are converted to probabilities using a parameterized
    logistic function,

    p(x) = lower + ((upper - lower) / (1 + exp(-rate*(x - midpoint))))

    with the following variable meanings:
    `lower`: The lower asymptote of the function's range.
    `upper`: The upper asymptote of the function's range.
    `midpoint`: The midpoint of the function's domain and point of
        maximum growth.
    `rate`: The growth rate of the logistic function.

    """

    def __init__(
            self, kernel=None, lower_initializer=None, upper_initializer=None,
            midpoint_initializer=None, rate_initializer=None,
            lower_trainable=False, upper_trainable=False,
            midpoint_trainable=True, rate_trainable=True, **kwargs):
        """Initialize.

        Args:
            kernel: A kernel layer.
            lower_initializer (optional): TensorFlow initializer.
            upper_initializer (optional): TensorFlow initializer.
            midpoint_initializer (optional): TensorFlow initializer.
            rate_initializer (optional): TensorFlow initializer.
            lower_trainable (optional): Boolean indicating if variable
                is trainable.
            upper_trainable (optional): Boolean indicating if variable
                is trainable.
            fid_midpoint (optional): Boolean indicating if variable is
                trainable.
            rate_trainable (optional): Boolean indicating if variable
                is trainable.
            kwargs (optional): Additional keyword arguments.

        """
        super(RateSimilarity, self).__init__(**kwargs)
        self.kernel = kernel

        # Satisfy `GroupsMixin` contract.
        self._pass_groups['kernel'] = self.check_supports_groups(kernel)
        self.supports_groups = True

        # Satisfy RNNCell contract.  TODO
        self.state_size = [
            tf.TensorShape([2]),
            tf.TensorShape([2, 2])
        ]

        self.lower_trainable = lower_trainable
        if lower_initializer is None:
            lower_initializer = tf.keras.initializers.Constant(0.)
        self.lower_initializer = tf.keras.initializers.get(lower_initializer)
        self.lower = self.add_weight(
            shape=[], initializer=self.lower_initializer,
            trainable=self.lower_trainable, name="lower", dtype=K.floatx(),
            constraint=pk_constraints.GreaterEqualThan(min_value=0.0)
        )

        self.upper_trainable = upper_trainable
        if upper_initializer is None:
            upper_initializer = tf.keras.initializers.Constant(1.)
        self.upper_initializer = tf.keras.initializers.get(upper_initializer)
        self.upper = self.add_weight(
            shape=[], initializer=self.upper_initializer,
            trainable=self.upper_trainable, name="upper", dtype=K.floatx(),
            constraint=pk_constraints.LessEqualThan(max_value=1.0)
        )

        self.midpoint_trainable = midpoint_trainable
        if midpoint_initializer is None:
            midpoint_initializer = tf.keras.initializers.Constant(.5)
        self.midpoint_initializer = tf.keras.initializers.get(
            midpoint_initializer
        )
        self.midpoint = self.add_weight(
            shape=[], initializer=self.midpoint_initializer,
            trainable=self.midpoint_trainable, name="midpoint",
            dtype=K.floatx(),
            constraint=pk_constraints.MinMax(0.0, 1.0)
        )

        self.rate_trainable = rate_trainable
        if rate_initializer is None:
            rate_initializer = tf.keras.initializers.Constant(5.)
        self.rate_initializer = tf.keras.initializers.get(rate_initializer)
        self.rate = self.add_weight(
            shape=[], initializer=self.rate_initializer,
            trainable=self.rate_trainable, name="rate", dtype=K.floatx(),
        )

    def _split_stimulus_set(self, z):
        """Call at the start of kernel operation.

        Args:
            z: A tensor of embeddings.
                shape=TensorShape(
                    [batch_size, n_sample, 2, n_dim]
                )

        Returns:
            z_0: A tensor of embeddings for one part of the pair.
                shape=TensorShape(
                    [batch_size, n_sample, 1, n_dim]
                )
            z_1: A tensor of embeddings for the other part of the pair.
                shape=TensorShape(
                    [batch_size, n_sample, 1, n_dim]
                )

        """
        # Divide up stimuli sets for kernel call.
        z_0 = z[:, :, 0]
        z_1 = z[:, :, 1]
        return z_0, z_1

    def call(self, inputs, states):
        """Return predicted rating of a trial.

        Args:
            inputs[0]: i.e., stimulus_set: A tensor containing indices
                that define the stimuli used in each trial.
                shape=(batch_size, n_sample, n_stimuli_per_trial)
            inputs[1]: i.e., z: A tensor containing the embeddings for
                the stimulus set.
                shape=(batch_size, n_sample, n_stimuli_per_trial, n_dim)
            inputs[-1]: i.e., groups (optional): A tensor containing
                group membership information.

        Returns:
            probs: The probabilites as determined by a parameterized
                logistic function.

        """
        # stimulus_set = inputs[0]
        z = inputs[1]

        # Prep retrieved embeddings for kernel op based on behavior.
        z_q, z_r = self._split_stimulus_set(z)

        if self._pass_groups['kernel']:
            groups = inputs[-1]
            sim_qr = self.kernel([z_q, z_r, groups])
        else:
            sim_qr = self.kernel([z_q, z_r])

        prob = self.lower + tf.math.divide(
            self.upper - self.lower,
            1 + tf.math.exp(-self.rate * (sim_qr - self.midpoint))
        )

        # TODO temporary hack
        if tf.rank(prob) > 2:
            prob = prob[:, :, 0]

        # Add singleton trailing dimension since MSE assumes rank-2 Tensors.
        # prob = tf.expand_dims(prob, axis=-1) TODO, not sure if this is necessary anymore since we have timestep version
        return prob, states

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        config.update({
            'kernel': tf.keras.utils.serialize_keras_object(self.kernel),
            'lower_trainable': self.lower_trainable,
            'upper_trainable': self.upper_trainable,
            'midpoint_trainable': self.midpoint_trainable,
            'rate_trainable': self.rate_trainable,
            'lower_initializer': tf.keras.initializers.serialize(
                self.lower_initializer
            ),
            'upper_initializer': tf.keras.initializers.serialize(
                self.upper_initializer
            ),
            'midpoint_initializer': tf.keras.initializers.serialize(
                self.midpoint_initializer
            ),
            'rate_initializer': tf.keras.initializers.serialize(
                self.rate_initializer
            ),
        })
        return config

    @classmethod
    def from_config(cls, config):
        kernel_serial = config['kernel']
        config['kernel'] = tf.keras.layers.deserialize(kernel_serial)
        return super().from_config(config)
