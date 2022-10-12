# -*- coding: utf-8 -*-
# Copyright 2022 The PsiZ Authors. All Rights Reserved.
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
    RateSimilarityCellV2: A rate similarity layer.

"""

import tensorflow as tf
from tensorflow.python.keras import backend as K

import psiz.keras.constraints as pk_constraints
from psiz.keras.layers.behaviors.behavior import Behavior
from psiz.keras.layers.gates.gate_adapter import GateAdapter


@tf.keras.utils.register_keras_serializable(
    package='psiz.keras.layers', name='RateSimilarityCellV2'
)
class RateSimilarityCellV2(Behavior):
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
        self,
        percept=None,
        kernel=None,
        percept_gating_keys=None,
        kernel_gating_keys=None,
        lower_initializer=None,
        upper_initializer=None,
        midpoint_initializer=None,
        rate_initializer=None,
        lower_trainable=False,
        upper_trainable=False,
        midpoint_trainable=True,
        rate_trainable=True,
        **kwargs
    ):
        """Initialize.

        Args:
            percept: A Keras Layer for computing perceptual embeddings.
            kernel: A Keras Layer for computing kernel similarity.
            percept_gating_keys (optional): A list of dictionary
                keys pointing to gate weights that should be passed to
                the `percept` layer. Since the `percept` layer assumes
                a tuple is passed to the `call` method, the weights are
                appended at the end of the "standard" Tensors, in the
                same order specified by the user.
            kernel_gating_keys (optional): A list of dictionary
                keys pointing to gate weights that should be passed to
                the `kernel` layer. Since the `kernel` layer assumes a
                tuple is passed to the `call` method, the weights are
                appended at the end of the "standard" Tensors, in the
                same order specified by the user.
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
        super(RateSimilarityCellV2, self).__init__(**kwargs)
        self.percept = percept
        self.kernel = kernel

        # Set up adapters based on provided gate weights.
        self._percept_adapter = GateAdapter(
            subnet=percept,
            input_keys=['rate_similarity_stimset_samples'],
            gating_keys=percept_gating_keys,
            format_inputs_as_tuple=True
        )
        self._kernel_adapter = GateAdapter(
            subnet=kernel,
            input_keys=['rate_similarity_z_q', 'rate_similarity_z_q'],
            gating_keys=kernel_gating_keys,
            format_inputs_as_tuple=True
        )

        # Satisfy RNNCell contract.
        # NOTE: A placeholder state.
        self.state_size = [
            tf.TensorShape([1])
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
        """Split stimulus set into pairs.

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

    def get_mask(self, inputs):
        """Return appropriate mask."""
        mask = tf.not_equal(inputs['rate_similarity_stimulus_set'], 0)
        return mask[:, :, 0, 0]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        """Get initial state."""
        initial_state = [
            tf.zeros([batch_size, 1], name='rate_cell_initial_state')
        ]
        return initial_state

    def call(self, inputs, states, training=None):
        """Return predicted rating of a trial.

        Args:
            inputs: A dictionary containing the following information:
                rate_similarity_stimulus_set: A tensor containing
                    indices that define the stimuli used in each trial.
                    shape=(batch_size, n_sample, n_stimuli_per_trial)
                gate_weights (optional): Tensor(s) containing gate
                    weights. The actual key value(s) will depend on how
                    the user initialized the layer.

        Returns:
            probs: The probabilites as determined by a parameterized
                logistic function.

        """
        stimulus_set = inputs['rate_similarity_stimulus_set']

        # Expand `sample_axis` of `stimulus_set` for stochastic
        # functionality (e.g., variational inference).
        stimulus_set = tf.repeat(
            stimulus_set, self.n_sample, axis=self.sample_axis_in_cell
        )

        # Embed stimuli indices in n-dimensional space.
        inputs.update({
            'rate_similarity_stimset_samples': stimulus_set
        })
        z = self._percept_adapter(inputs)
        # TensorShape=(batch_size, n_sample, 2, n_dim])

        # Prepare retrieved embeddings point for kernel and then compute
        # similarity.
        z_q, z_r = self._split_stimulus_set(z)
        inputs.update({
            'rate_similarity_z_q': z_q,
            'rate_similarity_z_r': z_r
        })
        sim_qr = self._kernel_adapter(inputs)

        prob = self.lower + tf.math.divide(
            self.upper - self.lower,
            1 + tf.math.exp(-self.rate * (sim_qr - self.midpoint))
        )

        return prob, states

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        config.update({
            'percept': tf.keras.utils.serialize_keras_object(self.percept),
            'kernel': tf.keras.utils.serialize_keras_object(self.kernel),
            'percept_gating_keys': (
                self._percept_adapter.gating_keys
            ),
            'kernel_gating_keys': self._kernel_adapter.gating_keys,
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
        percept_serial = config['percept']
        kernel_serial = config['kernel']
        config['percept'] = tf.keras.layers.deserialize(percept_serial)
        config['kernel'] = tf.keras.layers.deserialize(kernel_serial)
        return super().from_config(config)
