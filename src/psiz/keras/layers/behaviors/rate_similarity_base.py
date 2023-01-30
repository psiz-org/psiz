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
    RateSimilarityBase: A base layer for rate-similarity judgments.

"""

import tensorflow as tf
from tensorflow.keras import backend as K

import psiz.keras.constraints as pk_constraints
from psiz.keras.layers.gates.gate_adapter import GateAdapter


@tf.keras.utils.register_keras_serializable(
    package="psiz.keras.layers", name="RateSimilarityBase"
)
class RateSimilarityBase(tf.keras.layers.Layer):
    """A base layer for rate similarity behavior.

    Similarities are converted to ratings using a parameterized
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
        percept_adapter=None,
        kernel_adapter=None,
        lower_initializer=None,
        upper_initializer=None,
        midpoint_initializer=None,
        rate_initializer=None,
        lower_trainable=False,
        upper_trainable=False,
        midpoint_trainable=True,
        rate_trainable=True,
        data_scope=None,
        **kwargs
    ):
        """Initialize.

        Args:
            percept: A Keras Layer for computing perceptual embeddings.
            kernel: A Keras Layer for computing kernel similarity.
            percept_adapter (optional): A layer for adapting inputs
                to match the assumptions of the provided `percept`
                layer.
            kernel_adapter (optional): A layer for adapting inputs
                to match the assumptions of the provided `kernel`
                layer.
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
            data_scope (optional): String indicating the behavioral
                data that should be used for the layer.
            kwargs (optional): Additional keyword arguments.

        """
        super(RateSimilarityBase, self).__init__(**kwargs)
        self.percept = percept
        self.kernel = kernel

        # Derive prefix from configuration.
        n_stimuli = 2  # TODO Make an argument for layer.
        if data_scope is None:
            data_scope = "rate{0}".format(n_stimuli)
        self.data_scope = data_scope

        # Configure percept adapter.
        if percept_adapter is None:
            # Default adapter has no gating keys.
            percept_adapter = GateAdapter(format_inputs_as_tuple=True)
        self.percept_adapter = percept_adapter
        # Set required input keys.
        self.percept_adapter.input_keys = [self.data_scope + "_stimulus_set"]

        # Configure kernel adapter.
        if kernel_adapter is None:
            # Default adapter has not gating keys.
            kernel_adapter = GateAdapter(format_inputs_as_tuple=True)
        self.kernel_adapter = kernel_adapter
        self.kernel_adapter.input_keys = [
            self.data_scope + "_z_q",
            self.data_scope + "_z_r",
        ]

        self.lower_trainable = lower_trainable
        if lower_initializer is None:
            lower_initializer = tf.keras.initializers.Constant(0.0)
        self.lower_initializer = tf.keras.initializers.get(lower_initializer)
        self.lower = self.add_weight(
            shape=[],
            initializer=self.lower_initializer,
            trainable=self.lower_trainable,
            name="lower",
            dtype=K.floatx(),
            constraint=pk_constraints.GreaterEqualThan(min_value=0.0),
        )

        self.upper_trainable = upper_trainable
        if upper_initializer is None:
            upper_initializer = tf.keras.initializers.Constant(1.0)
        self.upper_initializer = tf.keras.initializers.get(upper_initializer)
        self.upper = self.add_weight(
            shape=[],
            initializer=self.upper_initializer,
            trainable=self.upper_trainable,
            name="upper",
            dtype=K.floatx(),
            constraint=pk_constraints.LessEqualThan(max_value=1.0),
        )

        self.midpoint_trainable = midpoint_trainable
        if midpoint_initializer is None:
            midpoint_initializer = tf.keras.initializers.Constant(0.5)
        self.midpoint_initializer = tf.keras.initializers.get(midpoint_initializer)
        self.midpoint = self.add_weight(
            shape=[],
            initializer=self.midpoint_initializer,
            trainable=self.midpoint_trainable,
            name="midpoint",
            dtype=K.floatx(),
            constraint=pk_constraints.MinMax(0.0, 1.0),
        )

        self.rate_trainable = rate_trainable
        if rate_initializer is None:
            rate_initializer = tf.keras.initializers.Constant(5.0)
        self.rate_initializer = tf.keras.initializers.get(rate_initializer)
        self.rate = self.add_weight(
            shape=[],
            initializer=self.rate_initializer,
            trainable=self.rate_trainable,
            name="rate",
            dtype=K.floatx(),
        )

    def build(self, input_shape):
        """Build."""
        # We assume axes semantics based on relative position from last axis.
        stimuli_axis = -1  # i.e., stimuli indices.
        # Convert from *relative* axis index to *absolute* axis index.
        n_axis = len(input_shape[self.data_scope + "_stimulus_set"])
        self._stimuli_axis = tf.constant(n_axis + stimuli_axis)

    def _split_stimulus_set(self, z):
        """Split stimulus set into pairs.

        Args:
            z: A tensor of embeddings.
                shape=TensorShape(
                    [batch_size, 2, n_dim]
                )

        Returns:
            z_0: A tensor of embeddings for one part of the pair.
                shape=TensorShape(
                    [batch_size, 1, n_dim]
                )
            z_1: A tensor of embeddings for the other part of the pair.
                shape=TensorShape(
                    [batch_size, 1, n_dim]
                )

        """
        # Divide up stimuli for kernel call.
        # NOTE: By using an array for `indices` we keep the stimuli axis. This
        # is useful because we need a singleton dimension when computing MSE
        # loss.
        z_0 = tf.gather(z, indices=tf.constant([0]), axis=self._stimuli_axis)
        z_1 = tf.gather(z, indices=tf.constant([1]), axis=self._stimuli_axis)
        return z_0, z_1

    def _pairwise_similarity(self, inputs_copied):
        """Compute pairwise similarity."""
        inputs_percept = self.percept_adapter(inputs_copied)
        z = self.percept(inputs_percept)
        # TensorShape=(batch_size, 2, n_dim])

        # Prepare retrieved embeddings point for kernel and then compute
        # similarity.
        z_q, z_r = self._split_stimulus_set(z)
        inputs_copied.update(
            {self.data_scope + "_z_q": z_q, self.data_scope + "_z_r": z_r}
        )
        inputs_kernel = self.kernel_adapter(inputs_copied)
        sim_qr = self.kernel(inputs_kernel)
        return sim_qr

    def get_config(self):
        """Return layer configuration."""
        config = super(RateSimilarityBase, self).get_config()
        config.update(
            {
                "percept": tf.keras.utils.serialize_keras_object(self.percept),
                "kernel": tf.keras.utils.serialize_keras_object(self.kernel),
                "percept_adapter": tf.keras.utils.serialize_keras_object(
                    self.percept_adapter
                ),
                "kernel_adapter": tf.keras.utils.serialize_keras_object(
                    self.kernel_adapter
                ),
                "lower_trainable": self.lower_trainable,
                "upper_trainable": self.upper_trainable,
                "midpoint_trainable": self.midpoint_trainable,
                "rate_trainable": self.rate_trainable,
                "lower_initializer": tf.keras.initializers.serialize(
                    self.lower_initializer
                ),
                "upper_initializer": tf.keras.initializers.serialize(
                    self.upper_initializer
                ),
                "midpoint_initializer": tf.keras.initializers.serialize(
                    self.midpoint_initializer
                ),
                "rate_initializer": tf.keras.initializers.serialize(
                    self.rate_initializer
                ),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        percept_serial = config["percept"]
        kernel_serial = config["kernel"]
        percept_adapter_serial = config["percept_adapter"]
        kernel_adapter_serial = config["kernel_adapter"]
        config["percept"] = tf.keras.layers.deserialize(percept_serial)
        config["kernel"] = tf.keras.layers.deserialize(kernel_serial)
        config["percept_adapter"] = tf.keras.layers.deserialize(percept_adapter_serial)
        config["kernel_adapter"] = tf.keras.layers.deserialize(kernel_adapter_serial)
        return super().from_config(config)
