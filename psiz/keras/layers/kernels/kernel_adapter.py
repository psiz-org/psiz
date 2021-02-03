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
    KernelAdapter: A input signature adapter.

"""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(
    package='psiz.keras.layers', name='KernelAdapter'
)
class KernelAdapter(tf.keras.layers.Layer):
    """A stochastic Minkowski distance layer."""
    def __init__(self, kernel=None, **kwargs):
        """Initialize."""
        super(KernelAdapter, self).__init__(**kwargs)
        self.kernel = kernel

    def call(self, inputs):
        """Call.

        Convert 3-tuple inputs to 2-tuple inputs.

        """
        z_q = inputs[0]
        z_r = inputs[1]
        group = inputs[2]
        z_q = tf.broadcast_to(z_q, tf.shape(z_r))
        z_qr = tf.stack([z_q, z_r], axis=-1)
        return self.kernel([z_qr, group])

    def build(self, input_shape):
        """Build."""
        input_shape_stacked = input_shape[1].concatenate(tf.TensorShape([2]))
        new_input_shape = [input_shape_stacked, input_shape[2]]
        self.kernel.build(new_input_shape)

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        config.update({
            'kernel': tf.keras.utils.serialize_keras_object(self.kernel),
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Create from configuration."""
        config['kernel'] = tf.keras.layers.deserialize(config['kernel'])
        return cls(**config)

    def compute_output_shape(self, input_shape):
        """Computes the output shape of the layer.

        Arguments:
            input_shape: Shape tuple (tuple of integers) or list of
                shape tuples (one per output tensor of the layer).
                Shape tuples can include None for free dimensions,
                instead of an integer.

        Returns:
            A tf.TensorShape representing the output shape.

        NOTE: This method overrides the TF default, since the default
        cannot infer the correct output shape.

        """

        # Compute output shape for a subnetwork without passing in group
        # information.
        return tf.TensorShape(input_shape[0][0:-1])
