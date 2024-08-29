# -*- coding: utf-8 -*-
# Copyright 2024 The PsiZ Authors. All Rights Reserved.
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
"""Module of Keras proximity layers.

Classes:
    Proximity: A pairwise proximity kernel layer.

"""


import keras


@keras.saving.register_keras_serializable(package="psiz.keras.layers", name="Proximity")
class Proximity(keras.layers.Layer):
    """Abstract base class for pairwise proximity kernel layer.

    A pairwise proximity layer that consumes the last axis of the input
        tensors (see `call` method).

    NOTE: It is assumed that both tensors have the same rank, are
    broadcast-compatible, and have the same size for the last axis.

    """

    def __init__(self, activation=None, **kwargs):
        """Initialize.

        Args:
            activation (optional): An activation function to apply to
                the output of the distance layer.

        """
        super(Proximity, self).__init__(**kwargs)

        if activation is None:
            activation = keras.layers.Activation("linear")
        self.activation = activation

    def call(self, inputs):
        """Call."""
        raise NotImplementedError

    def build(self, input_shape):
        """Build."""
        # Since activation is applied to the output of the proximity layer, only need input shape
        # of one of the inputs, less the last axis which will have been consumed by the proximity layer.
        self.activation.build(input_shape[0][:-1])

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        config.update(
            {
                "activation": keras.saving.serialize_keras_object(self.activation),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        """Create from configuration."""
        config["activation"] = keras.saving.deserialize_keras_object(
            config["activation"]
        )
        return cls(**config)
