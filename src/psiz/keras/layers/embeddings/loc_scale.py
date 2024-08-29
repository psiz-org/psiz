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
"""Module of TensorFlow embedding layers.

Classes:
    _EmbeddingLocScale: An abstract embedding class for location, scale
        family of distributions.

"""


import keras
import tensorflow_probability as tfp

from psiz.keras.layers.embeddings.stochastic_embedding import StochasticEmbedding


class _EmbeddingLocScale(StochasticEmbedding):
    """A private base class for a location-scale embedding.

    Each embedding point is characterized by a location-scale
    distribution.

    """

    def __init__(
        self,
        input_dim,
        output_dim,
        mask_zero=False,
        input_length=1,
        loc_initializer=None,
        scale_initializer=None,
        loc_regularizer=None,
        scale_regularizer=None,
        loc_constraint=None,
        scale_constraint=None,
        loc_trainable=True,
        scale_trainable=True,
        sample_shape=(),
        **kwargs
    ):
        """Initialize.

        Args:
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
            input_dim,
            output_dim,
            mask_zero=mask_zero,
            input_length=input_length,
            sample_shape=sample_shape,
            **kwargs
        )

        # Handle initializers.
        if loc_initializer is None:
            loc_initializer = keras.initializers.RandomUniform()
        self.loc_initializer = keras.initializers.get(loc_initializer)
        if scale_initializer is None:
            scale_initializer = keras.initializers.RandomNormal(
                mean=tfp.math.softplus_inverse(1.0).numpy(), stddev=0.001
            )
        self.scale_initializer = keras.initializers.get(scale_initializer)

        # Handle regularizers.
        self.loc_regularizer = keras.regularizers.get(loc_regularizer)
        self.scale_regularizer = keras.regularizers.get(scale_regularizer)

        # Handle constraints.
        self.loc_constraint = keras.constraints.get(loc_constraint)
        self.scale_constraint = keras.constraints.get(scale_constraint)

        self.loc_trainable = self.trainable and loc_trainable
        self.scale_trainable = self.trainable and scale_trainable

    def call(self, inputs):
        """Call."""
        inputs = super().call(inputs)
        # Delay reification until end of subclass call in order to
        # generate independent samples for each instance in batch_size.
        inputs_loc = keras.ops.take(self.loc, inputs, axis=0)
        inputs_scale = keras.ops.take(self.scale, inputs, axis=0)
        return [inputs_loc, inputs_scale]

    def get_config(self):
        """Return layer configuration."""
        config = super(_EmbeddingLocScale, self).get_config()
        config.update(
            {
                "loc_initializer": keras.initializers.serialize(self.loc_initializer),
                "scale_initializer": keras.initializers.serialize(
                    self.scale_initializer
                ),
                "loc_regularizer": keras.regularizers.serialize(self.loc_regularizer),
                "scale_regularizer": keras.regularizers.serialize(
                    self.scale_regularizer
                ),
                "loc_constraint": keras.constraints.serialize(self.loc_constraint),
                "scale_constraint": keras.constraints.serialize(self.scale_constraint),
                "loc_trainable": self.loc_trainable,
                "scale_trainable": self.scale_trainable,
            }
        )
        return config
