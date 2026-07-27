# -*- coding: utf-8 -*-
# Copyright 2026 The PsiZ Authors. All Rights Reserved.
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
"""Posterior factory strategies."""

import keras

from psiz.keras.layers.embeddings.non_centered_normal_diag import (
    EmbeddingNonCenteredNormalDiag,
)


@keras.saving.register_keras_serializable(
    package="psiz.keras.layers", name="PosteriorFactory"
)
class PosteriorFactory:
    """Base strategy object for constructing posterior layers."""

    def build(self, prior_minimal):
        """Build posterior layer given `prior_minimal`."""
        raise NotImplementedError("Subclasses must implement `build`.")

    def get_config(self):
        """Return configuration."""
        return {}

    @classmethod
    def from_config(cls, config):
        """Create factory from configuration."""
        return cls(**config)


@keras.saving.register_keras_serializable(
    package="psiz.keras.layers", name="NonCenteredPosteriorFactory"
)
class NonCenteredPosteriorFactory(PosteriorFactory):
    """Factory for `EmbeddingNonCenteredNormalDiag` posteriors."""

    def __init__(
        self,
        epsilon_loc_gradient_scale=None,
        epsilon_scale_gradient_scale=None,
        epsilon_loc_initializer=None,
        epsilon_scale_initializer=None,
        epsilon_loc_regularizer=None,
        epsilon_scale_regularizer=None,
        epsilon_loc_trainable=None,
        epsilon_scale_trainable=None,
    ):
        self.epsilon_loc_gradient_scale = epsilon_loc_gradient_scale
        self.epsilon_scale_gradient_scale = epsilon_scale_gradient_scale
        self.epsilon_loc_initializer = epsilon_loc_initializer
        self.epsilon_scale_initializer = epsilon_scale_initializer
        self.epsilon_loc_regularizer = epsilon_loc_regularizer
        self.epsilon_scale_regularizer = epsilon_scale_regularizer
        self.epsilon_loc_trainable = epsilon_loc_trainable
        self.epsilon_scale_trainable = epsilon_scale_trainable

    def build(self, prior_minimal):
        """Build posterior from a minimal prior embedding."""
        return EmbeddingNonCenteredNormalDiag(
            prior_minimal,
            epsilon_loc_gradient_scale=self.epsilon_loc_gradient_scale,
            epsilon_scale_gradient_scale=self.epsilon_scale_gradient_scale,
            epsilon_loc_initializer=self.epsilon_loc_initializer,
            epsilon_scale_initializer=self.epsilon_scale_initializer,
            epsilon_loc_regularizer=self.epsilon_loc_regularizer,
            epsilon_scale_regularizer=self.epsilon_scale_regularizer,
            epsilon_loc_trainable=self.epsilon_loc_trainable,
            epsilon_scale_trainable=self.epsilon_scale_trainable,
        )

    def get_config(self):
        """Return configuration."""
        return {
            "epsilon_loc_gradient_scale": self.epsilon_loc_gradient_scale,
            "epsilon_scale_gradient_scale": self.epsilon_scale_gradient_scale,
            "epsilon_loc_initializer": keras.initializers.serialize(
                keras.initializers.get(self.epsilon_loc_initializer)
            ),
            "epsilon_scale_initializer": keras.initializers.serialize(
                keras.initializers.get(self.epsilon_scale_initializer)
            ),
            "epsilon_loc_regularizer": keras.regularizers.serialize(
                keras.regularizers.get(self.epsilon_loc_regularizer)
            ),
            "epsilon_scale_regularizer": keras.regularizers.serialize(
                keras.regularizers.get(self.epsilon_scale_regularizer)
            ),
            "epsilon_loc_trainable": self.epsilon_loc_trainable,
            "epsilon_scale_trainable": self.epsilon_scale_trainable,
        }

    @classmethod
    def from_config(cls, config):
        """Create factory from configuration."""
        config = dict(config)
        config["epsilon_loc_initializer"] = keras.initializers.deserialize(
            config["epsilon_loc_initializer"]
        )
        config["epsilon_scale_initializer"] = keras.initializers.deserialize(
            config["epsilon_scale_initializer"]
        )
        config["epsilon_loc_regularizer"] = keras.regularizers.deserialize(
            config["epsilon_loc_regularizer"]
        )
        config["epsilon_scale_regularizer"] = keras.regularizers.deserialize(
            config["epsilon_scale_regularizer"]
        )
        return cls(**config)
