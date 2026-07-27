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
"""Non-centered variational embedding layer."""

import keras
import numpy as np
import tensorflow_probability as tfp

from psiz.keras.layers.embeddings.embedding_take import EmbeddingTake
from psiz.keras.layers.embeddings.non_centered_normal_diag import (
    EmbeddingNonCenteredNormalDiag,
)
from psiz.keras.layers.embeddings.normal_diag import EmbeddingNormalDiag
from psiz.keras.layers.posterior_factory import NonCenteredPosteriorFactory
from psiz.keras.layers.variational import Variational
from psiz.utils.drill_down import drill_down
from psiz.utils.generate_take_map import generate_take_map


@keras.saving.register_keras_serializable(
    package="psiz.keras.layers", name="EmbeddingNonCenteredVariational"
)
class EmbeddingNonCenteredVariational(Variational):
    """Variational embedding layer for a non-centered parameterization."""

    def __init__(
        self,
        prior_full=None,
        membership_parent=None,
        membership_current=None,
        posterior_factory=None,
        kl_weight=0.0,
        kl_use_exact=False,
        kl_n_sample=100,
        **kwargs,
    ):
        """Initialize."""
        self.prior_full = prior_full
        self.membership_parent = np.asarray(membership_parent, dtype="int32")
        self.membership_current = np.asarray(membership_current, dtype="int32")
        self.posterior_factory = posterior_factory

        if not self.prior_full.built:
            self.prior_full.build([None])

        mask_zero = self.prior_full.mask_zero

        membership_parent_for_take = self._build_parent_source_map(
            self.prior_full, self.membership_parent
        )

        prior_map_minimal = generate_take_map(
            membership_parent_for_take,
            membership_destination=self.membership_current,
            mode="minimal",
        )
        posterior_map_full = self._build_posterior_full_map(self.membership_current)

        prior_map_minimal = self._account_for_mask_zero(prior_map_minimal, mask_zero)
        posterior_map_full = self._account_for_mask_zero(posterior_map_full, mask_zero)

        prior_core = drill_down(
            self.prior_full,
            stop_layers=[
                EmbeddingNormalDiag,
                EmbeddingNonCenteredNormalDiag,
            ],
        )
        prior_minimal = EmbeddingTake(
            embedding=prior_core,
            input_map=prior_map_minimal,
        )

        posterior_minimal = self.posterior_factory.build(prior_minimal)
        posterior_full = EmbeddingTake(
            embedding=posterior_minimal,
            input_map=posterior_map_full,
        )

        super(EmbeddingNonCenteredVariational, self).__init__(
            prior=self.prior_full,
            posterior=posterior_full,
            kl_weight=kl_weight,
            kl_use_exact=kl_use_exact,
            kl_n_sample=kl_n_sample,
            **kwargs,
        )

        self.prior.build([None])
        self.posterior.build([None])

        epsilon_prior = tfp.distributions.Normal(
            keras.ops.zeros_like(self.posterior._embedding.epsilon_embeddings.mean()),
            keras.ops.ones_like(self.posterior._embedding.epsilon_embeddings.mean()),
        )
        batch_ndims = keras.ops.size(epsilon_prior.batch_shape_tensor())
        self.epsilon_prior = tfp.distributions.Independent(
            epsilon_prior, reinterpreted_batch_ndims=batch_ndims
        )

    def call(self, inputs, training=None):
        """Call."""
        outputs = self.posterior(inputs)
        _ = self.prior(inputs)
        self.add_kl_loss(
            self.posterior._embedding.epsilon_embeddings,
            self.epsilon_prior,
        )
        return outputs

    @property
    def input_dim(self):
        """Getter for embeddings input_dim."""
        return self.posterior.input_dim

    @property
    def output_dim(self):
        """Getter for embeddings output_dim."""
        return self.posterior.output_dim

    @property
    def mask_zero(self):
        """Getter for embeddings mask_zero."""
        return self.posterior.mask_zero

    @property
    def embeddings(self):
        """Getter for posterior embeddings."""
        return self.posterior.embeddings

    def get_config(self):
        """Return configuration."""
        config = keras.layers.Layer.get_config(self)
        config.update(
            {
                "prior_full": keras.saving.serialize_keras_object(self.prior_full),
                "membership_parent": self.membership_parent.tolist(),
                "membership_current": self.membership_current.tolist(),
                "posterior_factory_class_name": self.posterior_factory.__class__.__name__,
                "posterior_factory_config": self.posterior_factory.get_config(),
                "kl_weight": float(self.kl_weight),
                "kl_use_exact": bool(self.kl_use_exact),
                "kl_n_sample": int(self.kl_n_sample),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        """Create layer from configuration."""
        config = dict(config)
        config["prior_full"] = keras.saving.deserialize_keras_object(
            config["prior_full"]
        )
        config["membership_parent"] = np.asarray(
            config["membership_parent"], dtype="int32"
        )
        config["membership_current"] = np.asarray(
            config["membership_current"], dtype="int32"
        )
        posterior_factory_class_name = config.pop("posterior_factory_class_name")
        posterior_factory_config = config.pop("posterior_factory_config")
        if posterior_factory_class_name == "NonCenteredPosteriorFactory":
            config["posterior_factory"] = NonCenteredPosteriorFactory.from_config(
                posterior_factory_config
            )
        else:
            raise ValueError(
                "Unrecognized posterior factory class name: "
                f"{posterior_factory_class_name}"
            )
        return cls(**config)

    def _account_for_mask_zero(self, take_map, mask_zero):
        """Account for mask_zero in take map."""
        if mask_zero:
            take_map = np.hstack(
                [
                    np.zeros([1], dtype="int32"),
                    take_map + 1,
                ]
            )
        return take_map

    def _build_posterior_full_map(self, membership_current):
        """Build full map from destination IDs to minimal posterior indices."""
        _, first_indices, inverse = np.unique(
            membership_current, return_inverse=True, return_index=True
        )
        sorted_unique_at_minimal_index = np.argsort(first_indices)
        minimal_index_for_sorted_unique = np.empty_like(sorted_unique_at_minimal_index)
        minimal_index_for_sorted_unique[sorted_unique_at_minimal_index] = np.arange(
            sorted_unique_at_minimal_index.shape[0],
            dtype=sorted_unique_at_minimal_index.dtype,
        )
        posterior_map_full = minimal_index_for_sorted_unique[inverse]
        return posterior_map_full.astype("int32")

    def _build_parent_source_map(self, prior_full, membership_parent):
        """Build source-row map aligned to the parent layer row ordering."""
        try:
            source_map = np.asarray(prior_full.posterior.input_map, dtype="int32")
            if prior_full.mask_zero:
                source_map = source_map[1:] - 1
            if source_map.shape[0] != membership_parent.shape[0]:
                raise ValueError(
                    "Parent source map shape does not match membership shape."
                )
        except AttributeError:
            source_map = membership_parent
        return source_map.astype("int32")
