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
"""User-facing progressive builder API for hierarchical VI embeddings."""

from typing import Callable

import keras
import numpy as np

from psiz.keras.layers.embeddings.embedding_take import EmbeddingTake
from psiz.keras.layers.embeddings.non_centered_variational import (
    EmbeddingNonCenteredVariational,
)
from psiz.keras.layers.embeddings.normal_diag import EmbeddingNormalDiag
from psiz.keras.layers.hierarchical_specs import HierarchySpec
from psiz.keras.layers.hierarchical_specs import KLWeightingPolicy
from psiz.keras.layers.hierarchical_specs import LevelInitializationMode
from psiz.keras.layers.hierarchical_specs import MembershipInput
from psiz.keras.layers.hierarchical_specs import MembershipSourcePolicy
from psiz.keras.layers.hierarchical_specs import ParentMapPolicy
from psiz.keras.layers.hierarchical_specs import ScaleInitializationPolicy
from psiz.keras.layers.posterior_factory import NonCenteredPosteriorFactory
from psiz.keras.layers.posterior_factory import PosteriorFactory
from psiz.stochastic import softplus_inverse


class HierarchicalBuilderHooks:
    """Optional expert hooks for advanced customization."""

    def resolve_memberships(
        self,
        membership: MembershipInput,
        hierarchy: HierarchySpec,
    ) -> np.ndarray:
        raise NotImplementedError()

    def compute_gradient_scales(
        self,
        level_memberships: np.ndarray,
        n_sample_train: int,
        exponent: float,
    ) -> tuple[float, float]:
        raise NotImplementedError()

    def initialize_level_scale(
        self,
        n_dim: int,
        i_level: int,
        role: str,
        policy: ScaleInitializationPolicy,
        decay_ratio: float,
        floor_ratio: float,
    ) -> float:
        raise NotImplementedError()

    def compute_kl_weight(
        self,
        kl_scale: float,
        n_sample_train: int,
        n_branch: int,
        n_level: int,
        policy: KLWeightingPolicy,
    ) -> float:
        raise NotImplementedError()

    def build_level_factory(
        self,
        level_spec,
        i_level: int,
        memberships_level: np.ndarray,
        n_dim: int,
        n_sample_train: int,
        target_std: float,
        loc_gradient_scale: float,
        scale_gradient_scale: float,
        default_factory: PosteriorFactory,
        mask_zero: bool,
        variance_floor: float,
        scale_clip_min: float,
        scale_clip_max: float,
        warmstart_strength: float,
    ) -> PosteriorFactory | None:
        raise NotImplementedError()


@keras.saving.register_keras_serializable(
    package="psiz.keras.layers", name="HierarchicalVIEmbeddingBuilder"
)
class HierarchicalVIEmbeddingBuilder:
    """Builder for hierarchical variational embeddings.

    This class is the standard tier in the progressive-disclosure API.
    Use `build_hierarchical_vi_embedding` for a convenience wrapper.
    """

    def __init__(
        self,
        hierarchy: HierarchySpec,
        posterior_factory: PosteriorFactory,
        kl_policy: KLWeightingPolicy = KLWeightingPolicy.PER_SAMPLE_PER_BRANCH,
        kl_n_sample: int = 30,
        scale_policy: ScaleInitializationPolicy = ScaleInitializationPolicy.GEOMETRIC_DECAY,
        membership_source_policy: MembershipSourcePolicy = MembershipSourcePolicy.PRECOMPUTED,
        parent_map_policy: ParentMapPolicy = ParentMapPolicy.MINIMAL_FIRST_OCCURRENCE,
        scale_decay_ratio: float = 0.6,
        scale_floor_ratio: float = 0.15,
        gradient_scale_exponent: float = 0.25,
        variance_floor: float = 1e-4,
        scale_clip_min: float = 1e-6,
        scale_clip_max: float = 50.0,
        warmstart_strength: float = 0.1,
        custom_membership_resolver: Callable | None = None,
        custom_kl_weight_fn: Callable | None = None,
        custom_scale_init_fn: Callable | None = None,
    ):
        self.hierarchy = hierarchy
        self.posterior_factory = posterior_factory
        self.kl_policy = KLWeightingPolicy(kl_policy)
        self.kl_n_sample = int(kl_n_sample)
        self.scale_policy = ScaleInitializationPolicy(scale_policy)
        self.membership_source_policy = MembershipSourcePolicy(membership_source_policy)
        self.parent_map_policy = ParentMapPolicy(parent_map_policy)
        self.scale_decay_ratio = float(scale_decay_ratio)
        self.scale_floor_ratio = float(scale_floor_ratio)
        self.gradient_scale_exponent = float(gradient_scale_exponent)
        self.variance_floor = float(variance_floor)
        self.scale_clip_min = float(scale_clip_min)
        self.scale_clip_max = float(scale_clip_max)
        self.warmstart_strength = float(warmstart_strength)
        self.custom_membership_resolver = custom_membership_resolver
        self.custom_kl_weight_fn = custom_kl_weight_fn
        self.custom_scale_init_fn = custom_scale_init_fn

        self.hierarchy.validate(strict=True)
        self._validate_custom_requirements()

    def _validate_custom_requirements(self):
        """Enforce custom callback requirements for custom enum modes."""
        if (
            self.membership_source_policy == MembershipSourcePolicy.CUSTOM
            and self.custom_membership_resolver is None
        ):
            raise ValueError(
                "`custom_membership_resolver` is required when "
                "membership_source_policy='custom'."
            )
        if (
            self.kl_policy == KLWeightingPolicy.CUSTOM
            and self.custom_kl_weight_fn is None
        ):
            raise ValueError(
                "`custom_kl_weight_fn` is required when kl_policy='custom'."
            )
        if (
            self.scale_policy == ScaleInitializationPolicy.CUSTOM
            and self.custom_scale_init_fn is None
        ):
            raise ValueError(
                "`custom_scale_init_fn` is required when scale_policy='custom'."
            )

    def _resolve_memberships(self, membership: MembershipInput) -> np.ndarray:
        """Resolve memberships according to source policy."""
        membership.validate()

        if (
            self.membership_source_policy
            == MembershipSourcePolicy.STRICT_PRECOMPUTED_OVERRIDES_RESOLVER
        ):
            if membership.memberships is not None:
                return np.asarray(membership.memberships, dtype="int32")
            raise ValueError(
                "`memberships` must be provided for strict_precomputed_overrides_resolver."
            )

        if self.membership_source_policy == MembershipSourcePolicy.PRECOMPUTED:
            if membership.memberships is None:
                raise ValueError("`memberships` must be provided for precomputed mode.")
            return np.asarray(membership.memberships, dtype="int32")

        if self.membership_source_policy == MembershipSourcePolicy.DATAFRAME_RESOLVER:
            if membership.df_stimuli is None:
                raise ValueError(
                    "`df_stimuli` must be provided for dataframe_resolver mode."
                )
            memberships = np.zeros(
                [len(membership.df_stimuli), len(self.hierarchy.levels)], dtype="int32"
            )
            for i_level, level in enumerate(self.hierarchy.levels):
                if i_level == 0:
                    memberships[:, i_level] = np.zeros([len(membership.df_stimuli)])
                else:
                    if level.membership_key is None:
                        raise ValueError(
                            "Missing `membership_key` for non-root level in "
                            "dataframe_resolver mode."
                        )
                    memberships[:, i_level] = np.asarray(
                        membership.df_stimuli[level.membership_key], dtype="int32"
                    )
            return memberships

        if self.membership_source_policy == MembershipSourcePolicy.CUSTOM:
            memberships = self.custom_membership_resolver(membership, self.hierarchy)
            return np.asarray(memberships, dtype="int32")

        raise ValueError(
            "Unrecognized membership source policy: "
            f"{self.membership_source_policy.value}."
        )

    def _compute_kl_weight(
        self,
        kl_scale: float,
        n_sample_train: int,
        n_branch: int,
        n_level: int,
    ) -> float:
        """Compute KL weighting scalar according to configured policy."""
        if self.kl_policy == KLWeightingPolicy.PER_SAMPLE_PER_BRANCH:
            return float(kl_scale / (n_sample_train * n_branch))
        if self.kl_policy == KLWeightingPolicy.PER_SAMPLE:
            return float(kl_scale / n_sample_train)
        if self.kl_policy == KLWeightingPolicy.PER_SAMPLE_PER_LEVEL:
            return float(kl_scale / (n_sample_train * max(1, n_level - 1)))
        if self.kl_policy == KLWeightingPolicy.CUSTOM:
            return float(
                self.custom_kl_weight_fn(
                    kl_scale=kl_scale,
                    n_sample_train=n_sample_train,
                    n_branch=n_branch,
                    n_level=n_level,
                )
            )
        raise ValueError(f"Unrecognized KL policy: {self.kl_policy.value}")

    def _initialize_level_scale(self, n_dim: int, i_level: int, role: str) -> float:
        """Return target scale for initialization."""
        del role
        if self.scale_policy == ScaleInitializationPolicy.CONSTANT:
            return 1.0
        if self.scale_policy == ScaleInitializationPolicy.GEOMETRIC_DECAY:
            stddev_global = 1.0 / np.sqrt(n_dim)
            stddev = stddev_global * self.scale_decay_ratio**i_level
            return float(max(stddev, self.scale_floor_ratio * stddev_global))
        if self.scale_policy == ScaleInitializationPolicy.CUSTOM:
            return float(
                self.custom_scale_init_fn(
                    n_dim=n_dim,
                    i_level=i_level,
                    role=role,
                    decay_ratio=self.scale_decay_ratio,
                    floor_ratio=self.scale_floor_ratio,
                )
            )
        raise ValueError(f"Unrecognized scale policy: {self.scale_policy.value}")

    def _compute_gradient_scales(
        self,
        memberships_level: np.ndarray,
        n_sample_train: int,
    ) -> tuple[float, float]:
        """Compute non-centered gradient scales.

        Uses a coarse dataset-level proxy for expected class activity.
        """
        n_class_level = max(1, len(np.unique(memberships_level)))
        avg_hits_per_class = max(1.0, n_sample_train / n_class_level)
        loc_scale = 1.0 / (avg_hits_per_class**self.gradient_scale_exponent)
        scale_scale = 0.5 / (avg_hits_per_class**self.gradient_scale_exponent)
        return float(loc_scale), float(scale_scale)

    def _minimal_index_map(self, labels: np.ndarray) -> np.ndarray:
        """Return minimal-order class index for each row."""
        _, first_indices, inverse = np.unique(
            labels, return_index=True, return_inverse=True
        )
        sorted_unique_at_minimal_index = np.argsort(first_indices)
        minimal_index_for_sorted_unique = np.empty_like(sorted_unique_at_minimal_index)
        minimal_index_for_sorted_unique[sorted_unique_at_minimal_index] = np.arange(
            sorted_unique_at_minimal_index.shape[0],
            dtype=sorted_unique_at_minimal_index.dtype,
        )
        return minimal_index_for_sorted_unique[inverse].astype("int32")

    def _build_root_prior(self, n_stimuli: int, n_dim: int, memberships: np.ndarray):
        """Build the root prior mapped to full stimulus index space."""
        root_labels = memberships[:, 0]
        row_map = self._minimal_index_map(root_labels)
        n_root = int(np.max(row_map) + 1)

        target_std = self._initialize_level_scale(n_dim, 0, self.hierarchy.levels[0].role)
        untransformed_scale = keras.ops.convert_to_numpy(softplus_inverse(target_std))

        root_core = EmbeddingNormalDiag(
            input_dim=n_root + int(self.hierarchy.mask_zero),
            output_dim=n_dim,
            mask_zero=self.hierarchy.mask_zero,
            loc_initializer=keras.initializers.Constant(0.0),
            scale_initializer=keras.initializers.Constant(untransformed_scale),
            loc_trainable=self.hierarchy.levels[0].loc_trainable,
            scale_trainable=self.hierarchy.levels[0].scale_trainable,
        )

        if self.hierarchy.mask_zero:
            full_map = np.hstack(
                [np.zeros([1], dtype="int32"), row_map + 1]
            )
            if full_map.shape[0] != n_stimuli + 1:
                raise ValueError(
                    "Membership rows must match `n_stimuli` when mask_zero=True."
                )
        else:
            full_map = row_map
            if full_map.shape[0] != n_stimuli:
                raise ValueError("Membership rows must match `n_stimuli`.")

        return EmbeddingTake(embedding=root_core, input_map=full_map)

    def _build_level_factory(
        self,
        i_level: int,
        memberships_level: np.ndarray,
        n_dim: int,
        n_sample_train: int,
    ) -> PosteriorFactory:
        """Create a posterior factory for one non-root level."""
        level_spec = self.hierarchy.levels[i_level]
        target_std = self._initialize_level_scale(n_dim, i_level, level_spec.role)
        epsilon_loc_grad_scale, epsilon_scale_grad_scale = self._compute_gradient_scales(
            memberships_level, n_sample_train
        )

        if level_spec.initialization in [
            LevelInitializationMode.PRETRAINED,
            LevelInitializationMode.PRETRAINED_POINT_ESTIMATE,
        ]:
            raise NotImplementedError(
                "Pretrained initialization modes require a project-specific hook "
                "and are not implemented in the default builder yet."
            )

        if not isinstance(self.posterior_factory, NonCenteredPosteriorFactory):
            return self.posterior_factory

        return NonCenteredPosteriorFactory(
            epsilon_loc_gradient_scale=epsilon_loc_grad_scale,
            epsilon_scale_gradient_scale=epsilon_scale_grad_scale,
            epsilon_loc_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01),
            epsilon_scale_initializer=keras.initializers.RandomNormal(
                mean=keras.ops.convert_to_numpy(softplus_inverse(target_std)), stddev=0.001
            ),
            epsilon_loc_trainable=level_spec.loc_trainable,
            epsilon_scale_trainable=level_spec.scale_trainable,
        )

    def build(
        self,
        n_stimuli: int,
        n_dim: int,
        membership: MembershipInput,
        n_sample_train: int,
        kl_scale: float = 1.0,
        n_branch: int = 1,
        random_seed: int | None = None,
    ):
        """Build a hierarchical variational embedding stack."""
        if random_seed is not None:
            keras.utils.set_random_seed(random_seed)

        memberships = self._resolve_memberships(membership)
        expected_levels = len(self.hierarchy.levels)
        if memberships.shape[1] != expected_levels:
            raise ValueError(
                "Membership matrix has incompatible number of levels. "
                f"Expected {expected_levels}, found {memberships.shape[1]}."
            )
        if memberships.shape[0] != n_stimuli:
            raise ValueError(
                "Membership matrix row count must equal `n_stimuli`. "
                f"Expected {n_stimuli}, found {memberships.shape[0]}."
            )

        embedding_prior = self._build_root_prior(n_stimuli, n_dim, memberships)

        if expected_levels == 1:
            return embedding_prior

        vi_embedding = None
        for i_level in range(1, expected_levels):
            membership_parent = memberships[:, i_level - 1]
            membership_current = memberships[:, i_level]
            posterior_factory = self._build_level_factory(
                i_level=i_level,
                memberships_level=membership_current,
                n_dim=n_dim,
                n_sample_train=n_sample_train,
            )

            kl_weight = self._compute_kl_weight(
                kl_scale=kl_scale,
                n_sample_train=n_sample_train,
                n_branch=n_branch,
                n_level=expected_levels,
            )

            vi_embedding = EmbeddingNonCenteredVariational(
                prior_full=embedding_prior,
                membership_current=membership_current,
                membership_parent=membership_parent,
                posterior_factory=posterior_factory,
                kl_weight=kl_weight,
                kl_n_sample=self.kl_n_sample,
            )
            vi_embedding.build(None)
            embedding_prior = vi_embedding

        return vi_embedding

    def get_config(self):
        """Return serializable configuration."""
        return {
            "hierarchy": self.hierarchy.to_dict(),
            "posterior_factory": keras.saving.serialize_keras_object(
                self.posterior_factory
            ),
            "kl_policy": self.kl_policy.value,
            "kl_n_sample": int(self.kl_n_sample),
            "scale_policy": self.scale_policy.value,
            "membership_source_policy": self.membership_source_policy.value,
            "parent_map_policy": self.parent_map_policy.value,
            "scale_decay_ratio": float(self.scale_decay_ratio),
            "scale_floor_ratio": float(self.scale_floor_ratio),
            "gradient_scale_exponent": float(self.gradient_scale_exponent),
            "variance_floor": float(self.variance_floor),
            "scale_clip_min": float(self.scale_clip_min),
            "scale_clip_max": float(self.scale_clip_max),
            "warmstart_strength": float(self.warmstart_strength),
        }

    @classmethod
    def from_config(cls, config):
        """Create builder from configuration."""
        config = dict(config)
        config["hierarchy"] = HierarchySpec.from_dict(config["hierarchy"])
        config["posterior_factory"] = keras.saving.deserialize_keras_object(
            config["posterior_factory"]
        )
        return cls(**config)


class AdvancedHierarchicalVIEmbeddingBuilder(HierarchicalVIEmbeddingBuilder):
    """Advanced builder that delegates core computations to hooks when provided."""

    def __init__(self, *args, hooks: HierarchicalBuilderHooks | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.hooks = hooks

    def set_hooks(self, hooks: HierarchicalBuilderHooks) -> None:
        """Assign or replace hooks object."""
        self.hooks = hooks

    def _resolve_memberships(self, membership: MembershipInput) -> np.ndarray:
        if self.hooks is not None and hasattr(self.hooks, "resolve_memberships"):
            return np.asarray(
                self.hooks.resolve_memberships(membership, self.hierarchy), dtype="int32"
            )
        return super()._resolve_memberships(membership)

    def _compute_gradient_scales(
        self,
        memberships_level: np.ndarray,
        n_sample_train: int,
    ) -> tuple[float, float]:
        if self.hooks is not None and hasattr(self.hooks, "compute_gradient_scales"):
            return self.hooks.compute_gradient_scales(
                memberships_level,
                n_sample_train,
                self.gradient_scale_exponent,
            )
        return super()._compute_gradient_scales(memberships_level, n_sample_train)

    def _initialize_level_scale(self, n_dim: int, i_level: int, role: str) -> float:
        if self.hooks is not None and hasattr(self.hooks, "initialize_level_scale"):
            return float(
                self.hooks.initialize_level_scale(
                    n_dim,
                    i_level,
                    role,
                    self.scale_policy,
                    self.scale_decay_ratio,
                    self.scale_floor_ratio,
                )
            )
        return super()._initialize_level_scale(n_dim, i_level, role)

    def _compute_kl_weight(
        self,
        kl_scale: float,
        n_sample_train: int,
        n_branch: int,
        n_level: int,
    ) -> float:
        if self.hooks is not None and hasattr(self.hooks, "compute_kl_weight"):
            return float(
                self.hooks.compute_kl_weight(
                    kl_scale,
                    n_sample_train,
                    n_branch,
                    n_level,
                    self.kl_policy,
                )
            )
        return super()._compute_kl_weight(kl_scale, n_sample_train, n_branch, n_level)

    def _build_level_factory(
        self,
        i_level: int,
        memberships_level: np.ndarray,
        n_dim: int,
        n_sample_train: int,
    ) -> PosteriorFactory:
        level_spec = self.hierarchy.levels[i_level]
        target_std = self._initialize_level_scale(n_dim, i_level, level_spec.role)
        loc_gradient_scale, scale_gradient_scale = self._compute_gradient_scales(
            memberships_level, n_sample_train
        )

        if self.hooks is not None and hasattr(self.hooks, "build_level_factory"):
            factory = self.hooks.build_level_factory(
                level_spec=level_spec,
                i_level=i_level,
                memberships_level=memberships_level,
                n_dim=n_dim,
                n_sample_train=n_sample_train,
                target_std=target_std,
                loc_gradient_scale=loc_gradient_scale,
                scale_gradient_scale=scale_gradient_scale,
                default_factory=self.posterior_factory,
                mask_zero=self.hierarchy.mask_zero,
                variance_floor=self.variance_floor,
                scale_clip_min=self.scale_clip_min,
                scale_clip_max=self.scale_clip_max,
                warmstart_strength=self.warmstart_strength,
            )
            if factory is not None:
                return factory

        return super()._build_level_factory(
            i_level=i_level,
            memberships_level=memberships_level,
            n_dim=n_dim,
            n_sample_train=n_sample_train,
        )


def build_hierarchical_vi_embedding(
    n_stimuli: int,
    n_dim: int,
    hierarchy: HierarchySpec,
    membership: MembershipInput,
    posterior_factory: PosteriorFactory,
    n_sample_train: int,
    kl_scale: float = 1.0,
    n_branch: int = 1,
    kl_policy: KLWeightingPolicy = KLWeightingPolicy.PER_SAMPLE_PER_BRANCH,
    kl_n_sample: int = 30,
    scale_policy: ScaleInitializationPolicy = ScaleInitializationPolicy.GEOMETRIC_DECAY,
    membership_source_policy: MembershipSourcePolicy = MembershipSourcePolicy.PRECOMPUTED,
    parent_map_policy: ParentMapPolicy = ParentMapPolicy.MINIMAL_FIRST_OCCURRENCE,
    scale_decay_ratio: float = 0.6,
    scale_floor_ratio: float = 0.15,
    gradient_scale_exponent: float = 0.25,
    variance_floor: float = 1e-4,
    scale_clip_min: float = 1e-6,
    scale_clip_max: float = 50.0,
    warmstart_strength: float = 0.1,
    random_seed: int | None = None,
    custom_membership_resolver: Callable | None = None,
    custom_kl_weight_fn: Callable | None = None,
    custom_scale_init_fn: Callable | None = None,
):
    """Convenience wrapper around `HierarchicalVIEmbeddingBuilder`."""
    builder = HierarchicalVIEmbeddingBuilder(
        hierarchy=hierarchy,
        posterior_factory=posterior_factory,
        kl_policy=kl_policy,
        kl_n_sample=kl_n_sample,
        scale_policy=scale_policy,
        membership_source_policy=membership_source_policy,
        parent_map_policy=parent_map_policy,
        scale_decay_ratio=scale_decay_ratio,
        scale_floor_ratio=scale_floor_ratio,
        gradient_scale_exponent=gradient_scale_exponent,
        variance_floor=variance_floor,
        scale_clip_min=scale_clip_min,
        scale_clip_max=scale_clip_max,
        warmstart_strength=warmstart_strength,
        custom_membership_resolver=custom_membership_resolver,
        custom_kl_weight_fn=custom_kl_weight_fn,
        custom_scale_init_fn=custom_scale_init_fn,
    )
    return builder.build(
        n_stimuli=n_stimuli,
        n_dim=n_dim,
        membership=membership,
        n_sample_train=n_sample_train,
        kl_scale=kl_scale,
        n_branch=n_branch,
        random_seed=random_seed,
    )
