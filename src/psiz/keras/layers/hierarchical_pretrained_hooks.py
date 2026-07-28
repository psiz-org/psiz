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
"""Pretrained initialization hooks for hierarchical VI builders.

This module provides optional, project-agnostic hooks that enable
`LevelInitializationMode.PRETRAINED` and
`LevelInitializationMode.PRETRAINED_POINT_ESTIMATE` without modifying the
default builder behavior.
"""

import keras
import numpy as np

from psiz.keras.layers.hierarchical_specs import LevelInitializationMode
from psiz.keras.layers.posterior_factory import NonCenteredPosteriorFactory
from psiz.stochastic import softplus_inverse


class PretrainedNonCenteredFactoryHooks:
    """Hook object that builds non-centered posterior factories for pretrained modes.

    The hook reads optional arrays from `HierarchyLevelSpec.metadata`:
    - `pretrained_epsilon_loc`: array of shape `(n_class, n_dim)` or
      `(n_stimuli, n_dim)`.
    - `pretrained_epsilon_scale`: array of shape `(n_class, n_dim)` or
      `(n_stimuli, n_dim)`.

    If arrays are provided per-stimulus, they are reduced to minimal class order
    using first occurrence in `memberships_level`.
    """

    def _minimal_indices(self, memberships_level: np.ndarray) -> np.ndarray:
        _, first_indices = np.unique(memberships_level, return_index=True)
        return np.sort(first_indices)

    def _to_minimal_rows(
        self,
        values: np.ndarray,
        memberships_level: np.ndarray,
        n_class_minimal: int,
        mask_zero: bool,
    ) -> np.ndarray:
        values = np.asarray(values, dtype="float32")
        if values.ndim != 2:
            raise ValueError("Pretrained arrays must be rank-2 with shape (n, n_dim).")

        if values.shape[0] == n_class_minimal:
            minimal_values = values
        elif values.shape[0] == memberships_level.shape[0]:
            minimal_values = values[self._minimal_indices(memberships_level)]
        else:
            raise ValueError(
                "Pretrained array row count must match number of minimal classes "
                "or number of stimulus rows in current level."
            )

        if mask_zero:
            return np.vstack(
                [
                    np.zeros((1, minimal_values.shape[1]), dtype=minimal_values.dtype),
                    minimal_values,
                ]
            )
        return minimal_values

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
        default_factory,
        mask_zero: bool,
        variance_floor: float,
        scale_clip_min: float,
        scale_clip_max: float,
        warmstart_strength: float,
    ):
        del i_level, n_sample_train, scale_clip_min, scale_clip_max

        if not isinstance(default_factory, NonCenteredPosteriorFactory):
            return None

        if level_spec.initialization == LevelInitializationMode.DEFAULT:
            return None

        n_class_minimal = int(np.unique(memberships_level).shape[0])
        metadata = level_spec.metadata or {}

        if level_spec.initialization == LevelInitializationMode.PRETRAINED:
            if "pretrained_epsilon_loc" not in metadata:
                raise ValueError(
                    "Missing `pretrained_epsilon_loc` in level metadata for "
                    "PRETRAINED initialization."
                )
            if "pretrained_epsilon_scale" not in metadata:
                raise ValueError(
                    "Missing `pretrained_epsilon_scale` in level metadata for "
                    "PRETRAINED initialization."
                )

            epsilon_loc = self._to_minimal_rows(
                metadata["pretrained_epsilon_loc"],
                memberships_level,
                n_class_minimal,
                mask_zero,
            )
            epsilon_scale = self._to_minimal_rows(
                metadata["pretrained_epsilon_scale"],
                memberships_level,
                n_class_minimal,
                mask_zero,
            )

            if epsilon_loc.shape[1] != n_dim or epsilon_scale.shape[1] != n_dim:
                raise ValueError("Pretrained epsilon arrays must have width `n_dim`.")

            epsilon_scale = np.maximum(epsilon_scale, variance_floor)
            epsilon_scale_untransformed = keras.ops.convert_to_numpy(
                softplus_inverse(epsilon_scale)
            )

            return NonCenteredPosteriorFactory(
                epsilon_loc_gradient_scale=loc_gradient_scale,
                epsilon_scale_gradient_scale=scale_gradient_scale,
                epsilon_loc_initializer=keras.initializers.Constant(epsilon_loc),
                epsilon_scale_initializer=keras.initializers.Constant(
                    epsilon_scale_untransformed
                ),
                epsilon_loc_trainable=level_spec.loc_trainable,
                epsilon_scale_trainable=level_spec.scale_trainable,
            )

        if level_spec.initialization == LevelInitializationMode.PRETRAINED_POINT_ESTIMATE:
            if "pretrained_epsilon_loc" not in metadata:
                raise ValueError(
                    "Missing `pretrained_epsilon_loc` in level metadata for "
                    "PRETRAINED_POINT_ESTIMATE initialization."
                )

            epsilon_loc = self._to_minimal_rows(
                metadata["pretrained_epsilon_loc"],
                memberships_level,
                n_class_minimal,
                mask_zero,
            )
            if epsilon_loc.shape[1] != n_dim:
                raise ValueError("Pretrained epsilon loc array must have width `n_dim`.")

            point_std = max(variance_floor, warmstart_strength * target_std)
            point_scale = np.full_like(epsilon_loc, point_std, dtype="float32")
            point_scale_untransformed = keras.ops.convert_to_numpy(
                softplus_inverse(point_scale)
            )

            return NonCenteredPosteriorFactory(
                epsilon_loc_gradient_scale=loc_gradient_scale,
                epsilon_scale_gradient_scale=scale_gradient_scale,
                epsilon_loc_initializer=keras.initializers.Constant(epsilon_loc),
                epsilon_scale_initializer=keras.initializers.Constant(
                    point_scale_untransformed
                ),
                epsilon_loc_trainable=level_spec.loc_trainable,
                epsilon_scale_trainable=level_spec.scale_trainable,
            )

        return None
