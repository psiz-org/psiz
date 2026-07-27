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
"""Typed specs and enums for hierarchical VI embedding builders."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


class KLWeightingPolicy(str, Enum):
    """Policies for scaling KL loss across hierarchy construction."""

    PER_SAMPLE_PER_BRANCH = "per_sample_per_branch"
    PER_SAMPLE = "per_sample"
    PER_SAMPLE_PER_LEVEL = "per_sample_per_level"
    CUSTOM = "custom"


class ScaleInitializationPolicy(str, Enum):
    """Policies for initializing scale across hierarchy levels."""

    GEOMETRIC_DECAY = "geometric_decay"
    CONSTANT = "constant"
    CUSTOM = "custom"


class LevelInitializationMode(str, Enum):
    """Initialization mode for each hierarchy level."""

    DEFAULT = "default"
    PRETRAINED = "pretrained"
    PRETRAINED_POINT_ESTIMATE = "pretrained_point_estimate"


class MembershipSourcePolicy(str, Enum):
    """Policies for resolving hierarchy memberships."""

    PRECOMPUTED = "precomputed"
    DATAFRAME_RESOLVER = "dataframe_resolver"
    CUSTOM = "custom"
    STRICT_PRECOMPUTED_OVERRIDES_RESOLVER = (
        "strict_precomputed_overrides_resolver"
    )


class ParentMapPolicy(str, Enum):
    """Policies for mapping parent rows between hierarchy levels."""

    MINIMAL_FIRST_OCCURRENCE = "minimal_first_occurrence"
    FULL_IDENTITY = "full_identity"
    CUSTOM = "custom"


@dataclass
class HierarchyLevelSpec:
    """Typed specification for one hierarchy level."""

    role: str
    membership_key: str | None = None
    loc_trainable: bool = True
    scale_trainable: bool = True
    initialization: LevelInitializationMode = LevelInitializationMode.DEFAULT
    scale_regularizer_l1: float = 0.0
    scale_regularizer_l2: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "role": self.role,
            "membership_key": self.membership_key,
            "loc_trainable": bool(self.loc_trainable),
            "scale_trainable": bool(self.scale_trainable),
            "initialization": self.initialization.value,
            "scale_regularizer_l1": float(self.scale_regularizer_l1),
            "scale_regularizer_l2": float(self.scale_regularizer_l2),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, config: dict[str, Any]):
        """Deserialize from dictionary."""
        config = dict(config)
        config["initialization"] = LevelInitializationMode(config["initialization"])
        metadata = config.get("metadata")
        if metadata is None:
            config["metadata"] = {}
        return cls(**config)


@dataclass
class HierarchySpec:
    """Typed specification for a hierarchy."""

    levels: list[HierarchyLevelSpec]
    mask_zero: bool = True

    def validate(self, strict: bool = True) -> None:
        """Validate hierarchy specification."""
        if len(self.levels) < 1:
            raise ValueError("`levels` must contain at least one level.")
        for i_level, level in enumerate(self.levels):
            if not isinstance(level, HierarchyLevelSpec):
                raise ValueError(
                    "All entries in `levels` must be HierarchyLevelSpec objects. "
                    f"Found type={type(level)} at index {i_level}."
                )
            if strict and level.membership_key is None and i_level > 0:
                raise ValueError(
                    "Each non-root level must provide `membership_key` in strict mode."
                )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "levels": [level.to_dict() for level in self.levels],
            "mask_zero": bool(self.mask_zero),
        }

    @classmethod
    def from_dict(cls, config: dict[str, Any]):
        """Deserialize from dictionary."""
        config = dict(config)
        config["levels"] = [
            HierarchyLevelSpec.from_dict(level_cfg) for level_cfg in config["levels"]
        ]
        return cls(**config)


@dataclass
class MembershipInput:
    """Container for membership sources used by the builder."""

    memberships: np.ndarray | None = None
    df_stimuli: pd.DataFrame | None = None
    resolver_name: str | None = None

    def validate(self) -> None:
        """Validate container shape and types when present."""
        if self.memberships is not None:
            memberships = np.asarray(self.memberships)
            if memberships.ndim != 2:
                raise ValueError("`memberships` must be a 2D array.")
        if self.df_stimuli is not None and not isinstance(self.df_stimuli, pd.DataFrame):
            raise ValueError("`df_stimuli` must be a pandas DataFrame.")

    def to_dict(self) -> dict[str, Any]:
        """Serialize lightweight fields only."""
        return {
            "resolver_name": self.resolver_name,
            # membership arrays/dataframes are runtime data, not persisted here.
        }

    @classmethod
    def from_dict(cls, config: dict[str, Any]):
        """Deserialize lightweight fields only."""
        return cls(resolver_name=config.get("resolver_name"))
