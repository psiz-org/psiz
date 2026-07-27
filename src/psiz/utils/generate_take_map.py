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
"""Utilities for building EmbeddingTake mapping arrays."""

import numpy as np


def generate_take_map(membership_source, membership_destination=None, mode="full"):
    """Generate a map for an EmbeddingTake layer.

    Args:
        membership_source: Membership of parent level.
        membership_destination: Membership of current level.
        mode: Mode of the take map. Options are "full" and "minimal".

    Returns:
        take_map: Integer array used as EmbeddingTake input_map.

    Notes:
        In "minimal" mode, first occurrence ordering in
        `membership_destination` is preserved.
    """
    membership_source = np.asarray(membership_source)

    if mode == "full":
        take_map = membership_source
    elif mode == "minimal":
        if membership_destination is None:
            raise ValueError("`membership_destination` is required in minimal mode.")
        membership_destination = np.asarray(membership_destination)
        _, first_occurrence_indices = np.unique(
            membership_destination, return_index=True
        )
        first_occurrence_indices = np.sort(first_occurrence_indices)
        take_map = membership_source[first_occurrence_indices]
    else:
        raise ValueError(f"Invalid mode: {mode}")

    return np.asarray(take_map, dtype="int32")
