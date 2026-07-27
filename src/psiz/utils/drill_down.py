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
"""Layer traversal utilities."""

from typing import Iterable


def drill_down(emb, stop_layers: Iterable[type]):
    """Traverse nested embedding wrappers until reaching `stop_layers`.

    Notes:
        Variational wrappers are first peeled to their posterior branch,
        then `_embedding` wrappers (e.g., EmbeddingTake) are traversed.
    """
    # Avoid an import-time cycle with psiz.keras.layers by checking for the
    # Variational interface instead of importing the class here.
    if hasattr(emb, "posterior") and hasattr(emb, "prior"):
        emb = emb.posterior

    stop_layers = tuple(stop_layers)
    while not isinstance(emb, stop_layers):
        if not hasattr(emb, "_embedding"):
            raise ValueError(
                "Could not drill down to requested stop layer. "
                f"Encountered terminal layer type: {type(emb)}"
            )
        emb = emb._embedding

    return emb
