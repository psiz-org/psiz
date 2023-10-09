# -*- coding: utf-8 -*-
# Copyright 2023 The PsiZ Authors. All Rights Reserved.
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
"""Module of Keras layers."""

from psiz.keras.layers.activations.exponential import ExponentialSimilarity
from psiz.keras.layers.activations.heavy_tailed import HeavyTailedSimilarity
from psiz.keras.layers.activations.inverse import InverseSimilarity
from psiz.keras.layers.activations.students_t import StudentsTSimilarity
from psiz.keras.layers.drop import Drop
from psiz.keras.layers.variational import Variational
from psiz.keras.layers.behaviors.experimental.alcove_cell import ALCOVECell
from psiz.keras.layers.behaviors.logistic import Logistic
from psiz.keras.layers.behaviors.rank_similarity_base import RankSimilarityBase
from psiz.keras.layers.behaviors.rank_similarity import RankSimilarity
from psiz.keras.layers.behaviors.experimental.rank_similarity_cell import (
    RankSimilarityCell,
)
from psiz.keras.layers.behaviors.soft_rank_base import SoftRankBase
from psiz.keras.layers.behaviors.soft_rank import SoftRank
from psiz.keras.layers.behaviors.rate_similarity_base import RateSimilarityBase
from psiz.keras.layers.behaviors.rate_similarity import RateSimilarity
from psiz.keras.layers.behaviors.experimental.rate_similarity_cell import (
    RateSimilarityCell,
)
from psiz.keras.layers.embeddings.gamma_diag import EmbeddingGammaDiag
from psiz.keras.layers.embeddings.laplace_diag import EmbeddingLaplaceDiag
from psiz.keras.layers.embeddings.log_normal_diag import EmbeddingLogNormalDiag
from psiz.keras.layers.embeddings.logit_normal_diag import EmbeddingLogitNormalDiag
from psiz.keras.layers.embeddings.normal_diag import EmbeddingNormalDiag
from psiz.keras.layers.embeddings.shared import EmbeddingShared
from psiz.keras.layers.embeddings.stochastic_embedding import StochasticEmbedding
from psiz.keras.layers.embeddings.trunc_normal_diag import EmbeddingTruncatedNormalDiag
from psiz.keras.layers.embeddings.variational import EmbeddingVariational
from psiz.keras.layers.behaviors.experimental.soft_rank_cell import SoftRankCell
from psiz.keras.layers.gates.gate import Gate
from psiz.keras.layers.gates.braid_gate import BraidGate
from psiz.keras.layers.gates.branch_gate import BranchGate
from psiz.keras.layers.gates.gate_adapter import GateAdapter
from psiz.keras.layers.gates.split_gate import SplitGate
from psiz.keras.layers.proximities.distance_based import DistanceBased
from psiz.keras.layers.proximities.experimental.cosine_similarity import (
    CosineSimilarity,
)
from psiz.keras.layers.proximities.experimental.inner_product import InnerProduct
from psiz.keras.layers.proximities.experimental.generalized_inner_product import (
    GeneralizedInnerProduct,
)
from psiz.keras.layers.proximities.mink import Minkowski
from psiz.keras.layers.proximities.mink_stochastic import MinkowskiStochastic
from psiz.keras.layers.proximities.mink_variational import MinkowskiVariational
from psiz.keras.layers.proximities.proximity import Proximity

__all__ = [
    "ExponentialSimilarity",
    "HeavyTailedSimilarity",
    "InverseSimilarity",
    "StudentsTSimilarity",
    "RankSimilarityBase",
    "RankSimilarity",
    "RankSimilarityCell",
    "SoftRankBase",
    "SoftRank",
    "RateSimilarityBase",
    "RateSimilarity",
    "RateSimilarityCell",
    "ALCOVECell",
    "Logistic",
    "EmbeddingGammaDiag",
    "EmbeddingLaplaceDiag",
    "EmbeddingLogNormalDiag",
    "EmbeddingLogitNormalDiag",
    "EmbeddingNormalDiag",
    "EmbeddingShared",
    "StochasticEmbedding",
    "EmbeddingTruncatedNormalDiag",
    "EmbeddingVariational",
    "SoftRankCell",
    "Gate",
    "BraidGate",
    "BranchGate",
    "GateAdapter",
    "SplitGate",
    "Drop",
    "DistanceBased",
    "CosineSimilarity",
    "InnerProduct",
    "GeneralizedInnerProduct",
    "Variational",
    "Minkowski",
    "MinkowskiStochastic",
    "MinkowskiVariational",
    "Proximity",
]
