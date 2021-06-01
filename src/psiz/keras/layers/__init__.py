# -*- coding: utf-8 -*-
# Copyright 2020 The PsiZ Authors. All Rights Reserved.
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
"""Keras layers initialization file."""

from psiz.keras.layers.variational import Variational
from psiz.keras.layers.group_level import GroupLevel
from psiz.keras.layers.gate_multi import GateMulti
from psiz.keras.layers.gate import Gate
from psiz.keras.layers.stimuli import Stimuli
from psiz.keras.layers.distances.minkowski import WeightedMinkowski
from psiz.keras.layers.distances.mink import Minkowski
from psiz.keras.layers.distances.mink_stochastic import MinkowskiStochastic
from psiz.keras.layers.distances.mink_variational import MinkowskiVariational
from psiz.keras.layers.similarities.exponential import ExponentialSimilarity
from psiz.keras.layers.similarities.heavy_tailed import HeavyTailedSimilarity
from psiz.keras.layers.similarities.inverse import InverseSimilarity
from psiz.keras.layers.similarities.students_t import StudentsTSimilarity
from psiz.keras.layers.kernels.distance_based import DistanceBased
from psiz.keras.layers.kernels.kernel import Kernel
from psiz.keras.layers.kernels.kernel import AttentionKernel
from psiz.keras.layers.behaviors.rank import RankBehavior
from psiz.keras.layers.behaviors.rate import RateBehavior
from psiz.keras.layers.behaviors.sort import SortBehavior
from psiz.keras.layers.embeddings.gamma_diag import EmbeddingGammaDiag
from psiz.keras.layers.embeddings.group_attention import GroupAttention
from psiz.keras.layers.embeddings.group_attn_variational import GroupAttentionVariational
from psiz.keras.layers.embeddings.laplace_diag import EmbeddingLaplaceDiag
from psiz.keras.layers.embeddings.log_normal_diag import EmbeddingLogNormalDiag
from psiz.keras.layers.embeddings.logit_normal_diag import EmbeddingLogitNormalDiag
from psiz.keras.layers.embeddings.nd import EmbeddingND
from psiz.keras.layers.embeddings.normal_diag import EmbeddingNormalDiag
from psiz.keras.layers.embeddings.shared import EmbeddingShared
from psiz.keras.layers.embeddings.stochastic_embedding import StochasticEmbedding
from psiz.keras.layers.embeddings.trunc_normal_diag import EmbeddingTruncatedNormalDiag
from psiz.keras.layers.embeddings.variational import EmbeddingVariational
