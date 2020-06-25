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
"""Custom TensorFlow layers initialization file."""

from psiz.keras.layers.variational import Variational
from psiz.keras.layers.kernel import WeightedMinkowski
from psiz.keras.layers.kernel import GroupAttention
from psiz.keras.layers.kernel import InverseSimilarity
from psiz.keras.layers.kernel import ExponentialSimilarity
from psiz.keras.layers.kernel import HeavyTailedSimilarity
from psiz.keras.layers.kernel import StudentsTSimilarity
from psiz.keras.layers.kernel import Kernel
from psiz.keras.layers.kernel import AttentionKernel
from psiz.keras.layers.kernel import GroupAttentionVariational
from psiz.keras.layers.behavior import RankBehavior
from psiz.keras.layers.behavior import RateBehavior
from psiz.keras.layers.embeddings import EmbeddingNormalDiag
from psiz.keras.layers.embeddings import EmbeddingLaplaceDiag
from psiz.keras.layers.embeddings import EmbeddingLogNormalDiag
from psiz.keras.layers.embeddings import EmbeddingLogitNormalDiag
from psiz.keras.layers.embeddings import EmbeddingGammaDiag
from psiz.keras.layers.embeddings import EmbeddingVariational
