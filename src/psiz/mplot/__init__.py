# -*- coding: utf-8 -*-
# Copyright 2021 The PsiZ Authors. All Rights Reserved.
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
"""Module of plotting functionality based on matplotlib."""

from psiz.mplot.bvn_ellipse import bvn_ellipse
from psiz.mplot.hdi_bvn import hdi_bvn
from psiz.mplot.heatmap_embeddings import heatmap_embeddings
from psiz.mplot.embedding_input_dimension import embedding_input_dimension
from psiz.mplot.embedding_output_dimension import embedding_output_dimension

__all__ = [
    "bvn_ellipse",
    "hdi_bvn",
    "heatmap_embeddings",
    "embedding_input_dimension",
    "embedding_output_dimension",
]
