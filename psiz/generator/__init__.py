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
"""Generators initialization."""

from psiz.generator.similarity.base import DocketGenerator
from psiz.generator.similarity.rank.random_rank import RandomRank
from psiz.generator.similarity.rank.active_rank import ActiveRank
from psiz.generator.similarity.rank.active_rank import expected_information_gain_rank
from psiz.generator.similarity.rate.random_rate import RandomRate
