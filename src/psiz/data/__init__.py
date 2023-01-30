# -*- coding: utf-8 -*-
# Copyright 2022 The PsiZ Authors. All Rights Reserved.
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
"""Module of convenience data classes."""


from psiz.data.dataset_component import DatasetComponent
from psiz.data.dataset import Dataset
from psiz.data.contents.content import Content
from psiz.data.contents.categorize import Categorize
from psiz.data.contents.rank import Rank
from psiz.data.contents.rate import Rate
from psiz.data.groups.group import Group
from psiz.data.outcomes.outcome import Outcome
from psiz.data.outcomes.continuous import Continuous
from psiz.data.outcomes.sparse_categorical import SparseCategorical
from psiz.data.sample_qr_sets import sample_qr_sets
from psiz.data.unravel_timestep import unravel_timestep

__all__ = [
    "DatasetComponent",
    "Dataset",
    "Content",
    "Categorize",
    "Rank",
    "Rate",
    "Group",
    "Outcome",
    "Continuous",
    "SparseCategorical",
    "sample_qr_sets",
    "unravel_timestep",
]
