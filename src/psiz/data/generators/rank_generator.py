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

"""Module for data.

Classes:
    RankGenerator

"""


import numpy as np
from psiz.data.generators.content_generator import ContentGenerator


class RankGenerator(ContentGenerator):
    """Abstract class for generating Rank content."""

    def __init__(self, n_reference=None, n_select=None, **kwargs):
        """Initialize.

        Args:
            n_reference: A scalar indicating the number of references
                for each trial.
            n_select: A scalar indicating the number of selections an
                agent must make.

        """
        super(RankGenerator, self).__init__(**kwargs)

        if n_reference > self.n_element:
            raise ValueError(
                "The argument `n_reference` must be less than the number of "
                "elements in `elements`."
            )
        if n_select > n_reference:
            raise ValueError(
                "The argument `n_select` must less than or equal to `n_reference`."
            )

        # Sanitize inputs.
        self.n_reference = np.int32(n_reference)
        self.n_select = np.int32(n_select)
