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
    ContentGenerator

"""

from abc import ABCMeta, abstractmethod

import numpy as np


class ContentGenerator(metaclass=ABCMeta):
    """Abstract class for trial content data."""

    def __init__(self, elements=None, verbose=0):
        """Initialize.

        Args:
            elements: An array-like object indicating the elements
                eligible to be used in the generation process.
            verbose (optional): The verbosity of output.

        """
        # Check elements.
        try:
            n_element = len(elements)
            raw_elements = elements
        except TypeError:
            raise TypeError(
                "The argument `elements` must be an array-like object that "
                "specificies the set of elements eligible to be used during "
                "content generation."
            )

        self.n_element = n_element
        self.raw_elements = raw_elements
        self.element_indices = np.arange(n_element, dtype=np.int32)

        self.verbose = verbose

    @classmethod
    @abstractmethod
    def generate(cls, n_sample, **kwargs):
        """Generate samples of the corresponding content.

        Returns:
            content

        """
        pass
