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
"""Base functionality of generators.

Classes:
    DocketGenerator: Base class for generating a docket of unjudged
        similarity trials.

"""

from abc import ABCMeta, abstractmethod
import warnings


class DocketGenerator(object):
    """Abstract base class for generating similarity judgment trials.

    Methods:
        generate: Generate unjudged trials.

    """

    __metaclass__ = ABCMeta

    def __init_subclass__(cls, **kwargs):
        """Subclassing initialization."""
        warnings.warn(
            (
                f"{cls.__name__} is deprecated and will be removed. "
                "Please use the `psiz.data` module for creating datasets; "
                "version_announced=0.8.0; version_scheduled=0.9.0"
            ),
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init_subclass__(**kwargs)

    def __init__(self):
        """Initialize."""
        warnings.warn(
            (
                f"{self.__class__.__name__} is deprecated and will be "
                "removed. "
                "Please use the `psiz.data` module for creating datasets; "
                "version_announced=0.8.0; version_scheduled=0.9.0"
            ),
            DeprecationWarning,
            stacklevel=2,
        )

    @abstractmethod
    def generate(self, *args, **kwargs):
        """Return generated trials based on provided arguments.

        Args:
            n_stimuli

        Returns:
            A Docket object.

        """
