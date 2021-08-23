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
"""Base functionality for simulating agent behavior.

Classes:
    Agent: Base class for simulating agent behavior.

"""

from abc import ABCMeta, abstractmethod


class Agent():  # pylint: disable=too-few-public-methods
    """Abstract base class for simulating agent behavior.

    Methods:
        simulate: Simulate behavior.

    """

    __metaclass__ = ABCMeta

    def __init__(self):
        """Initialize."""

    @abstractmethod
    def simulate(self, docket, batch_size=None):
        """Return simulated trials.

        Arguments:
            docket: A trial docket.
            batch_size (optional): The batch size.

        Returns:
            An Observations object.

        """
