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
"""Module for trials.

Classes:
    TrialComponent: Abstract class for trial component.

"""

from abc import ABCMeta, abstractmethod


class TrialComponent(metaclass=ABCMeta):
    """Abstract class for trial content data."""

    def __init__(self):
        """Initialize."""
        # Attributes determined by concrete class.
        self.n_sequence = None
        self.max_timestep = None

    @abstractmethod
    def stack(self, component_list):
        """Return new object with sequence-stacked data.

        Arguments:
            component_list: A tuple of TrialComponent objects to be
                stacked. All objects must be the same class.

        Returns:
            A new object.

        """

    @abstractmethod
    def subset(self, idx):
        """Return subset of data as a new object.

        Arguments:
            index: The indices corresponding to the subset.

        Returns:
            A new object.

        """

    @abstractmethod
    def _for_dataset(self, format='tf', timestep=True):
        """Return appropriately formatted data."""

    @abstractmethod
    def _save(self, grp):
        """Add relevant datasets to group.

        Example:
        grp.create_dataset("my_data_name", data=my_data)

        """

    @abstractmethod
    def _load(self, grp):
        """Retrieve relevant datasets from group."""
