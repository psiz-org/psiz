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
"""Module for data.

Classes:
    DatasetComponent: Abstract class for dataset component.

"""

from abc import ABCMeta, abstractmethod


class DatasetComponent(metaclass=ABCMeta):
    """Abstract class for dataset component."""

    def __init__(self):
        """Initialize."""
        self.n_sequence = None
        self.sequence_length = None
        self._timestep_axis = 1

    @property
    def timestep_axis(self):
        return self._timestep_axis

    @abstractmethod
    def export(self, export_format='tfds', with_timestep_axis=True):
        """Return appropriately formatted data.

        Args:
            export_format (optional): The output format of the dataset.
                By default the dataset is formatted as a
                    tf.data.Dataset object.
            with_timestep_axis (optional): Boolean indicating if data
                should be returned with a timestep axis. If `False`,
                data is reshaped.

        """
