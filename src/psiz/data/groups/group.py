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

import warnings

import numpy as np
import tensorflow as tf

from psiz.data.trial_component import TrialComponent
from psiz.data.unravel_timestep import unravel_timestep


class Group(TrialComponent):
    """Base class for group membership data."""

    def __init__(self, group_values, name=None):
        """Initialize.

        Args:
            group_values: An np.ndarray that must be rank-2 or
                rank-3.
                shape=(samples, [sequence_length], n_col)
            name: A string indicating the variable name of the group.

        """
        TrialComponent.__init__(self)
        group_values = self._rectify_shape(group_values)
        group_values = self._validate_group_values(
            name, group_values
        )
        self.n_sequence = group_values.shape[0]
        self.sequence_length = group_values.shape[1]
        self.name = name
        self.group_values = group_values

    def _rectify_shape(self, group_values):
        """Rectify shape of group weights."""
        if group_values.ndim == 2:
            # Assume independent trials and add singleton timestep axis.
            group_values = np.expand_dims(
                group_values, axis=self.timestep_axis
            )
        return group_values

    def _validate_group_values(self, group_key, group_values):
        """Validate group weights."""
        # Check rank of `group_values`.
        if not (group_values.ndim == 3):
            raise ValueError(
                "The values for '{0}' must be a rank-2 or rank-3 ND "
                "array. If using a sparse coding format, make sure you have "
                "a trailing singleton dimension to meet this "
                "requirement.".format(group_key)
            )

        # If `group_values` looks like sparse coding format, check data type.
        if group_values.shape[-1] == 1:
            if not isinstance(group_values[0, 0, 0], (int, np.integer)):
                # TODO more careful handling
                warnings.warn(
                    "The values for '{0}' appear to use a sparse "
                    "coding. To improve efficiency, these weights should "
                    "have an integer dtype.".format(group_key)
                )

        return group_values

    def export(self, export_format='tfds', with_timestep_axis=True):
        """Export.

        Args:
            export_format (optional): The output format of the dataset.
                By default the dataset is formatted as a
                    tf.data.Dataset object.
            with_timestep_axis (optional): Boolean indicating if data should be
                returned with a timestep axis. If `False`, data is
                reshaped.

        """
        group_values = self.group_values
        if with_timestep_axis is False:
            group_values = unravel_timestep(group_values)

        if export_format == 'tfds':
            group_values = tf.constant(
                group_values, name=('group/' + self.name)
            )
        else:
            raise ValueError(
                "Unrecognized `export_format` '{0}'.".format(export_format)
            )
        return {
            self.name: group_values
        }
