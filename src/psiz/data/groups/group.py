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

    def __init__(self, group_weights=None, name=None):
        """Initialize.

        Args:
            groups_weights: An np.ndarray that must be rank-2 or
                rank-3.
                shape=(samples, [sequence_length], n_col)
            name: A string indicating the variable name of the group.

        """
        TrialComponent.__init__(self)
        # TODO rename group_weights to weights?
        group_weights = self._rectify_shape(group_weights)
        group_weights = self._validate_group_weights(
            name, group_weights
        )
        self.n_sequence = group_weights.shape[0]
        self.sequence_length = group_weights.shape[1]
        self.name = name
        self.group_weights = group_weights

    def _rectify_shape(self, group_weights):
        """Rectify shape of group weights."""
        if group_weights.ndim == 2:
            # Assume independent trials and add singleton timestep axis.
            group_weights = np.expand_dims(
                group_weights, axis=self.timestep_axis
            )
        return group_weights

    def _validate_group_weights(self, group_key, group_weights):
        """Validate group weights."""
        # Check rank of `group_weights`.
        if not (group_weights.ndim == 3):
            raise ValueError(
                "The group weights for '{0}' must be a rank-2 or rank-3 ND "
                "array. If using a sparse coding format, make sure you have "
                "a trailing singleton dimension to meet this "
                "requirement.".format(group_key)
            )

        # If `group_weights` looks like sparse coding format, check data type.
        if group_weights.shape[-1] == 1:
            if not isinstance(group_weights[0, 0, 0], (int, np.integer)):
                warnings.warn(
                    "The group weights for '{0}' appear to use a sparse "
                    "coding. To improve efficiency, these weights should "
                    "have an integer dtype.".format(group_key)
                )

        # Check lowerbound support limit.
        bad_locs = group_weights < 0
        n_bad = np.sum(bad_locs)
        if n_bad != 0:
            raise ValueError(
                "The group weights for '{0}' contain values less than 0. "
                "Found {1} bad trial(s).".format(
                    group_key, n_bad
                )
            )
        return group_weights

    def export(self, export_format='tf', with_timestep_axis=True):
        """Export.

        Args:
            export_format (optional): The output format of the dataset.
                By default the dataset is formatted as a
                    tf.data.Dataset object.
            with_timestep_axis (optional): Boolean indicating if data should be
                returned with a timestep axis. If `False`, data is
                reshaped.

        """
        if with_timestep_axis is False:
            group_weights = unravel_timestep(self.group_weights)
        else:
            group_weights = self.group_weights

        if export_format == 'tf':
            group_weights = tf.constant(group_weights)
        else:
            raise ValueError(
                "Unrecognized `export_format` '{0}'.".format(export_format)
            )
        return {
            self.name: group_weights
        }

    def save(self, h5_grp):
        """Add relevant data to H5 group.

        Args:
            h5_grp: H5 group for saving data.

        """
        h5_grp.create_dataset("class_name", data="psiz.data.Group")
        h5_grp.create_dataset("group_weights", data=self.group_weights)
        h5_grp.create_dataset("name", data=self.name)

    def subset(self, idx):
        """Return subset of data as a new object.

        Args:
            index: The indices corresponding to the subset.

        Returns:
            A new object.

        """
        return Group(group_weights=self.group_weights[idx], name=self.name)

    # TODO delete stack
    # def stack(self, trials_list, sequence_length):
    #     """Stack `groups` data."""

    #     # First check that groups keys are compatible.
    #     # NOTE: It is not safe to simply pad an missing key with zeros, since
    #     # zero likely has user-defined semantics.
    #     group_keys = trials_list[0].groups.keys()
    #     for i_trials in trials_list[1:]:
    #         i_group_keys = i_trials.groups.keys()
    #         if group_keys != i_group_keys:
    #             raise ValueError(
    #                 'The dictionary keys of `groups` must be identical '
    #                 'for all TrialDatasets. Got a mismatch: {0} and '
    #                 '{1}.'.format(str(group_keys), str(i_group_keys))
    #             )

    #     # Loop over each key in groups.
    #     groups_stacked = {}
    #     for key in group_keys:
    #         # Check that shapes are compatible.
    #         value_shape = trials_list[0].groups[key].shape
    #         for i_trials in trials_list[1:]:
    #             i_value_shape = i_trials.groups[key].shape
    #             is_axis_2_ok = value_shape[2] == i_value_shape[2]
    #             if not is_axis_2_ok:
    #                 raise ValueError(
    #                     "The shape of 'groups's '{0}' is not compatible. They "
    #                     "must be identical on axis=2.".format(key)
    #                 )

    #         # Start by padding first entry in list.
    #         timestep_pad = sequence_length - trials_list[0].sequence_length
    #         pad_width = ((0, 0), (0, timestep_pad), (0, 0))
    #         groups = np.pad(
    #             trials_list[0].groups[key],
    #             pad_width, mode='constant', constant_values=0
    #         )

    #         # Loop over remaining list.
    #         for i_trials in trials_list[1:]:
    #             timestep_pad = sequence_length - i_trials.sequence_length
    #             pad_width = ((0, 0), (0, timestep_pad), (0, 0))
    #             curr_groups = np.pad(
    #                 i_trials.groups[key],
    #                 pad_width, mode='constant', constant_values=0
    #             )

    #             groups = np.concatenate(
    #                 (groups, curr_groups), axis=0
    #             )
    #         groups_stacked[key] = groups

    #     return groups_stacked

    @classmethod
    def load(cls, h5_grp):
        """Retrieve relevant datasets from group.

        Args:
            h5_grp: H5 group from which to load data.

        """
        group_weights = h5_grp["group_weights"][()]
        name = h5_grp["name"].asstr()[()]
        return cls(group_weights=group_weights, name=name)
