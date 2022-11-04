
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
"""Trials module.


Classes:
    RateSimilarity: Trial content requiring similarity ratings.

"""

import numpy as np
import tensorflow as tf

from psiz.data.contents.content import Content
from psiz.data.unravel_timestep import unravel_timestep


class RateSimilarity(Content):
    """Trial content requiring similarity ratings."""

    def __init__(self, stimulus_set):
        """Initialize.

        Args:
            stimulus_set:  A np.ndarray of non-negative integers
                indicating specific stimuli. The value "0" can be used
                as a placeholder. Must be rank-2 or rank-3. If rank-2,
                it is assumed that sequence_length=1 and a singleton
                dimension is added.
                shape=
                    (samples, 2)
                    OR
                    (samples, sequence_length, 2)

        Raises:
            ValueError if improper arguments are provided.

        """
        Content.__init__(self)
        stimulus_set = self._rectify_shape(stimulus_set)
        self.n_sequence = stimulus_set.shape[0]
        self.sequence_length = stimulus_set.shape[1]
        stimulus_set = self._validate_stimulus_set(stimulus_set)
        self.stimulus_set = stimulus_set

    @property
    def is_actual(self):
        """Return 2D Boolean array indicating trials with actual content."""
        return np.not_equal(self.stimulus_set[:, :, 0], self.mask_value)

    def stack(self, component_list):
        """Return new object with sequence-stacked data.

        Args:
            component_list: A tuple of TrialComponent objects to be
                stacked. All objects must be the same class.

        Returns:
            A new object.

        """
        # Determine maximum number of timesteps.
        sequence_length = 0
        for i_component in component_list:
            if i_component.sequence_length > sequence_length:
                sequence_length = i_component.sequence_length

        # Start by padding first entry in list.
        timestep_pad = sequence_length - component_list[0].sequence_length
        pad_width = ((0, 0), (0, timestep_pad), (0, 0))
        stimulus_set = np.pad(
            component_list[0].stimulus_set,
            pad_width, mode='constant', constant_values=0
        )

        # Loop over remaining list.
        for i_component in component_list[1:]:

            timestep_pad = sequence_length - i_component.sequence_length
            pad_width = ((0, 0), (0, timestep_pad), (0, 0))
            curr_stimulus_set = np.pad(
                i_component.stimulus_set,
                pad_width, mode='constant', constant_values=0
            )

            stimulus_set = np.concatenate(
                (stimulus_set, curr_stimulus_set), axis=0
            )

        return RateSimilarity(stimulus_set)

    def subset(self, idx):
        """Return subset of data as a new object.

        Args:
            index: The indices corresponding to the subset.

        Returns:
            A new object.

        """
        stimulus_set_sub = self.stimulus_set[idx]
        return RateSimilarity(stimulus_set_sub)

    def _rectify_shape(self, stimulus_set):
        """Rectify shape of `stimulus_set`."""
        if stimulus_set.ndim == 2:
            # Assume trials are independent and add singleton dimension for
            # timestep axis.
            stimulus_set = np.expand_dims(
                stimulus_set, axis=self.timestep_axis
            )
        return stimulus_set

    def _validate_stimulus_set(self, stimulus_set):
        """Validate `stimulus_set`.

        Raises:
            ValueError

        """
        # Check that provided values are integers.
        if not issubclass(stimulus_set.dtype.type, np.integer):
            raise ValueError(
                "The argument `stimulus_set` must be an np.ndarray of "
                "integers."
            )

        # Check that all values are greater than or equal to placeholder.
        if np.sum(np.less(stimulus_set, self.mask_value)) > 0:
            raise ValueError(
                "The argument `stimulus_set` must contain integers "
                "greater than or equal to 0."
            )

        # Check shape.
        if not stimulus_set.ndim == 3:
            raise ValueError(
                "The argument `stimulus_set` must be a rank-2 or rank-3 "
                "ndarray with a shape corresponding to (samples, "
                "sequence_length, n_stimuli_per_trial)."
            )

        # TODO is this really the right policy?
        # Check values are in int32 range.
        ii32 = np.iinfo(np.int32)
        if np.sum(np.greater(stimulus_set, ii32.max)) > 0:
            raise ValueError((
                "The argument `stimulus_set` must only contain integers "
                "in the int32 range."
            ))
        return stimulus_set.astype(np.int32)

    def export(self, export_format='tf', with_timestep_axis=True):
        """Prepare trial content data for dataset.

        Args:
            export_format (optional): The output format of the dataset.
                By default the dataset is formatted as a
                    tf.data.Dataset object.
            with_timestep_axis (optional): Boolean indicating if data should be
                returned with a timestep axis. If `False`, data is
                reshaped.

        """
        if export_format == 'tf':
            stimulus_set = self.stimulus_set
            if with_timestep_axis is False:
                stimulus_set = unravel_timestep(stimulus_set)
            x = {
                'rate_similarity_stimulus_set': tf.constant(
                    stimulus_set, dtype=tf.int32
                )
            }
        else:
            raise ValueError(
                "Unrecognized `export_format` '{0}'.".format(export_format)
            )
        return x

    def save(self, h5_grp):
        """Add relevant data to H5 group.

        Args:
            h5_grp: H5 group for saving data.

        """
        h5_grp.create_dataset("class_name", data="psiz.data.RateSimilarity")
        h5_grp.create_dataset("stimulus_set", data=self.stimulus_set)

    @classmethod
    def load(cls, h5_grp):
        """Retrieve relevant datasets from group.

        Args:
            h5_grp: H5 group from which to load data.

        """
        stimulus_set = h5_grp["stimulus_set"][()]
        return cls(stimulus_set)
