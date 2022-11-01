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
    SparseCategorical: A categorical outcome.

"""

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K

from psiz.data.outcomes.outcome import Outcome
from psiz.data.unravel_timestep import unravel_timestep


class SparseCategorical(Outcome):
    """A categorical outcome."""

    def __init__(self, index, depth=None):
        """Initialize.

        Args:
            index: A 2D np.ndarray of integers indicating the
                outcomes (as positional indices).
            depth: Integer indicating the maximum number of
                outcomes. This value determines the length of the
                one-hot encoding.

        Raises:
            ValueError if improper arguments are provided.

        """
        Outcome.__init__(self)
        self.index = self._validate_index(index)
        self.n_sequence = self.index.shape[0]
        self.sequence_length = self.index.shape[1]
        self.depth = depth

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
        max_depth = 0
        for i_component in component_list:
            if i_component.sequence_length > sequence_length:
                sequence_length = i_component.sequence_length
            if i_component.depth > max_depth:
                max_depth = i_component.depth

        # Start by padding first entry in list.
        timestep_pad = sequence_length - component_list[0].sequence_length
        pad_width = ((0, 0), (0, timestep_pad))
        index = np.pad(
            component_list[0].index,
            pad_width, mode='constant', constant_values=0
        )

        # Loop over remaining list.
        for i_component in component_list[1:]:

            timestep_pad = sequence_length - i_component.sequence_length
            pad_width = ((0, 0), (0, timestep_pad))
            curr_index = np.pad(
                i_component.index,
                pad_width, mode='constant', constant_values=0
            )

            index = np.concatenate(
                (index, curr_index), axis=0
            )

        return SparseCategorical(index, depth=max_depth)

    def subset(self, idx):
        """Return subset of data as a new object.

        Args:
            index: The indices corresponding to the subset.

        Returns:
            A new object.

        """
        index_sub = self.index[idx]
        return SparseCategorical(index_sub, depth=self.depth)

    def _validate_index(self, index):
        """Validate `index`."""
        # Check that provided values are integers.
        if not issubclass(index.dtype.type, np.integer):
            raise ValueError(
                "The argument `index` must be an np.ndarray of "
                "integers."
            )

        # Check that all values are greater than or equal to placeholder.
        if np.sum(np.less(index, 0)) > 0:
            raise ValueError(
                "The argument `index` must contain non-negative "
                "integers."
            )

        if index.ndim == 1:
            # Assume trials are independent and add singleton dimension for
            # `timestep`.
            index = np.expand_dims(index, axis=1)

        # Check shape.
        if not (index.ndim == 2):
            raise ValueError(
                "The argument `index` must be a rank-2 ndarray with a "
                "shape corresponding to (samples, sequence_length)."
            )

        return index

    def export(self, export_format='tf', with_timestep_axis=True):
        """Return appropriately formatted data.

        Args:
            export_format (optional): The output format of the dataset.
                By default the dataset is formatted as a
                    tf.data.Dataset object.
            with_timestep_axis (optional): Boolean indicating if data
                should be returned with a timestep axis. If `False`,
                data is reshaped.

        """
        if export_format == 'tf':
            # Convert from sparse to one-hot-encoding (along new trailing
            # axis).
            index = self.index
            if with_timestep_axis is False:
                index = unravel_timestep(index)
            # pylint: disable=unexpected-keyword-arg
            y = tf.one_hot(
                index, self.depth, on_value=1.0, off_value=0.0,
                dtype=K.floatx()
            )
        else:
            raise ValueError(
                "Unrecognized `export_format` '{0}'.".format(export_format)
            )
        return y

    def save(self, h5_grp):
        """Add relevant data to H5 group.

        Args:
            h5_grp: H5 group for saving data.

        """
        h5_grp.create_dataset("class_name", data="SparseCategorical")
        h5_grp.create_dataset("index", data=self.index)
        h5_grp.create_dataset("depth", data=self.depth)

    @classmethod
    def load(cls, h5_grp):
        """Retrieve relevant datasets from group.

        Args:
            h5_grp: H5 group from which to load data.

        """
        index = h5_grp["index"][()]
        depth = h5_grp["depth"][()]
        return cls(index, depth=depth)
