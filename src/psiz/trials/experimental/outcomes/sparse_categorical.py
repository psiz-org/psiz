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
    SparseCategorical: A categorical outcome.

"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

from psiz.trials.experimental.outcomes.outcome import Outcome
from psiz.trials.experimental.unravel_timestep import unravel_timestep


class SparseCategorical(Outcome):
    """A categorical outcome."""

    def __init__(self, index, depth=None):
        """Initialize.

        Arguments:
            index: A 2D np.ndarray of integers indicating the
                outcomes (as positional indices).
            depth: Integer indicating the maximum number of
                outcomes. This value determines the length of the
                one-hot encoding.

        Raises:
            ValueError if improper arguments are provided.

        """
        Outcome.__init__(self)
        self.index = self._check_index(index)
        self.n_sequence = self.index.shape[0]
        self.max_timestep = self.index.shape[1]
        self.depth = depth

    def stack(self, component_list):
        """Return new object with sequence-stacked data.

        Arguments:
            component_list: A tuple of TrialComponent objects to be
                stacked. All objects must be the same class.

        Returns:
            A new object.

        """
        # Determine maximum number of timesteps.
        max_timestep = 0
        max_depth = 0
        for i_component in component_list:
            if i_component.max_timestep > max_timestep:
                max_timestep = i_component.max_timestep
            if i_component.depth > max_depth:
                max_depth = i_component.depth

        # Start by padding first entry in list.
        timestep_pad = max_timestep - component_list[0].max_timestep
        pad_width = ((0, 0), (0, timestep_pad))
        index = np.pad(
            component_list[0].index,
            pad_width, mode='constant', constant_values=0
        )

        # Loop over remaining list.
        for i_component in component_list[1:]:

            timestep_pad = max_timestep - i_component.max_timestep
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

        Arguments:
            index: The indices corresponding to the subset.

        Returns:
            A new object.

        """
        index_sub = self.index[idx]
        return SparseCategorical(index_sub, depth=self.depth)

    def _check_index(self, index):
        """Check validity of `index`."""
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
                "shape corresponding to (n_sequence, n_timestep)."
            )

        return index

    def _for_dataset(self, format='tf', timestep=True):
        """Return appropriately formatted data."""
        if format == 'tf':
            # Convert from sparse to one-hot-encoding (along new trailing
            # axis).
            index = self.index
            if timestep is False:
                index = unravel_timestep(index)
            # pylint: disable=unexpected-keyword-arg
            y = tf.one_hot(
                index, self.depth, on_value=1.0, off_value=0.0,
                dtype=K.floatx()
            )
        else:
            raise ValueError(
                "Unrecognized format '{0}'.".format(format)
            )
        return y

    def _save(self, grp):
        """Add relevant datasets to H5 group.

        Example:
        grp.create_dataset("my_data_name", data=my_data)

        """
        grp.create_dataset("class_name", data="SparseCategorical")
        grp.create_dataset("index", data=self.index)
        grp.create_dataset("depth", data=self.depth)
        return None

    @classmethod
    def _load(cls, grp):
        """Retrieve relevant datasets from group."""
        index = grp["index"][()]
        depth = grp["depth"][()]
        return cls(index, depth=depth)
