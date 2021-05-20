
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

from psiz.trials.experimental.contents.content import Content
from psiz.trials.experimental.unravel_timestep import unravel_timestep


class RateSimilarity(Content):
    """Trial content requiring similarity ratings."""

    def __init__(
            self, stimulus_set):
        """Initialize.

        Arguments:
            stimulus_set:  A np.ndarray of non-negative integers
                indicating specific stimuli. The value "0" can be used
                as a placeholder. Must be rank-2 or rank-3. If rank-2,
                it is assumed that n_timestep=1 and a singleton
                dimension is added.
                shape=
                    (n_sequence, 2)
                    OR
                    (n_sequence, n_timestep, 2)

        Raises:
            ValueError if improper arguments are provided.

        """
        Content.__init__(self)
        stimulus_set = self._check_stimulus_set(stimulus_set)

        # Trim excess placeholder padding off of axis=1.
        is_present = np.not_equal(stimulus_set, self.placeholder)
        # Logical or across last `set` dimension.
        is_present = np.any(is_present, axis=2)
        n_timestep = np.sum(is_present, axis=1, dtype=np.int32)
        self.n_timestep = self._check_n_timestep(n_timestep)
        max_timestep = np.amax(self.n_timestep)
        self.max_timestep = max_timestep
        stimulus_set = stimulus_set[:, 0:max_timestep, :]

        self.stimulus_set = stimulus_set
        self.n_sequence = stimulus_set.shape[0]

    def is_actual(self):
        """Return 2D Boolean array indicating trials with actual content."""
        return np.not_equal(self.stimulus_set[:, :, 0], self.placeholder)

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
        for i_component in component_list:
            if i_component.max_timestep > max_timestep:
                max_timestep = i_component.max_timestep

        # Start by padding first entry in list.
        timestep_pad = max_timestep - component_list[0].max_timestep
        pad_width = ((0, 0), (0, timestep_pad), (0, 0))
        stimulus_set = np.pad(
            component_list[0].stimulus_set,
            pad_width, mode='constant', constant_values=0
        )

        # Loop over remaining list.
        for i_component in component_list[1:]:

            timestep_pad = max_timestep - i_component.max_timestep
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

        Arguments:
            index: The indices corresponding to the subset.

        Returns:
            A new object.

        """
        stimulus_set_sub = self.stimulus_set[idx]
        return RateSimilarity(stimulus_set_sub)

    def _check_stimulus_set(self, stimulus_set):
        """Check validity of `stimulus_set`.

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
        if np.sum(np.less(stimulus_set, self.placeholder)) > 0:
            raise ValueError(
                "The argument `stimulus_set` must contain integers "
                "greater than or equal to 0."
            )

        # Check shape.
        if not ((stimulus_set.ndim == 2) or (stimulus_set.ndim == 3)):
            raise ValueError(
                "The argument `stimulus_set` must be a rank-2 or rank-3 "
                "ndarray with a shape corresponding to (n_sequence, "
                "n_timestep, n_stimuli_per_trial)."
            )

        if stimulus_set.ndim == 2:
            # Assume trials are independent and add singleton dimension for
            # `timestep`.
            stimulus_set = np.expand_dims(stimulus_set, axis=1)

        # Check values are in int32 range.
        ii32 = np.iinfo(np.int32)
        if np.sum(np.greater(stimulus_set, ii32.max)) > 0:
            raise ValueError((
                "The argument `stimulus_set` must only contain integers "
                "in the int32 range."
            ))
        return stimulus_set.astype(np.int32)

    def _check_n_timestep(self, n_timestep):
        """Check validity of `n_timestep`.

        Arguments:
            n_stimstep: A 1D np.ndarray.

        Raises:
            ValueError

        """
        if np.sum(np.equal(n_timestep, 0)) > 0:
            raise ValueError((
                "The argument `stimulus_set` must contain at least one "
                "valid timestep per sequence."))
        return n_timestep

    def _for_dataset(self, format='tf', timestep=True):
        """Prepare trial content data for dataset."""
        if format == 'tf':
            stimulus_set = self.stimulus_set
            if timestep is False:
                stimulus_set = unravel_timestep(stimulus_set)
            x = {
                'stimulus_set': tf.constant(stimulus_set, dtype=tf.int32)
            }
        else:
            raise ValueError(
                "Unrecognized format '{0}'.".format(format)
            )
        return x

    def _save(self, grp):
        """Add relevant data to H5 group."""
        grp.create_dataset("class_name", data="RateSimilarity")
        grp.create_dataset("stimulus_set", data=self.stimulus_set)
        return None

    @classmethod
    def _load(cls, grp):
        """Retrieve relevant datasets from group."""
        stimulus_set = grp["stimulus_set"][()]
        return cls(stimulus_set)
