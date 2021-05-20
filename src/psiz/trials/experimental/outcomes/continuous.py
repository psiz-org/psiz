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
    Continuous: A continuous outcome.

"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

from psiz.trials.experimental.outcomes.outcome import Outcome
from psiz.trials.experimental.unravel_timestep import unravel_timestep


class Continuous(Outcome):
    """A continuous outcome."""

    def __init__(self, value):
        """Initialize.

        Arguments:
            value: Ab np.ndarray of floats indicating the outcome
                values. Must be rank-2 or rank-3. If rank-2,
                it is assumed that n_timestep=1 and a singleton
                dimension is added. If rank-3, the first two axis are
                interpretted as `n_sequence` and `n_timestep`.

        Raises:
            ValueError if improper arguments are provided.

        """
        Outcome.__init__(self)
        self.value = self._check_value(value)
        self.n_sequence = self.value.shape[0]
        self.max_timestep = self.value.shape[1]
        self.n_unit = self.value.shape[2]

    def stack(self, component_list):
        """Return new object with sequence-stacked data.

        Arguments:
            component_list: A tuple of TrialComponent objects to be
                stacked. All objects must be the same class.

        Returns:
            A new object.

        Raises:
            ValueError if `n_unit` does not agree.

        """
        # Before doing anything, check that `n_unit` agree.
        n_unit = component_list[0].n_unit
        for i_component in component_list[1:]:
            if component_list[0].n_unit != n_unit:
                raise ValueError(
                    'The `n_unit` for the different components must be'
                    'identical to stack `Continous` output.'
                )

        # Determine maximum number of timesteps.
        max_timestep = 0
        for i_component in component_list:
            if i_component.max_timestep > max_timestep:
                max_timestep = i_component.max_timestep

        # Start by padding first entry in list.
        timestep_pad = max_timestep - component_list[0].max_timestep
        pad_width = ((0, 0), (0, timestep_pad), (0, 0))
        value = np.pad(
            component_list[0].value,
            pad_width, mode='constant', constant_values=0
        )

        # Loop over remaining list.
        for i_component in component_list[1:]:
            timestep_pad = max_timestep - i_component.max_timestep
            pad_width = ((0, 0), (0, timestep_pad), (0, 0))
            curr_value = np.pad(
                i_component.value,
                pad_width, mode='constant', constant_values=0
            )

            value = np.concatenate(
                (value, curr_value), axis=0
            )

        return Continuous(value)

    def subset(self, idx):
        """Return subset of data as a new object.

        Arguments:
            index: The indices corresponding to the subset.

        Returns:
            A new object.

        """
        value_sub = self.value[idx]
        return Continuous(value_sub)

    def _check_value(self, value):
        """Check validity of `value`."""
        if value.ndim == 2:
            # Assume trials are independent and add singleton dimension for
            # `timestep`.
            value = np.expand_dims(value, axis=1)

        # Check shape.
        if not value.ndim == 3:
            raise ValueError(
                "The argument `value` must be a rank-2 or rank-3 "
                "ndarray with a shape corresponding to (n_sequence, "
                "[n_timestep,] n_unit)."
            )

        return value

    def _for_dataset(self, format='tf', timestep=True):
        """Return appropriately formatted data."""
        if format == 'tf':
            value = self.value
            if timestep is False:
                value = unravel_timestep(value)
            y = tf.constant(value, dtype=K.floatx())
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
        grp.create_dataset("class_name", data="Continuous")
        grp.create_dataset("value", data=self.value)
        return None

    @classmethod
    def _load(cls, grp):
        """Retrieve relevant datasets from group."""
        value = grp["value"][()]
        return cls(value)
