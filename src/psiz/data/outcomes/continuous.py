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
    Continuous: A continuous outcome.

"""

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K

from psiz.data.outcomes.outcome import Outcome
from psiz.data.unravel_timestep import unravel_timestep


class Continuous(Outcome):
    """A continuous outcome."""

    def __init__(self, value, **kwargs):
        """Initialize.

        Args:
            value: Ab np.ndarray of floats indicating the outcome
                values. Must be rank-2 or rank-3. If rank-2,
                it is assumed that sequence_length=1 and a singleton
                dimension is added. If rank-3, the first two axis are
                interpretted as `samples` and `sequence_length`.
            kwargs: Additional key-word arguments.

        Raises:
            ValueError if improper arguments are provided.

        """
        Outcome.__init__(self, **kwargs)
        value = self._rectify_shape(value)
        self.n_sequence = value.shape[0]
        self.sequence_length = value.shape[1]
        value = self._validate_value(value)
        self.value = value
        self.n_unit = self.value.shape[2]
        self.process_sample_weight()

    def _rectify_shape(self, value):
        """Rectify shape of value."""
        if value.ndim == 2:
            # Assume trials are independent and add singleton dimension for
            # timestep axis.
            value = np.expand_dims(value, axis=self.timestep_axis)
        return value

    def _validate_value(self, value):
        """Validate `value`."""
        # Check shape.
        if not value.ndim == 3:
            raise ValueError(
                "The argument `value` must be a rank-2 or rank-3 "
                "ndarray with a shape corresponding to (samples, "
                "[sequence_length,] n_unit)."
            )

        return value

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
        w = super(Continuous, self).export(
            export_format=export_format, with_timestep_axis=with_timestep_axis
        )

        value = self.value
        if with_timestep_axis is False:
            value = unravel_timestep(value)

        if export_format == 'tf':
            y = tf.constant(value, dtype=K.floatx())
        else:
            raise ValueError(
                "Unrecognized `export_format` '{0}'.".format(export_format)
            )
        return {self.name: y}, w

    def save(self, h5_grp):
        """Add relevant data to H5 group.

        Args:
            h5_grp: H5 group for saving data.

        """
        h5_grp.create_dataset("class_name", data="psiz.data.Continuous")
        h5_grp.create_dataset("value", data=self.value)
        h5_grp.create_dataset("name", data=self.name)
        h5_grp.create_dataset("sample_weight", data=self.sample_weight)

    @classmethod
    def load(cls, h5_grp):
        """Retrieve relevant datasets from group.

        Args:
            h5_grp: H5 group from which to load data.

        """
        value = h5_grp["value"][()]
        name = h5_grp["name"].asstr()[()]
        sample_weight = h5_grp["sample_weight"][()]
        return cls(value, name=name, sample_weight=sample_weight)

    def subset(self, idx):
        """Return subset of data as a new object.

        Args:
            index: The indices corresponding to the subset.

        Returns:
            A new object.

        """
        value_sub = self.value[idx]
        sample_weight_sub = self._subset_sample_weight(idx)
        return Continuous(
            value_sub, sample_weight=sample_weight_sub, name=self.name
        )

    # TODO delete stack
    # def stack(self, component_list):
    #     """Return new object with sequence-stacked data.

    #     Args:
    #         component_list: A tuple of TrialComponent objects to be
    #             stacked. All objects must be the same class.

    #     Returns:
    #         A new object.

    #     Raises:
    #         ValueError if `n_unit` does not agree.

    #     """
    #     # Before doing anything, check that `n_unit` agree.
    #     n_unit = component_list[0].n_unit
    #     for i_component in component_list[1:]:
    #         if i_component.n_unit != n_unit:
    #             raise ValueError(
    #                 'The `n_unit` for the different components must be'
    #                 'identical to stack `Continous` outcomes.'
    #             )

    #     # Determine maximum number of timesteps.
    #     sequence_length = 0
    #     for i_component in component_list:
    #         if i_component.sequence_length > sequence_length:
    #             sequence_length = i_component.sequence_length

    #     # Start by padding first entry in list.
    #     timestep_pad = sequence_length - component_list[0].sequence_length
    #     pad_width = ((0, 0), (0, timestep_pad), (0, 0))
    #     value = np.pad(
    #         component_list[0].value,
    #         pad_width, mode='constant', constant_values=0
    #     )

    #     # Loop over remaining list.
    #     for i_component in component_list[1:]:
    #         timestep_pad = sequence_length - i_component.sequence_length
    #         pad_width = ((0, 0), (0, timestep_pad), (0, 0))
    #         curr_value = np.pad(
    #             i_component.value,
    #             pad_width, mode='constant', constant_values=0
    #         )

    #         value = np.concatenate(
    #             (value, curr_value), axis=0
    #         )

    #     return Continuous(value)
