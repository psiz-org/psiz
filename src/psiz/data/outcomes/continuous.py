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
