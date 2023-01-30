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

# import warnings

# import numpy as np
import tensorflow as tf

from psiz.data.dataset_component import DatasetComponent
from psiz.data.unravel_timestep import unravel_timestep


class Group(DatasetComponent):
    """Base class for group membership data."""

    def __init__(self, value, name=None):
        """Initialize.

        Args:
            value: An np.ndarray that must be rank-2 or rank-3.
                shape=(samples, [sequence_length], n_col)
            name: A string indicating the variable name of the group.

        """
        DatasetComponent.__init__(self)
        value = self._standardize_shape(value)
        value = self._validate_value(name, value)
        self.n_sample = value.shape[0]
        self.sequence_length = value.shape[1]
        if name is None:
            raise ValueError("Argument `name` must be provided.")
        self.name = name
        self.value = value

    def _validate_value(self, group_key, value):
        """Validate group weights."""
        # Check rank of `value`.
        if not (value.ndim == 3):
            raise ValueError(
                "The values for '{0}' must be a rank-2 or rank-3 ND "
                "array. If using a sparse coding format, make sure you have "
                "a trailing singleton dimension to meet this "
                "requirement.".format(group_key)
            )

        # If `value` looks like sparse coding format, check data type. TODO
        # if value.shape[-1] == 1:
        #     if not isinstance(
        #         value[0, 0, 0], (float, np.float32, np.float64, np.float128)
        #     ):
        #         # NOTE: We check if float, because integer or string is ok.
        #         warnings.warn(
        #             "The values for '{0}' appear to use a sparse "
        #             "coding. To improve efficiency, these weights should "
        #             "have an integer dtype, not a float dtype.".format(
        #                 group_key
        #             )
        #         )

        return value

    def export(self, export_format="tfds", with_timestep_axis=None):
        """Export.

        Args:
            export_format (optional): The output format of the dataset.
                By default the dataset is formatted as a
                `tf.data.Dataset` object.
            with_timestep_axis (optional): Boolean indicating if data
                should be returned with a timestep axis. By default,
                data is exported in the same format as it was
                provided at initialization. Callers can override
                default behavior by setting this argument.

        """
        if with_timestep_axis is None:
            with_timestep_axis = self._export_with_timestep_axis

        value = self.value
        if with_timestep_axis is False:
            value = unravel_timestep(value)

        if export_format == "tfds":
            value = tf.constant(value, name=("group_" + self.name))
        else:
            raise ValueError(
                "Unrecognized `export_format` '{0}'.".format(export_format)
            )
        return {self.name: value}
