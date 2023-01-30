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
from tensorflow.keras import backend as K

from psiz.data.outcomes.outcome import Outcome
from psiz.data.unravel_timestep import unravel_timestep


class SparseCategorical(Outcome):
    """A categorical outcome."""

    def __init__(self, index, depth=None, **kwargs):
        """Initialize.

        Args:
            index: A 2D np.ndarray of integers indicating the
                outcomes (as positional indices).
            depth: Integer indicating the maximum number of
                outcomes. This value determines the length of the
                one-hot encoding.
            kwargs: Additional key-word arguments.

        Raises:
            ValueError if improper arguments are provided.

        """
        Outcome.__init__(self, **kwargs)
        index = self._standardize_shape(index)
        self.n_sample = index.shape[0]
        self.sequence_length = index.shape[1]
        index = self._validate_index(index)
        self.index = index
        self.depth = depth
        self.process_sample_weight()

    def _standardize_shape(self, x):
        """Standardize shape of `x`.

        The attribute `_export_with_timestep_axis` is set based on the
        shape.

        Args:
            x: A numpy.ndarray object with rank-2 or rank-3 shape.

        Returns:
            x: a numpy.ndarray object with rank-3 shape.

        """
        if x.ndim == 1:
            # Assume trials are independent and add singleton dimension for
            # timestep axis.
            x = np.expand_dims(x, axis=self.timestep_axis)
            self._export_with_timestep_axis = False
        else:
            self._export_with_timestep_axis = True
        return x

    def _validate_index(self, index):
        """Validate `index`."""
        # Check that provided values are integers.
        if not issubclass(index.dtype.type, np.integer):
            raise ValueError(
                "The argument `index` must be an np.ndarray of " "integers."
            )

        # Check that all values are greater than or equal to placeholder.
        if np.sum(np.less(index, 0)) > 0:
            raise ValueError(
                "The argument `index` must contain non-negative " "integers."
            )

        # Check shape.
        if not (index.ndim == 2):
            raise ValueError(
                "The argument `index` must be a rank-2 ndarray with a "
                "shape corresponding to (samples, sequence_length)."
            )

        return index

    def export(self, export_format="tfds", with_timestep_axis=None):
        """Return appropriately formatted data.

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

        w_dict = super(SparseCategorical, self).export(
            export_format=export_format, with_timestep_axis=with_timestep_axis
        )

        index = self.index
        if with_timestep_axis is False:
            index = unravel_timestep(index)

        if export_format == "tfds":
            # Convert from sparse to one-hot-encoding, creating new trailing
            # axis.
            # pylint: disable=unexpected-keyword-arg
            # NOTE: A float for loss computation.
            y = tf.one_hot(
                index, self.depth, on_value=1.0, off_value=0.0, dtype=K.floatx()
            )
        else:
            raise ValueError(
                "Unrecognized `export_format` '{0}'.".format(export_format)
            )
        return {self.name: y}, w_dict
