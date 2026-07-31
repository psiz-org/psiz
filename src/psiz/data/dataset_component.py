# -*- coding: utf-8 -*-
# Copyright 2024 The PsiZ Authors. All Rights Reserved.
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
    DatasetComponent: Abstract class for dataset component.

"""

from abc import ABCMeta, abstractmethod
import warnings

import numpy as np


class DatasetComponent(metaclass=ABCMeta):
    """Abstract class for dataset component."""

    def __init__(self):
        """Initialize."""
        self.n_sample = None
        self.sequence_length = None
        self._timestep_axis = 1
        self._export_with_timestep_axis = None

    @property
    def timestep_axis(self):
        return self._timestep_axis

    def _standardize_shape(self, x):
        """Standardize shape of `x`.

        The attribute `_export_with_timestep_axis` is set based on the
        shape.

        Args:
            x: A numpy.ndarray object with rank-2 or rank-3 shape.

        Returns:
            x: a numpy.ndarray object with rank-3 shape.

        """
        if x.ndim == 2:
            # Add singleton dimension for timestep axis.
            x = np.expand_dims(x, axis=self.timestep_axis)
            self._export_with_timestep_axis = False
        else:
            self._export_with_timestep_axis = True
        return x

    @abstractmethod
    def numpy(self, with_timestep_axis=None):
        """Return appropriately formatted NumPy data.

        Args:
            with_timestep_axis (optional): Boolean indicating if data
                should be returned with a timestep axis. By default,
                data is exported in the same format as it was
                provided at initialization. Callers can override
                default behavior by setting this argument.

        """

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
        if export_format != "tfds":
            raise ValueError(
                "Unrecognized `export_format` '{0}'.".format(export_format)
            )

        warnings.warn(
            "`{0}.export(export_format='tfds')` is deprecated and will be "
            "removed in a future release. Use `{0}.numpy(...)` instead.".format(
                self.__class__.__name__
            ),
            DeprecationWarning,
            stacklevel=2,
        )

        payload = self.numpy(with_timestep_axis=with_timestep_axis)
        return self._numpy_to_tf(payload)

    @staticmethod
    def _numpy_to_tf(payload):
        """Convert nested NumPy payloads to TensorFlow tensors."""
        import tensorflow as tf

        if isinstance(payload, dict):
            return {k: DatasetComponent._numpy_to_tf(v) for k, v in payload.items()}
        if isinstance(payload, tuple):
            return tuple(DatasetComponent._numpy_to_tf(v) for v in payload)
        if isinstance(payload, list):
            return [DatasetComponent._numpy_to_tf(v) for v in payload]
        return tf.constant(payload)
