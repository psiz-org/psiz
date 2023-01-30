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
"""Data module.

Classes:
    Rate: Trial content requiring ratings.

"""

import numpy as np
import tensorflow as tf

from psiz.data.contents.content import Content
from psiz.data.unravel_timestep import unravel_timestep


class Rate(Content):
    """Trial content requiring similarity ratings."""

    def __init__(self, stimulus_set, name=None):
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
            name (optional): A string indicating a name that can be
                used to identify the content when exported.

        Raises:
            ValueError if improper arguments are provided.

        """
        Content.__init__(self)
        stimulus_set = self._standardize_shape(stimulus_set)
        self.n_sample = stimulus_set.shape[0]
        self.sequence_length = stimulus_set.shape[1]
        stimulus_set = self._validate_stimulus_set(stimulus_set)
        self.stimulus_set = stimulus_set

        if name is None:
            name = "rate2"
        self.name = name

    @property
    def is_actual(self):
        """Return 2D Boolean array indicating trials with actual content."""
        return np.not_equal(self.stimulus_set[:, :, 0], self.mask_value)

    def _validate_stimulus_set(self, stimulus_set):
        """Validate `stimulus_set`.

        Raises:
            ValueError

        """
        # Check that provided values are integers.
        if not issubclass(stimulus_set.dtype.type, np.integer):
            raise ValueError(
                "The argument `stimulus_set` must be an np.ndarray of " "integers."
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

        return stimulus_set

    def export(self, export_format="tfds", with_timestep_axis=None):
        """Prepare trial content data for dataset.

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

        if export_format == "tfds":
            stimulus_set = self.stimulus_set
            if with_timestep_axis is False:
                stimulus_set = unravel_timestep(stimulus_set)

            x = {
                self.name
                + "_stimulus_set": tf.constant(
                    stimulus_set, name=(self.name + "_stimulus_set")
                )
            }
        else:
            raise ValueError(
                "Unrecognized `export_format` '{0}'.".format(export_format)
            )
        return x
