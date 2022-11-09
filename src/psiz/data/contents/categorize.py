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
    Categorize: Trial content requiring categorization judgments.

"""

import numpy as np
import tensorflow as tf

from psiz.data.contents.content import Content
from psiz.data.unravel_timestep import unravel_timestep


class Categorize(Content):
    """Content for categorization judgments."""

    def __init__(self, stimulus_set=None, correct_label=None):
        """Initialize.

        Args:
            stimulus_set: An np.ndarray of non-negative integers
                indicating specific stimuli. The value "0" can be used
                as a placeholder. Must be rank-2 or rank-3. If rank-2,
                it is assumed that sequence_length=1 and a singleton
                timestep axis is added.
                shape=
                    (samples, [sequence_length,] 1)
            correct_label: An np.ndarray of non-negative integers
                indicating the correct label. Must be rank-2 or rank-3.
                If rank-2, it is assumed that sequence_length=1 and a
                singleton timestep axis is added.Can use sparse or
                one-hot encoding.
                shape=
                    (samples, [sequence_length,] 1)
                    OR
                    (samples, [sequence_length,] n_class)

        Raises:
            ValueError if improper arguments are provided.

        """
        Content.__init__(self)
        stimulus_set = self._rectify_shape(stimulus_set)
        correct_label = self._rectify_shape(correct_label)
        self.n_sequence = stimulus_set.shape[0]
        self.sequence_length = stimulus_set.shape[1]
        self.stimulus_set = self._validate_stimulus_set(stimulus_set)
        self.correct_label = self._validate_correct_label(correct_label)

    @property
    def is_actual(self):
        """Return 2D Boolean array indicating trials with actual content."""
        return np.not_equal(self.stimulus_set[:, :, 0], self.mask_value)

    def _rectify_shape(self, x):
        """Rectify shape of `stimulus_set`."""
        if x.ndim == 2:
            # Add singleton dimension for timestep axis.
            x = np.expand_dims(x, axis=self.timestep_axis)
        return x

    def _validate_stimulus_set(self, stimulus_set):
        """Validate `stimulus_set`.

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

        # TODO enforce or delete
        # Check values are in int32 range.
        # ii32 = np.iinfo(np.int32)
        # if np.sum(np.greater(stimulus_set, ii32.max)) > 0:
        #     raise ValueError((
        #         "The argument `stimulus_set` must only contain integers "
        #         "in the int32 range."
        #     ))
        # stimulus_set = stimulus_set.astype(np.int32)
        return stimulus_set

    def _validate_correct_label(self, correct_label):
        """Validate `correct_label`.

        Raises:
            ValueError

        """
        # Check that provided values are integers.
        if not issubclass(correct_label.dtype.type, np.integer):
            raise ValueError(
                "The argument `correct_label` must be an np.ndarray of "
                "integers."
            )

        # Check that all values are greater than or equal to placeholder.
        if np.sum(np.less(correct_label, self.mask_value)) > 0:
            raise ValueError(
                "The argument `correct_label` must contain integers "
                "greater than or equal to 0."
            )

        # Check shape.
        if not correct_label.ndim == 3:
            raise ValueError(
                "The argument `correct_label` must be a rank-2 or rank-3 "
                "ndarray with a shape corresponding to (samples, "
                "sequence_length, n_stimuli_per_trial)."
            )
        if correct_label.shape[0] != self.n_sequence:
            raise ValueError(
                "The argument `correct_label` must be a rank-2 or rank-3 "
                "ndarray with a shape corresponding to (samples, "
                "sequence_length, n_stimuli_per_trial). Provided value has "
                "incorrect `samples`."
            )
        if correct_label.shape[1] != self.sequence_length:
            raise ValueError(
                "The argument `correct_label` must be a rank-2 or rank-3 "
                "ndarray with a shape corresponding to (samples, "
                "sequence_length, n_stimuli_per_trial). Provided value has "
                "incorrect `sequence_length`."
            )
        return correct_label

    def export(self, export_format='tfds', with_timestep_axis=True):
        """Prepare trial content data for dataset.

        Args:
            export_format (optional): The output format of the dataset.
                By default the dataset is formatted as a
                    tf.data.Dataset object.
            with_timestep_axis (optional): Boolean indicating if data should be
                returned with a timestep axis. If `False`, data is
                reshaped.

        """
        name_prefix = 'categorize'

        if export_format == 'tfds':
            stimulus_set = self.stimulus_set
            correct_label = self.correct_label

            if with_timestep_axis is False:
                stimulus_set = unravel_timestep(stimulus_set)
                correct_label = unravel_timestep(correct_label)

            x = {
                name_prefix + '/stimulus_set': tf.constant(
                    stimulus_set, name=(name_prefix + '/stimulus_set')
                ),
                name_prefix + '/correct_label': tf.constant(
                    correct_label, name=(name_prefix + '/correct_label')
                ),
            }
        else:
            raise ValueError(
                "Unrecognized `export_format` '{0}'.".format(export_format)
            )
        return x
