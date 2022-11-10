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
    Rank: Trial content requiring ranked judgments.

"""

from itertools import permutations

import numpy as np
import tensorflow as tf

from psiz.data.contents.content import Content
from psiz.data.unravel_timestep import unravel_timestep


class Rank(Content):
    """Content for ranked similarity judgments."""

    def __init__(self, stimulus_set, n_select=None):
        """Initialize.

        Args:
            stimulus_set: An np.ndarray of non-negative integers
                indicating specific stimuli. The value "0" can be used
                as a placeholder. Must be rank-2 or rank-3. If rank-2,
                it is assumed that sequence_length=1 and a singleton
                timestep axis is added.
                shape=
                    (samples, max(n_reference) + 1)
                    OR
                    (samples, sequence_length, max(n_reference) + 1)
            n_select (optional): An integer indicating how many
                references are selected on non-placeholder trials.

        Raises:
            ValueError if improper arguments are provided.

        """
        Content.__init__(self)
        self._reference_axis = 2
        stimulus_set = self._rectify_shape(stimulus_set)
        self.n_sequence = stimulus_set.shape[0]
        self.sequence_length = stimulus_set.shape[1]
        stimulus_set = self._validate_stimulus_set(stimulus_set)

        # Validate references and create private array-based attribute and
        # public scalar attribute.
        self.n_reference, self._n_reference = self._validate_n_reference(
            stimulus_set
        )

        # Trim any excess placeholder padding off of reference axis before
        # setting public attribute.
        stimulus_set = stimulus_set[:, :, 0:self.n_reference + 1]
        self.stimulus_set = stimulus_set

        self.n_select, self._n_select = self._validate_n_select(n_select)

    @property
    def max_outcome(self):
        """Getter method for `max_outcome`.

        Returns:
            The maximum number of outcomes for a trial.

        """
        _, df_config = self.unique_configurations

        max_n_outcome = 0
        for _, row in df_config.iterrows():
            outcome_idx = self.possible_outcomes(
                row['_n_reference'], row['_n_select']
            )
            n_outcome = outcome_idx.shape[0]
            if n_outcome > max_n_outcome:
                max_n_outcome = n_outcome
        return max_n_outcome

    @property
    def is_actual(self):
        """Return 2D Boolean array indicating trials with actual content."""
        return np.not_equal(self.stimulus_set[:, :, 0], self.mask_value)

    def _validate_n_reference(self, stimulus_set):
        """Validate implied `n_reference` in `stimulus_set`.

        Raises:
            ValueError

        """
        is_present = np.not_equal(stimulus_set, self.mask_value)
        n_reference_arr = np.sum(
            is_present, axis=self._reference_axis, dtype=np.int32
        ) - 1
        # NOTE: At this point n_reference_arr=-1 for trials that are
        # placeholders (i.e., completely empty trials).

        # Restrict check to non-placeholder trials.
        bidx = np.not_equal(n_reference_arr, -1)
        n_reference_not_empty = n_reference_arr[bidx]

        # First check there are a sufficient number of references for each
        # trial.
        if np.sum(np.less(n_reference_not_empty, 2)) > 0:
            raise ValueError(
                "The argument `stimulus_set` must contain at least three "
                "positive integers per a trial, i.e. one query and at "
                "least two reference stimuli."
            )

        # Check how many unique (non-placeholder) referenes there are.
        unique_n_reference = len(np.unique(n_reference_not_empty))
        if unique_n_reference != 1:
            raise ValueError(
                "When creating a `psiz.data.Rank` object, all non-placeholder "
                "trials must have the same number of references. Detected "
                "{0} different reference counts.".format(
                    unique_n_reference
                )
            )

        # Floor at zero since timestep placeholders yield -1.
        n_reference_arr = np.maximum(n_reference_arr, 0)

        n_reference_sclar = np.amax(n_reference_arr)
        return n_reference_sclar, n_reference_arr

    def _validate_n_select(self, n_select_scalar):
        """Validate `n_select_scalar`.

        Raises:
            ValueError

        """
        if n_select_scalar is None:
            # Assume `n_select_scalar` is 1 for all actual trials.
            n_select_scalar = 1
        else:
            # Cast in order to generate TypeError if non-int was given.
            try:
                n_select_scalar = int(n_select_scalar)
            except TypeError:
                raise ValueError(
                    "The argument `n_select` must be an integer."
                )

        # Check lowerbound support limit.
        if n_select_scalar < 1:
            raise ValueError(
                "The argument `n_select` must be greater than 0."
            )
        # Check upperbound support limit.
        if n_select_scalar > self.n_reference:
            raise ValueError(
                "The argument `n_select` can not be greater than "
                "`n_reference`."
            )

        # Derive array which indicates `n_select` for every trial.
        # NOTE: Array creation uses same integer dtype as `stimulus_set`.
        n_select_arr = (
            self.is_actual.astype(self.stimulus_set.dtype) * n_select_scalar
        )
        return n_select_scalar, n_select_arr

    def _rectify_shape(self, stimulus_set):
        """Rectify shape of `stimulus_set`."""
        if stimulus_set.ndim == 2:
            # Assume trials are independent and add singleton dimension for
            # timestep axis.
            stimulus_set = np.expand_dims(
                stimulus_set, axis=self.timestep_axis
            )
        return stimulus_set

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

        return stimulus_set

    @classmethod
    def _config_attrs(cls):
        """Return attributes that govern trial configurations."""
        return ['_n_reference', '_n_select']

    def _is_select(self, compress=False):
        """Indicate if a stimulus was selected.

        This method has two modes that return 2D arrays of different
        shapes.

        Args:
            compress (optional): A Boolean indicating if the returned
                2D array should be compressed such that the first
                column corresponding to the query is removed, and any
                trailing columns with no selected stimuli are also
                removed. This results in a 2D array with a shape that
                implies the maximum number of selected references.

        Returns:
            is_select: A 3D Boolean array indicating the stimuli that
                were selected. By default, this will be a 3D array that
                has the same shape as `stimulus_set`. See the
                `compress` option for non-default behavior.
                shape=(n_trial, n_max_reference + 1) if compress=False
                shape=(n_trial, n_max_select) if compress=True.

        """
        is_select = np.zeros(self.stimulus_set.shape, dtype=bool)
        max_n_select = np.max(self._n_select)
        for n_select in range(1, max_n_select + 1):
            locs = np.less_equal(n_select, self._n_select)
            is_select[locs, n_select] = True

        if compress:
            is_select = is_select[:, :, 1:max_n_select + 1]

        return is_select

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
        name_prefix = '{0}rank{1}'.format(self.n_reference, self.n_select)

        if export_format == 'tfds':
            stimulus_set = self.stimulus_set
            is_select = self._is_select(compress=False)

            if with_timestep_axis is False:
                stimulus_set = unravel_timestep(stimulus_set)
                is_select = unravel_timestep(is_select)

            x = {
                name_prefix + '/stimulus_set': tf.constant(
                    stimulus_set, name=(name_prefix + '/stimulus_set')
                ),
                name_prefix + '/is_select': tf.constant(
                    is_select,
                    dtype=tf.bool,
                    name=(name_prefix + '/is_select')
                ),
            }
        else:
            raise ValueError(
                "Unrecognized `export_format` '{0}'.".format(export_format)
            )
        return x

    @staticmethod
    def possible_outcomes(n_reference, n_select):
        """Return the possible outcomes of a ranked trial.

        Args:
            n_reference: Integer
            n_select: Integer

        Returns:
            An 2D array indicating all possible outcomes where the
                values indicate indices of the reference stimuli. Each
                row corresponds to one outcome. Note the indices refer
                to references only and does not include an index for
                the query. Also note that the unpermuted index is
                returned first.

        """
        # Cast if necessary.
        n_reference = int(n_reference)
        n_select = int(n_select)

        reference_list = range(n_reference)

        # Get all permutations of length n_select.
        perm = permutations(reference_list, n_select)

        selection = list(perm)
        n_outcome = len(selection)

        outcomes = np.empty((n_outcome, n_reference), dtype=np.int32)
        for i_outcome in range(n_outcome):
            # Fill in selections.
            outcomes[i_outcome, 0:n_select] = selection[i_outcome]
            # Fill in unselected.
            dummy_idx = np.arange(n_reference)
            for i_selected in range(n_select):
                loc = dummy_idx != outcomes[i_outcome, i_selected]
                dummy_idx = dummy_idx[loc]

            outcomes[i_outcome, n_select:] = dummy_idx

        return outcomes
