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

import numpy as np
import tensorflow as tf

from psiz.data.contents.content import Content
from psiz.data.unravel_timestep import unravel_timestep
from psiz.utils.m_prefer_n import m_prefer_n


class Rank(Content):
    """Content for ranked similarity judgments."""

    def __init__(self, stimulus_set, n_select=None, name=None):
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
                Defaults to `1`.
            name (optional): A string indicating a name that can be
                used to identify the content when exported.

        Raises:
            ValueError if improper arguments are provided.

        """
        Content.__init__(self)
        self._reference_axis = 2
        stimulus_set = self._standardize_shape(stimulus_set)
        self.n_sample = stimulus_set.shape[0]
        self.sequence_length = stimulus_set.shape[1]
        stimulus_set = self._validate_stimulus_set(stimulus_set)

        # Validate references and create private array-based attribute and
        # public scalar attribute.
        self.n_reference, self._n_reference = self._validate_n_reference(stimulus_set)

        # Trim any excess placeholder padding off of reference axis before
        # setting public attribute.
        stimulus_set = stimulus_set[:, :, 0 : self.n_reference + 1]
        self.stimulus_set = stimulus_set

        self.n_select, self._n_select_arr = self._validate_n_select(n_select)

        if name is None:
            name = "given{0}rank{1}".format(self.n_reference, self.n_select)
        self.name = name

    @property
    def n_outcome(self):
        """Getter method for `n_outcome`.

        Returns:
            The number of outcomes for a trial.

        """
        outcomes = self.possible_outcomes(self.n_reference, self.n_select)
        return outcomes.shape[0]

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
        n_reference_arr = (
            np.sum(is_present, axis=self._reference_axis, dtype=np.int32) - 1
        )
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
                "{0} different reference counts.".format(unique_n_reference)
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
                raise ValueError("The argument `n_select` must be an integer.")

        # Check lowerbound support limit.
        if n_select_scalar < 1:
            raise ValueError("The argument `n_select` must be greater than 0.")
        # Check upperbound support limit.
        if n_select_scalar > self.n_reference:
            raise ValueError(
                "The argument `n_select` can not be greater than " "`n_reference`."
            )

        # Derive array which indicates `n_select` for every trial.
        # NOTE: Array creation uses same integer dtype as `stimulus_set`.
        n_select_arr = self.is_actual.astype(self.stimulus_set.dtype) * n_select_scalar
        return n_select_scalar, n_select_arr

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

    @classmethod
    def _config_attrs(cls):
        """Return attributes that govern trial configurations."""
        return ["_n_reference", "_n_select_arr"]

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
        max_n_select = np.max(self._n_select_arr)
        for n_select in range(1, max_n_select + 1):
            locs = np.less_equal(n_select, self._n_select_arr)
            is_select[locs, n_select] = True

        if compress:
            is_select = is_select[:, :, 1 : max_n_select + 1]

        return is_select

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
            n_reference: Integer indicating number of references
            n_select: Integer indicating number of ranked selections.

        Returns:
            An 2D array indicating all possible outcomes where the
            values indicate indices of the reference stimuli. Each row
            corresponds to one outcome. Note the indices refer to
            references only and does not include an index for the
            query. Also note that the unpermuted index is returned
            first.

        """
        return m_prefer_n(n_reference, n_select)

    @staticmethod
    def as_sparse_outcome(n_reference, selection_indices):
        """Convert from selection indices to an outcome index.

        Args:
            n_reference: Integer indicating the number of references.
            selection_indices: Array-like of integers indicating the
                stimulus indices that were selected. The order of
                indices is assumed to correspond to the order that the
                selections were made.

        Returns:
            An integer representing a sparse encoding of the outcome.

        """
        # Cast as array.
        selection_indices = np.array(selection_indices)
        # Add axis for broadcasting.
        selection_indices = np.expand_dims(selection_indices, 0)

        n_select = selection_indices.shape[1]
        outcomes = m_prefer_n(n_reference, n_select)
        n_outcome = outcomes.shape[0]
        dmy_idx = np.arange(n_outcome)
        # Drop non-selected references.
        outcomes = outcomes[:, 0:n_select]
        # Find outcome idx by finding a match.
        is_equal = np.equal(outcomes, selection_indices)
        is_equal = np.product(is_equal, 1).astype(bool)
        # There should only be one True location.
        outcome_idx = dmy_idx[is_equal][0]
        return outcome_idx
