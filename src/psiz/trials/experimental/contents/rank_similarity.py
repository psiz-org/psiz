
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
"""Trials module.


Classes:
    RankSimilarity: Trial content requiring ranked similarity judgments.

"""

from itertools import permutations

import numpy as np
import tensorflow as tf

from psiz.trials.experimental.contents.content import Content
from psiz.trials.experimental.unravel_timestep import unravel_timestep


class RankSimilarity(Content):
    """Trial content requiring ranked similarity judgments."""

    def __init__(self, stimulus_set, n_select=None):
        """Initialize.

        Arguments:
            stimulus_set: An np.ndarray of non-negative integers
                indicating specific stimuli. The value "0" can be used
                as a placeholder. Must be rank-2 or rank-3. If rank-2,
                it is assumed that n_timestep=1 and a singleton
                dimension is added.
                shape=
                    (n_sequence, max(n_reference) + 1)
                    OR
                    (n_sequence, n_timestep, max(n_reference) + 1)
            n_select (optional): A np.ndarray of non-negative integers
                indicating how many references are selected on a given
                trial. Must be rank-1 or rank-2. If rank-1, it is
                assumed that n_timestep=1 and a singleton dimension is
                added.
                shape=
                    (n_sequence,)
                    OR
                    (n_sequence, n_timestep)

        Raises:
            ValueError if improper arguments are provided.

        """
        Content.__init__(self)
        stimulus_set = self._check_stimulus_set(stimulus_set)

        # Trim excess placeholder padding off of timestep (axis=1).
        is_present = np.not_equal(stimulus_set, self.placeholder)
        # Logical or across last `set` dimension.
        is_present = np.any(is_present, axis=2)
        n_timestep = np.sum(is_present, axis=1, dtype=np.int32)
        self.n_timestep = self._check_n_timestep(n_timestep)
        max_timestep = np.amax(self.n_timestep)
        self.max_timestep = max_timestep
        stimulus_set = stimulus_set[:, 0:max_timestep, :]

        # Trim any excess placeholder padding off of n_reference (axis=2).
        is_present = np.not_equal(stimulus_set, self.placeholder)
        n_reference = np.sum(is_present, axis=2, dtype=np.int32) - 1
        self.n_reference = self._check_n_reference(n_reference)
        self.max_n_reference = np.amax(self.n_reference)
        stimulus_set = stimulus_set[:, :, 0:self.max_n_reference + 1]

        self.stimulus_set = stimulus_set
        self.n_sequence = stimulus_set.shape[0]

        if n_select is None:
            n_select = np.ones(
                (self.n_sequence, self.max_timestep), dtype=np.int32
            )
        else:
            n_select = self._check_n_select(n_select)
        self.n_select = n_select

    @property
    def max_outcome(self):
        """Getter method for `max_outcome`.

        Returns:
            The max number of outcomes for any trial in the dataset.

        """
        _, df_config = self.unique_configurations()

        max_n_outcome = 0
        for _, row in df_config.iterrows():
            outcome_idx = self._possible_outcomes(
                row['n_reference'], row['n_select']
            )
            n_outcome = outcome_idx.shape[0]
            if n_outcome > max_n_outcome:
                max_n_outcome = n_outcome
        return max_n_outcome

    def is_actual(self):
        """Return 2D Boolean array indicating trials with actual content."""
        return np.not_equal(self.stimulus_set[:, :, 0], self.placeholder)

    def stack(self, component_list):
        """Return new object with sequence-stacked data.

        Arguments:
            component_list: A tuple of TrialComponent objects to be
                stacked. All objects must be the same class.

        Returns:
            A new object.

        """
        # Determine maximum number of references and maximum number of
        # timesteps.
        max_n_reference = 0
        max_timestep = 0
        for i_component in component_list:
            if i_component.max_n_reference > max_n_reference:
                max_n_reference = i_component.max_n_reference
            if i_component.max_timestep > max_timestep:
                max_timestep = i_component.max_timestep

        # Start by padding first entry in list.
        timestep_pad = max_timestep - component_list[0].max_timestep
        n_pad = max_n_reference - component_list[0].max_n_reference
        pad_width = ((0, 0), (0, timestep_pad), (0, n_pad))
        stimulus_set = np.pad(
            component_list[0].stimulus_set,
            pad_width, mode='constant', constant_values=0
        )
        pad_width = ((0, 0), (0, timestep_pad))
        n_select = np.pad(
            component_list[0].n_select,
            pad_width, mode='constant', constant_values=0
        )

        # Loop over remaining list.
        for i_component in component_list[1:]:

            timestep_pad = max_timestep - i_component.max_timestep
            n_pad = max_n_reference - i_component.max_n_reference
            pad_width = ((0, 0), (0, timestep_pad), (0, n_pad))
            curr_stimulus_set = np.pad(
                i_component.stimulus_set,
                pad_width, mode='constant', constant_values=0
            )

            pad_width = ((0, 0), (0, timestep_pad))
            curr_n_select = np.pad(
                i_component.n_select,
                pad_width, mode='constant', constant_values=0
            )

            stimulus_set = np.concatenate(
                (stimulus_set, curr_stimulus_set), axis=0
            )
            n_select = np.concatenate((n_select, curr_n_select), axis=0)

        return RankSimilarity(stimulus_set, n_select=n_select)

    def subset(self, idx):
        """Return subset of data as a new object.

        Arguments:
            index: The indices corresponding to the subset.

        Returns:
            A new object.

        """
        stimulus_set_sub = self.stimulus_set[idx]
        n_select_sub = self.n_select[idx]
        return RankSimilarity(stimulus_set_sub, n_select=n_select_sub)

    def _stimulus_set_with_outcomes(self):
        """Inflate `stimulus_set` for all possible outcomes."""
        config_idx, df_config = self.unique_configurations()

        # Precompute possible outcomes for each content configuration.
        outcome_idx_hash = {}
        max_n_outcome = 0
        for index, row in df_config.iterrows():
            outcome_idx = self._possible_outcomes(
                row['n_reference'], row['n_select']
            )
            outcome_idx_hash[index] = outcome_idx
            n_outcome = outcome_idx.shape[0]
            if n_outcome > max_n_outcome:
                max_n_outcome = n_outcome

        # Combine `n_sequence` and `max_timestep` axis for `stimulus_set` and
        # corresponding `config_idx` to enable reuse of existing code.
        n_trial = self.n_sequence * self.max_timestep
        stimulus_set_flat = np.reshape(
            self.stimulus_set, [n_trial, self.max_n_reference + 1]
        )
        config_idx = np.reshape(config_idx, [n_trial])

        # Pre-allocate `stimulus_set` that has additional axis for outcomes.
        stimulus_set_flat_expand = np.full(
            [n_trial, self.max_n_reference + 1, max_n_outcome],
            self.placeholder, dtype=np.int32
        )

        for index, row in df_config.iterrows():
            # Identify relevant trials.
            trial_locs = config_idx == index
            n_trial_config = np.sum(trial_locs)

            outcome_idx = outcome_idx_hash[index]
            n_outcome = outcome_idx.shape[0]

            if n_outcome != 1:
                # NOTE: if n_outcome == 1, this is a placeholder trial, which
                # we would fill with placeholder values. Since the pre-
                # allocation used placeholder values, we do not need to do
                # anything.

                # Add query index, increment references to accommodate query.
                stimulus_set_idx = np.hstack(
                    [np.zeros([n_outcome, 1], dtype=int), outcome_idx + 1]
                )

                curr_stimulus_set_copy = stimulus_set_flat[trial_locs, :]
                curr_stimulus_set_expand = np.full(
                    [n_trial_config, self.max_n_reference + 1, max_n_outcome],
                    self.placeholder, dtype=np.int32
                )
                for i_outcome in range(n_outcome):
                    curr_stimulus_set_idx = stimulus_set_idx[i_outcome, :]
                    # Append placeholder indices.
                    curr_idx = np.hstack([
                        curr_stimulus_set_idx,
                        np.arange(
                            np.max(curr_stimulus_set_idx) + 1,
                            self.max_n_reference + 1
                        )
                    ])
                    curr_stimulus_set_expand[:, :, i_outcome] = (
                        curr_stimulus_set_copy[:, curr_idx]
                    )
                stimulus_set_flat_expand[trial_locs] = curr_stimulus_set_expand

        stimulus_set_expand = np.reshape(
            stimulus_set_flat_expand,
            [
                self.n_sequence, self.max_timestep, self.max_n_reference + 1,
                max_n_outcome
            ]
        )
        return stimulus_set_expand

    def _check_n_reference(self, n_reference):
        """Check validity of `n_reference`.

        NOTE: At this point n_reference=-1 for trials that are timestep
        placeholders (i.e., completely empty trials).

        Raises:
            ValueError

        """
        # Restrict check to non-placeholder trials.
        bidx = np.not_equal(n_reference, -1)
        n_reference_not_empty = n_reference[bidx]

        if np.sum(np.less(n_reference_not_empty, 2)) > 0:
            raise ValueError(
                "The argument `stimulus_set` must contain at least three "
                "positive integers per a trial, i.e. one query and at "
                "least two reference stimuli."
            )

        # Floor at zero since timestep placeholders yield -1.
        n_reference = np.maximum(n_reference, 0)

        return n_reference

    def _check_n_select(self, n_select):
        """Check validity of `n_select`.

        Raises:
            ValueError

        """
        # Cast if necessary.
        n_select = n_select.astype(np.int32)

        # Check rank.
        if n_select.ndim == 1:
            # Assume passed in trials are all independent.
            n_select = np.expand_dims(n_select, axis=1)

        if n_select.ndim != 2:
            raise ValueError(
                "The argument `n_select` must be a rank 2 np.ndarray."
            )

        # Trim empty timesteps if necessary.
        n_select = n_select[:, 0:self.max_timestep]

        # Check shape agreement.
        if not (n_select.shape[0] == self.n_sequence):
            raise ValueError(
                "The argument `n_select` must have the same shape as the "
                "first two axes of the argument 'stimulus_set'."
            )

        # Check lowerbound support limit, but restrict to trials that
        # that actually have references.
        bidx = np.not_equal(self.n_reference, 0)
        bad_locs = n_select[bidx] < 1
        n_bad = np.sum(bad_locs)
        if n_bad != 0:
            raise ValueError(
                "The argument `n_select` contains integers less than 1. "
                "Found {0} bad trial(s).".format(n_bad)
            )
        # Check upperbound support limit.
        bad_locs = np.greater_equal(n_select[bidx], self.n_reference[bidx])
        n_bad = np.sum(bad_locs)
        if n_bad != 0:
            raise ValueError(
                "The argument `n_select` contains integers greater than "
                "or equal to the corresponding 'n_reference'. Found {0} bad "
                "trial(s).".format(n_bad)
            )

        return n_select

    def _check_n_timestep(self, n_timestep):
        """Check validity of `n_timestep`.

        Arguments:
            n_stimstep: A 1D np.ndarray.

        Raises:
            ValueError

        """
        if np.sum(np.equal(n_timestep, 0)) > 0:
            raise ValueError((
                "The argument `stimulus_set` must contain at least one "
                "valid timestep per sequence."))
        return n_timestep

    def _check_stimulus_set(self, stimulus_set):
        """Check validity of `stimulus_set`.

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
        if np.sum(np.less(stimulus_set, self.placeholder)) > 0:
            raise ValueError(
                "The argument `stimulus_set` must contain integers "
                "greater than or equal to 0."
            )

        if stimulus_set.ndim == 2:
            # Assume trials are independent and add singleton dimension for
            # `timestep`.
            stimulus_set = np.expand_dims(stimulus_set, axis=1)

        # Check shape.
        if not stimulus_set.ndim == 3:
            raise ValueError(
                "The argument `stimulus_set` must be a rank-2 or rank-3 "
                "ndarray with a shape corresponding to (n_sequence, "
                "n_timestep, n_stimuli_per_trial)."
            )

        # Check values are in int32 range.
        ii32 = np.iinfo(np.int32)
        if np.sum(np.greater(stimulus_set, ii32.max)) > 0:
            raise ValueError((
                "The argument `stimulus_set` must only contain integers "
                "in the int32 range."
            ))
        return stimulus_set.astype(np.int32)

    @classmethod
    def _config_attrs(cls):
        """Return attributes that govern trial configurations."""
        return ['n_reference', 'n_select']

    def _is_select(self, compress=False):
        """Indicate if a stimulus was selected.

        This method has two modes that return 2D arrays of different
        shapes.

        Arguments:
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
        max_n_select = np.max(self.n_select)
        for n_select in range(1, max_n_select + 1):
            locs = np.less_equal(n_select, self.n_select)
            is_select[locs, n_select] = True

        if compress:
            is_select = is_select[:, :, 1:max_n_select + 1]

        return is_select

    def _for_dataset(self, format='tf', timestep=True):
        """Prepare trial content data for dataset."""
        if format == 'tf':
            # Create appropriate `stimulus_set` for all possible outcomes.
            stimulus_set = self._stimulus_set_with_outcomes()
            # Expand `is_select` to add axis for outcomes.
            is_select = self._is_select(compress=False)
            is_select = np.expand_dims(is_select, axis=-1)

            if timestep is False:
                stimulus_set = unravel_timestep(stimulus_set)
                is_select = unravel_timestep(is_select)
            x = {
                'stimulus_set': tf.constant(stimulus_set, dtype=tf.int32),
                'is_select': tf.constant(is_select, dtype=tf.bool)
            }
        else:
            raise ValueError(
                "Unrecognized format '{0}'.".format(format)
            )
        return x

    def _save(self, grp):
        """Add relevant data to H5 group."""
        grp.create_dataset("class_name", data="RankSimilarity")
        grp.create_dataset("stimulus_set", data=self.stimulus_set)
        grp.create_dataset("n_select", data=self.n_select)
        return None

    @classmethod
    def _load(cls, grp):
        """Retrieve relevant datasets from group."""
        stimulus_set = grp['stimulus_set'][()]
        n_select = grp['n_select'][()]
        return cls(stimulus_set, n_select=n_select)

    @staticmethod
    def _possible_outcomes(n_reference, n_select):
        """Return the possible outcomes of a ranked trial.

        Arguments:
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
