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
"""Rank trials module.

On each similarity judgment trial, an agent judges the similarity
between a single query stimulus and multiple reference stimuli.

Classes:
    RankTrials: Abstract base class for 'Rank' trials.

"""

from abc import ABCMeta
from itertools import permutations

import numpy as np

from psiz.trials.similarity.similarity_trials import SimilarityTrials


class RankTrials(SimilarityTrials, metaclass=ABCMeta):
    """Abstract base class for rank-type trials."""

    def __init__(self, stimulus_set, n_select=None, is_ranked=None, mask_zero=False):
        """Initialize.

        Args:
            stimulus_set: An integer matrix containing indices that
                indicate the set of stimuli used in each trial. Each
                row indicates the stimuli used in one trial. The first
                column is the query stimulus. The remaining columns
                indicate reference stimuli. It is assumed that stimuli
                indices are composed of non-negative integers.
                shape = (n_trial, max(n_reference) + 1)
            n_select (optional): An integer array indicating the number
                of references selected in each trial. Values must be
                greater than zero but less than the number of
                references for the corresponding trial.
                shape = n_trial,)
            is_ranked (optional): A Boolean array indicating which
                trials require reference selections to be ranked.
                shape = (n_trial,)
            mask_zero (optional): See SimilarityTrials.

        """
        SimilarityTrials.__init__(self, stimulus_set, mask_zero=mask_zero)
        self.n_present = self._check_n_present(self.n_present)

        # Format stimulus set.
        self.n_reference = self.n_present - 1
        self.max_n_reference = np.amax(self.n_reference)
        self.stimulus_set = self.stimulus_set[:, 0 : self.max_n_reference + 1]

        if n_select is None:
            n_select = np.ones((self.n_trial), dtype=np.int32)
        else:
            n_select = self._check_n_select(n_select)
        self.n_select = n_select

        if is_ranked is None:
            is_ranked = np.full((self.n_trial), True)
        else:
            is_ranked = self._check_is_ranked(is_ranked)
        self.is_ranked = is_ranked

    def _check_n_present(self, n_present):
        """Check the argument `n_present`.

        Valid rank similarity trials must have at least three stimuli
        (one query and two references).

        Returns:
            n_present: An integer array indicating the number of
                stimuli present in each trial.
                shape = [n_trial, 1]

        Raises:
            ValueError

        """
        if np.sum(np.less(n_present, 3)) > 0:
            raise ValueError(
                (
                    "The argument `stimulus_set` must contain at least three "
                    "non-negative integers per a row, i.e. one query and at least "
                    "two reference stimuli per trial."
                )
            )
        return n_present

    def _check_n_select(self, n_select):
        """Check the argument `n_select`.

        Raises:
            ValueError

        """
        n_select = n_select.astype(np.int32)
        # Check shape agreement.
        if not (n_select.shape[0] == self.n_trial):
            raise ValueError(
                (
                    "The argument `n_select` must have the same length as the "
                    "number of rows in the argument 'stimulus_set'."
                )
            )
        # Check lowerbound support limit.
        bad_locs = n_select < 1
        n_bad = np.sum(bad_locs)
        if n_bad != 0:
            raise ValueError(
                (
                    "The argument `n_select` contains integers less than 1. "
                    "Found {0} bad trial(s)."
                ).format(n_bad)
            )
        # Check upperbound support limit.
        bad_locs = np.greater_equal(n_select, self.n_reference)
        n_bad = np.sum(bad_locs)
        if n_bad != 0:
            raise ValueError(
                (
                    "The argument `n_select` contains integers greater than "
                    "or equal to the corresponding 'n_reference'. Found {0} bad "
                    "trial(s)."
                ).format(n_bad)
            )
        return n_select

    def _check_is_ranked(self, is_ranked):
        """Check the argument `is_ranked`.

        Raises:
            ValueError

        """
        if not (is_ranked.shape[0] == self.n_trial):
            raise ValueError(
                (
                    "The argument `n_select` must have the same length as the "
                    "number of rows in the argument 'stimulus_set'."
                )
            )
        bad_locs = np.not_equal(is_ranked, True)
        n_bad = np.sum(bad_locs)
        if n_bad != 0:
            raise ValueError(
                (
                    "The unranked version is not implemented, Found {0} bad "
                    "trial(s)."
                ).format(n_bad)
            )
        return is_ranked

    def is_select(self, compress=False):
        """Indicate if a stimulus was selected.

        This method has two modes that return 2D arrays of different
        shapes.

        Assumes `stimulus_set` is in sorted order with chosen
        references first. The output of this method is combined with a
        `stimulus_set` that has been expanded for all possible
        outcomes.

        Returns:
            is_select: A 2D Boolean array indicating the stimuli that
                were selected. By default, this will be a 2D array that
                has the same shape as `stimulus_set`. See the
                `compress` option for non-default behavior.
                shape=(n_trial, n_max_reference + 1) if compress=False
                shape=(n_trial, n_max_select) if compress=True
            compress (optional): A Boolean indicating if the returned
                2D array should be compressed such that the first
                column corresponding to the query is removed, and any
                trailing columns with no selected stimuli are also
                removed. This results in a 2D array with a shape that
                implies the maximum number of selected references.

        """
        is_select = np.zeros(self.stimulus_set.shape, dtype=bool)
        max_n_select = np.max(self.n_select)

        # Iterate over columns, marking trials that have at least that
        # many references selected. Start at 1 since query is not
        # selected.
        for n_select in range(1, max_n_select + 1):
            locs = np.less_equal(n_select, self.n_select)
            is_select[locs, n_select] = True

        if compress:
            is_select = is_select[:, 1 : max_n_select + 1]

        return is_select

    def all_outcomes(self):
        """Inflate stimulus set for all possible outcomes."""
        outcome_idx_list = self.outcome_idx_list
        n_outcome_list = self.config_list["n_outcome"].values
        max_n_outcome = np.max(n_outcome_list)
        n_config = self.config_list.shape[0]

        # Can use mask value regardless of whether `mask_zero=True` since if
        # not True, all cells will be updated.
        stimulus_set_expand = np.full(
            [self.n_trial, self.max_n_reference + 1, max_n_outcome],
            self._mask_value,
            dtype=np.int32,
        )
        for i_config in range(n_config):
            # Identify relevant trials.
            trial_locs = self.config_idx == i_config
            n_trial_config = np.sum(trial_locs)

            outcome_idx = outcome_idx_list[i_config]
            n_outcome = outcome_idx.shape[0]

            # Assemble relative indices (not absolute stimuli indices) for
            # possible outcomes. The query stimulus stays in the same spot
            # regardless of outcome, so we append a column of zeros to the
            # possible orderings of the references. The indices into the
            # reference positions are incremented by one to take into account
            # the query.
            stimulus_set_rel_idx = np.hstack(
                [np.zeros([n_outcome, 1], dtype=int), outcome_idx + 1]
            )
            # Grab absolute stimuli indices associated with current trial
            # configuration.
            curr_stimulus_set_copy = self.stimulus_set[trial_locs, :]
            # Initialize array taking into account max outcome configuration.
            curr_stimulus_set_expand = np.full(
                [n_trial_config, self.max_n_reference + 1, max_n_outcome],
                self._mask_value,
                dtype=int,
            )
            for i_outcome in range(n_outcome):
                curr_stimulus_set_idx = stimulus_set_rel_idx[i_outcome, :]
                # Append placeholder indices.
                curr_idx = np.hstack(
                    [
                        curr_stimulus_set_idx,
                        np.arange(
                            np.max(curr_stimulus_set_idx) + 1, self.max_n_reference + 1
                        ),
                    ]
                )
                # Convert relative indices to absolute indices.
                curr_stimulus_set_expand[:, :, i_outcome] = curr_stimulus_set_copy[
                    :, curr_idx
                ]
            stimulus_set_expand[trial_locs] = curr_stimulus_set_expand
        return stimulus_set_expand

    @staticmethod
    def _possible_rank_outcomes(trial_configuration):
        """Return the possible outcomes of a ranked trial.

        Args:
            trial_configuration: A trial configuration Pandas Series.

        Returns:
            An 2D array indicating all possible outcomes where the
                values indicate indices of the reference stimuli. Each
                row corresponds to one outcome. Note the indices refer
                to references only and does not include an index for
                the query. Also note that the unpermuted index is
                returned first.

        """
        n_reference = int(trial_configuration["n_reference"])
        n_select = int(trial_configuration["n_select"])

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
